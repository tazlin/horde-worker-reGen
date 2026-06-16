"""Tests for background model downloads and on-disk availability gating."""

from __future__ import annotations

import asyncio
import queue
import time
from unittest.mock import Mock

from horde_worker_regen.process_management.action_ledger import ActionLedger
from horde_worker_regen.process_management.download_process import DOWNLOAD_PROCESS_ID
from horde_worker_regen.process_management.fake_worker_processes import FakeDownloadProcess
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.messages import (
    HordeDownloadAvailabilityMessage,
    HordeDownloadControlMessage,
    HordeProcessMessage,
)
from horde_worker_regen.process_management.model_availability import ModelAvailability
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_mock_bridge_data,
    make_test_model_metadata,
    make_test_runtime_config,
    make_testable_process_manager,
)


def _drain_availability(message_queue: queue.Queue) -> list[HordeDownloadAvailabilityMessage]:  # type: ignore[type-arg]
    """Pull every queued ``HordeDownloadAvailabilityMessage`` off a (stdlib) queue."""
    messages: list[HordeDownloadAvailabilityMessage] = []
    while not message_queue.empty():
        item = message_queue.get_nowait()
        if isinstance(item, HordeDownloadAvailabilityMessage):
            messages.append(item)
    return messages


def _availability_message(available: list[str], **extra: object) -> HordeDownloadAvailabilityMessage:
    return HordeDownloadAvailabilityMessage(
        process_id=DOWNLOAD_PROCESS_ID,
        process_launch_identifier=0,
        info="test",
        available_model_names=available,
        **extra,  # type: ignore[arg-type]
    )


class TestModelAvailability:
    """The on-disk availability holder."""

    def test_unknown_until_first_report(self) -> None:
        """Availability is unknown until the first report, treating all models as present."""
        availability = ModelAvailability()
        assert availability.is_known is False
        assert availability.present is None
        # While unknown, everything is treated as present so legacy workers are unaffected.
        assert availability.is_present("anything") is True
        assert availability.filter_present({"a", "b"}) == {"a", "b"}

    def test_known_after_update_filters_to_present(self) -> None:
        """Once reported, only the present set is considered available."""
        availability = ModelAvailability()
        availability.update(present={"a"}, currently_downloading="b", pending=("b",), failed=())
        assert availability.is_known is True
        assert availability.present == {"a"}
        assert availability.is_present("a") is True
        assert availability.is_present("b") is False
        assert availability.filter_present({"a", "b", "c"}) == {"a"}
        assert availability.currently_downloading == "b"
        assert availability.pending == ("b",)

    def test_empty_present_filters_to_nothing(self) -> None:
        """An empty present set means no models are available."""
        availability = ModelAvailability()
        availability.update(present=set(), currently_downloading=None, pending=(), failed=())
        assert availability.is_known is True
        assert availability.filter_present({"a", "b"}) == set()

    def test_status_and_scan_complete_round_trip(self) -> None:
        """An early (scanning) report is known but not scan-complete, and carries the rich status."""
        availability = ModelAvailability()
        status = DownloadStatusSnapshot(phase=DownloadPhase.SCANNING)
        availability.update(
            present=set(),
            currently_downloading=None,
            pending=(),
            failed=(),
            status=status,
            scan_complete=False,
        )
        assert availability.is_known is True
        assert availability.scan_complete is False
        assert availability.status is status

        availability.update(present={"a"}, currently_downloading=None, pending=(), failed=())
        assert availability.scan_complete is True


class TestSelectModelsForPopGating:
    """``_select_models_for_pop`` must only advertise models that are on disk."""

    def _bridge(self, **overrides: object) -> Mock:
        return make_mock_bridge_data(**overrides)

    def test_no_availability_is_unchanged(self) -> None:
        """With no availability holder, every configured model is advertised."""
        bridge = self._bridge(image_models_to_load=["a", "b"])
        models = _select_models_for_pop(
            bridge,  # type: ignore[arg-type]
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
            model_availability=None,
        )
        assert models == {"a", "b"}

    def test_unknown_availability_is_unchanged(self) -> None:
        """An unreported holder advertises every configured model."""
        bridge = self._bridge(image_models_to_load=["a", "b"])
        models = _select_models_for_pop(
            bridge,  # type: ignore[arg-type]
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
            model_availability=ModelAvailability(),
        )
        assert models == {"a", "b"}

    def test_filters_to_present_models(self) -> None:
        """Only on-disk models are advertised."""
        availability = ModelAvailability()
        availability.update(present={"a"}, currently_downloading="b", pending=("b",), failed=())
        bridge = self._bridge(image_models_to_load=["a", "b"])
        models = _select_models_for_pop(
            bridge,  # type: ignore[arg-type]
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
            model_availability=availability,
        )
        assert models == {"a"}

    def test_returns_none_when_nothing_present(self) -> None:
        """No on-disk models means no pop is attempted."""
        availability = ModelAvailability()
        availability.update(present=set(), currently_downloading="a", pending=(), failed=())
        bridge = self._bridge(image_models_to_load=["a"])
        models = _select_models_for_pop(
            bridge,  # type: ignore[arg-type]
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=1,
            last_pop_had_no_jobs=False,
            model_availability=availability,
        )
        assert models is None

    def test_custom_models_bypass_disk_gating(self) -> None:
        """Custom models are advertised regardless of disk gating."""
        availability = ModelAvailability()
        availability.update(present=set(), currently_downloading=None, pending=(), failed=())
        bridge = self._bridge(image_models_to_load=["a"], custom_models=[{"name": "my_custom"}])
        models = _select_models_for_pop(
            bridge,  # type: ignore[arg-type]
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=1,
            last_pop_had_no_jobs=False,
            model_availability=availability,
        )
        assert models == {"my_custom"}


class TestManagerDownloadHandling:
    """The manager's reaction to download-process availability reports."""

    def _manager_in_download_mode(self, **bridge_overrides: object) -> Mock:
        manager = make_testable_process_manager(**bridge_overrides)  # type: ignore
        manager._enable_background_downloads = True
        manager._download_wait_started = time.time()
        manager._process_lifecycle = Mock()
        return manager  # type: ignore[return-value]

    def test_first_report_requests_missing_and_starts_inference(self) -> None:
        """The first report requests the missing models and starts inference once one is present."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._on_download_availability(_availability_message(["a"]))

        assert manager._model_availability.present == {"a"}
        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == ["b"]
        assert kwargs["download_aux"] is True
        manager._process_lifecycle.start_inference_processes.assert_called_once()
        assert manager._inference_processes_started is True

    def test_empty_report_defers_inference_but_still_requests(self) -> None:
        """An empty first report requests downloads but defers inference startup."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._on_download_availability(_availability_message([]))

        manager._process_lifecycle.request_downloads.assert_called_once()
        assert sorted(manager._process_lifecycle.request_downloads.call_args.args[0]) == ["a", "b"]
        manager._process_lifecycle.start_inference_processes.assert_not_called()
        assert manager._inference_processes_started is False

    def test_all_present_skips_request_and_starts_inference(self) -> None:
        """When everything is already present, no download is requested and inference starts."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        manager._on_download_availability(_availability_message(["a"]))

        manager._process_lifecycle.request_downloads.assert_not_called()
        manager._process_lifecycle.start_inference_processes.assert_called_once()

    def test_subsequent_reports_do_not_re_request_or_double_start(self) -> None:
        """Later reports neither re-request downloads nor restart inference."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._on_download_availability(_availability_message([]))
        manager._on_download_availability(_availability_message(["a"]))
        manager._on_download_availability(_availability_message(["a", "b"]))

        # The download request is only sent once (on the first report).
        manager._process_lifecycle.request_downloads.assert_called_once()
        # Inference starts exactly once, when the first model lands.
        manager._process_lifecycle.start_inference_processes.assert_called_once()
        assert manager._model_availability.present == {"a", "b"}

    def test_pre_scan_report_does_not_request_or_start(self) -> None:
        """An early scanning report (scan_complete False) defers both the request and inference."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        scanning = DownloadStatusSnapshot(phase=DownloadPhase.SCANNING)
        manager._on_download_availability(_availability_message([], scan_complete=False, status=scanning))

        manager._process_lifecycle.request_downloads.assert_not_called()
        manager._process_lifecycle.start_inference_processes.assert_not_called()
        assert manager._initial_download_requested is False

        # The first authoritative (scan-complete) report then drives the request and startup.
        manager._on_download_availability(_availability_message(["a"]))
        manager._process_lifecycle.request_downloads.assert_called_once()
        manager._process_lifecycle.start_inference_processes.assert_called_once()


class TestDispatcherRoutesDownloadMessages:
    """The dispatcher must route download-process messages without raising (they are out of map)."""

    def _make_dispatcher(self, handler: object) -> tuple[MessageDispatcher, queue.Queue]:  # type: ignore[type-arg]
        message_queue: queue.Queue = queue.Queue()  # type: ignore[type-arg]

        async def _noop_unload(_info: object) -> None:
            return None

        dispatcher = MessageDispatcher(
            process_map=ProcessMap({}),
            horde_model_map=HordeModelMap(root={}),
            job_tracker=JobTracker(),
            process_message_queue=message_queue,  # type: ignore[arg-type]
            runtime_config=make_test_runtime_config(),
            model_metadata=make_test_model_metadata(),
            action_ledger=ActionLedger(),
            on_unload_vram=_noop_unload,  # type: ignore[arg-type]
            state=WorkerState(),
        )
        dispatcher.set_download_availability_handler(handler)  # type: ignore[arg-type]
        return dispatcher, message_queue

    def test_availability_message_routed_to_handler(self) -> None:
        """Availability messages from the download pid reach the registered handler."""
        handler = Mock()
        dispatcher, message_queue = self._make_dispatcher(handler)
        message = _availability_message(["a"])
        message_queue.put(message)

        asyncio.run(dispatcher.receive_and_handle_process_messages())

        handler.assert_called_once_with(message)

    def test_unknown_download_message_does_not_raise(self) -> None:
        """Non-availability messages from the download pid are dropped, not errored."""
        # A non-availability message from the download pid must be dropped, not treated as an
        # unknown-process error (which would raise for any pid missing from the process map).
        dispatcher, message_queue = self._make_dispatcher(Mock())
        message_queue.put(
            HordeProcessMessage(process_id=DOWNLOAD_PROCESS_ID, process_launch_identifier=0, info="stray"),
        )

        asyncio.run(dispatcher.receive_and_handle_process_messages())


class TestFakeDownloadProcessProtocol:
    """The fake download process must speak the same availability protocol as the real one."""

    def _make_process(self, scripted_present: list[str]) -> tuple[FakeDownloadProcess, queue.Queue]:  # type: ignore[type-arg]
        message_queue: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        process = FakeDownloadProcess(
            process_id=DOWNLOAD_PROCESS_ID,
            process_message_queue=message_queue,  # type: ignore[arg-type]
            pipe_connection=Mock(),
            disk_lock=Mock(),
            process_launch_identifier=0,
            scripted_present=scripted_present,
        )
        return process, message_queue

    def test_reports_initial_present_set(self) -> None:
        """The fake reports its scripted present set on startup."""
        _process, message_queue = self._make_process(["a"])
        availability = _drain_availability(message_queue)
        assert availability, "expected an initial availability report"
        assert availability[-1].available_model_names == ["a"]

    def test_download_request_marks_model_present(self) -> None:
        """A download request makes the model present in a later availability report."""
        process, message_queue = self._make_process(["a"])
        _drain_availability(message_queue)

        process._receive_and_handle_control_message(HordeDownloadControlMessage(model_names=["b"]))
        process.worker_cycle()

        availability = _drain_availability(message_queue)
        assert availability, "expected availability reports after the download"
        assert "b" in availability[-1].available_model_names

    def test_pause_holds_downloads_until_resumed(self) -> None:
        """While paused the queue is held; resuming lets the model download."""
        process, message_queue = self._make_process(["a"])
        process._receive_and_handle_control_message(
            HordeDownloadControlMessage(model_names=["b"], set_paused=True),
        )
        process.worker_cycle()

        held = _drain_availability(message_queue)[-1]
        assert "b" not in held.available_model_names
        assert held.status is not None and held.status.paused is True

        process._receive_and_handle_control_message(HordeDownloadControlMessage(set_paused=False))
        process.worker_cycle()
        resumed = _drain_availability(message_queue)[-1]
        assert "b" in resumed.available_model_names

    def test_rate_limit_is_reflected_in_status(self) -> None:
        """A set-rate-limit control is reflected in the emitted status snapshot."""
        process, message_queue = self._make_process(["a"])
        process._receive_and_handle_control_message(HordeDownloadControlMessage(set_rate_limit_kbps=4096))
        status = _drain_availability(message_queue)[-1].status
        assert status is not None and status.rate_limit_kbps == 4096

        process._receive_and_handle_control_message(HordeDownloadControlMessage(set_rate_limit_kbps=0))
        status = _drain_availability(message_queue)[-1].status
        assert status is not None and status.rate_limit_kbps is None


class TestDownloadMessageRoundTrips:
    """The download status/plan and supervisor control messages must serialize losslessly."""

    def test_status_snapshot_round_trip(self) -> None:
        """A populated DownloadStatusSnapshot survives a model_dump/model_validate round trip."""
        status = DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name="Flux",
                feature="image model",
                target_dir="models/compvis",
                downloaded_bytes=10,
                total_bytes=40,
            ),
            pending=[DownloadItem(model_name="next", feature="image model", size_bytes=5)],
            failures=[DownloadFailure(model_name="bad", feature="LoRa", reason="disk full")],
            paused=True,
            rate_limit_kbps=2048,
        )
        restored = DownloadStatusSnapshot.model_validate(status.model_dump())
        assert restored == status
        assert restored.current is not None and restored.current.percent == 25.0

    def test_supervisor_rate_limit_command_round_trip(self) -> None:
        """The SET_DOWNLOAD_RATE_LIMIT command carries its KB/s value through serialization."""
        message = SupervisorControlMessage(
            command=SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT,
            download_rate_limit_kbps=3000,
        )
        restored = SupervisorControlMessage.model_validate(message.model_dump())
        assert restored.command is SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT
        assert restored.download_rate_limit_kbps == 3000
