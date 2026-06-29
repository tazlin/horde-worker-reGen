"""Tests for background model downloads and on-disk availability gating."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
import types
from collections import Counter
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import Mock

import pytest

from horde_worker_regen.model_download_core import ChunkPacer, CompVisLike, DownloadAborted
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import (
    HordeDownloadAvailabilityMessage,
    HordeDownloadControlMessage,
    HordeProcessMessage,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.download_scheduler import DownloadKind, DownloadTask
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.simulation.fake_worker_processes import FakeDownloadProcess
from horde_worker_regen.process_management.workers.download_process import (
    DOWNLOAD_PROCESS_ID,
    FEATURE_IMAGE_MODEL,
    FEATURE_SAFETY,
    HordeDownloadProcess,
    _TaskRuntime,
)
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_test_model_metadata,
    make_test_runtime_config,
    make_testable_process_manager,
)


class _DrainableQueue(Protocol):
    """The minimal queue surface :func:`_drain_availability` reads; stdlib and multiprocessing queues both fit."""

    def empty(self) -> bool: ...

    def get_nowait(self) -> object: ...


def _drain_availability(message_queue: _DrainableQueue) -> list[HordeDownloadAvailabilityMessage]:
    """Pull every queued ``HordeDownloadAvailabilityMessage`` off a stdlib or multiprocessing queue."""
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

    def test_background_download_active_requires_downloading_current(self) -> None:
        """Only an active current download suppresses LoRA pops."""
        availability = ModelAvailability()
        assert availability.background_download_active is False

        availability.update(
            present={"a"},
            currently_downloading=None,
            pending=("b",),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.PAUSED, pending=[DownloadItem(model_name="b", feature="image model")]
            ),
        )
        assert availability.background_download_active is False

        availability.update(
            present={"a"},
            currently_downloading="b",
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                current=CurrentDownloadStatus(model_name="b", feature="image model", target_dir=""),
            ),
        )
        assert availability.background_download_active is True

    def test_safety_present_round_trip(self) -> None:
        """The required-safety-models flag defaults False and round-trips through an update."""
        availability = ModelAvailability()
        assert availability.safety_present is False
        availability.update(present={"a"}, currently_downloading=None, pending=(), failed=())
        assert availability.safety_present is False
        availability.update(
            present={"a"},
            currently_downloading=None,
            pending=(),
            failed=(),
            safety_present=True,
        )
        assert availability.safety_present is True


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
        manager._download_coordinator._enable_background_downloads = True
        manager._download_coordinator.download_wait_started = time.time()
        manager._process_lifecycle = Mock(download_process_info=None)
        manager._process_lifecycle._num_process_recoveries = 0
        manager._download_coordinator._process_lifecycle = manager._process_lifecycle
        return manager  # type: ignore[return-value]

    def test_first_report_requests_missing_and_starts_inference(self) -> None:
        """The first report requests the missing models and starts inference once one is present."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._download_coordinator.on_download_availability(_availability_message(["a"]))

        assert manager._model_availability.present == {"a"}
        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == ["b"]
        assert kwargs["download_aux"] is True
        manager._process_lifecycle.start_inference_processes.assert_called_once()
        assert manager._download_coordinator.inference_processes_started is True

    def test_empty_report_defers_inference_but_still_requests(self) -> None:
        """An empty first report requests downloads but defers inference startup."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._download_coordinator.on_download_availability(_availability_message([]))

        manager._process_lifecycle.request_downloads.assert_called_once()
        assert sorted(manager._process_lifecycle.request_downloads.call_args.args[0]) == ["a", "b"]
        manager._process_lifecycle.start_inference_processes.assert_not_called()
        assert manager._download_coordinator.inference_processes_started is False

    def test_all_present_skips_request_and_starts_inference(self) -> None:
        """When everything is already present, no download is requested and inference starts."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        manager._download_coordinator.on_download_availability(_availability_message(["a"]))

        manager._process_lifecycle.request_downloads.assert_not_called()
        manager._process_lifecycle.start_inference_processes.assert_called_once()

    def test_subsequent_reports_do_not_re_request_or_double_start(self) -> None:
        """Later reports neither re-request downloads nor restart inference."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        manager._download_coordinator.on_download_availability(_availability_message([]))
        manager._download_coordinator.on_download_availability(_availability_message(["a"]))
        manager._download_coordinator.on_download_availability(_availability_message(["a", "b"]))

        # The download request is only sent once (on the first report).
        manager._process_lifecycle.request_downloads.assert_called_once()
        # Inference starts exactly once, when the first model lands.
        manager._process_lifecycle.start_inference_processes.assert_called_once()
        assert manager._model_availability.present == {"a", "b"}

    def test_pre_scan_report_does_not_request_or_start(self) -> None:
        """An early scanning report (scan_complete False) defers both the request and inference."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        scanning = DownloadStatusSnapshot(phase=DownloadPhase.SCANNING)
        manager._download_coordinator.on_download_availability(
            _availability_message([], scan_complete=False, status=scanning)
        )

        manager._process_lifecycle.request_downloads.assert_not_called()
        manager._process_lifecycle.start_inference_processes.assert_not_called()
        assert manager._download_coordinator.initial_download_requested is False

        # The first authoritative (scan-complete) report then drives the request and startup.
        manager._download_coordinator.on_download_availability(_availability_message(["a"]))
        manager._process_lifecycle.request_downloads.assert_called_once()
        manager._process_lifecycle.start_inference_processes.assert_called_once()

    def test_snapshot_marks_lora_blocked_by_active_download(self) -> None:
        """The supervisor snapshot explains temporary LoRA pop suppression."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"], allow_lora=True)
        manager._download_coordinator.on_download_availability(
            _availability_message(
                ["a"],
                status=DownloadStatusSnapshot(
                    phase=DownloadPhase.DOWNLOADING,
                    current=CurrentDownloadStatus(model_name="b", feature="image model", target_dir=""),
                ),
            )
        )

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.config.allow_lora is True
        assert snapshot.config.effective_allow_lora is False
        assert snapshot.lora_pops_blocked_by_downloads is True

    def test_snapshot_does_not_mark_lora_blocked_when_lora_disabled(self) -> None:
        """Active downloads do not imply a temporary LoRA override when LoRA is off by config."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"], allow_lora=False)
        manager._download_coordinator.on_download_availability(
            _availability_message(
                ["a"],
                status=DownloadStatusSnapshot(
                    phase=DownloadPhase.DOWNLOADING,
                    current=CurrentDownloadStatus(model_name="b", feature="image model", target_dir=""),
                ),
            )
        )

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.config.allow_lora is False
        assert snapshot.config.effective_allow_lora is False
        assert snapshot.lora_pops_blocked_by_downloads is False


class TestConfigReloadTriggersDownloads:
    """A config change that adds a model must background-download it without a restart."""

    def _manager_in_download_mode(self, **bridge_overrides: object) -> Mock:
        manager = make_testable_process_manager(**bridge_overrides)  # type: ignore
        manager._enable_background_downloads = True
        manager._download_coordinator._enable_background_downloads = True
        manager._process_lifecycle = Mock()
        manager._download_coordinator._process_lifecycle = manager._process_lifecycle
        # Past the one-shot startup trigger, so only the reload path can drive a new request.
        manager._download_coordinator.initial_download_requested = True
        return manager  # type: ignore[return-value]

    def _mark_present(self, manager: Mock, present: set[str]) -> None:
        manager._model_availability.update(present=present, currently_downloading=None, pending=(), failed=())

    def test_reload_requests_newly_configured_missing_model(self) -> None:
        """Adding a model to the config fetches just the new one, without the heavy aux pass."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})

        new_bridge = make_mock_bridge_data(image_models_to_load=["a", "b"], dry_run_skip_inference=True)
        manager._apply_reloaded_bridge_data(new_bridge)

        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == ["b"]
        assert kwargs["download_aux"] is False

    def test_reload_with_all_models_present_requests_nothing(self) -> None:
        """A reload that adds no missing model triggers no download."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(image_models_to_load=["a"], dry_run_skip_inference=True),
        )

        manager._process_lifecycle.request_downloads.assert_not_called()

    def test_helper_no_op_when_background_downloads_disabled(self) -> None:
        """Without a download process, the trigger is a silent no-op (no spurious request)."""
        manager = make_testable_process_manager(image_models_to_load=["a", "b"])  # type: ignore
        manager._enable_background_downloads = False
        manager._process_lifecycle = Mock()
        self._mark_present(manager, {"a"})

        manager._download_coordinator.reconcile_downloads(run_aux_if_incomplete=False)

        manager._process_lifecycle.request_downloads.assert_not_called()

    def test_reload_removing_a_model_sends_authoritative_desired_set(self) -> None:
        """Dropping a model from config sends the now-authoritative set so its download is stopped."""
        manager = self._manager_in_download_mode(image_models_to_load=["a", "b"])
        # Both present on disk: the reload adds nothing, so without the desired-set reconcile the old
        # short-circuit would send nothing and leave b downloading.
        self._mark_present(manager, {"a", "b"})

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(image_models_to_load=["a"], dry_run_skip_inference=True),
        )

        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == []  # nothing new to fetch
        assert kwargs["desired_image_models"] == ["a"]

    def test_reload_adding_a_model_also_carries_desired_set(self) -> None:
        """An add still attaches the authoritative set (harmless dedup) alongside the missing model."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(image_models_to_load=["a", "b"], dry_run_skip_inference=True),
        )

        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == ["b"]
        assert kwargs["desired_image_models"] == ["a", "b"]

    def test_reload_changing_an_aux_flag_forwards_gating_live(self) -> None:
        """Toggling a gating flag (e.g. allow_controlnet) is forwarded to the download process live, no restart."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})
        # Normalize to a known baseline, then flip exactly one gating flag. (purge is pinned because the mock
        # bridge data leaves it an auto-Mock that would never compare equal.)
        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(
                image_models_to_load=["a"],
                allow_controlnet=False,
                purge_loras_on_download=False,
                dry_run_skip_inference=True,
            ),
        )
        manager._process_lifecycle.set_download_gating.reset_mock()

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(
                image_models_to_load=["a"],
                allow_controlnet=True,
                purge_loras_on_download=False,
                dry_run_skip_inference=True,
            ),
        )

        manager._process_lifecycle.set_download_gating.assert_called_once()
        assert manager._process_lifecycle.set_download_gating.call_args.kwargs["allow_controlnet"] is True
        # The disruptive restart is gone: a newly-enabled category is fetched by the live aux re-arm.
        manager._process_lifecycle.restart_download_process.assert_not_called()

    def test_reload_without_download_flag_change_does_not_forward_gating(self) -> None:
        """A reload that leaves the gating flags unchanged forwards nothing (and never restarts)."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})
        # Establish the baseline flags, then reload with the same flags (only the model set changes).
        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(
                image_models_to_load=["a"],
                allow_controlnet=True,
                purge_loras_on_download=False,
                dry_run_skip_inference=True,
            ),
        )
        manager._process_lifecycle.set_download_gating.reset_mock()

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(
                image_models_to_load=["a", "b"],
                allow_controlnet=True,
                purge_loras_on_download=False,
                dry_run_skip_inference=True,
            ),
        )

        manager._process_lifecycle.set_download_gating.assert_not_called()
        manager._process_lifecycle.restart_download_process.assert_not_called()


class TestDownloadsOnlyMode:
    """The download-only posture: pre-fetch models with the GPU uncommitted, then GO_LIVE."""

    def _manager(self, **bridge_overrides: object) -> Mock:
        manager = make_testable_process_manager(**bridge_overrides)  # type: ignore
        manager._enable_background_downloads = True
        manager._download_coordinator._enable_background_downloads = True
        manager._process_lifecycle = Mock()
        manager._download_coordinator._process_lifecycle = manager._process_lifecycle
        manager._download_coordinator.initial_download_requested = True
        return manager  # type: ignore[return-value]

    def _mark_present(self, manager: Mock, present: set[str]) -> None:
        manager._model_availability.update(present=present, currently_downloading=None, pending=(), failed=())

    def test_enter_hold_sets_state_and_starts_download_process(self) -> None:
        """Entering the hold flags the state and ensures the download process is running."""
        manager = self._manager(image_models_to_load=["a"])
        manager._download_coordinator.enter_downloads_only_hold()

        assert manager._state.downloads_only_hold is True
        manager._process_lifecycle.start_download_process.assert_called_once()

    def test_hold_blocks_inference_start_even_with_a_model_present(self) -> None:
        """While held, inference does not start despite a present, scanned model."""
        manager = self._manager(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})
        manager._state.downloads_only_hold = True

        manager._download_coordinator.maybe_start_inference_processes()

        manager._process_lifecycle.start_inference_processes.assert_not_called()

    def test_go_live_clears_hold_and_starts_inference(self) -> None:
        """GO_LIVE lifts the hold and brings inference up (a model is present)."""
        manager = self._manager(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})
        manager._state.downloads_only_hold = True

        manager._download_coordinator.leave_downloads_only_hold()

        assert manager._state.downloads_only_hold is False
        manager._process_lifecycle.start_inference_processes.assert_called_once()

    def test_alchemist_only_worker_starts_inference_without_image_models(self) -> None:
        """An alchemist-only worker (no image models) starts inference once the on-disk scan completes."""
        manager = self._manager(image_models_to_load=[], alchemist=True)
        self._mark_present(manager, set())  # scan complete, zero image models present

        manager._download_coordinator.maybe_start_inference_processes()

        manager._process_lifecycle.start_inference_processes.assert_called_once()

    def test_no_models_and_no_alchemist_does_not_start_inference(self) -> None:
        """With neither image models nor alchemist there is nothing to serve, so inference does not start."""
        manager = self._manager(image_models_to_load=[], alchemist=False)
        self._mark_present(manager, set())

        manager._download_coordinator.maybe_start_inference_processes()

        manager._process_lifecycle.start_inference_processes.assert_not_called()

    def test_on_demand_download_adds_to_desired_set_and_kicks_aux(self) -> None:
        """A picker request joins the one desired set and fetches what is missing, optionally running aux."""
        manager = self._manager(image_models_to_load=["a"])
        # The configured model is already on disk (the picker is used after the first scan), so only the
        # picker's own models are still to fetch.
        self._mark_present(manager, {"a"})

        manager._download_coordinator.download_models_on_demand(["x", "y"], include_aux=True)

        manager._process_lifecycle.request_downloads.assert_called_once()
        args, kwargs = manager._process_lifecycle.request_downloads.call_args
        assert args[0] == ["x", "y"]
        assert kwargs["download_aux"] is True
        # Declarative: the authoritative desired set now carries the picker's models alongside config, so a
        # later config reconcile keeps fetching them instead of pruning them.
        assert kwargs["desired_image_models"] == ["a", "x", "y"]

    def test_picker_additions_survive_a_config_reconcile(self) -> None:
        """The bug fix: a config reconcile after a picker add must not prune the picker's models.

        The former additive picker sent no desired set, so the very next config reconcile (which sends the
        configured-only authoritative set) cancelled the picker's still-queued downloads. Now both share one
        desired set, so the picker's models stay in the authoritative set the config reconcile sends.
        """
        manager = self._manager(image_models_to_load=["a"])
        self._mark_present(manager, {"a"})

        manager._download_coordinator.download_models_on_demand(["x"], include_aux=False)
        # A subsequent config-driven reconcile (e.g. the reload path) with no config change at all.
        manager._process_lifecycle.request_downloads.reset_mock()
        manager._download_coordinator.reconcile_downloads(run_aux_if_incomplete=False)

        manager._process_lifecycle.request_downloads.assert_called_once()
        kwargs = manager._process_lifecycle.request_downloads.call_args.kwargs
        assert kwargs["desired_image_models"] == ["a", "x"]
        assert manager._process_lifecycle.request_downloads.call_args.args[0] == ["x"]


class TestDownloadEntryPointSignatures:
    """The download entry points must forward every download-process constructor kwarg.

    Regression guard: a constructor kwarg added to the lifecycle's launch dict but not to the entry-point
    function makes the spawned process die with ``TypeError: ... unexpected keyword argument`` before it
    can load managers or report availability. With no inference process started yet, the empty process map
    then makes the hung-detector falsely declare "all processes unresponsive".
    """

    def test_real_entry_point_forwards_every_constructor_kwarg(self) -> None:
        """``start_download_process`` accepts (and forwards) every keyword-only HordeDownloadProcess arg."""
        import inspect

        from horde_worker_regen.process_management.worker_entry_points import start_download_process
        from horde_worker_regen.process_management.workers.download_process import HordeDownloadProcess

        ctor_kwargs = {
            name
            for name, param in inspect.signature(HordeDownloadProcess.__init__).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        entry_kwargs = {
            name
            for name, param in inspect.signature(start_download_process).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        missing = ctor_kwargs - entry_kwargs
        assert not missing, f"download entry point does not forward constructor kwargs: {missing}"

    def test_fake_entry_point_is_signature_compatible_with_the_real_one(self) -> None:
        """The fake receives the same launch kwargs, so it must accept every real entry-point keyword."""
        import inspect

        from horde_worker_regen.process_management.simulation.fake_worker_processes import start_fake_download_process
        from horde_worker_regen.process_management.worker_entry_points import start_download_process

        real_kwargs = {
            name
            for name, param in inspect.signature(start_download_process).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        fake_kwargs = {
            name
            for name, param in inspect.signature(start_fake_download_process).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        missing = real_kwargs - fake_kwargs
        assert not missing, f"fake download entry point is missing kwargs the parent forwards: {missing}"


class TestManagerSafetyDeferral:
    """The manager defers the safety-process launch until the download process provides its models.

    Starting the safety process before the safety models (DeepDanbooru + CLIP, ~2.3GB) are on disk would
    make it download them synchronously in its constructor, which reads as a hung worker.
    """

    def _manager_in_download_mode(self, **bridge_overrides: object) -> Mock:
        manager = make_testable_process_manager(**bridge_overrides)  # type: ignore
        manager._enable_background_downloads = True
        manager._download_coordinator._enable_background_downloads = True
        manager._download_coordinator.download_wait_started = time.time()
        manager._process_lifecycle = Mock()
        manager._download_coordinator._process_lifecycle = manager._process_lifecycle
        return manager  # type: ignore[return-value]

    def test_safety_present_report_starts_safety(self) -> None:
        """A report that the safety models are present starts the safety process once."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        manager._download_coordinator.on_download_availability(
            _availability_message(["a"], safety_models_present=True)
        )

        manager._process_lifecycle.start_safety_processes.assert_called_once()
        assert manager._download_coordinator.safety_processes_started is True

    def test_safety_absent_and_not_yet_attempted_defers(self) -> None:
        """A transient post-scan report (ensure not yet attempted) must not trip the launch."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        downloading = DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(model_name="safety models", feature="safety models", target_dir=""),
        )
        manager._download_coordinator.on_download_availability(
            _availability_message(
                ["a"],
                safety_models_present=False,
                safety_models_attempted=False,
                status=downloading,
            ),
        )

        manager._process_lifecycle.start_safety_processes.assert_not_called()
        assert manager._download_coordinator.safety_processes_started is False

    def test_safety_started_when_attempted_without_success(self) -> None:
        """If the ensure finished without producing them, start safety to self-fetch/surface the error."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        idle = DownloadStatusSnapshot(phase=DownloadPhase.IDLE)
        manager._download_coordinator.on_download_availability(
            _availability_message(
                ["a"],
                safety_models_present=False,
                safety_models_attempted=True,
                status=idle,
            ),
        )

        manager._process_lifecycle.start_safety_processes.assert_called_once()
        assert manager._download_coordinator.safety_processes_started is True

    def test_safety_grace_fallback_when_no_report(self) -> None:
        """A download process that never reports cannot wedge startup past the grace window."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        manager._download_coordinator.download_wait_started = time.time() - (
            manager._download_coordinator.DOWNLOAD_STARTUP_GRACE_SECONDS + 1.0
        )

        manager._download_coordinator.maybe_start_safety_processes()

        manager._process_lifecycle.start_safety_processes.assert_called_once()
        assert manager._download_coordinator.safety_processes_started is True

    def test_safety_no_grace_fallback_before_window(self) -> None:
        """Before the grace window elapses (and with no report), the safety launch stays deferred."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])

        manager._download_coordinator.maybe_start_safety_processes()

        manager._process_lifecycle.start_safety_processes.assert_not_called()
        assert manager._download_coordinator.safety_processes_started is False

    def test_safety_starts_only_once(self) -> None:
        """Repeated present reports do not relaunch the safety pool."""
        manager = self._manager_in_download_mode(image_models_to_load=["a"])
        manager._download_coordinator.on_download_availability(
            _availability_message(["a"], safety_models_present=True)
        )
        manager._download_coordinator.on_download_availability(
            _availability_message(["a"], safety_models_present=True)
        )

        manager._process_lifecycle.start_safety_processes.assert_called_once()


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

    def test_desired_set_prunes_queue_and_drops_in_flight(self) -> None:
        """A model removed from the authoritative set is dropped from the fake's queue and in-flight slot."""
        process, message_queue = self._make_process(["a"])
        process._receive_and_handle_control_message(HordeDownloadControlMessage(model_names=["b", "c"]))
        process._currently_downloading = "b"

        process._receive_and_handle_control_message(HordeDownloadControlMessage(desired_image_models=["a"]))

        assert process._pending == []
        assert process._currently_downloading is None


class TestRealDownloadProcessReconcile:
    """The real download process reconciles its staged/queued/in-flight work against the authoritative set.

    These drive ``_handle_control_message`` directly (no managers/hordelib needed): the reconcile is pure
    state (the staging buffer, the host-aware scheduler, and per-task runtimes) plus a status emit, so it
    can be unit-tested without a GPU or a real download.
    """

    def _make_process(self) -> HordeDownloadProcess:
        return HordeDownloadProcess(
            process_id=DOWNLOAD_PROCESS_ID,
            process_message_queue=queue.Queue(),  # type: ignore[arg-type]
            pipe_connection=Mock(),
            disk_lock=Mock(),
            download_bandwidth_semaphore=Mock(),
            process_launch_identifier=0,
        )

    @staticmethod
    def _insert_active(process: HordeDownloadProcess, kind: DownloadKind, model: str, feature: str) -> _TaskRuntime:
        """Register a fake in-flight task runtime under the scheduler's dedup key for that task."""
        runtime = _TaskRuntime(
            status=CurrentDownloadStatus(model_name=model, feature=feature, target_dir=""),
            pacer=ChunkPacer(),
        )
        process._active[(kind, "", model)] = runtime
        return runtime

    def test_desired_set_prunes_staged_and_queued_model(self) -> None:
        """A removed model is dropped from the staging buffer and from the scheduler queue."""
        process = self._make_process()
        process._pending_image_models = ["a", "b"]
        process._scheduler.enqueue(
            DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name="b", host="h", feature=FEATURE_IMAGE_MODEL),
        )

        process._handle_control_message(HordeDownloadControlMessage(desired_image_models=["a"]))

        assert process._pending_image_models == ["a"]
        assert all(task.model_name != "b" for task in process._scheduler.pending_snapshot())

    def test_desired_set_cancels_in_flight_image_model(self) -> None:
        """Removing the in-flight image model flips its runtime cancel flag, so its callback aborts."""
        process = self._make_process()
        runtime = self._insert_active(process, DownloadKind.IMAGE_MODEL, "b", FEATURE_IMAGE_MODEL)

        process._handle_control_message(HordeDownloadControlMessage(desired_image_models=["a"]))

        assert runtime.cancelled is True
        # The cancel reaches the download via the per-task chunk-callback abort predicate.
        task = DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name="b", host="h", feature=FEATURE_IMAGE_MODEL)
        with pytest.raises(DownloadAborted):
            process._make_callback(task, runtime)(10, 100)

    def test_desired_set_does_not_cancel_safety_download(self) -> None:
        """A model removal must never abort an in-flight required safety (or aux) download."""
        process = self._make_process()
        runtime = self._insert_active(process, DownloadKind.SAFETY, "safety models", FEATURE_SAFETY)

        process._handle_control_message(HordeDownloadControlMessage(desired_image_models=["a"]))

        assert runtime.cancelled is False

    def test_re_adding_a_model_uncancels_in_flight(self) -> None:
        """Re-adding the in-flight model (config flap) clears the cancel so it keeps downloading."""
        process = self._make_process()
        runtime = self._insert_active(process, DownloadKind.IMAGE_MODEL, "b", FEATURE_IMAGE_MODEL)

        process._handle_control_message(HordeDownloadControlMessage(desired_image_models=["a"]))
        assert runtime.cancelled is True
        process._handle_control_message(HordeDownloadControlMessage(desired_image_models=["a", "b"]))

        assert runtime.cancelled is False

    def test_live_gating_enable_rearms_the_aux_pass(self) -> None:
        """Flipping a gating flag on live updates it and re-arms the one-shot aux pass (replaces the restart)."""
        process = self._make_process()
        process._allow_lora = False
        process._aux_requested = False
        process._aux_enqueued = True  # the aux pass already ran once

        process._handle_control_message(HordeDownloadControlMessage(set_allow_lora=True))

        assert process._allow_lora is True
        assert process._aux_requested is True
        assert process._aux_enqueued is False

    def test_live_gating_unchanged_value_does_not_rearm_aux(self) -> None:
        """A gating flag set to its current value is a no-op: the aux pass is not needlessly replayed."""
        process = self._make_process()
        process._allow_lora = True
        process._aux_enqueued = True

        process._handle_control_message(HordeDownloadControlMessage(set_allow_lora=True))

        assert process._aux_enqueued is True


class TestFirstClassAnnotators:
    """The first-class ``controlnet_annotator`` manager path: per-file downloads, then a verify with recovery.

    With a hordelib that exposes a ``controlnet_annotator`` manager, each detector checkpoint is a per-file
    aux download (size/progress/checksum/presence like any model), and a single ComfyUI verify confirms the
    preprocessors run. A verify failure re-downloads once and, if it still fails, disables ControlNet and
    notifies the operator instead of faulting every ControlNet job.
    """

    def _process(self) -> HordeDownloadProcess:
        return HordeDownloadProcess(
            process_id=DOWNLOAD_PROCESS_ID,
            process_message_queue=queue.Queue(),  # type: ignore[arg-type]
            pipe_connection=Mock(),
            disk_lock=Mock(),
            download_bandwidth_semaphore=Mock(),
            process_launch_identifier=0,
            allow_lora=False,
            allow_post_processing=False,
            allow_sdxl_controlnet=False,
            allow_controlnet=True,
        )

    @staticmethod
    def _annotator_manager(*, present: bool, names: tuple[str, ...] = ("annotator_hed", "annotator_depth")) -> object:
        """A fake first-class annotator manager: per-record presence plus the taint/download recovery hooks."""
        return SimpleNamespace(
            model_reference={name: object() for name in names},
            model_folder_path="/cn",
            is_model_available=lambda _name: present,
            get_model_download=lambda _name: [{"file_url": "https://huggingface.co/lllyasviel/Annotators"}],
            taint_models=lambda _names: None,
            download_model=lambda _name, callback=None, connections=1: True,
        )

    def _inject(self, monkeypatch: pytest.MonkeyPatch, *, annotators_present: bool) -> object:
        """Inject a SharedModelManager exposing ControlNet plus a first-class annotator manager; return it."""
        annotator_manager = self._annotator_manager(present=annotators_present)
        controlnet = SimpleNamespace(
            model_reference={},
            model_folder_path="/cn",
            is_model_available=lambda _name: True,
            get_model_download=lambda _name: [],
        )
        manager = SimpleNamespace(
            lora=None,
            gfpgan=None,
            esrgan=None,
            codeformer=None,
            miscellaneous=None,
            controlnet=controlnet,
            controlnet_annotator=annotator_manager,
        )
        fake_api = types.ModuleType("hordelib.api")
        fake_api.SharedModelManager = SimpleNamespace(manager=manager)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)
        return manager

    def test_missing_annotators_enqueue_per_file_aux_tasks_not_the_opaque_preload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each missing annotator checkpoint becomes its own AUX_MODEL task (no opaque preload task)."""
        process = self._process()
        self._inject(monkeypatch, annotators_present=False)

        process._enqueue_aux_tasks()

        tasks = process._scheduler.pending_snapshot()
        annotator_tasks = [task for task in tasks if task.manager_key == "controlnet_annotator"]
        assert {task.model_name for task in annotator_tasks} == {"annotator_hed", "annotator_depth"}
        # Every annotator fetch is a per-file aux download; the coarse preload task no longer exists.
        assert all(task.kind is DownloadKind.AUX_MODEL for task in annotator_tasks)

    def test_present_annotators_enqueue_a_single_verify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a cold marker (verify due), exactly one ANNOTATOR_VERIFY is enqueued (idempotent)."""
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        monkeypatch.setattr(process, "_annotators_verified_for_pin", lambda: False)  # marker cold: verify is due

        process._maybe_enqueue_annotator_verify(manager, True)
        process._maybe_enqueue_annotator_verify(manager, True)  # second call must not re-enqueue

        verifies = [t for t in process._scheduler.pending_snapshot() if t.kind is DownloadKind.ANNOTATOR_VERIFY]
        assert len(verifies) == 1
        assert verifies[0].exclusive is True

    def test_warm_marker_skips_the_verify_without_booting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A warm marker (already verified for the pin) marks the verify done and never enqueues the boot.

        The verify boots a whole ComfyUI/torch stack in the otherwise-offline download process, so a marker
        that records a prior successful run for the pinned annotator commit must short-circuit it entirely.
        """
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        monkeypatch.setattr(process, "_annotators_verified_for_pin", lambda: True)  # marker warm: nothing to do

        process._maybe_enqueue_annotator_verify(manager, True)

        verifies = [t for t in process._scheduler.pending_snapshot() if t.kind is DownloadKind.ANNOTATOR_VERIFY]
        assert verifies == []  # no verify task, so no ComfyUI boot is ever paid
        assert process._annotator_verify_done is True  # recorded as done so it is not re-evaluated this session
        assert process._annotator_verify_enqueued is False

    def test_run_annotator_preload_honors_warm_marker_before_initialise(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_run_annotator_preload`` returns True on a warm marker without importing/booting hordelib."""
        process = self._process()
        monkeypatch.setattr(process, "_annotators_verified_for_pin", lambda: True)
        # If the marker is honored first, hordelib is never imported; make any import of it explode so a
        # regression that boots anyway fails loudly rather than silently paying the cost.
        monkeypatch.setitem(sys.modules, "hordelib", None)  # ``import hordelib`` -> ImportError if reached

        assert process._run_annotator_preload() is True

    def test_verify_success_marks_done_without_killing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A passing verify records completion and never touches the kill switch (no re-download)."""
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        redownloaded: list[bool] = []
        monkeypatch.setattr(process, "_run_annotator_preload", lambda: True)
        monkeypatch.setattr(process, "_redownload_annotators", lambda _m: redownloaded.append(True))
        monkeypatch.setattr(process, "_refresh_feature_presence", lambda: None)

        process._verify_annotators(manager)

        assert process._annotator_verify_done is True
        assert process._controlnet_killed is False
        assert redownloaded == []  # success path does not re-download

    def test_verify_failure_then_recovers_after_one_redownload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A first-pass failure re-downloads once and, if the re-verify passes, re-enables ControlNet."""
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        results = iter([False, True])  # fail, then pass after the re-download
        redownloaded: list[bool] = []
        monkeypatch.setattr(process, "_run_annotator_preload", lambda: next(results))
        monkeypatch.setattr(process, "_redownload_annotators", lambda _m: redownloaded.append(True))
        monkeypatch.setattr(process, "_refresh_feature_presence", lambda: None)

        process._verify_annotators(manager)

        assert redownloaded == [True]  # exactly one re-download (max retry of 1)
        assert process._annotator_verify_done is True
        assert process._controlnet_killed is False

    def test_verify_permanent_failure_kills_controlnet_and_reports_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two failures (even after a re-download) disable ControlNet and report the failure to the parent."""
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        redownloaded: list[bool] = []
        monkeypatch.setattr(process, "_run_annotator_preload", lambda: False)
        monkeypatch.setattr(process, "_redownload_annotators", lambda _m: redownloaded.append(True))
        monkeypatch.setattr(process, "_refresh_feature_presence", lambda: None)

        process._verify_annotators(manager)

        assert redownloaded == [True]  # tried recovery exactly once
        assert process._controlnet_killed is True
        assert process._annotator_verify_done is True  # do not keep retrying a dead feature
        # The kill is surfaced to the parent as ``controlnet_failed`` (drives the FAILED readiness + pop gate).
        process._send_status(DownloadPhase.IDLE, force=True)
        messages = _drain_availability(process.process_message_queue)
        assert messages and messages[-1].controlnet_failed is True

    def test_killed_controlnet_is_reported_absent_to_the_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A killed ControlNet reports annotators not-present, so the pop gate and TUI withhold it."""
        process = self._process()
        manager = self._inject(monkeypatch, annotators_present=True)
        process._controlnet_killed = True

        assert process._annotators_present_now(manager) is False


class TestFeaturePresenceValidation:
    """The feature 'offer' gate is existence + sha256, and it stays coherent with the offline plan.

    A feature is offered only once its models are on disk *and* validate: an existence-only check would
    wrongly offer a truncated or corrupt pre-existing file (which then faults at job time). The live
    presence check and the offline existence authority agree on present/absent; the one intended divergence
    is that the live gate additionally withholds a file that is on disk but fails its checksum.
    """

    def _process(self) -> HordeDownloadProcess:
        return HordeDownloadProcess(
            process_id=DOWNLOAD_PROCESS_ID,
            process_message_queue=queue.Queue(),  # type: ignore[arg-type]
            pipe_connection=Mock(),
            disk_lock=Mock(),
            download_bandwidth_semaphore=Mock(),
            process_launch_identifier=0,
            allow_controlnet=True,
        )

    @staticmethod
    def _manager(*, on_disk: dict[str, bool], checksum: dict[str, bool | None]) -> tuple[CompVisLike, Counter[str]]:
        """A fake feature manager plus a per-name validate counter.

        ``on_disk`` is the existence answer (``is_model_available``); ``checksum`` is what ``validate_model``
        returns per name (True passes, False is a corrupt file, None means the record has no checksum).
        """
        validate_calls: Counter[str] = Counter()

        def validate_model(name: str, skip_checksum: bool = False) -> bool | None:
            validate_calls[name] += 1
            return checksum.get(name)

        manager = SimpleNamespace(
            model_reference={name: object() for name in on_disk},
            model_folder_path="/cn",
            is_model_available=lambda name: on_disk.get(name, False),
            validate_model=validate_model,
        )
        return manager, validate_calls  # type: ignore[return-value]  # duck-typed fake satisfies CompVisLike

    def test_present_and_valid_model_is_offered(self) -> None:
        """A model on disk whose checksum verifies counts as present."""
        process = self._process()
        manager, _ = self._manager(on_disk={"cn_a": True}, checksum={"cn_a": True})
        assert process._feature_model_present(manager, "controlnet", "cn_a") is True

    def test_present_but_corrupt_model_is_withheld(self) -> None:
        """A model on disk whose checksum fails is reported not-present and recorded as invalid, not cached."""
        process = self._process()
        manager, _ = self._manager(on_disk={"cn_a": True}, checksum={"cn_a": False})
        assert process._feature_model_present(manager, "controlnet", "cn_a") is False
        assert ("controlnet", "cn_a") in process._invalid_feature_files
        assert ("controlnet", "cn_a") not in process._validated_feature_files

    def test_no_checksum_falls_back_to_existence(self) -> None:
        """A record without a checksum cannot be hashed, so existence alone makes it present."""
        process = self._process()
        manager, _ = self._manager(on_disk={"cn_a": True}, checksum={"cn_a": None})
        assert process._feature_model_present(manager, "controlnet", "cn_a") is True

    def test_absent_model_is_not_present_and_never_hashed(self) -> None:
        """A model not on disk is not present, and its (nonexistent) bytes are never hashed."""
        process = self._process()
        manager, calls = self._manager(on_disk={"cn_a": False}, checksum={"cn_a": True})
        assert process._feature_model_present(manager, "controlnet", "cn_a") is False
        assert calls["cn_a"] == 0

    def test_validation_is_cached_after_first_success(self) -> None:
        """A known-good file is validated once; later presence reads are served from the session cache."""
        process = self._process()
        manager, calls = self._manager(on_disk={"cn_a": True}, checksum={"cn_a": True})
        assert process._feature_model_present(manager, "controlnet", "cn_a") is True
        assert process._feature_model_present(manager, "controlnet", "cn_a") is True
        assert calls["cn_a"] == 1

    def test_manager_all_present_withholds_when_any_file_is_corrupt(self) -> None:
        """One corrupt file is enough to withhold the whole feature (all-or-nothing presence)."""
        process = self._process()
        controlnet, _ = self._manager(
            on_disk={"cn_a": True, "cn_b": True},
            checksum={"cn_a": True, "cn_b": False},
        )
        manager_obj = SimpleNamespace(controlnet=controlnet)
        assert process._manager_all_present(manager_obj, "controlnet") is False

    def test_offline_and_live_agree_when_no_file_is_corrupt(self) -> None:
        """Coherence lock: with every present file valid, the live verdict matches the offline existence one.

        The offline plan reports a feature present iff all its files are on disk; the live composite reaches
        the same answer when nothing is corrupt, so the two presence paths cannot silently drift apart.
        """
        on_disk = {"cn_a": True, "cn_b": True}
        process = self._process()
        controlnet, _ = self._manager(on_disk=on_disk, checksum={"cn_a": True, "cn_b": None})
        manager_obj = SimpleNamespace(controlnet=controlnet)

        offline_present = all(on_disk.values())  # the existence authority the offline plan uses
        assert process._manager_all_present(manager_obj, "controlnet") is offline_present


class TestDownloadProcessConcurrencyFixes:
    """The threaded download path's correctness guards: per-manager locking, retry, bandwidth, pool growth."""

    def _make_process(
        self,
        *,
        semaphore: object | None = None,
        max_parallel_downloads: int = 4,
    ) -> HordeDownloadProcess:
        return HordeDownloadProcess(
            process_id=DOWNLOAD_PROCESS_ID,
            process_message_queue=queue.Queue(),  # type: ignore[arg-type]
            pipe_connection=Mock(),
            disk_lock=Mock(),
            download_bandwidth_semaphore=semaphore or Mock(),  # type: ignore[arg-type]
            process_launch_identifier=0,
            max_parallel_downloads=max_parallel_downloads,
        )

    @staticmethod
    def _inject_aux_managers(monkeypatch: pytest.MonkeyPatch, managers: dict[str, object]) -> None:
        """Make ``from hordelib.api import SharedModelManager`` yield a manager exposing *managers* by key."""
        manager = SimpleNamespace(**managers)
        fake_api = types.ModuleType("hordelib.api")
        fake_api.SharedModelManager = SimpleNamespace(manager=manager)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)

    def test_same_manager_downloads_serialize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two AUX tasks on the *same* manager never run that manager's download_model concurrently."""
        probe = _ConcurrencyProbe()
        gfpgan = SimpleNamespace(download_model=lambda _name, callback=None, connections=1: probe.run(0.05))
        self._inject_aux_managers(monkeypatch, {"gfpgan": gfpgan})
        process = self._make_process()

        task_a = _aux_task("a", "h1", "gfpgan")
        task_b = _aux_task("b", "h2", "gfpgan")
        self._run_dispatch_concurrently(process, [task_a, task_b])

        assert probe.max_active == 1  # the per-manager lock serialized them

    def test_different_managers_download_in_parallel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AUX tasks on *different* managers run truly in parallel (independent shared state)."""
        probe = _ConcurrencyProbe()
        gfpgan = SimpleNamespace(download_model=lambda _name, callback=None, connections=1: probe.run(0.05))
        esrgan = SimpleNamespace(download_model=lambda _name, callback=None, connections=1: probe.run(0.05))
        self._inject_aux_managers(monkeypatch, {"gfpgan": gfpgan, "esrgan": esrgan})
        process = self._make_process()

        task_a = _aux_task("a", "h1", "gfpgan")
        task_b = _aux_task("b", "h2", "esrgan")
        self._run_dispatch_concurrently(process, [task_a, task_b])

        assert probe.max_active == 2  # distinct manager locks, so both ran at once

    @staticmethod
    def _run_dispatch_concurrently(process: HordeDownloadProcess, tasks: list[DownloadTask]) -> None:
        def noop(_downloaded: int, _total: int) -> None:
            return

        threads = [threading.Thread(target=lambda t=task: process._dispatch_task(t, noop)) for task in tasks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

    def test_failed_image_fetch_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed per-file fetch is re-queued (bounded), so a transient failure is not abandoned."""
        monkeypatch.setattr(
            "horde_worker_regen.process_management.workers.download_process._RETRY_BACKOFF_SECONDS",
            0.0,
        )
        process = self._make_process()
        task = DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name="m", host="h", feature=FEATURE_IMAGE_MODEL)

        process._maybe_retry(task, "boom")

        assert process._attempts[task.dedup_key] == 1
        assert any(t.model_name == "m" for t in process._scheduler.pending_snapshot())

    def test_retry_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After the attempt ceiling the task is no longer re-queued (no infinite retry loop)."""
        monkeypatch.setattr(
            "horde_worker_regen.process_management.workers.download_process._RETRY_BACKOFF_SECONDS",
            0.0,
        )
        from horde_worker_regen.process_management.workers import download_process as dp

        process = self._make_process()
        task = DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name="m", host="h", feature=FEATURE_IMAGE_MODEL)
        process._attempts[task.dedup_key] = dp._MAX_DOWNLOAD_ATTEMPTS

        process._maybe_retry(task, "boom")

        assert not process._scheduler.pending_snapshot()

    def test_coarse_kinds_are_not_retried(self) -> None:
        """The coarse kinds (safety/LoRa/annotators) own their own retry and are not re-queued here."""
        process = self._make_process()
        task = DownloadTask(
            kind=DownloadKind.SAFETY,
            model_name="safety models",
            host="unknown",
            feature=FEATURE_SAFETY,
        )

        process._maybe_retry(task, "boom")

        assert not process._scheduler.pending_snapshot()
        assert task.dedup_key not in process._attempts

    def test_failure_cleared_on_later_success(self) -> None:
        """A recorded failure is dropped once the model is later marked successful."""
        process = self._make_process()
        process._record_failure("m", FEATURE_IMAGE_MODEL, "boom")
        assert any(f.model_name == "m" for f in process._failures)

        process._clear_failure("m")

        assert not any(f.model_name == "m" for f in process._failures)

    def test_bandwidth_slot_acquired_once_and_released_last(self) -> None:
        """The cross-process slot is acquired by the first task and released only when the last finishes."""
        semaphore = _CountingSemaphore()
        process = self._make_process(semaphore=semaphore)

        process._acquire_bandwidth_slot()  # first task
        process._acquire_bandwidth_slot()  # second concurrent task
        assert semaphore.acquired == 1  # only one real acquire for both tasks
        assert process._bandwidth_held is True

        process._release_bandwidth_slot()  # one task still active
        assert semaphore.released == 0
        process._release_bandwidth_slot()  # last task finishes
        assert semaphore.released == 1
        assert process._bandwidth_held is False

    def test_executor_pool_grows_but_never_shrinks(self) -> None:
        """Raising the limit grows the pool to use the new ceiling; a lower limit leaves threads idle."""
        process = self._make_process(max_parallel_downloads=2)
        try:
            process._ensure_executor_threads(2)
            assert sum(1 for t in process._executor_threads if t.is_alive()) == 2
            process._ensure_executor_threads(5)
            assert sum(1 for t in process._executor_threads if t.is_alive()) == 5
            process._ensure_executor_threads(3)  # a lower limit must not drop threads
            assert sum(1 for t in process._executor_threads if t.is_alive()) == 5
        finally:
            process._end_process = True
            process._scheduler.close()
            for thread in process._executor_threads:
                thread.join(timeout=1.0)

    def test_dead_executor_thread_is_respawned(self) -> None:
        """A thread that died is pruned and replaced, so download capacity self-heals (oracle safety)."""
        process = self._make_process(max_parallel_downloads=1)
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        process._executor_threads = [dead]
        try:
            process._ensure_executor_threads(1)
            assert dead not in process._executor_threads  # the dead thread was pruned
            assert sum(1 for thread in process._executor_threads if thread.is_alive()) == 1
        finally:
            process._end_process = True
            process._scheduler.close()
            for thread in process._executor_threads:
                thread.join(timeout=1.0)

    def test_executor_loop_survives_a_task_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unexpected error in one task is contained: the loop keeps running rather than dying."""
        process = self._make_process(max_parallel_downloads=1)
        process._scheduler.enqueue(_aux_task("a", "h1", "gfpgan"))
        process._scheduler.enqueue(_aux_task("b", "h1", "gfpgan"))
        calls = {"n": 0}

        def boom(_task: DownloadTask) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            process._end_process = True  # a second task ran, so the loop survived the first error

        monkeypatch.setattr(process, "_run_task", boom)
        thread = threading.Thread(target=process._executor_loop)
        thread.start()
        thread.join(timeout=3.0)

        assert not thread.is_alive()  # the loop exited cleanly via _end_process, it did not crash out
        assert calls["n"] >= 2  # it processed a second task after the first one raised


def _aux_task(model: str, host: str, manager_key: str) -> DownloadTask:
    """A compact AUX_MODEL task builder for the concurrency tests."""
    return DownloadTask(
        kind=DownloadKind.AUX_MODEL,
        model_name=model,
        host=host,
        feature="f",
        manager_key=manager_key,
    )


class _ConcurrencyProbe:
    """Records the peak number of overlapping ``run`` calls, to assert (non-)concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, seconds: float) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(seconds)
        with self._lock:
            self.active -= 1


class _CountingSemaphore:
    """A stand-in for the cross-process bandwidth semaphore that counts acquire/release calls."""

    def __init__(self) -> None:
        self.acquired = 0
        self.released = 0

    def acquire(self, timeout: float | None = None) -> bool:
        self.acquired += 1
        return True

    def release(self) -> None:
        self.released += 1


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
