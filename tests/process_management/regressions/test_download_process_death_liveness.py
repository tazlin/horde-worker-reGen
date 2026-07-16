"""Reproductions for a silently-dead background download process and the fallout it left unhandled.

The download process (reserved id ``DOWNLOAD_PROCESS_ID``) lives outside the process map, so nothing swept
it: when it died mid-prefetch the parent never noticed, its frozen final status kept reporting a transfer at
a fixed byte count forever, and the aux-prefetch coordinator deferred a job's deadline against that corpse to
the full deferral cap (three budgets of GPU idleness) instead of faulting on the first. These encode the
contracts that close the gap:

- the parent detects the death, reports it once (with the exit code), forgets the corpse, and restarts within
  a crash-loop bound, giving up loudly (and withholding LoRA/aux advertising) past the bound;
- the in-flight provider yields nothing while the downloader is dead, so a deadline serves the job without the
  reference on its first budget (the death detection and restart are separately owned) rather than deferring
  against a process that can never progress;
- a file whose reported bytes have not advanced between two consecutive deadline expiries defers at most once,
  regardless of any in-flight-provider flicker between those expiries (the stall-memory hole);
- a pending aux job survives the downloader's death and is re-fetched against the fresh downloader within a
  single download-timeout budget, not the full deferral cap;
- the RESTART_PROCESS supervisor command revives the download process, the operator path to a stuck downloader.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import AuxModelRef, AuxPrefetchEntry
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    FEATURE_LORA_ADHOC,
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


class _Clock:
    """A hand-advanceable clock so deadline and deferral behaviour is deterministic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _SenderSpy:
    """Records the (entries, pins) of each prefetch request the coordinator would send."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[AuxPrefetchEntry], list[AuxModelRef]]] = []

    def __call__(self, entries: list[AuxPrefetchEntry], pins: list[AuxModelRef]) -> None:
        self.calls.append((entries, pins))


class _InFlightSpy:
    """A settable in-flight provider standing in for the downloader's live progress snapshot."""

    def __init__(self) -> None:
        self.map: dict[str, tuple[int, int]] = {}

    def __call__(self) -> dict[str, tuple[int, int]]:
        return dict(self.map)


class _LogCapture:
    """Capture loguru messages at or above a level for the duration of a with-block."""

    def __init__(self, level: str = "WARNING") -> None:
        self._level = level
        self.messages: list[str] = []
        self._sink_id: int | None = None

    def __enter__(self) -> _LogCapture:
        self._sink_id = logger.add(lambda message: self.messages.append(message.record["message"]), level=self._level)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._sink_id is not None:
            logger.remove(self._sink_id)

    def containing(self, needle: str) -> list[str]:
        return [message for message in self.messages if needle in message]


def _lora(name: str) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=False)


def _job(loras: list[LorasPayloadEntry]) -> ImageGenerateJobPopResponse:
    return make_job_pop_response("some-model", loras=loras)


def _coordinator(
    tracker: JobTracker,
    *,
    clock: _Clock,
    in_flight: _InFlightSpy,
    timeout: float = 60.0,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy]:
    sender = _SenderSpy()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=WorkerState(),
        prefetch_sender=sender,
        download_timeout_provider=lambda: timeout,
        pin_sender=lambda _pins: None,
        in_flight_provider=in_flight,
        clock=clock,
    )
    return coordinator, sender


def _adhoc_prefetch_snapshot(name: str, downloaded: int, total: int) -> DownloadStatusSnapshot:
    """A download-status snapshot showing one ad-hoc LoRA prefetch in flight at a fixed byte count."""
    current = CurrentDownloadStatus(
        model_name=name,
        feature=FEATURE_LORA_ADHOC,
        target_dir="/tmp",
        downloaded_bytes=downloaded,
        total_bytes=total,
    )
    return DownloadStatusSnapshot(phase=DownloadPhase.DOWNLOADING, current=current, active=[current])


def _inject_download_process(manager: object, *, alive: bool, exitcode: int | None) -> None:
    """Own a mock download process on the manager's lifecycle with the given liveness and exit code."""
    info = make_mock_process_info(
        process_id=DOWNLOAD_PROCESS_ID,
        process_type=HordeProcessType.DOWNLOAD,
        model_name=None,
    )
    info.mp_process.is_alive.return_value = alive
    info.mp_process.exitcode = exitcode  # pyrefly: ignore
    manager._process_lifecycle._download_process_info = info  # type: ignore[attr-defined]


def _stub_download_lifecycle(manager: object, *, restart_alive: bool) -> tuple[Mock, Callable[[], int]]:
    """Stub the lifecycle so a restart re-owns a fresh downloader and is counted; return the call counter.

    ``restart_alive`` models the restarted process's fate: True for a healthy revival, False for one that
    itself dies immediately (a crash loop).
    """
    restart_calls = {"n": 0}

    def _restart() -> None:
        restart_calls["n"] += 1
        _inject_download_process(manager, alive=restart_alive, exitcode=None if restart_alive else -9)

    restart = Mock(side_effect=_restart)
    manager._process_lifecycle.restart_download_process = restart  # type: ignore[attr-defined]
    return restart, lambda: restart_calls["n"]


# --- (#2) the in-flight provider must yield nothing while the downloader is dead --------------------------


def test_dead_download_process_yields_empty_in_flight() -> None:
    """A dead downloader's frozen snapshot never reports an in-flight file, so no deadline defers on it."""
    manager = make_testable_process_manager()
    manager._enable_background_downloads = True
    manager._model_availability._status = _adhoc_prefetch_snapshot("styleA", 7_340_032, 20_000_000)

    _inject_download_process(manager, alive=True, exitcode=None)
    assert manager._aux_prefetch_in_flight_downloads() == {"styleA": (7_340_032, 20_000_000)}

    # The very same frozen snapshot, once the process is dead, must be reported as nothing in flight.
    info = manager._process_lifecycle.download_process_info
    assert info is not None
    info.mp_process.is_alive.return_value = False
    assert manager._aux_prefetch_in_flight_downloads() == {}


async def test_expired_deadline_salvages_now_when_downloader_dead() -> None:
    """With the downloader dead the in-flight set is empty, so an expired deadline serves the job without the file.

    A dead download process is the prefetch lane's own unavailability, not evidence the reference is bad, and the
    inference child fetches the file itself, so on the first budget the deadline dispatches the job without the
    reference rather than faulting it. Detecting the death and restarting the downloader is separately owned
    machinery this salvage neither performs nor replaces; the salvage announces itself on its own log line.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    in_flight = _InFlightSpy()  # empty, as the parent's provider yields for a dead downloader
    coordinator, _sender = _coordinator(tracker, clock=clock, in_flight=in_flight)

    job = _job([_lora("styleA")])
    assert job.id_ is not None
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)

    clock.now += 61.0
    with _LogCapture(level="INFO") as logs:
        coordinator.scan_deadlines()

    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.is_lora_skipped(_lora("styleA")) is True
    assert logs.containing("never in flight"), "deadline salvage must emit its own distinguishable signal"


# --- (#3) the stall-memory hole: provider flicker between expiries must not license a second deferral -----


async def test_frozen_bytes_defer_once_despite_inflight_provider_flicker() -> None:
    """Bytes frozen across two expiries defer at most once even when the provider flickers empty between them.

    This is the exact incident shape: the file's reported byte count never advances, but a tick on which the
    provider momentarily reports nothing must not wipe the remembered count and reset the file to
    "progressing by default", which would license a second deferral (and a third budget of idleness).
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    in_flight = _InFlightSpy()
    coordinator, _sender = _coordinator(tracker, clock=clock, in_flight=in_flight)

    job = _job([_lora("styleA")])
    assert job.id_ is not None
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)

    # First expiry: bytes present, no prior observation, so it defers and records 7,340,032.
    in_flight.map = {"styleA": (7_340_032, 20_000_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    # A flicker tick between expiries: the provider reports nothing (no deadline expires here). The remembered
    # byte count must survive this, keyed by the job's still-live deadline rather than the momentary in-flight
    # set.
    in_flight.map = {}
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    # Second expiry: the frozen snapshot reports the same 7,340,032, so the stall is detected and the job
    # faults. Without the fix the flicker would have wiped the memory and deferred a second time.
    in_flight.map = {"styleA": (7_340_032, 20_000_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


# --- (#1) death detection, restart-within-bound, and re-fetch within one budget --------------------------


async def test_pending_aux_job_refetched_within_one_budget_after_reset() -> None:
    """A downloader reset drops the corpse's deadlines so the job is re-requested within a single fresh budget.

    A job already deferred once against the (now dead) downloader must not have to wait out the remaining
    deferral cap. After the reset the periodic reconcile re-requests it exactly as a fresh pop would, arming
    one fresh budget, and that fresh budget resolves the job on a single expiry with no inherited deferral debt:
    with nothing in flight it is served without the reference rather than deferring again against a dead transfer.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    in_flight = _InFlightSpy()
    coordinator, sender = _coordinator(tracker, clock=clock, in_flight=in_flight)

    job = _job([_lora("styleA")])
    assert job.id_ is not None
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)

    # Defer once against a download that looked alive and progressing.
    in_flight.map = {"styleA": (1_000, 20_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert coordinator.has_live_deadline(job.id_) is True
    requests_before = len(sender.calls)

    # The downloader dies and is replaced: the coordinator forgets the corpse's deadlines.
    coordinator.on_downloader_reset()
    assert coordinator.has_live_deadline(job.id_) is False

    # Same tick, the reconcile re-requests the still-pending job against the fresh downloader (nothing in
    # flight yet), arming one fresh budget.
    in_flight.map = {}
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == requests_before + 1
    assert coordinator.has_live_deadline(job.id_) is True

    # The fresh budget carries no inherited deferral debt: a single expiry with nothing in flight resolves the
    # job, serving it without the reference rather than deferring again against a transfer that never resumed.
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert tracker.are_job_aux_models_prepared(job) is True


def test_sweep_detects_death_reports_once_and_restarts() -> None:
    """The liveness sweep reports the death (with exit code) exactly once and restarts the download process."""
    manager = make_testable_process_manager()
    manager._enable_background_downloads = True
    restart, _count = _stub_download_lifecycle(manager, restart_alive=True)
    manager._aux_prefetch_coordinator.on_downloader_reset = Mock()  # type: ignore[method-assign]

    _inject_download_process(manager, alive=False, exitcode=-9)

    with _LogCapture(level="ERROR") as captured:
        manager._sweep_download_process_liveness()

    died = captured.containing("Download process died unexpectedly")
    assert len(died) == 1
    assert "-9" in died[0]
    restart.assert_called_once()
    manager._aux_prefetch_coordinator.on_downloader_reset.assert_called_once()

    # The restart revived a healthy process, so a second sweep finds it alive and neither re-logs nor
    # restarts again (the death report is edge-triggered per incident, not repeated every tick).
    with _LogCapture(level="ERROR") as captured_again:
        manager._sweep_download_process_liveness()
    assert captured_again.containing("Download process died unexpectedly") == []
    restart.assert_called_once()


def test_sweep_crash_loop_bound_gives_up_and_withholds_advertising() -> None:
    """Past the restart bound the sweep stops restarting, reports the give-up once, and withholds aux features."""
    manager = make_testable_process_manager()
    manager._enable_background_downloads = True
    _restart, count = _stub_download_lifecycle(manager, restart_alive=False)

    with _LogCapture(level="ERROR") as captured:
        # Four consecutive deaths: the first three restart, the fourth is past the bound.
        for _ in range(4):
            _inject_download_process(manager, alive=False, exitcode=-9)
            manager._sweep_download_process_liveness()
        # A fifth death past the bound must not re-log the give-up (edge-triggered).
        _inject_download_process(manager, alive=False, exitcode=-9)
        manager._sweep_download_process_liveness()

    assert count() == 3
    assert manager._model_availability.downloader_lost is True
    assert len(captured.containing("restart bound is exhausted")) == 1


def test_downloader_lost_withholds_lora_advertising() -> None:
    """A lost downloader withholds LoRA advertising, exactly as a worker with no background downloader would."""
    manager = make_testable_process_manager(allow_lora=True)
    manager._enable_background_downloads = True
    popper = manager._job_popper
    popper._background_downloads_enabled = True

    manager._model_availability.note_downloader_present()
    assert popper._lora_disk_permits is True

    manager._model_availability.mark_downloader_lost()
    assert popper._lora_disk_permits is False

    # A fresh report from a restarted downloader restores advertising.
    manager._model_availability.note_downloader_present()
    assert popper._lora_disk_permits is True


# --- (#4) bounded shutdown: every registered child-facing queue feeder is detached so exit cannot wedge ---


def test_neutralize_message_queue_feeder_detaches_and_closes() -> None:
    """Neutralizing the feeder cancels its join thread and closes the queue, so atexit cannot block on it."""
    manager = make_testable_process_manager()
    queue = Mock()
    manager._process_lifecycle._child_facing_queues = [queue]

    manager._process_lifecycle.neutralize_message_queue_feeder()

    queue.cancel_join_thread.assert_called_once()
    queue.close.assert_called_once()


def test_neutralize_message_queue_feeder_covers_every_registered_queue() -> None:
    """Every parent-created child-facing queue in the registry is detached and closed, not just the first.

    A parent that grows a second child-facing queue must have it neutralized at teardown too; otherwise a
    feeder left mid-send toward a dead child would reintroduce the atexit wedge the single-queue path fixed.
    """
    manager = make_testable_process_manager()
    queue_a = Mock()
    queue_b = Mock()
    manager._process_lifecycle._child_facing_queues = [queue_a, queue_b]

    manager._process_lifecycle.neutralize_message_queue_feeder()

    for queue in (queue_a, queue_b):
        queue.cancel_join_thread.assert_called_once()
        queue.close.assert_called_once()


def test_neutralize_message_queue_feeder_continues_after_one_queue_raises() -> None:
    """One queue failing to close must not skip neutralizing the rest (teardown is best-effort per queue)."""
    manager = make_testable_process_manager()
    failing = Mock()
    failing.cancel_join_thread.side_effect = RuntimeError("already torn down")
    healthy = Mock()
    manager._process_lifecycle._child_facing_queues = [failing, healthy]

    manager._process_lifecycle.neutralize_message_queue_feeder()

    healthy.cancel_join_thread.assert_called_once()
    healthy.close.assert_called_once()


def test_status_queue_is_registered_for_neutralization() -> None:
    """The status queue the parent creates is registered so the default teardown neutralizes it."""
    manager = make_testable_process_manager()
    assert manager._process_message_queue in manager._process_lifecycle._child_facing_queues


def test_atexit_handler_neutralizes_message_queue_feeder() -> None:
    """The atexit backstop detaches the feeder (before multiprocessing's own join finalizer runs)."""
    manager = make_testable_process_manager()
    manager._process_lifecycle.kill_owned_children = Mock(return_value=[])  # type: ignore[method-assign]
    manager._process_lifecycle.neutralize_message_queue_feeder = Mock()  # type: ignore[method-assign]

    manager._kill_owned_children_on_exit()

    manager._process_lifecycle.neutralize_message_queue_feeder.assert_called_once()


# --- (#5) the operator path to revive a stuck/dead downloader --------------------------------------------


def test_restart_process_command_accepts_download_process_id() -> None:
    """A RESTART_PROCESS command targeting the download id routes to the dedicated download restart path."""
    manager = make_testable_process_manager()
    manager._process_lifecycle.restart_download_process = Mock()  # type: ignore[method-assign]

    manager._apply_supervisor_command(
        SupervisorControlMessage(command=SupervisorCommand.RESTART_PROCESS, process_id=DOWNLOAD_PROCESS_ID),
    )

    manager._process_lifecycle.restart_download_process.assert_called_once()


def test_restart_process_command_still_rejects_unknown_id() -> None:
    """A RESTART_PROCESS command for an id that is neither an inference slot nor the downloader is ignored."""
    manager = make_testable_process_manager()
    manager._process_lifecycle.restart_download_process = Mock()  # type: ignore[method-assign]
    manager._process_lifecycle._replace_inference_process = Mock()  # type: ignore[method-assign]

    manager._apply_supervisor_command(
        SupervisorControlMessage(command=SupervisorCommand.RESTART_PROCESS, process_id=4242),
    )

    manager._process_lifecycle.restart_download_process.assert_not_called()
    manager._process_lifecycle._replace_inference_process.assert_not_called()
