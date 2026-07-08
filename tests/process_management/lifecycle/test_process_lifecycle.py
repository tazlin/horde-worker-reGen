"""Tests for ProcessLifecycleManager."""

from __future__ import annotations

import multiprocessing
import sys
import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
    track_popped_job_async,
)


def _make_plm(
    *,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    ctx: object | None = None,
    device_free_mb_provider: object | None = None,
    device_total_vram_mb_provider: object | None = None,
    device_governor_state_provider: object | None = None,
    gpu_start_context_mb_provider: object | None = None,
) -> ProcessLifecycleManager:
    """Helper to build a PLM with mostly-mocked dependencies."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.enable_pipeline_disaggregation = False
    bridge_data.safety_on_gpu = False
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.inference_first_step_timeout = 120
    bridge_data.inference_stuck_step_repeat_limit = 20
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    return ProcessLifecycleManager(
        ctx=ctx if ctx is not None else multiprocessing.get_context("spawn"),  # type: ignore[arg-type]
        process_map=process_map or ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=job_tracker or JobTracker(),
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(target_process_count=2),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
        device_free_mb_provider=device_free_mb_provider,  # type: ignore[arg-type]
        device_total_vram_mb_provider=device_total_vram_mb_provider,  # type: ignore[arg-type]
        device_governor_state_provider=device_governor_state_provider,  # type: ignore[arg-type]
        gpu_start_context_mb_provider=gpu_start_context_mb_provider,  # type: ignore[arg-type]
    )


def test_inference_child_is_created_from_the_injected_context() -> None:
    """Children must be spawned via the injected context, not the process-global multiprocessing.Process.

    Using the global default would fork on POSIX, killing any child that touches CUDA after the parent
    initialized it ("Cannot re-initialize CUDA in forked subprocess").
    """
    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345

    plm = _make_plm(ctx=fake_ctx)
    plm._start_inference_process(0)

    fake_ctx.Process.assert_called_once()


def test_inference_replacement_defers_start_until_gpu_headroom_recovers() -> None:
    """A reclaim replacement kills the old slot but does not respawn into a pressured card."""
    free_mb = 500.0
    governor_state = GovernorState.PRESSURE

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_process = Mock()
    fake_process.pid = 22222
    fake_process.is_alive.return_value = True
    fake_ctx.Process.return_value = fake_process

    old_process = make_mock_process_info(
        2,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.INFERENCE,
    )
    old_process.ram_usage_bytes = 12 * 1024 * 1024 * 1024
    process_map = ProcessMap({2: old_process})
    plm = _make_plm(
        ctx=fake_ctx,
        process_map=process_map,
        device_free_mb_provider=lambda _device_index: free_mb,
        device_total_vram_mb_provider=lambda _device_index: 24_564.0,
        device_governor_state_provider=lambda _device_index: governor_state,
        gpu_start_context_mb_provider=lambda: 243.0,
    )

    plm._replace_inference_process(old_process, intentional_reclaim=True)

    assert 2 not in process_map
    assert plm.has_pending_inference_starts() is True
    fake_ctx.Process.assert_not_called()

    free_mb = 4_000.0
    governor_state = GovernorState.HEALTHY

    assert plm.drain_pending_gpu_starts() == 1
    assert 2 in process_map
    assert process_map[2].last_process_state == HordeProcessState.PROCESS_STARTING
    fake_ctx.Process.assert_called_once()


def test_pending_inference_start_is_recoverable_capacity_during_backoff() -> None:
    """Save-our-ship should not treat bounded deferred starts as an unrecoverable pool."""
    bridge_data = Mock()
    bridge_data.max_threads = 2
    lifecycle = Mock()
    lifecycle.has_pending_inference_starts.return_value = True
    lifecycle.pending_gpu_starts_backing_off.return_value = True
    lifecycle.quarantined_inference_slots = frozenset({1, 2})

    dispatcher = Mock()
    dispatcher.get_deadlock_snapshot.return_value.indicates_structural_wedge.return_value = False
    scheduler = Mock()
    scheduler.whole_card_residency_grace_active.return_value = False
    scheduler.heavy_head_load_grace_active.return_value = False
    scheduler.ram_reclaim_cycle_grace_active.return_value = False

    coordinator = WorkerRecoveryCoordinator(
        state=WorkerState(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        job_tracker=JobTracker(),
        process_map=ProcessMap({}),
        process_lifecycle=lifecycle,
        message_dispatcher=dispatcher,
        inference_scheduler=scheduler,
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        bridge_data_provider=lambda: bridge_data,
        max_inference_processes_provider=lambda: 2,
        abort_callback=Mock(),
    )

    assert coordinator.is_inference_capacity_available() is True
    assert coordinator.is_inference_pool_unrecoverable() is False


def test_inference_pids_never_take_the_reserved_safety_slot() -> None:
    """Inference process ids are allocated from 1 upward; slot 0 is reserved for the safety process.

    Allocation must skip 0 even when the map is empty (the inference pool can start before the safety pool),
    and must keep returning the lowest free slot at or above 1 as processes come and go.
    """
    plm = _make_plm(process_map=ProcessMap({}))
    assert plm._allocate_inference_pid() == 1, "the first inference process must not take the reserved safety slot 0"

    plm._process_map[1] = make_mock_process_info(1, process_type=HordeProcessType.INFERENCE)
    assert plm._allocate_inference_pid() == 2

    # Even with the safety process resident at slot 0, allocation stays in the inference range.
    plm._process_map[0] = make_mock_process_info(0, process_type=HordeProcessType.SAFETY)
    assert plm._allocate_inference_pid() == 2


def test_safety_process_takes_slot_zero_without_clobbering_inference() -> None:
    """The safety process claims the reserved slot 0, never an existing inference process's id.

    Inference and safety start on independent on-disk gates, so on a multi-GPU host the inference processes
    can register first (in the inference range 1..N). The safety process must still land on the reserved slot
    0 rather than overwriting an inference slot, so no card is stranded.
    """
    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345

    # Two inference processes already occupy slots 1 and 2 (the multi-GPU inference-first ordering).
    process_map = ProcessMap(
        {
            1: make_mock_process_info(1, process_type=HordeProcessType.INFERENCE, device_index=0),
            2: make_mock_process_info(2, process_type=HordeProcessType.INFERENCE, device_index=1),
        },
    )
    plm = _make_plm(process_map=process_map, ctx=fake_ctx)

    plm.start_safety_processes()

    assert process_map.num_inference_processes() == 2, "an inference slot was clobbered by the safety process"
    assert process_map.num_safety_processes() == 1
    assert process_map[1].process_type is HordeProcessType.INFERENCE
    assert process_map[2].process_type is HordeProcessType.INFERENCE
    safety_ids = [pid for pid, info in process_map.items() if info.process_type is HordeProcessType.SAFETY]
    assert safety_ids == [0], "the safety process must occupy the reserved slot 0"


def test_non_spawn_context_is_rejected_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fork (or forkserver) context must fail loudly outside tests rather than crash-loop every child."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("AI_HORDE_TESTING", raising=False)

    fork_ctx = Mock()
    fork_ctx.get_start_method.return_value = "fork"

    with pytest.raises(RuntimeError, match="spawn"):
        _make_plm(ctx=fork_ctx)


class TestModelLoadFailureQuarantine:
    """A model that deterministically fails to load is quarantined, instead of churning the process pool."""

    def test_quarantines_only_after_threshold(self) -> None:
        """A model is quarantined only after the configured failure threshold."""
        from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
            MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD,
        )

        plm = _make_plm()
        model = "Z-Image-Turbo"
        # Each failure lands on a different slot, mirroring the round-robin re-dispatch that the per-slot
        # crash-loop breaker cannot catch.
        for attempt in range(1, MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            assert plm.record_model_load_failure(process_id=attempt, model_name=model) is False
            assert plm.is_model_load_quarantined(model) is False
        assert plm.record_model_load_failure(process_id=99, model_name=model) is True
        assert plm.is_model_load_quarantined(model) is True
        assert model in plm.quarantined_models()

    def test_distinct_models_do_not_pool_failures(self) -> None:
        """Failures for different models are counted independently."""
        plm = _make_plm()
        # Two different models each failing under the threshold must not combine to a quarantine.
        plm.record_model_load_failure(process_id=1, model_name="model_a")
        plm.record_model_load_failure(process_id=2, model_name="model_b")
        assert plm.is_model_load_quarantined("model_a") is False
        assert plm.is_model_load_quarantined("model_b") is False

    def test_none_model_is_never_quarantined(self) -> None:
        """A missing model name is never treated as a quarantine candidate."""
        plm = _make_plm()
        assert plm.is_model_load_quarantined(None) is False

    def test_load_failure_reap_is_labelled_and_skips_slot_breaker(self) -> None:
        """A reported load failure labels the recovery as a model-load failure and spares the slot breaker.

        The fault is the model's, not the slot's, so a poison model must not quarantine a healthy slot.
        """
        from horde_worker_regen.process_management.lifecycle.process_lifecycle import CRASH_LOOP_MAX_REPLACEMENTS

        plm = _make_plm()
        # Don't touch real OS processes; only the recovery-classification logic is under test.
        plm._end_inference_process = Mock()  # type: ignore[method-assign]
        plm._start_inference_process = Mock()  # type: ignore[method-assign]
        captured: list[str] = []
        plm.set_process_recovery_observer(lambda _info, reason: captured.append(reason))

        # Many consecutive load-failure replacements of the SAME slot must never trip its crash-loop breaker.
        for _ in range(CRASH_LOOP_MAX_REPLACEMENTS + 2):
            process_info = make_mock_process_info(
                3,
                model_name=None,
                state=HordeProcessState.PROCESS_ENDED,
            )
            process_info.mp_process = Mock(is_alive=Mock(return_value=False), exitcode=0)
            plm.record_model_load_failure(process_id=3, model_name="Z-Image-Turbo")
            plm._replace_inference_process(process_info)

        assert captured, "a recovery should have been reported"
        assert all("failed to load model" in reason for reason in captured)
        assert 3 not in plm._quarantined_inference_slots
        # The slot's crash-loop counter must be untouched: load failures are the model's fault, not the slot's.
        assert plm._slot_recovery_history.get(3, []) == []


class TestStuckOnNonAdvancingStep:
    """The stuck-step watchdog reaps a slot looping on one sampling step, which silence cannot catch.

    Reproduces the live wedge: an inference slot whose ComfyUI generation loops on the final step keeps
    receiving identical progress callbacks and keeps emitting heartbeats, so ``last_heartbeat_timestamp``
    stays fresh and the silence-based hang watchdog never fires. The slot sits in ``INFERENCE_STARTING``
    indefinitely, holding VRAM and a queue slot while never returning a result. The fix reaps it once the
    child-forwarded non-advancing-repeat count crosses the configured limit.
    """

    def _starting_slot(self, *, repeats: int) -> HordeProcessInfo:
        """An INFERENCE_STARTING slot with a fresh heartbeat (not silent) and the given repeat count."""
        proc = make_mock_process_info(1, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
        now = time.time()
        # Fresh liveness on every clock the silence watchdog reads, so only the repeat count can reap it.
        proc.last_heartbeat_timestamp = now
        proc.last_received_timestamp = now
        proc.last_process_state_started_at = now
        proc.last_current_step = 24
        proc.last_total_steps = 25
        proc.nonadvancing_step_repeats = repeats
        return proc

    def test_wedged_slot_is_reaped_despite_fresh_heartbeats(self) -> None:
        """A non-silent slot past the repeat limit is replaced (the bug: it never was)."""
        proc = self._starting_slot(repeats=21)
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_called_once_with(proc)

    def test_slot_below_the_limit_is_left_alone(self) -> None:
        """A stray duplicate report (under the limit) is not a wedge and must not be reaped."""
        proc = self._starting_slot(repeats=2)
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()

    def test_idle_slot_with_a_stale_count_is_not_reaped(self) -> None:
        """The count only reaps a slot that is actually sampling (INFERENCE_STARTING), not an idle one."""
        proc = self._starting_slot(repeats=20)
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()


def test_empty_process_map_is_not_declared_all_unresponsive() -> None:
    """With no inference/safety process running, the hung-detector must not fire (``all([])`` is True).

    During the startup download-and-scan window, and throughout download-only mode, the process map is
    legitimately empty. The all-timed-out verdict over an empty map is vacuously True, which previously
    declared "all processes unresponsive" and tried to recover nothing.
    """
    plm = _make_plm(process_map=ProcessMap({}))

    plm.replace_hung_processes()

    assert plm._hung_processes_detected is False


def test_broadcast_reload_model_database_targets_inference_and_download() -> None:
    """The reload broadcast reaches every inference process and the download process."""
    from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeControlMessage

    process_map = ProcessMap({})
    inf0 = make_mock_process_info(0)
    inf1 = make_mock_process_info(1)
    process_map[0] = inf0
    process_map[1] = inf1

    plm = _make_plm(process_map=process_map)
    download_info = make_mock_process_info(9000, process_type=HordeProcessType.DOWNLOAD, model_name=None)
    plm._download_process_info = download_info

    plm.broadcast_reload_model_database()

    for proc in (inf0, inf1, download_info):
        proc.pipe_connection.send.assert_called_once()  # type: ignore
        sent = proc.pipe_connection.send.call_args.args[0]  # pyrefly: ignore
        assert isinstance(sent, HordeControlMessage)
        assert sent.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE


def test_init_stores_references() -> None:
    """Test that the constructor properly stores references to its dependencies."""
    plm = _make_plm()
    assert plm.num_processes_launched == 0
    assert plm._num_process_recoveries == 0
    assert plm._safety_processes_should_be_replaced is False
    assert plm._safety_processes_ending is False
    assert plm._recently_recovered is False
    assert plm._hung_processes_detected is False
    assert plm._hung_processes_detected_time == 0.0


def test_reset_recovery_counter_zeroes_count_but_keeps_crash_loop_history() -> None:
    """The level-boundary reset zeroes the cumulative counter without forgetting the crash-loop window.

    The warm benchmark worker reuses one pool across levels, so the per-level recovery count must
    reset; but the slot-recovery history that backs the crash-loop breaker must survive so a genuine
    crash loop spanning levels is still caught.
    """
    plm = _make_plm()
    plm._num_process_recoveries = 3
    plm._slot_recovery_history = {1: [time.time()]}

    plm.reset_recovery_counter()

    assert plm._num_process_recoveries == 0
    assert plm._slot_recovery_history == {1: [pytest.approx(plm._slot_recovery_history[1][0])]}


def test_intentional_reclaim_is_not_counted_as_a_crash_recovery() -> None:
    """Cycling a healthy idle slot to reclaim RAM must not feed the crash bookkeeping.

    The RAM budget cycles an idle, model-less process to return allocator-retained RAM to the OS. That
    is a deliberate reclaim, not a crash or hang, so it must not bump ``process_recoveries`` or the
    per-slot crash-loop history; otherwise sustained RAM pressure (3 reclaim-cycles of one slot within
    the window) would spuriously quarantine a perfectly healthy slot.
    """
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    idle.ram_usage_bytes = 5 * 1024 * 1024 * 1024
    plm._process_map[1] = idle

    plm._replace_inference_process(idle, intentional_reclaim=True)

    assert plm._num_process_recoveries == 0
    assert plm._slot_recovery_history.get(1, []) == []
    plm._start_inference_process.assert_called_once_with(1, device_index=0)


def test_maintenance_reload_is_not_counted_or_labelled_as_a_crash() -> None:
    """A deliberate maintenance-mode pool reload must not masquerade as a crash recovery.

    When the horde forces the worker into maintenance, the manager reloads every (healthy, idle) inference
    slot. That is an operational replacement, not a crash or hang: it must not bump ``process_recoveries``,
    must not feed the per-slot crash-loop history, and must be labelled with its real reason in the ledger
    rather than the misleading "crashed or hung" that would make a routine maintenance episode look like a
    crash storm in the recovery diagnostics.
    """
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]
    plm._action_ledger = Mock()

    healthy = make_mock_process_info(1, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
    plm._process_map[1] = healthy

    plm._replace_inference_process(healthy, intentional_reason="maintenance-mode pool reload")

    assert plm._num_process_recoveries == 0
    assert plm._slot_recovery_history.get(1, []) == []
    plm._start_inference_process.assert_called_once_with(1, device_index=0)
    ledger_reason = plm._action_ledger.record.call_args.kwargs["reason"]
    assert ledger_reason == "maintenance-mode pool reload"
    assert "crashed or hung" not in ledger_reason


def test_crash_replacement_still_counts_as_a_recovery() -> None:
    """The ordinary (crash/hang) replacement path must still record a recovery and crash-loop history.

    Guards the intentional-reclaim carve-out against over-reach: a real crash replacement keeps feeding
    the breakers it always did.
    """
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    crashed = make_mock_process_info(1, model_name=None, state=HordeProcessState.INFERENCE_STARTING)
    plm._process_map[1] = crashed

    plm._replace_inference_process(crashed)

    assert plm._num_process_recoveries == 1
    assert len(plm._slot_recovery_history.get(1, [])) == 1


def test_aux_download_teardown_registers_backoff_strike() -> None:
    """Reaping a slot stuck downloading aux models arms the LoRA-pop backoff."""
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    stuck = make_mock_process_info(1, model_name="CyberRealistic Pony", state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    plm._process_map[1] = stuck

    assert plm._state.lora_download_backoff.strikes == 0

    plm._replace_inference_process(stuck)

    assert plm._state.lora_download_backoff.strikes == 1
    assert plm._state.lora_download_backoff.pops_suppressed(time.time())


async def test_aux_stall_retryable_first_then_dropped_during_incident() -> None:
    """The first aux stall keeps its ordinary retry; a stall during the active incident is dropped."""
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]
    plm._job_tracker.handle_job_fault_now = Mock()  # type: ignore[method-assign]

    job = make_job_pop_response()
    await track_popped_job_async(plm._job_tracker, job)

    first = make_mock_process_info(1, model_name="WAI", state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    first.last_job_referenced = job
    plm._process_map[1] = first
    plm._replace_inference_process(first)

    # No incident was active before this strike, so the lone stall keeps its ordinary retry.
    assert plm._job_tracker.handle_job_fault_now.call_args.kwargs["retryable"] is True

    # The strike above made the incident active; a subsequent aux stall is faulted terminally.
    second = make_mock_process_info(1, model_name="WAI", state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    second.last_job_referenced = job
    plm._process_map[1] = second
    plm._replace_inference_process(second)

    assert plm._job_tracker.handle_job_fault_now.call_args.kwargs["retryable"] is False


def test_intentional_reclaim_does_not_register_backoff_strike() -> None:
    """A deliberate idle-slot reclaim is not a download failure and must not arm the backoff."""
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    plm._process_map[1] = idle

    plm._replace_inference_process(idle, intentional_reclaim=True)

    assert plm._state.lora_download_backoff.strikes == 0


def test_effective_aux_download_timeout_shortens_under_backoff() -> None:
    """The stuck-aux grace is the configured timeout until a strike, then the shortened fast-fault value."""
    from horde_worker_regen.process_management.lifecycle.process_lifecycle import FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS

    plm = _make_plm()
    bridge_data = plm._runtime_config.bridge_data

    assert plm._effective_aux_download_timeout(bridge_data) == bridge_data.download_timeout

    plm._state.lora_download_backoff.register_timeout(time.time())
    assert plm._effective_aux_download_timeout(bridge_data) == min(
        bridge_data.download_timeout,
        FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS,
    )


def test_aux_download_deadline_for_dispatch_tracks_watchdog_minus_margin() -> None:
    """The child-side deadline is the (backoff-aware) watchdog timeout minus the safety margin, floored."""
    from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
        AUX_DOWNLOAD_DEADLINE_MARGIN_SECONDS,
        FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS,
        MIN_AUX_DOWNLOAD_DEADLINE_SECONDS,
    )

    plm = _make_plm()
    bridge_data = plm._runtime_config.bridge_data

    expected_idle = bridge_data.download_timeout - AUX_DOWNLOAD_DEADLINE_MARGIN_SECONDS
    assert plm.aux_download_deadline_for_dispatch(bridge_data) == expected_idle

    plm._state.lora_download_backoff.register_timeout(time.time())
    expected_incident = FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS - AUX_DOWNLOAD_DEADLINE_MARGIN_SECONDS
    assert plm.aux_download_deadline_for_dispatch(bridge_data) == expected_incident
    assert plm.aux_download_deadline_for_dispatch(bridge_data) >= MIN_AUX_DOWNLOAD_DEADLINE_SECONDS


def test_get_processes_with_model_for_queued_job_empty() -> None:
    """If there are no processes or no jobs pending inference, the result should be empty."""
    plm = _make_plm()
    result = plm.get_processes_with_model_for_queued_job()
    assert result == []


async def test_get_processes_with_model_for_queued_job_matches() -> None:
    """If there is a waiting process with the needed model, it should be returned."""
    process_map = ProcessMap({})
    job_tracker = JobTracker()

    proc = Mock()
    proc.process_id = 0
    proc.loaded_horde_model_name = "stable_diffusion"
    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    process_map[0] = proc

    queued_job = Mock()
    queued_job.id_ = "queued-job"
    queued_job.model = "stable_diffusion"
    await track_popped_job_async(job_tracker, queued_job)

    plm = _make_plm(process_map=process_map, job_tracker=job_tracker)
    result = plm.get_processes_with_model_for_queued_job()

    assert 0 in result


def test_get_processes_with_model_for_queued_job_preloaded() -> None:
    """If there is a preloaded process with the needed model, it should be returned."""
    process_map = ProcessMap({})

    proc = Mock()
    proc.process_id = 1
    proc.loaded_horde_model_name = "some_other_model"
    proc.last_process_state = HordeProcessState.PRELOADED_MODEL
    process_map[1] = proc

    plm = _make_plm(process_map=process_map)
    result = plm.get_processes_with_model_for_queued_job()

    assert 1 in result


def test_recently_recovered_property() -> None:
    """Test the recently_recovered property getter and setter."""
    plm = _make_plm()
    assert plm.recently_recovered is False

    plm._recently_recovered = True
    assert plm.recently_recovered is True


def test_safety_processes_should_be_replaced_property() -> None:
    """Test the safety_processes_should_be_replaced property getter and setter."""
    plm = _make_plm()
    assert plm.safety_processes_should_be_replaced is False

    plm.safety_processes_should_be_replaced = True
    assert plm.safety_processes_should_be_replaced is True


def test_pause_and_restore_safety_on_gpu_toggles_override_and_arms_replacement() -> None:
    """Pausing safety-on-GPU sets the cpu_only override and arms a replacement; restoring clears it.

    This is how a whole-card (single-residency) job frees the safety process's CUDA context: a context is
    only reclaimed by the process exiting, so the safety process is cycled to come back up off-GPU.
    """
    plm = _make_plm()
    plm._runtime_config.bridge_data.safety_on_gpu = True

    assert plm.is_safety_gpu_paused is False
    assert plm.pause_safety_on_gpu() is True
    assert plm.is_safety_gpu_paused is True
    # The existing safety-replacement state machine was armed (so the on-GPU process is cycled off-GPU)...
    assert plm.safety_processes_should_be_replaced is True
    # ...and the cycle is marked intentional so its completion is not counted as a crash recovery.
    assert plm._safety_replacement_intentional is True
    # Idempotent: a second pause does nothing.
    assert plm.pause_safety_on_gpu() is False

    assert plm.restore_safety_on_gpu() is True
    assert plm.is_safety_gpu_paused is False
    assert plm._safety_replacement_intentional is True
    # Idempotent: restoring when not paused does nothing.
    assert plm.restore_safety_on_gpu() is False


def test_pause_safety_on_gpu_is_noop_when_safety_not_on_gpu() -> None:
    """With safety not configured on-GPU there is no context to free, so the pause is a no-op."""
    plm = _make_plm()  # _make_plm defaults safety_on_gpu to False
    assert plm.pause_safety_on_gpu() is False
    assert plm.is_safety_gpu_paused is False


def test_pause_and_restore_post_process_lane_stops_and_restarts_it_for_whole_card() -> None:
    """A whole-card model stops the post-processing lane (freeing its context), then restarts it after.

    Unlike safety (which cycles cpu_only), the lane has no useful CPU fallback, so it is stopped outright:
    the per-tick start hook must stay a no-op while paused so the lane does not resurrect, and clearing the
    pause lets the next start bring it back. The pause/restart is marked intentional so it is not counted as
    a lane crash.
    """
    running_lane = make_mock_process_info(1, process_type=HordeProcessType.POST_PROCESS)
    plm = _make_plm(process_map=ProcessMap({1: running_lane}))
    plm._runtime_config.bridge_data.post_processing_lane_enabled = True

    assert plm.is_post_process_gpu_paused is False
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.is_post_process_gpu_paused is True
    # The lane-replacement state machine was armed (to end the running lane) and marked intentional.
    assert plm.post_process_processes_should_be_replaced is True
    assert plm._post_process_replacement_intentional is True
    # Idempotent: a second pause does nothing.
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is False

    # The per-tick start hook must not resurrect the lane while it is paused.
    plm.start_post_process_processes()
    assert plm._process_map.num_post_process_processes() == 1  # only the still-ending original, no new start

    assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.is_post_process_gpu_paused is False
    # Idempotent: restoring when not paused does nothing.
    assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is False


def test_restore_post_process_lane_starts_it_when_none_running() -> None:
    """Restoring after a whole-card pause must itself relaunch the lane.

    By restore time the replacement state machine has already ended and deleted the paused lane process
    and consumed its flag (its final start call was suppressed by the pause gate), and the bring-up
    callers are one-shot latches. If the restore did not start the lane directly, no code path would
    ever bring it back, and every post-processing job for the rest of the session would queue against a
    lane that never returns.
    """
    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345
    fake_ctx.Process.return_value.exitcode = None

    plm = _make_plm(ctx=fake_ctx)
    plm._runtime_config.bridge_data.post_processing_lane_enabled = True
    plm._runtime_config.bridge_data.dry_run_skip_post_processing = False

    assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    # The paused-lane teardown already ran to completion: no lane process remains in the map.
    assert plm._process_map.num_post_process_processes() == 0

    assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm._process_map.num_post_process_processes() == 1


def test_pause_post_process_lane_is_noop_when_lane_disabled() -> None:
    """With the dedicated lane disabled there is no lane to stop, so the pause is a no-op."""
    plm = _make_plm()
    plm._runtime_config.bridge_data.post_processing_lane_enabled = False
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_post_process_gpu_paused is False


def test_pause_and_restore_vae_lane_stops_and_restarts_it_for_whole_card() -> None:
    """A whole-card model stops the disaggregation VAE lane (freeing its context), then restarts it after.

    Mirrors the post-processing lane's whole-card yield: the lane is stopped outright (its CUDA context is
    only reclaimable by the process exiting), the per-tick start hook stays a no-op while paused, and the
    pause/restart is marked intentional so it is not counted as a lane crash.
    """
    running_lane = make_mock_process_info(1, process_type=HordeProcessType.VAE_LANE)
    plm = _make_plm(process_map=ProcessMap({1: running_lane}))
    plm._runtime_config.bridge_data.enable_pipeline_disaggregation = True

    assert plm.is_vae_lane_gpu_paused is False
    assert plm.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.is_vae_lane_gpu_paused is True
    # The lane-replacement state machine was armed (to end the running lane) and marked intentional.
    assert plm.vae_lane_processes_should_be_replaced is True
    assert plm._vae_lane_replacement_intentional is True
    # Idempotent: a second pause does nothing.
    assert plm.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is False

    # The per-tick start hook must not resurrect the lane while it is paused.
    plm.start_vae_lane_processes()
    assert plm._process_map.num_vae_lane_processes() == 1  # only the still-ending original, no new start

    assert plm.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.is_vae_lane_gpu_paused is False
    # Idempotent: restoring when not paused does nothing.
    assert plm.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is False


def test_restore_vae_lane_starts_it_when_none_running() -> None:
    """Restoring after a whole-card pause must itself relaunch the VAE lane.

    By restore time the replacement state machine has already ended and deleted the paused lane and
    consumed its flag, and the bring-up callers are one-shot latches. If the restore did not start the lane
    directly, every VAE stage for the rest of the session would queue against a lane that never returns.
    """
    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345
    fake_ctx.Process.return_value.exitcode = None

    plm = _make_plm(ctx=fake_ctx)
    plm._runtime_config.bridge_data.enable_pipeline_disaggregation = True
    plm._runtime_config.bridge_data.dry_run_skip_inference = False

    assert plm.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    # No lane was running when paused, so nothing remains in the map.
    assert plm._process_map.num_vae_lane_processes() == 0

    assert plm.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm._process_map.num_vae_lane_processes() == 1


def test_pause_vae_lane_is_noop_when_disaggregation_disabled() -> None:
    """With disaggregation disabled the VAE lane never runs, so the pause is a no-op."""
    plm = _make_plm()  # _make_plm defaults enable_pipeline_disaggregation to False
    assert plm.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_vae_lane_gpu_paused is False


def test_post_process_lane_pause_ownership_is_isolated() -> None:
    """A ladder-owned PP pause is not lifted by a whole-card restore, and vice versa: only the owner clears it.

    The two initiators of a lane pause (the whole-card residency and the reclaim ladder) must not clear each
    other's hold. A ladder pause cleared by the residency completion loop (which has no grant to complete)
    would be the original wedge; a residency pause cleared by the ladder's unwind would restart the lane while
    a heavy model still needs the card.
    """
    running_lane = make_mock_process_info(1, process_type=HordeProcessType.POST_PROCESS)
    plm = _make_plm(process_map=ProcessMap({1: running_lane}))
    plm._runtime_config.bridge_data.post_processing_lane_enabled = True

    # A ladder-owned pause records the ladder as owner and refuses a whole-card restore.
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.post_process_pause_owner is PauseOwner.RECLAIM_LADDER
    assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_post_process_gpu_paused is True
    # The ladder's own restore clears it.
    assert plm.restore_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_post_process_gpu_paused is False
    assert plm.post_process_pause_owner is None

    # Symmetric: a whole-card pause refuses a ladder unwind and is cleared only by the whole-card restore.
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.post_process_pause_owner is PauseOwner.WHOLE_CARD
    assert plm.restore_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is False
    assert plm.is_post_process_gpu_paused is True
    assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    assert plm.is_post_process_gpu_paused is False


def test_vae_lane_pause_ownership_is_isolated() -> None:
    """The VAE lane honors the same pause ownership: only the initiating owner's restore clears it."""
    running_lane = make_mock_process_info(1, process_type=HordeProcessType.VAE_LANE)
    plm = _make_plm(process_map=ProcessMap({1: running_lane}))
    plm._runtime_config.bridge_data.enable_pipeline_disaggregation = True

    assert plm.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.vae_lane_pause_owner is PauseOwner.RECLAIM_LADDER
    assert plm.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_vae_lane_gpu_paused is True
    assert plm.restore_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_vae_lane_gpu_paused is False


def test_component_lane_pause_ownership_is_isolated() -> None:
    """The component lane honors the same pause ownership: only the initiating owner's restore clears it."""
    running_lane = make_mock_process_info(1, process_type=HordeProcessType.COMPONENT)
    plm = _make_plm(process_map=ProcessMap({1: running_lane}))
    plm._runtime_config.bridge_data.enable_pipeline_disaggregation = True

    assert plm.pause_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.component_pause_owner is PauseOwner.RECLAIM_LADDER
    assert plm.restore_component_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_component_gpu_paused is True
    assert plm.restore_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_component_gpu_paused is False


def _captured_safety_cpu_only(plm: ProcessLifecycleManager) -> bool:
    """Start a safety process with a mocked spawn and return the cpu_only arg it was launched with."""
    # A real integer pid so the owned-process ledger event validates (it records os_pid).
    plm._new_process = Mock(return_value=Mock(pid=12345))  # type: ignore[method-assign]
    plm.start_safety_processes()
    # The safety entry point's positional args are (pid, queue, pipe, lock, launch_id, cpu_only).
    return bool(plm._new_process.call_args.kwargs["args"][5])


def test_safety_forced_cpu_only_on_cpu_install_even_with_safety_on_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only torch build has no CUDA device, so the safety process must come up off-GPU.

    Without this the safety models load on 'cuda' and raise during init; the configured safety_on_gpu=true
    must be overridden by the install reality.
    """
    monkeypatch.setattr(
        "horde_worker_regen.process_management.lifecycle.process_lifecycle.is_cpu_only_install",
        lambda: True,
    )
    plm = _make_plm()
    plm._runtime_config.bridge_data.safety_on_gpu = True

    assert _captured_safety_cpu_only(plm) is True


def test_safety_on_gpu_respected_on_gpu_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a GPU install with safety_on_gpu=true and no pause, the safety process comes up on-GPU."""
    monkeypatch.setattr(
        "horde_worker_regen.process_management.lifecycle.process_lifecycle.is_cpu_only_install",
        lambda: False,
    )
    plm = _make_plm()
    plm._runtime_config.bridge_data.safety_on_gpu = True

    assert _captured_safety_cpu_only(plm) is False


def test_intentional_safety_cycle_not_counted_as_recovery() -> None:
    """A whole-card safety pause/restore cycle completing must not bump recoveries or the crash-loop breaker.

    Otherwise a burst of whole-card jobs cycling safety off/on reads as a safety crash loop and trips
    save-our-ship (the observed instability after the overflow fix).
    """
    plm = _make_plm()
    plm.start_safety_processes = Mock()  # type: ignore[method-assign]  # avoid spawning a real process on completion

    # Drive the replacement state machine to its completion branch with the intentional flag set.
    plm._safety_replacement_intentional = True
    plm._safety_processes_should_be_replaced = True
    plm._safety_processes_ending = True  # already in the ending phase; map is empty so it completes now
    before = plm._num_process_recoveries

    plm._replace_all_safety_process()

    assert plm._num_process_recoveries == before  # not counted as a recovery
    assert plm._safety_recovery_history == []  # crash-loop breaker not fed
    assert plm._safety_replacement_intentional is False  # consumed
    assert plm._safety_replacement_intentional_until_ready is True

    plm._process_map[0] = make_mock_process_info(
        0,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    plm._clear_completed_intentional_safety_replacement()
    assert plm._safety_replacement_intentional_until_ready is False
    plm._process_map.clear()

    # A crash-driven rebuild (flag clear) DOES count.
    plm._safety_processes_should_be_replaced = True
    plm._safety_processes_ending = True
    plm._replace_all_safety_process()
    assert plm._num_process_recoveries == before + 1
    assert len(plm._safety_recovery_history) == 1


def test_intentional_safety_replacement_startup_churn_not_counted_as_recovery() -> None:
    """Safety replacements before the new safety process is loaded remain part of the placement change."""
    plm = _make_plm()
    plm.start_safety_processes = Mock()  # type: ignore[method-assign]
    before = plm._num_process_recoveries

    plm._safety_replacement_intentional = True
    plm._safety_processes_should_be_replaced = True
    plm._safety_processes_ending = True
    plm._replace_all_safety_process()

    assert plm._num_process_recoveries == before
    assert plm._safety_replacement_intentional_until_ready is True

    plm._process_map[0] = make_mock_process_info(
        0,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    plm._safety_processes_should_be_replaced = True
    plm._safety_processes_ending = True
    plm._replace_all_safety_process()

    assert plm._num_process_recoveries == before
    assert plm._safety_recovery_history == []
    assert plm._safety_replacement_intentional_until_ready is True


def test_soft_reset_safety_rebuild_not_counted_as_recovery() -> None:
    """A Save-our-ship soft reset rebuilding the (healthy) safety pool must not bump recoveries.

    The soft reset rebuilds both pools to give a wedged worker a clean start; the safety pool is usually
    healthy collateral, so counting its deliberate rebuild double-counts a single wedge (two process
    recoveries from one soft reset, when only the inference slot was actually wedged). Like the inference
    rebuild, it is a supervised rebuild: it clears the safety crash-loop history and is not a crash
    recovery.
    """
    plm = _make_plm()
    plm.start_safety_processes = Mock()  # type: ignore[method-assign]
    plm.end_safety_processes = Mock()  # type: ignore[method-assign]
    plm._safety_recovery_history = [time.time()]  # stale history a deliberate rebuild must clear
    before = plm._num_process_recoveries

    plm.rebuild_safety_pool(reason="soft reset #1")
    # Drive the replacement state machine to completion (empty map => it finishes on the next tick).
    plm._replace_all_safety_process()

    assert plm._num_process_recoveries == before  # the deliberate rebuild is not a crash recovery
    assert plm._safety_recovery_history == []  # crash-loop breaker reset, mirroring rebuild_inference_pool
    assert plm._safety_replacement_intentional is False  # consumed


def test_end_safety_processes_stops_starting_process_and_marks_intent() -> None:
    """Shutdown must send END_PROCESS even if safety has not reached WAITING_FOR_JOB yet."""
    safety = make_mock_process_info(
        0,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    plm = _make_plm(process_map=ProcessMap({0: safety}))

    assert safety.end_intended is False

    plm.end_safety_processes()

    assert safety.end_intended is True
    assert safety.last_process_state == HordeProcessState.PROCESS_ENDING
    safety.pipe_connection.send.assert_called_once()  # type: ignore[attr-defined]
    sent = safety.pipe_connection.send.call_args.args[0]  # pyrefly: ignore
    assert sent.control_flag == HordeControlFlag.END_PROCESS


def _patch_spawn_with_stub(plm: ProcessLifecycleManager) -> None:
    """Replace real process spawning with a stub that adds an idle mock process to the map."""

    def _fake_start(pid: int, *, device_index: int = 0) -> HordeProcessInfo:
        info = make_mock_process_info(pid, model_name=None, process_type=HordeProcessType.INFERENCE)
        info.device_index = device_index
        plm._process_map[pid] = info
        plm.num_processes_launched += 1
        return info

    plm._start_inference_process = _fake_start  # type: ignore[method-assign]


def test_allocate_inference_pid_picks_lowest_free() -> None:
    """The pid allocator returns the lowest unused slot id, reusing freed ones."""
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, process_type=HordeProcessType.SAFETY),
            1: make_mock_process_info(1, model_name=None),
        },
    )
    plm = _make_plm(process_map=process_map)
    assert plm._allocate_inference_pid() == 2

    process_map.pop(1)
    assert plm._allocate_inference_pid() == 1


def test_scale_up_starts_processes_up_to_ceiling() -> None:
    """Scaling up spawns processes, bounded by max_inference_processes."""
    plm = _make_plm()  # max_inference_processes=2
    _patch_spawn_with_stub(plm)

    assert plm.scale_inference_processes(2) == 2
    assert plm._process_map.num_inference_processes() == 2
    # Inference processes occupy the range from 1 upward; slot 0 is reserved for the safety process.
    assert sorted(plm._process_map.keys()) == [1, 2]

    # Requests beyond the launched ceiling are capped.
    assert plm.scale_inference_processes(5) == 2


def test_scale_down_stops_idle_processes() -> None:
    """Scaling down ends idle inference processes and removes them from the map."""
    plm = _make_plm()
    _patch_spawn_with_stub(plm)
    plm.scale_inference_processes(2)
    before = dict(plm._process_map)

    assert plm.scale_inference_processes(1) == 1
    assert plm._process_map.num_inference_processes() == 1
    retired_pid = (set(before) - set(plm._process_map)).pop()
    retired_process = before[retired_pid]
    assert plm._process_map.is_retired_launch(
        retired_process.process_id,
        retired_process.process_launch_identifier,
    )


def test_scale_down_never_kills_busy_processes() -> None:
    """A busy (mid-inference) process is retained even when scaling toward zero."""
    busy = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
    idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    plm = _make_plm(process_map=ProcessMap({0: busy, 1: idle}))

    plm.scale_inference_processes(0)

    remaining = list(plm._process_map.keys())
    assert remaining == [0]  # the busy process survives


def test_pid_reused_after_scale_down_then_up() -> None:
    """A slot freed by scaling down is reused on the next scale up (no collision)."""
    plm = _make_plm()
    _patch_spawn_with_stub(plm)
    plm.scale_inference_processes(2)  # pids 1, 2 (slot 0 reserved for safety)
    plm.scale_inference_processes(1)  # removes an idle slot
    plm.scale_inference_processes(2)  # re-allocates the freed slot

    assert sorted(plm._process_map.keys()) == [1, 2]


def test_stuck_starting_safety_arms_replacement() -> None:
    """A safety process stuck in PROCESS_STARTING must actually arm its replacement.

    Regression: the stuck-detection used to call `_replace_all_safety_process()` without first
    setting the flag it gates on, so the safety branch was a silent no-op that logged "replacing it"
    forever while leaving the wedged process in place.
    """
    safety = make_mock_process_info(
        0, model_name=None, state=HordeProcessState.PROCESS_STARTING, process_type=HordeProcessType.SAFETY
    )
    # Age the process's last-seen timestamps so the elapsed-time check trips against timeout=0.
    safety.last_received_timestamp = time.time() - 1000
    safety.last_heartbeat_timestamp = time.time() - 1000
    plm = _make_plm(process_map=ProcessMap({0: safety}))

    assert plm.safety_processes_should_be_replaced is False
    replaced = plm._check_and_replace_process(safety, 0.0, HordeProcessState.PROCESS_STARTING, "stuck")

    assert replaced is True
    assert plm.safety_processes_should_be_replaced is True


def test_aux_download_timeout_uses_state_duration_not_recent_liveness() -> None:
    """AUX download replacement is bounded by time in state, not by heartbeat silence.

    The child now emits liveness while blocked in the LoRA download path. That should keep the worker
    from looking globally unresponsive, but it must not make a download unkillable if it exceeds the
    configured operation timeout.
    """
    aux = make_mock_process_info(0, model_name="m", state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    now = time.time()
    aux.last_process_state_started_at = now - 1000
    aux.last_received_timestamp = now
    aux.last_heartbeat_timestamp = now
    plm = _make_plm(process_map=ProcessMap({0: aux}))
    plm._replace_inference_process = Mock()  # type: ignore[method-assign]

    replaced = plm._check_and_replace_process(
        aux,
        120.0,
        HordeProcessState.DOWNLOADING_AUX_MODEL,
        "stuck downloading",
        use_state_duration=True,
    )

    assert replaced is True
    plm._replace_inference_process.assert_called_once_with(aux)


def test_silence_timeout_still_uses_recent_liveness_by_default() -> None:
    """Non-operation checks keep their existing silence-based behavior."""
    aux = make_mock_process_info(0, model_name="m", state=HordeProcessState.DOWNLOADING_AUX_MODEL)
    now = time.time()
    aux.last_process_state_started_at = now - 1000
    aux.last_received_timestamp = now
    aux.last_heartbeat_timestamp = now
    plm = _make_plm(process_map=ProcessMap({0: aux}))
    plm._replace_inference_process = Mock()  # type: ignore[method-assign]

    replaced = plm._check_and_replace_process(
        aux,
        120.0,
        HordeProcessState.DOWNLOADING_AUX_MODEL,
        "stuck downloading",
    )

    assert replaced is False
    plm._replace_inference_process.assert_not_called()


def test_reap_if_crashed_recovers_dead_inference() -> None:
    """A dead inference child (no longer alive) is recovered without waiting on a state timer."""
    dead = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = 1  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({1: dead}))
    plm._replace_inference_process = Mock()  # type: ignore[method-assign]

    assert plm._reap_if_crashed(dead) is True
    plm._replace_inference_process.assert_called_once_with(dead)


def test_reap_if_crashed_recovers_dead_safety() -> None:
    """A dead safety child arms the safety-replacement state machine."""
    dead = make_mock_process_info(
        0, model_name=None, state=HordeProcessState.PROCESS_STARTING, process_type=HordeProcessType.SAFETY
    )
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = -9  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({0: dead}))

    assert plm._reap_if_crashed(dead) is True
    assert plm.safety_processes_should_be_replaced is True


def test_reap_if_crashed_ignores_live_and_intentionally_ended_processes() -> None:
    """A live process, or one we deliberately ended, is never reaped as a crash."""
    live = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    plm = _make_plm(process_map=ProcessMap({1: live}))
    assert plm._reap_if_crashed(live) is False

    # A dead slot whose end the supervisor *intended* (shutdown/scale-down/replacement) is left alone,
    # even though its OS process has exited and it reports an ending state.
    intended = make_mock_process_info(2, model_name=None, state=HordeProcessState.PROCESS_ENDING)
    intended.mp_process.is_alive.return_value = False
    intended.end_intended = True
    assert plm._reap_if_crashed(intended) is False


def test_reap_if_crashed_recovers_unintended_ended_process() -> None:
    """A dead slot reporting an ending state that we did *not* intend is recovered, not left wedged.

    The soak wedge: a child exited during preload and reported PROCESS_ENDED via its graceful shutdown
    path (it was never sent END_PROCESS), so state alone is indistinguishable from an intended end. With
    intent tracked separately, the reaper recovers it.
    """
    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    crashed = make_mock_process_info(3, model_name=None, state=HordeProcessState.PROCESS_ENDED)
    crashed.mp_process.is_alive.return_value = False
    crashed.mp_process.exitcode = 0  # pyrefly: ignore
    plm._process_map[3] = crashed
    assert crashed.end_intended is False

    assert plm._reap_if_crashed(crashed) is True
    plm._start_inference_process.assert_called_once_with(3, device_index=0)
    assert plm._num_process_recoveries == 1


def test_oom_kill_is_labelled_oom_and_spares_the_slot_crash_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``-9`` exit while system RAM is critically low is an OS OOM kill, not a slot crash or hang.

    When the kernel OOM-killer terminates an inference process (``exitcode=-9``) while system RAM is at or
    below the danger floor, the current reaper labels it "inference process replaced (crashed or hung)":
    (a) misleading, because the process was fine (the *host* ran out of memory) and (b) harmful, because
    it feeds the per-slot crash-loop breaker, which would quarantine a perfectly healthy slot for a host-wide
    RAM problem no slot teardown can fix.

    A ``-9`` exit with critically-low system RAM should be labelled an OS OOM kill (a recoverable resource
    failure) and kept out of the per-slot crash-loop history. The host-memory governor and pop throttle
    address the cause, not slot quarantine. A ``-9`` with healthy RAM stays an ordinary crash (covered by
    ``test_crash_replacement_still_counts_as_a_recovery``).
    """
    import psutil

    plm = _make_plm()
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    # Spy on the real ledger's record (a full Mock ledger would break _log_recovery_diagnostics, which
    # reads recent_actions); capture the PROCESS_REPLACED reason while still recording for real.
    recorded: list[dict[str, object]] = []
    real_record = plm._action_ledger.record

    def _capture(*args: object, **kwargs: object) -> object:
        recorded.append(kwargs)
        return real_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(plm._action_ledger, "record", _capture)

    # The host is out of RAM: ~0.9 GB available of 31.3 GiB (well below any sane danger floor).
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: Mock(available=int(900 * 1024 * 1024), total=int(31.3 * 1024 * 1024 * 1024), percent=97.0),
    )

    killed = make_mock_process_info(
        1,
        model_name="Flux.1-Schnell fp8 (Compact)",
        state=HordeProcessState.PRELOADING_MODEL,
    )
    killed.mp_process = Mock(is_alive=Mock(return_value=False), exitcode=-9, pid=100001)
    plm._process_map[1] = killed

    plm._replace_inference_process(killed)

    reason = next(str(k["reason"]) for k in recorded if "reason" in k)
    assert "crashed or hung" not in reason, "an OS OOM-kill must not be mislabelled a slot crash or hang"
    assert "oom" in reason.lower() or "out of memory" in reason.lower(), (
        "a -9 exit with critically-low system RAM should be labelled an OS OOM kill"
    )
    # A host-RAM OOM is not slot sickness: repeated OOM kills must not quarantine an otherwise-healthy slot.
    assert plm._slot_recovery_history.get(1, []) == [], "OS OOM kills must not feed the per-slot crash-loop breaker"


class TestPagedSlowdownWatchdog:
    """The last reclaim rung: a crawling sampler is replaced once the card is wedged over the paging cliff.

    A paged sampling process keeps emitting steps, so its heartbeat stays fresh and the silence-based hang
    watchdog never trips; the card is lost for minutes. The kill is the terminal reclaim rung, gated on
    device-level truth rather than per-PID attribution: the card has been continuously SATURATED past the
    kill horizon, the verified reclaim ladder has exhausted itself without relieving it, and the slot is
    crawling (its per-step floor tripped, or its whole-job elapsed grade reached WARN). The per-PID PDH
    victim map no longer gates anything: WDDM demotes the least-recently-touched allocator, so the slow slot
    and the demoted slot are usually different pids, making that gate structurally unsatisfiable.
    """

    def _crawling_slot(
        self,
        *,
        per_step_floor_tripped: bool = True,
        slowdown_level: int = 0,
        expected_seconds: float = 10.0,
        elapsed_seconds: float = 40.0,
    ) -> HordeProcessInfo:
        proc = make_mock_process_info(1, model_name="Flux.1", state=HordeProcessState.INFERENCE_STARTING)
        proc.current_job_slowdown_level = slowdown_level
        proc.current_job_per_step_floor_tripped = per_step_floor_tripped
        proc.current_job_expected_sampling_seconds = expected_seconds
        proc.current_first_step_at = time.time() - elapsed_seconds
        proc.last_heartbeat_timestamp = time.time()
        proc.last_received_timestamp = time.time()
        return proc

    @staticmethod
    def _arm_saturation(plm: ProcessLifecycleManager, *, seconds: float = 15.0, unresolved: bool = True) -> None:
        plm._device_saturation_duration_provider = lambda _device_index: seconds
        plm._saturation_unresolved_provider = lambda _device_index: unresolved

    def test_saturated_exhausted_and_crawling_slot_is_replaced(self) -> None:
        """All three device-level gates met (SATURATED past horizon, ladder exhausted, slot crawling) replace it."""
        proc = self._crawling_slot()
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_called_once()
        assert plm._replace_inference_process.call_args.args[0] is proc
        reason = plm._replace_inference_process.call_args.kwargs["resource_fault_reason"]
        assert "SATURATED" in reason and "4.0x" in reason
        assert plm._paging_victim_replacements == 1

    def test_warn_grade_satisfies_the_crawl_condition(self) -> None:
        """The whole-job WARN elapsed grade also counts as crawling, even without a per-step-floor trip."""
        proc = self._crawling_slot(per_step_floor_tripped=False, slowdown_level=2)
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_called_once()
        assert plm._paging_victim_replacements == 1

    def test_not_saturated_long_enough_is_not_replaced(self) -> None:
        """A card that has not been SATURATED past the kill horizon does not reach the last rung."""
        proc = self._crawling_slot()
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm, seconds=3.0)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()
        assert plm._paging_victim_replacements == 0

    def test_ladder_not_exhausted_is_not_replaced(self) -> None:
        """While the reclaim ladder is still yielding (not exhausted), a softer rung owns the card, not a kill."""
        proc = self._crawling_slot()
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm, unresolved=False)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()
        assert plm._paging_victim_replacements == 0

    def test_not_crawling_is_not_replaced(self) -> None:
        """A saturated, ladder-exhausted card with a slot that is NOT crawling leaves the slot alone.

        The slot is well within its expected sampling time (no per-step-floor trip and no WARN grade), so the
        elapsed-ratio grader also leaves it below WARN.
        """
        proc = self._crawling_slot(per_step_floor_tripped=False, slowdown_level=0, elapsed_seconds=5.0)
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()
        assert plm._paging_victim_replacements == 0

    def test_pdh_victim_map_no_longer_gates(self) -> None:
        """With every device-level gate met, the slot is replaced even though the PDH victim map is empty.

        This is the crux of the rework: the old contract required the slot's own pid in the paging-victim
        set, which the LRU physics made structurally unsatisfiable. That gate is gone; the map is hint-only.
        """
        proc = self._crawling_slot()
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        plm._wddm_paging_victims_provider = lambda _max_age: {}
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_called_once()

    def test_pdh_victim_pid_alone_does_not_replace(self) -> None:
        """The per-PID paging attribution alone (device not saturated/exhausted) never triggers the kill now."""
        proc = self._crawling_slot()
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        plm._wddm_paging_victims_provider = lambda _max_age: {proc.os_pid: 512.0}
        # Device-level gates left at their non-firing defaults (0s saturation, ladder not exhausted).

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()
        assert plm._paging_victim_replacements == 0

    def test_idle_slot_is_never_replaced(self) -> None:
        """A slot that is not actively sampling (not INFERENCE_STARTING) is never the last rung."""
        proc = self._crawling_slot()
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        plm = _make_plm(process_map=ProcessMap({1: proc}))
        plm._replace_inference_process = Mock()  # type: ignore[method-assign]
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        plm._replace_inference_process.assert_not_called()

    async def test_replaced_job_takes_the_degraded_retry_path(self) -> None:
        """The killed job faults as a resource failure, so it earns the one degraded/isolated retry."""
        plm = _make_plm()
        plm._end_inference_process = Mock()  # type: ignore[method-assign]
        plm._start_inference_process = Mock()  # type: ignore[method-assign]
        plm._job_tracker.set_retry_policy(2)

        job = make_job_pop_response(model="Flux.1")
        await plm._job_tracker.record_popped_job(job)
        await plm._job_tracker.mark_inference_started(job)

        proc = self._crawling_slot()
        proc.last_job_referenced = job
        plm._process_map[1] = proc
        self._arm_saturation(plm)

        plm.replace_hung_processes()

        assert plm._paging_victim_replacements == 1
        assert plm._job_tracker.is_degraded_dispatch_pending(job) is True
