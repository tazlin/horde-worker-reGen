"""Tests for ProcessLifecycleManager."""

from __future__ import annotations

import multiprocessing
import sys
import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
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
) -> ProcessLifecycleManager:
    """Helper to build a PLM with mostly-mocked dependencies."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.safety_on_gpu = False
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
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
        from horde_worker_regen.process_management.process_lifecycle import (
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
        plm = _make_plm()
        # Two different models each failing under the threshold must not combine to a quarantine.
        plm.record_model_load_failure(process_id=1, model_name="model_a")
        plm.record_model_load_failure(process_id=2, model_name="model_b")
        assert plm.is_model_load_quarantined("model_a") is False
        assert plm.is_model_load_quarantined("model_b") is False

    def test_none_model_is_never_quarantined(self) -> None:
        plm = _make_plm()
        assert plm.is_model_load_quarantined(None) is False

    def test_load_failure_reap_is_labelled_and_skips_slot_breaker(self) -> None:
        """A reported load failure labels the recovery as a model-load failure and spares the slot breaker.

        The fault is the model's, not the slot's, so a poison model must not quarantine a healthy slot.
        """
        from horde_worker_regen.process_management.process_lifecycle import CRASH_LOOP_MAX_REPLACEMENTS

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
    from horde_worker_regen.process_management.messages import HordeControlFlag, HordeControlMessage

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

    # A crash-driven rebuild (flag clear) DOES count.
    plm._safety_processes_should_be_replaced = True
    plm._safety_processes_ending = True
    plm._replace_all_safety_process()
    assert plm._num_process_recoveries == before + 1
    assert len(plm._safety_recovery_history) == 1


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
    assert sorted(plm._process_map.keys()) == [0, 1]

    # Requests beyond the launched ceiling are capped.
    assert plm.scale_inference_processes(5) == 2


def test_scale_down_stops_idle_processes() -> None:
    """Scaling down ends idle inference processes and removes them from the map."""
    plm = _make_plm()
    _patch_spawn_with_stub(plm)
    plm.scale_inference_processes(2)
    retired_process = plm._process_map[0]

    assert plm.scale_inference_processes(1) == 1
    assert plm._process_map.num_inference_processes() == 1
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
    plm.scale_inference_processes(2)  # pids 0, 1
    plm.scale_inference_processes(1)  # removes the first idle slot (pid 0)
    plm.scale_inference_processes(2)  # should re-allocate pid 0

    assert sorted(plm._process_map.keys()) == [0, 1]


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
