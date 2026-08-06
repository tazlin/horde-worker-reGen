"""Generated chaos against real child processes: the same seeded compositions, at full-worker altitude.

``tests/process_management/liveness/test_chaos_generated.py`` drives generated scenarios through the
scheduling loop on a fake clock. This module drives a bounded production-topology projection of the same
seeds through the whole worker: the real
process manager, real OS child processes running the protocol-faithful fakes, the real message pump,
recovery supervisor, safety and submit paths, and real wall-clock time. Where that module can express a
lane death and an outside reclaim, this one can express what a misbehaving child does: crashing mid-job,
running slow, reporting an out-of-memory result, emitting a misrouted message.

The property is the same one, read through what a whole run reports:

    Every job the scenario queues is completed, no job is given up on by scheduling recovery, no lifecycle
    invariant is violated, and no job's queue wait exceeds a bound derived from the scenario's own shape.

Every child disturbance also needs a receipt: replacement for a lane death, retry for a resource fault,
dispatcher rejection for a stale message, or an elevated sampling duration for a slow child. Generated
child faults use one lane and one event so the fake's process-local ordinal has one global target.

The give-up is read from the worker's own action ledger rather than inferred from a fault count, because a
give-up faults accepted work for reissue and that is exactly the outcome the campaign counts as the
failure rather than the remedy.

Each scenario boots a worker and spawns real child processes, so this module carries both ``slow`` and
``chaos_sweep`` and runs as a pre-release gate and nightly rather than in the default suite:

    pytest tests/e2e/test_chaos_generated_e2e.py -m "chaos_sweep and slow"

``HORDE_CHAOS_SEEDS`` overrides which seeds run (``HORDE_CHAOS_SEEDS=1007`` to replay one), and every
failure message carries the seed and the whole scenario.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from collections.abc import Iterator

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen import harness as harness_module
from horde_worker_regen.harness import HarnessConfig, HarnessResult, HarnessStageDeadlines, run_harness_async
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager, SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.simulation._canned_scenarios import ArrivalSchedule, make_canned_job
from horde_worker_regen.process_management.simulation.chaos_scenarios import (
    DISCLOSED_BOUNDS,
    FULL_WORKER_CANDIDATE_SEEDS,
    SEED_ENV_VAR,
    SWEEP_FULL_WORKER_SEEDS,
    ChaosArrival,
    ChaosAuxKind,
    ChaosControlKind,
    ChaosEventKind,
    ChaosJob,
    ChaosPostProcessing,
    ChaosSamplerProfile,
    ChaosScenario,
    ChaosSourceMode,
    generate_full_worker_scenarios,
    parse_seed_spec,
)
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile

# Spawning a fresh child re-imports the worker package and remains slower on Windows. The fake utilities
# lane and model reference are in-memory, however, so the multiplier covers interpreter startup only; it
# must not hide a stalled queue behind allowances for external environments or network work.
_SPAWN_SLOWDOWN = 1.5 if sys.platform == "win32" else 1.0

_BASE_TIMEOUT_SECONDS = 25.0
"""Boot, model preload, and teardown for a run that queues nothing unusual."""

_TIMEOUT_SECONDS_PER_JOB = 2.0
"""Added per queued job."""

_TIMEOUT_SECONDS_PER_EVENT = 12.0
"""Added per disturbance: a crashed child is detected immediately, but its replacement is a fresh spawn."""

_QUEUE_WAIT_SECONDS_PER_JOB = 3.0
"""Queue wait allowed per job ahead of one in the queue."""

_QUEUE_WAIT_BASE_SECONDS = 20.0
"""Queue wait allowed for the boot and first preload every job's wait includes."""

_QUEUE_WAIT_SECONDS_PER_EVENT = 12.0
"""Queue wait allowed per disturbance, which a job may sit through while its lane is replaced."""

_SLOW_CHILD_FACTOR = 4.0
"""How much longer a slow child takes than its expected per-job time."""

_JOB_DELAY_SECONDS = 0.02
"""What a fake job pretends to take, so a slow child is measurably slower without costing the clock."""

_CONTROL_LOOP_INTERVAL_SECONDS = 0.02
"""Synthetic stage cadence; preserves control-loop ordering without production wall-clock pacing."""


def _seeds() -> tuple[int, ...]:
    """Return the seeds to run: the environment override when set, otherwise the committed range."""
    override = os.environ.get(SEED_ENV_VAR)
    return parse_seed_spec(override) if override else SWEEP_FULL_WORKER_SEEDS


_SCENARIOS = generate_full_worker_scenarios(_seeds())


def _system_resources(scenario: ChaosScenario) -> SystemResources:
    """Build the synthetic host the scenario's card profile describes."""
    return SystemResources(
        total_ram_bytes=scenario.card.host_ram_gb * 1024 * 1024 * 1024,
        device_map=TorchDeviceMap(
            root={
                0: TorchDeviceInfo(
                    device_name=f"Chaos {scenario.card.label}",
                    device_index=0,
                    total_memory=scenario.card.total_vram_mb * 1024 * 1024,
                    kind="cuda",
                ),
            },
        ),
        per_process_overhead_mb=scenario.card.per_process_overhead_mb,
        marginal_process_overhead_mb=scenario.card.marginal_process_overhead_mb,
    )


def _jobs(scenario: ChaosScenario) -> list[ImageGenerateJobPopResponse]:
    """Expand the scenario's queue into canned pop responses."""
    return [_make_job(job, ordinal=ordinal) for ordinal, job in enumerate(scenario.jobs)]


def _make_job(job: ChaosJob, *, ordinal: int) -> ImageGenerateJobPopResponse:
    """Translate one generated payload description into a full-worker canned response."""
    has_lora = job.aux_kind in {ChaosAuxKind.LORA, ChaosAuxKind.BOTH}
    has_ti = job.aux_kind in {ChaosAuxKind.TEXTUAL_INVERSION, ChaosAuxKind.BOTH}
    post_processing = {
        ChaosPostProcessing.NONE: None,
        ChaosPostProcessing.FACE_FIX: ["GFPGAN"],
        ChaosPostProcessing.UPSCALE: ["RealESRGAN_x4plus"],
        ChaosPostProcessing.CHAIN: ["GFPGAN", "RealESRGAN_x4plus"],
    }[job.post_processing]
    sampler_name, scheduler = {
        ChaosSamplerProfile.EULER_NORMAL: ("k_euler", "normal"),
        ChaosSamplerProfile.DPM_KARRAS: ("k_dpmpp_2m", "karras"),
        ChaosSamplerProfile.LCM_SIMPLE: ("lcm", "simple"),
    }[job.sampler_profile]
    has_control = job.control_kind is not ChaosControlKind.NONE
    return make_canned_job(
        job.model.harness_name,
        width=job.width,
        height=job.height,
        ddim_steps=job.steps,
        n_iter=job.n_iter,
        loras=[LorasPayloadEntry(name=f"generated-lora-{ordinal % 3}")] if has_lora else None,
        tis=[TIPayloadEntry(name=f"generated-ti-{ordinal % 2}", inject_ti="prompt")] if has_ti else None,
        control_type="canny" if has_control else None,
        return_control_map=job.control_kind is ChaosControlKind.RETURN_MAP,
        image_is_control=job.control_kind is ChaosControlKind.PREANNOTATED,
        post_processing=post_processing,
        hires_fix=job.hires_fix,
        source_processing=job.source_mode.value,
        sampler_name=sampler_name,
        scheduler=scheduler,
    )


def _arrival(scenario: ChaosScenario) -> ArrivalSchedule | None:
    """Return the arrival schedule, or None when the whole queue is available at once."""
    if scenario.arrival is ChaosArrival.BURSTS:
        return ArrivalSchedule(
            kind="bursts",
            burst_size=scenario.burst_size,
            burst_interval_seconds=0.2,
        )
    if scenario.arrival is ChaosArrival.STEADY:
        return ArrivalSchedule(kind="steady", rate_per_minute=600.0)
    return None


def _fault_profile(scenario: ChaosScenario) -> FaultProfile | None:
    """Script the inference children from the disturbances this altitude can express.

    Returns:
        The profile, or None when the scenario draws no child-side disturbance.
    """
    events = scenario.child_events()
    if not events:
        return None
    if scenario.lanes != 1 or len(events) != 1:
        raise ValueError(
            "child fault ordinals are process-local, so generated subprocess faults require one lane and one event",
        )
    fields: dict[str, object] = {}
    for event in events:
        if event.kind is ChaosEventKind.LANE_DEATH:
            fields["crash_on_job_n"] = event.at_job_ordinal
        elif event.kind is ChaosEventKind.SLOW_CHILD:
            fields["slow_factor"] = _SLOW_CHILD_FACTOR
            fields["slow_on_job_n"] = event.at_job_ordinal
        elif event.kind is ChaosEventKind.RESOURCE_FAULT:
            fields["oom_on_job_n"] = event.at_job_ordinal
        elif event.kind is ChaosEventKind.MISROUTED_MESSAGE:
            fields["corrupt_on_job_n"] = event.at_job_ordinal
    return FaultProfile(**fields)  # type: ignore[arg-type]


def _timeout_seconds(scenario: ChaosScenario) -> float:
    """Return the wall-clock budget for the whole run, derived from the scenario's shape."""
    budget = (
        _BASE_TIMEOUT_SECONDS
        + _TIMEOUT_SECONDS_PER_JOB * scenario.job_count
        + _TIMEOUT_SECONDS_PER_EVENT * len(scenario.child_events())
    )
    return budget * _SPAWN_SLOWDOWN


def _queue_wait_bound_seconds(scenario: ChaosScenario) -> float:
    """Return the longest a job of this scenario may wait between being popped and reaching inference."""
    sequential = math.ceil(scenario.job_count / scenario.max_threads)
    budget = (
        _QUEUE_WAIT_BASE_SECONDS
        + _QUEUE_WAIT_SECONDS_PER_JOB * sequential
        + _QUEUE_WAIT_SECONDS_PER_EVENT * len(scenario.child_events())
    )
    return budget * _SPAWN_SLOWDOWN


def _stage_deadlines(scenario: ChaosScenario) -> HarnessStageDeadlines:
    """Return stage bounds that identify where a full-worker row stopped advancing."""
    stage_window = (
        _BASE_TIMEOUT_SECONDS + _TIMEOUT_SECONDS_PER_EVENT * len(scenario.child_events())
    ) * _SPAWN_SLOWDOWN
    return HarnessStageDeadlines(
        process_starting_seconds=stage_window,
        pending_inference_seconds=_queue_wait_bound_seconds(scenario),
        inference_seconds=stage_window,
        post_processing_seconds=stage_window,
        safety_seconds=stage_window,
        submission_seconds=stage_window,
        terminal_accounting_seconds=1.0,
    )


@contextlib.contextmanager
def _capture_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[HordeWorkerProcessManager]]:
    """Record the manager a harness run builds, so its ledger can be read once the run has ended.

    The run's result reports faults, not who issued them, and a give-up is a fault the worker issued
    against work it had accepted. The manager's action ledger records that decision directly, so the
    verdict reads the decision rather than inferring it from a count.
    """
    built: list[HordeWorkerProcessManager] = []
    original = harness_module.build_harness_process_manager

    def _recording(config: HarnessConfig) -> tuple[HordeWorkerProcessManager, int]:
        manager, expected = original(config)
        built.append(manager)
        return manager, expected

    monkeypatch.setattr(harness_module, "build_harness_process_manager", _recording)
    yield built


def _give_up_events(manager: HordeWorkerProcessManager) -> list[str]:
    """Return the reasons recorded for every give-up on accepted work during the run."""
    return [
        event.reason
        for event in manager._action_ledger.recent(limit=100_000)
        if event.event_type is LedgerEventType.RECOVERY_ABANDONED
    ]


def _assert_child_event_was_observed(
    manager: HordeWorkerProcessManager,
    result: HarnessResult,
    scenario: ChaosScenario,
) -> None:
    """Require observable evidence for the scenario's single child-side disturbance."""
    events = scenario.child_events()
    if not events:
        return
    event = events[0]
    ledger_types = {entry.event_type for entry in manager._action_ledger.recent(limit=100_000)}
    if event.kind is ChaosEventKind.LANE_DEATH:
        assert LedgerEventType.PROCESS_REPLACED in ledger_types, _message(
            scenario,
            result,
            "the requested lane death produced no process-replacement receipt",
        )
        return
    if event.kind is ChaosEventKind.RESOURCE_FAULT:
        assert LedgerEventType.INFERENCE_RETRIED in ledger_types, _message(
            scenario,
            result,
            "the requested resource fault produced no inference-retry receipt",
        )
        return
    if event.kind is ChaosEventKind.MISROUTED_MESSAGE:
        assert manager._message_dispatcher.stale_messages_ignored > 0, _message(
            scenario,
            result,
            "the requested stale message produced no dispatcher rejection receipt",
        )
        return

    assert result.metrics is not None
    sampling_seconds = [
        record.phase_metrics.sampling.duration_seconds
        for record in result.metrics.jobs
        if record.phase_metrics is not None and record.phase_metrics.sampling is not None
    ]
    minimum_slow_seconds = _JOB_DELAY_SECONDS * _SLOW_CHILD_FACTOR * 0.75
    assert sampling_seconds and max(sampling_seconds) >= minimum_slow_seconds, _message(
        scenario,
        result,
        "the requested slow child produced no elevated sampling-duration receipt",
    )


def _message(scenario: ChaosScenario, result: HarnessResult, complaint: str) -> str:
    """Return a failure message carrying the complaint, the seed, the scenario, and the run's own summary."""
    return (
        f"{complaint}\n"
        f"  scenario: {scenario.summary()}\n"
        f"  replay:   {SEED_ENV_VAR}={scenario.seed} pytest tests/e2e/test_chaos_generated_e2e.py "
        f'-m "chaos_sweep and slow"\n'
        f"  run:      {result.failure_summary()}"
    )


# Every scenario boots a worker and spawns real OS child processes, and the sweep runs many of them, so the
# module is opt-in on both counts.
pytestmark = [pytest.mark.slow, pytest.mark.chaos_sweep]


def test_child_fault_scripts_have_a_bijective_scenario_target() -> None:
    """Every child event maps to one profile field on a topology where its ordinal is unambiguous."""
    child_scenarios = [scenario for scenario in _SCENARIOS if scenario.child_events()]
    if os.environ.get(SEED_ENV_VAR) and not child_scenarios:
        pytest.skip("the requested replay seed carries no child-side disturbance")
    assert child_scenarios, "the subprocess corpus contains no child-side disturbance"
    assert all(scenario.lanes == 1 for scenario in child_scenarios)
    assert all(len(scenario.child_events()) == 1 for scenario in child_scenarios)

    for scenario in child_scenarios:
        event = scenario.child_events()[0]
        profile = _fault_profile(scenario)
        assert profile is not None
        if event.kind is ChaosEventKind.LANE_DEATH:
            assert profile.crash_on_job_n == event.at_job_ordinal
        elif event.kind is ChaosEventKind.SLOW_CHILD:
            assert profile.slow_on_job_n == event.at_job_ordinal
        elif event.kind is ChaosEventKind.RESOURCE_FAULT:
            assert profile.oom_on_job_n == event.at_job_ordinal
        else:
            assert profile.corrupt_on_job_n == event.at_job_ordinal


def test_full_worker_corpus_spans_every_generated_payload_axis() -> None:
    """The spawned projection retains every axis value from its larger candidate corpus."""
    if os.environ.get(SEED_ENV_VAR):
        pytest.skip(f"{SEED_ENV_VAR} overrides the committed sweep, so its span is the caller's to choose")

    candidates = generate_full_worker_scenarios(FULL_WORKER_CANDIDATE_SEEDS)
    jobs = [job for scenario in _SCENARIOS for job in scenario.jobs]
    candidate_jobs = [job for scenario in candidates for job in scenario.jobs]
    assert {scenario.card.label for scenario in _SCENARIOS} == {scenario.card.label for scenario in candidates}
    assert {scenario.shape for scenario in _SCENARIOS} == {scenario.shape for scenario in candidates}
    assert {scenario.demand_shape for scenario in _SCENARIOS} == {scenario.demand_shape for scenario in candidates}
    assert {scenario.initial_residency for scenario in _SCENARIOS} == {
        scenario.initial_residency for scenario in candidates
    }
    assert {scenario.arrival for scenario in _SCENARIOS} == {scenario.arrival for scenario in candidates}
    assert {scenario.max_threads for scenario in _SCENARIOS} == {scenario.max_threads for scenario in candidates}
    assert {scenario.queue_size for scenario in _SCENARIOS} == {scenario.queue_size for scenario in candidates}
    assert {scenario.lanes for scenario in _SCENARIOS} == {scenario.lanes for scenario in candidates}
    assert {scenario.enable_vram_budget for scenario in _SCENARIOS} == {
        scenario.enable_vram_budget for scenario in candidates
    }
    assert {scenario.whole_card_enabled for scenario in _SCENARIOS} == {
        scenario.whole_card_enabled for scenario in candidates
    }
    assert {scenario.performance for scenario in _SCENARIOS} == {scenario.performance for scenario in candidates}
    assert {scenario.unload_models_from_vram_often for scenario in _SCENARIOS} == {
        scenario.unload_models_from_vram_often for scenario in candidates
    }
    assert {event.kind for scenario in _SCENARIOS for event in scenario.events} == {
        event.kind for scenario in candidates for event in scenario.events
    }
    assert any(not scenario.events for scenario in _SCENARIOS)

    assert {job.source_mode for job in jobs} == set(ChaosSourceMode)
    assert {job.aux_kind for job in jobs} == set(ChaosAuxKind)
    assert {job.control_kind for job in jobs} == set(ChaosControlKind)
    assert {job.post_processing for job in jobs} == set(ChaosPostProcessing)
    assert {job.n_iter for job in jobs} == {1, 2, 4}
    assert {job.hires_fix for job in jobs} == {False, True}
    assert {job.sampler_profile for job in jobs} == set(ChaosSamplerProfile)
    assert {job.model for job in jobs} == {job.model for job in candidate_jobs}

    materialized = [_make_job(job, ordinal=index) for index, job in enumerate(jobs)]
    assert any(job.source_mask is not None for job in materialized)
    assert any(job.payload.loras and job.payload.tis for job in materialized)
    assert any(job.payload.return_control_map for job in materialized)
    assert any(job.payload.image_is_control for job in materialized)
    assert any(len(job.payload.post_processing or []) > 1 for job in materialized)


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[scenario.label for scenario in _SCENARIOS])
async def test_generated_scenario_completes_against_real_children(
    scenario: ChaosScenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated composition run against real children completes, with nothing given up on."""
    jobs = _jobs(scenario)
    with _capture_manager(monkeypatch) as managers:
        result = await run_harness_async(
            HarnessConfig(
                scenario=jobs,
                arrival=_arrival(scenario),
                process_mode="fake",
                skip_api=True,
                job_delay_seconds=_JOB_DELAY_SECONDS,
                control_loop_interval_seconds=_CONTROL_LOOP_INTERVAL_SECONDS,
                timeout_seconds=_timeout_seconds(scenario),
                stage_deadlines=_stage_deadlines(scenario),
                system_resources=_system_resources(scenario),
                inference_fault_profile=_fault_profile(scenario),
                bridge_data_overrides={
                    "max_threads": scenario.max_threads,
                    "queue_size": scenario.queue_size,
                    "enable_vram_budget": scenario.enable_vram_budget,
                    "whole_card_exclusive_residency": scenario.whole_card_enabled,
                    "high_performance_mode": scenario.performance.value == "high",
                    "moderate_performance_mode": scenario.performance.value == "moderate",
                    "unload_models_from_vram_often": scenario.unload_models_from_vram_often,
                },
            ),
        )

    assert not result.timed_out, _message(scenario, result, "the run did not finish inside its budget")
    assert result.num_jobs_completed == len(jobs), _message(
        scenario,
        result,
        f"{len(jobs) - result.num_jobs_completed} of {len(jobs)} queued jobs did not complete",
    )
    assert result.num_jobs_faulted == 0, _message(scenario, result, "queued work was faulted rather than served")
    assert result.audit_failures == [], _message(scenario, result, "a job lifecycle invariant was violated")

    assert managers, "the harness built no manager, so the give-up verdict would read nothing"
    assert managers[0]._card_runtimes[0].target_process_count == scenario.lanes, _message(
        scenario,
        result,
        "the full worker resolved a different lane topology than the scenario declared",
    )
    _assert_child_event_was_observed(managers[0], result, scenario)
    give_ups = _give_up_events(managers[0])
    assert give_ups == [], _message(scenario, result, f"the worker gave up on accepted work: {give_ups}")

    assert result.metrics is not None, _message(scenario, result, "the run reported no metrics to judge")
    bound = _queue_wait_bound_seconds(scenario)
    waits = [record.queue_wait_seconds for record in result.metrics.jobs if record.queue_wait_seconds is not None]
    assert waits, _message(scenario, result, "no job reported a queue wait, so the age bound judged nothing")
    assert max(waits) <= bound, _message(
        scenario,
        result,
        f"a job waited {max(waits):.0f}s to reach inference, past the {bound:.0f}s this scenario's shape allows",
    )


def test_the_full_worker_sweep_discloses_its_coverage(pytestconfig: pytest.Config) -> None:
    """The suite prints what it ran and what it did not explore, so no truncation is silent."""
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    expressible = sorted(kind.value for scenario in _SCENARIOS for kind in {event.kind for event in scenario.events})
    lines = [
        f"generated chaos (full worker): {len(_SCENARIOS)} scenarios, seeds "
        f"{','.join(str(seed) for seed in _seeds())} "
        f"(override with {SEED_ENV_VAR})",
        f"  disturbance kinds drawn: {sorted(set(expressible))}",
        "  not explored:",
        *(f"    - {axis}: {reason}" for axis, reason in DISCLOSED_BOUNDS),
    ]
    if reporter is not None:
        reporter.write_line("")
        for line in lines:
            reporter.write_line(line)

    assert DISCLOSED_BOUNDS, "the generated space's truncations must be listed, never silent"
