"""Reproductions of the threads=2 false-hang-kill recovery storm, rooted in job-weight-blind watchdogs.

Under threads=2 contention, hung-kill recoveries accumulate against healthy jobs: each has
``exitcode=None`` (the child is alive), ``last_state=INFERENCE_STARTING``, and a preceding
``SLOWDOWN_DETECTED`` at 4x-8x "expected sampling time". The reaped jobs, when re-run, sample at normal
speed -- they were killed during the startup/loading window, not during a genuine sampling slowdown, and
the watchdogs that judged them are blind to what a job's features make legitimate.

Two job-feature-driven mechanisms, which the live data shows compounding:

1. **The slowdown grader mis-attributes startup overhead to sampling.**
   ``_grade_running_inference`` stamps the clock at *dispatch* (``current_inference_started_at`` is set in
   ``start_inference`` before the child loads the model) and divides ``now - dispatch`` by
   ``current_job_expected_sampling_seconds``, which is the perf model's **sampling-only** estimate. Every
   non-sampling phase therefore inflates the ratio: cold VRAM load, **aux-model / ControlNet download**,
   prompt encode, **post-processing** setup, the hires-fix second-pass framing. The grader fires *before the
   first sampling step has even been emitted* (proven: a still-loading job at 18 s elapsed vs ~4 s expected
   is graded level 2 / "4.5x; watching for a hang"). The heavier the job's features, the longer that
   startup window, so a ControlNet / post-processed / hires / large-batch job is the most likely to be
   mislabeled as a hang candidate.

2. **The hang-kill timeout is a flat constant, blind to job weight.**
   ``replace_hung_processes`` kills on ``inference_step_timeout`` (20 s live) of heartbeat silence once a
   step is seen, widened by ``_effective_inference_step_timeout`` *only* for over-budget/exclusive admits.
   Resolution, batch size, ControlNet and hires-fix all raise a job's per-step wall time and its VRAM
   pressure (which, on a contended threads=2 card, stalls a step far longer); none of that feeds the
   timeout. A heavy-but-healthy job, or a normal job the grader already measured as contention-slowed, is
   killed by a budget calibrated for a light vanilla one.

Both reduce to the same root: the watchdogs do not account for the work a job's features (ControlNet /
SDXL-ControlNet, batching, resolution, hires-fix, post-processing) make legitimate, nor for the non-sampling
startup phases those features lengthen. The perf-model signature already buckets every one of those
features; the fix is to feed that expectation into the grader (grade sampling against sampling, only once
sampling has started) and into the kill timeout (scale the per-step grace with the job's expected work and
any measured contention, bounded so a genuine wedge is still reaped).

These scenarios are RED against the current scheduler. Guards assert the fix stays bounded and leaves
light / cold-start / idle / uncalibrated cases untouched.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeHeartbeatType, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SLOWDOWN_WARN_RATIO,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.performance_model import signature_from_job

from .conftest import make_job_pop_response, make_mock_bridge_data, make_mock_process_info

# Representative real-world knobs: a 20 s per-step timeout and a 90 s first-step grace.
_STEP_TIMEOUT = 20
_FIRST_STEP_TIMEOUT = 90
_OVERBUDGET_TIMEOUT = 120  # the existing widened ceiling, reused here as the natural bound

# A light vanilla SDXL job samples for ~5 s; a heavy one (high-res + batch + hires + ControlNet) samples for
# tens of seconds. The watchdog must tell these apart, since their legitimate per-step wall times differ.
_LIGHT_SAMPLING_SECONDS = 5.0
_HEAVY_SAMPLING_SECONDS = 45.0


def _feature_job(
    model: str = "SDXL",
    *,
    width: int = 1024,
    height: int = 1024,
    ddim_steps: int = 30,
    n_iter: int = 1,
    control_type: str | None = None,
    hires_fix: bool = False,
) -> ImageGenerateJobPopResponse:
    """Build a real pop response carrying the features the perf-model signature keys on.

    ``make_job_pop_response`` covers resolution/steps/batch, but the payload is frozen after construction, so
    ControlNet and hires-fix must be set here at build time.
    """
    job_id = uuid.uuid4()
    payload: dict[str, object] = {
        "prompt": "test prompt",
        "width": width,
        "height": height,
        "ddim_steps": ddim_steps,
        "n_iter": n_iter,
        "seed": "42",
        "sampler_name": "k_euler",
        "hires_fix": hires_fix,
    }
    if control_type is not None:
        payload["control_type"] = control_type
    data: dict[str, object] = {
        "id": str(job_id),
        "ids": [str(job_id)],
        "model": model,
        "payload": payload,
        "skipped": {},
        "source_processing": "txt2img" if control_type is None else "img2img",
    }
    return ImageGenerateJobPopResponse(**data)  # pyrefly: ignore - pydantic validates the shape


def _admitted_job_tracker(*models: str) -> tuple[JobTracker, dict[str, object]]:
    """A JobTracker with the given models popped (normal admits), plus a name->job map."""
    tracker = JobTracker()
    loop = asyncio.new_event_loop()
    jobs: dict[str, object] = {}
    try:
        for model in models:
            job = make_job_pop_response(model=model)
            loop.run_until_complete(tracker.record_popped_job(job))
            jobs[model] = job
    finally:
        loop.close()
    return tracker, jobs


def _inference_proc(
    process_id: int,
    model: str,
    job: object,
    *,
    slowdown_level: int = 0,
    expected_sampling_seconds: float = _LIGHT_SAMPLING_SECONDS,
    seconds_since_heartbeat: float = 0.0,
    seconds_since_dispatch: float | None = None,
    has_emitted_step: bool = True,
    state: HordeProcessState = HordeProcessState.INFERENCE_STARTING,
    baseline: str | None = "stable_diffusion_xl",
    heartbeat_type: HordeHeartbeatType = HordeHeartbeatType.INFERENCE_STEP,
) -> object:
    """An inference slot with controllable graded level, expected work, heartbeat age, and step progress."""
    proc = make_mock_process_info(process_id, model_name=model, state=state)
    proc.last_job_referenced = job  # type: ignore[assignment]
    proc.loaded_horde_model_baseline = baseline
    proc.current_job_slowdown_level = slowdown_level
    proc.last_heartbeat_type = heartbeat_type
    dispatch_age = seconds_since_dispatch if seconds_since_dispatch is not None else seconds_since_heartbeat
    proc.current_inference_started_at = time.time() - dispatch_age
    # Sampling is graded from the first step, not from dispatch; stamp the first-step clock so the elapsed
    # the grader sees is the intended sampling window. Before any step the slot is still loading (None).
    proc.current_first_step_at = (time.time() - dispatch_age) if has_emitted_step else None
    proc.current_job_expected_sampling_seconds = expected_sampling_seconds
    proc.last_current_step = 1 if has_emitted_step else None
    proc.last_heartbeat_timestamp = time.time() - seconds_since_heartbeat
    proc.last_received_timestamp = time.time() - seconds_since_heartbeat
    return proc


def _timeout_stub(job_tracker: JobTracker) -> Mock:
    """Minimal stub for calling ``_effective_inference_step_timeout`` unbound (matches the overbudget repro)."""
    stub = Mock()
    stub._job_tracker = job_tracker
    return stub


def _contention_bridge_data() -> Mock:
    """Bridge data with representative hang-timeout knobs: 20 s step, 90 s first step, 120 s ceiling."""
    return make_mock_bridge_data(
        inference_step_timeout=_STEP_TIMEOUT,
        inference_first_step_timeout=_FIRST_STEP_TIMEOUT,
        overbudget_step_timeout=_OVERBUDGET_TIMEOUT,
    )


def _effective_timeout(job_tracker: JobTracker, bridge_data: Mock, proc: object) -> int:
    return ProcessLifecycleManager._effective_inference_step_timeout(_timeout_stub(job_tracker), bridge_data, proc)


def _grade(process_map: ProcessMap) -> None:
    """Run ``_grade_running_inference`` with a stub exposing only the map, the counter, and the ledger."""
    stub = Mock()
    stub._process_map = process_map
    stub._num_slowdown_events = 0
    stub._action_ledger = Mock()
    ProcessLifecycleManager._grade_running_inference(stub)


def _is_stuck(job_tracker: JobTracker, bridge_data: Mock, proc: object) -> bool:
    """Run the exact watchdog decision: the effective per-step timeout fed to ``is_stuck_on_inference``."""
    process_map = ProcessMap({proc.process_id: proc})  # type: ignore[attr-defined]
    effective = _effective_timeout(job_tracker, bridge_data, proc)
    return process_map.is_stuck_on_inference(
        proc.process_id,  # type: ignore[attr-defined]
        effective,
        bridge_data.inference_first_step_timeout,
    )


class TestStartupOverheadIsNotASamplingSlowdown:
    """THE FIX (mechanism 1): startup/loading time must not be graded as a sampling slowdown.

    ``current_inference_started_at`` is stamped at dispatch, before the child loads the model, downloads
    ControlNet/aux models, encodes the prompt, or frames post-processing. Dividing that elapsed by the
    sampling-only expectation mislabels a healthy heavy/cold job as a hang candidate.
    """

    def test_still_loading_job_is_not_graded_as_slow(self) -> None:
        """A slot that has emitted no sampling step yet (still loading) must not be graded a slowdown.

        18 s elapsed since dispatch with no step is cold VRAM load + encode, not 4.5x slow sampling. The
        grader currently flags it level 2; pre-first-step time belongs to the first-step grace, not the
        slowdown signal.
        """
        job = _feature_job("SDXL")
        proc = _inference_proc(
            1,
            "SDXL",
            job,
            expected_sampling_seconds=4.0,
            seconds_since_dispatch=18.0,
            has_emitted_step=False,
        )
        _grade(ProcessMap({1: proc}))

        assert proc.current_job_slowdown_level == 0, (
            "a still-loading job (no sampling step emitted) was graded as a sampling slowdown; its startup "
            "time was mis-attributed to slow sampling"
        )

    def test_controlnet_aux_load_window_is_not_graded_as_slow(self) -> None:
        """A ControlNet job loading its aux model (no step yet) must not be graded a slowdown.

        ControlNet adds an aux-model download/load before sampling. That window is legitimately long and is
        the first-step grace's job, not the slowdown grader's; the grader must not flag it merely because the
        aux load outran the sampling-only expectation.
        """
        job = _feature_job("SDXL", control_type="canny")
        proc = _inference_proc(
            1,
            "SDXL",
            job,
            expected_sampling_seconds=4.0,
            seconds_since_dispatch=30.0,
            has_emitted_step=False,
        )
        _grade(ProcessMap({1: proc}))

        assert proc.current_job_slowdown_level == 0

    def test_sampling_slowdown_after_first_step_is_still_graded(self) -> None:
        """Once sampling has actually started, a genuine 4x sampling slowdown must still be graded.

        The fix narrows *what* is graded (sampling, not startup); it must not blind the grader to a real
        sampling slowdown. A slot that has emitted a step and is 4x over its sampling expectation is the
        contention signal the grader exists to catch.
        """
        job = _feature_job("SDXL")
        proc = _inference_proc(
            1,
            "SDXL",
            job,
            expected_sampling_seconds=_LIGHT_SAMPLING_SECONDS,
            seconds_since_dispatch=_LIGHT_SAMPLING_SECONDS * SLOWDOWN_WARN_RATIO,
            has_emitted_step=True,
        )
        _grade(ProcessMap({1: proc}))

        assert proc.current_job_slowdown_level == 2


class TestHangBudgetIsFeatureAndWeightAware:
    """THE FIX (mechanism 2): the per-step hang budget must scale with the job's expected work.

    Resolution, batch, ControlNet and hires-fix raise a job's legitimate per-step wall time; the flat 20 s
    timeout kills the heavy ones. The perf model already estimates this work per signature.
    """

    def test_heavy_job_earns_larger_step_budget_than_light_job(self) -> None:
        """A high-work job (large expected sampling) must get a longer per-step grace than a light one.

        Both jobs are healthy and normal-admit; only their expected work differs. The watchdog currently
        gives both the flat 20 s, so the heavy job is reaped on a budget sized for the light one.
        """
        bridge_data = _contention_bridge_data()
        tracker, _ = _admitted_job_tracker()
        light = _inference_proc(1, "SDXL", _feature_job("SDXL"), expected_sampling_seconds=_LIGHT_SAMPLING_SECONDS)
        heavy = _inference_proc(2, "SDXL", _feature_job("SDXL"), expected_sampling_seconds=_HEAVY_SAMPLING_SECONDS)

        light_budget = _effective_timeout(tracker, bridge_data, light)
        heavy_budget = _effective_timeout(tracker, bridge_data, heavy)

        assert heavy_budget > light_budget, (
            "a heavy job (high resolution / batch / hires / ControlNet) got the same flat hang budget as a "
            "light one and will be false-killed"
        )
        assert heavy_budget <= _OVERBUDGET_TIMEOUT, "the budget must stay bounded so a genuine wedge is still reaped"

    def test_light_job_keeps_tight_budget(self) -> None:
        """A light job with no measured slowdown keeps the tight timeout: the fix must not widen everything."""
        bridge_data = _contention_bridge_data()
        tracker, _ = _admitted_job_tracker()
        light = _inference_proc(1, "SDXL", _feature_job("SDXL"), expected_sampling_seconds=_LIGHT_SAMPLING_SECONDS)

        assert _effective_timeout(tracker, bridge_data, light) == _STEP_TIMEOUT

    def test_heavy_job_progressing_one_stretched_step_is_not_killed(self) -> None:
        """A heavy job silent for 25 s (one legitimately long step) must not be judged a hang once weight-aware.

        25 s between heartbeats is abnormal for a light job but ordinary for a batched high-res hires step.
        Under the flat 20 s timeout it is killed; under a weight-aware budget it survives.
        """
        bridge_data = _contention_bridge_data()
        tracker, _ = _admitted_job_tracker()
        heavy = _inference_proc(
            1,
            "SDXL",
            _feature_job("SDXL", width=1216, height=1216, n_iter=4, hires_fix=True),
            expected_sampling_seconds=_HEAVY_SAMPLING_SECONDS,
            seconds_since_heartbeat=25.0,
        )

        assert _is_stuck(tracker, bridge_data, heavy) is False, (
            "a heavy job progressing one legitimately long step was killed as a hang"
        )


class TestContentionSlowdownDuringSampling:
    """THE FIX (compounding): a normal job the grader flagged as contention-slowed earns the same grace.

    Independent of features, two SDXL models co-sampling on a contended card slow each other 4x-8x. The
    grader detects it (``current_job_slowdown_level``) but the kill timeout ignores the flag.
    """

    def test_contention_slowed_job_widens_step_timeout(self) -> None:
        """A normal-admit job flagged at level 2 (>=4x) must not keep the tight per-step timeout."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("AlbedoBase XL (SDXL)")
        proc = _inference_proc(1, "AlbedoBase XL (SDXL)", jobs["AlbedoBase XL (SDXL)"], slowdown_level=2)

        effective = _effective_timeout(tracker, bridge_data, proc)
        assert effective > _STEP_TIMEOUT, (
            "a job measured at 4x contention slowdown kept the tight per-step timeout"
        )
        assert effective <= _OVERBUDGET_TIMEOUT, (
            "the contention grace must stay bounded so a true hang is still reaped"
        )

    def test_contention_slowed_progressing_slot_is_not_killed(self) -> None:
        """A 4x slot silent for 25 s (one contention-stretched step) must not be judged a hang once widened."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("AlbedoBase XL (SDXL)")
        proc = _inference_proc(
            3,
            "AlbedoBase XL (SDXL)",
            jobs["AlbedoBase XL (SDXL)"],
            slowdown_level=2,
            seconds_since_heartbeat=25.0,
        )

        assert _is_stuck(tracker, bridge_data, proc) is False


class TestFeatureAndPhaseWidening:
    """THE FIX (job features are first-class): a feature-laden job or a heavy non-step phase earns the grace.

    Two reap patterns exist that a measured slowdown alone does not explain: a job whose features
    (ControlNet / hires-fix / batch / large resolution) lengthen its steps and add heartbeat-silent phases,
    and a slot whose last beat was a ``PIPELINE_STATE_CHANGE`` (it is inside the hires second pass / VAE
    decode / post-processing, none of which emit a sampling step).
    """

    def test_feature_heavy_job_with_light_expectation_still_widens(self) -> None:
        """A ControlNet + hires + large-resolution job widens on its features even with a light expectation.

        The widening must respond to the job's features, not only its expected sampling seconds: a feature
        job can carry a modest sampling estimate yet long heartbeat-silent feature phases.
        """
        bridge_data = _contention_bridge_data()
        tracker, _ = _admitted_job_tracker()
        job = _feature_job("SDXL", width=1216, height=1216, control_type="canny", hires_fix=True)
        proc = _inference_proc(1, "SDXL", job, expected_sampling_seconds=_LIGHT_SAMPLING_SECONDS)

        assert _effective_timeout(tracker, bridge_data, proc) == _OVERBUDGET_TIMEOUT, (
            "a feature-heavy job (ControlNet + hires + large resolution) did not earn the widened per-step grace"
        )

    def test_pipeline_phase_widens_step_timeout(self) -> None:
        """A slot whose last beat was a PIPELINE_STATE_CHANGE (a heavy non-step phase) widens its grace."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("Juggernaut XL")
        proc = _inference_proc(
            1,
            "Juggernaut XL",
            jobs["Juggernaut XL"],
            heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
        )

        assert _effective_timeout(tracker, bridge_data, proc) > _STEP_TIMEOUT

    def test_pipeline_phase_progressing_slot_is_not_killed(self) -> None:
        """A slot silent 25 s inside a hires/VAE/post phase must not be reaped."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("WAI-NSFW-illustrious-SDXL")
        proc = _inference_proc(
            2,
            "WAI-NSFW-illustrious-SDXL",
            jobs["WAI-NSFW-illustrious-SDXL"],
            heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
            seconds_since_heartbeat=25.0,
        )

        assert _is_stuck(tracker, bridge_data, proc) is False


class TestFeaturesProduceHeavierExpectation:
    """The perf-model signature keys on the named features, so a weight-aware budget actually responds to them.

    These pin the input the watchdog fix must consume: ControlNet, hires-fix, batch and resolution each
    produce a distinct, heavier job signature than a light vanilla one.
    """

    def test_resolution_batch_controlnet_and_hires_are_all_in_the_signature(self) -> None:
        """A high-res, batched, ControlNet, hires job must produce a signature distinct from a light one."""
        light = signature_from_job(_feature_job("SDXL", width=768, height=768, ddim_steps=20), "stable_diffusion_xl")
        heavy_job = _feature_job(
            "SDXL",
            width=1216,
            height=1216,
            ddim_steps=30,
            n_iter=4,
            control_type="canny",
            hires_fix=True,
        )
        heavy = signature_from_job(heavy_job, "stable_diffusion_xl")

        assert light is not None and heavy is not None
        assert heavy.has_controlnet is True and light.has_controlnet is False
        assert heavy.has_hires_fix is True and light.has_hires_fix is False
        assert heavy.batch_bucket != light.batch_bucket
        assert heavy.resolution_bucket != light.resolution_bucket
        # Hires-fix doubles the sampling iterations, so the heavy job's modelled work is strictly larger.
        assert heavy.total_sampling_iterations > light.total_sampling_iterations


class TestQueueShapeContention:
    """The grader must flag a sampling slowdown across plausible and implausible queue shapes.

    The contention is GPU/host pressure, not model diversity, so detection (hence the earned grace) must be
    identical whether the queue is heterogeneous, an anomalous run of one model, or a lone slot.
    """

    def _sampling_slowed(self, process_id: int, model: str, job: object, *, ratio: float) -> object:
        """An INFERENCE_STARTING slot that has emitted a step and whose sampling elapsed/expected is ``ratio``."""
        return _inference_proc(
            process_id,
            model,
            job,
            expected_sampling_seconds=_LIGHT_SAMPLING_SECONDS,
            seconds_since_dispatch=_LIGHT_SAMPLING_SECONDS * ratio,
            has_emitted_step=True,
        )

    def test_heterogeneous_two_distinct_sdxl_coresidence_flags_and_widens(self) -> None:
        """Two distinct SDXL models co-sampling at 4x (the observed case): the slowed slot must be flagged."""
        tracker, jobs = _admitted_job_tracker("AlbedoBase XL (SDXL)", "WAI-NSFW-illustrious-SDXL")
        proc_a = self._sampling_slowed(
            1,
            "AlbedoBase XL (SDXL)",
            jobs["AlbedoBase XL (SDXL)"],
            ratio=SLOWDOWN_WARN_RATIO,
        )
        proc_b = make_mock_process_info(
            2,
            model_name="WAI-NSFW-illustrious-SDXL",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        _grade(ProcessMap({1: proc_a, 2: proc_b}))

        assert proc_a.current_job_slowdown_level == 2, "the grader failed to flag the 4x co-residence slowdown"
        assert _effective_timeout(tracker, _contention_bridge_data(), proc_a) > _STEP_TIMEOUT, (
            "the flagged heterogeneous-queue slot kept the tight timeout and will be false-killed"
        )

    def test_homogeneous_same_model_two_slots_still_widens(self) -> None:
        """An anomalous run of one model on two slots still contends on compute and must earn the grace."""
        tracker, jobs = _admitted_job_tracker("Juggernaut XL")
        job = jobs["Juggernaut XL"]
        proc_a = self._sampling_slowed(1, "Juggernaut XL", job, ratio=SLOWDOWN_WARN_RATIO)
        proc_b = make_mock_process_info(2, model_name="Juggernaut XL", state=HordeProcessState.INFERENCE_STARTING)
        _grade(ProcessMap({1: proc_a, 2: proc_b}))

        assert proc_a.current_job_slowdown_level == 2
        assert _effective_timeout(tracker, _contention_bridge_data(), proc_a) > _STEP_TIMEOUT

    def test_lone_slot_transient_slowdown_is_not_false_killed(self) -> None:
        """A single slot that slows for any reason (no sibling) must still earn the grace.

        Implausible but possible: only one slot busy and it slows from a host stall or thermal throttle. The
        widening keys on the measured slowdown, not on counting siblings.
        """
        tracker, jobs = _admitted_job_tracker("CyberRealistic Pony")
        proc = self._sampling_slowed(
            1,
            "CyberRealistic Pony",
            jobs["CyberRealistic Pony"],
            ratio=SLOWDOWN_WARN_RATIO + 4.0,
        )
        _grade(ProcessMap({1: proc}))

        assert proc.current_job_slowdown_level == 2
        assert _effective_timeout(tracker, _contention_bridge_data(), proc) > _STEP_TIMEOUT


class TestHangKillGuards:
    """The fix must stay bounded and leave genuine wedges, idle slots, cold starts, and the uncalibrated alone."""

    def test_genuine_wedge_still_detected_despite_widening(self) -> None:
        """A slot silent far beyond any legitimate stretched step must still be reaped even when slowed/heavy."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("WAI-NSFW-illustrious-SDXL")
        proc = _inference_proc(
            1,
            "WAI-NSFW-illustrious-SDXL",
            jobs["WAI-NSFW-illustrious-SDXL"],
            slowdown_level=2,
            expected_sampling_seconds=_HEAVY_SAMPLING_SECONDS,
            seconds_since_heartbeat=_OVERBUDGET_TIMEOUT + 30.0,
        )

        assert _is_stuck(tracker, bridge_data, proc) is True

    def test_overbudget_admit_takes_precedence_over_contention_grace(self) -> None:
        """An over-budget admit keeps its dedicated overbudget grace, distinct from the contention ceiling.

        The new widening must only raise floors, never override the budget-gated over-budget path; with a
        distinct overbudget value the over-budget branch must win.
        """
        bridge_data = _contention_bridge_data()
        bridge_data.overbudget_step_timeout = 150  # distinct from the 120 s contended ceiling
        tracker, jobs = _admitted_job_tracker("AlbedoBase XL (SDXL)")
        job = jobs["AlbedoBase XL (SDXL)"]
        tracker.mark_admitted_over_budget(job)  # type: ignore[attr-defined]
        proc = _inference_proc(1, "AlbedoBase XL (SDXL)", job)

        assert _effective_timeout(tracker, bridge_data, proc) == 150

    def test_pre_first_step_cold_load_uses_first_step_grace(self) -> None:
        """Before the first step the long cold-load grace applies and is unaffected by the fix."""
        bridge_data = _contention_bridge_data()
        tracker, jobs = _admitted_job_tracker("AlbedoBase XL (SDXL)")
        proc = _inference_proc(
            2,
            "AlbedoBase XL (SDXL)",
            jobs["AlbedoBase XL (SDXL)"],
            seconds_since_heartbeat=45.0,
            has_emitted_step=False,
        )

        assert _is_stuck(tracker, bridge_data, proc) is False

    def test_idle_slot_is_never_judged_stuck(self) -> None:
        """A slot not in INFERENCE_STARTING is never a hang candidate, whatever its heartbeat age."""
        bridge_data = _contention_bridge_data()
        tracker, _ = _admitted_job_tracker()
        proc = make_mock_process_info(4, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.last_heartbeat_timestamp = time.time() - 300.0

        assert _is_stuck(tracker, bridge_data, proc) is False

    def test_uncalibrated_slot_is_not_flagged(self) -> None:
        """A job with no expected sampling time (cold/uncalibrated) must not be graded as slow."""
        job = _feature_job("Deliberate")
        proc = make_mock_process_info(1, model_name="Deliberate", state=HordeProcessState.INFERENCE_STARTING)
        proc.last_job_referenced = job  # type: ignore[assignment]
        proc.current_job_expected_sampling_seconds = None
        proc.current_inference_started_at = time.time() - 60.0
        proc.current_first_step_at = time.time() - 60.0
        proc.current_job_slowdown_level = 0
        proc.last_current_step = 1
        _grade(ProcessMap({1: proc}))

        assert proc.current_job_slowdown_level == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
