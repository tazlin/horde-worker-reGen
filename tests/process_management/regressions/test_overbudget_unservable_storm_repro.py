"""Reproduction and fix for the over-budget *unservable-model* crash storm that triggered server maintenance.

Failure mode:
    On a 16 GB card a Flux.1-Schnell fp8 (Compact) head job whose predicted peak exceeds achievable free
    VRAM is best-effort *admitted over budget* (``inference_scheduler`` admit path). It loads onto a
    near-full device (``Free VRAM: 52 MB``); during sampling its weights spill to system RAM, so a step
    takes ~83 s ("Job slowdown detected ... 83.77 s/it", 4.0x expected). The ``inference_step_timeout``
    watchdog kills the "stuck" slot (``TIMEOUT_DETECTED ... stuck mid inference``). Because the job was
    ``admitted_over_budget`` the kill is classified ``resource/OOM`` and requeued for a degraded/isolated
    retry. *Isolating the job does not shrink Flux's footprint*, so it re-thrashes, is killed again,
    faulted, and dropped. This recurs every ~2-3 min; the steady drop stream trips the horde server's
    "dropping too many jobs" guard, which forced the worker into maintenance.

What the earlier over-budget handling did and did not cover:
    An earlier layer added the isolated/degraded retry classification and a 90 s first-step grace, and
    ``test_oversized_job_crash_storm_repro`` pins that the retry is *isolated*. Neither addresses a model
    the device genuinely cannot run: the isolated retry still thrashes, and nothing stops the worker
    popping/dropping the model until the server forces maintenance. This module reproduces that gap and
    pins the fix.

The fix pinned here:
    - **Circuit-breaker**: terminal over-budget faults are counted per model; after
      ``unservable_model_fault_threshold`` consecutive faults the model is held back (not admitted, not
      popped) for ``unservable_model_cooldown_seconds``. A successful generation resets the streak.
    - **Self-throttle backstop**: terminal resource faults across all models are counted in a rolling
      window; once they approach the server's tolerance the worker enters a local pop-pause itself.
    - **Exclusive-first**: an over-budget admit runs with the device to itself (no concurrent staging) and
      gets a per-step grace, so a slow-but-progressing heavy job completes instead of being killed.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobTracker,
)
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import (
    is_model_locally_unservable,
    predict_job_vram_mb,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_UNSERVABLE_MODEL = "Flux.1-Schnell fp8 (Compact)"


async def _terminal_overbudget_fault(job_tracker: JobTracker, model: str) -> InferenceFailureResolution:
    """Pop a job for ``model``, admit it over budget, and fault its slot terminally (retry policy 1).

    Mirrors one turn of the live storm: a best-effort over-budget admit whose slot is killed and, with no
    attempts left, faulted. The ``admitted_over_budget`` tag (not an explicit resource signal) is what
    classifies the kill as a resource fault, exactly as the lifecycle's hung-slot replacement does.
    """
    job = make_job_pop_response(model=model)
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    job_tracker.mark_admitted_over_budget(job)
    slot = make_mock_process_info(1, model_name=model)
    return job_tracker.handle_job_fault_now(faulted_job=job, process_info=slot)  # pyrefly: ignore


class TestUnservablePolicy:
    """The pure breaker policy over raw counters (shared by the scheduler and popper)."""

    def test_below_threshold_is_servable(self) -> None:
        """A model under the fault threshold is not held back."""
        assert (
            is_model_locally_unservable(
                overbudget_fault_count=2,
                last_overbudget_fault_time=time.time(),
                threshold=3,
                cooldown_seconds=900,
            )
            is False
        )

    def test_at_threshold_within_cooldown_is_unservable(self) -> None:
        """At/over the threshold and within the cooldown, the model is held back."""
        assert (
            is_model_locally_unservable(
                overbudget_fault_count=3,
                last_overbudget_fault_time=time.time(),
                threshold=3,
                cooldown_seconds=900,
            )
            is True
        )

    def test_after_cooldown_is_servable_again(self) -> None:
        """Once the cooldown elapses since the last fault, the model is tried again."""
        assert (
            is_model_locally_unservable(
                overbudget_fault_count=5,
                last_overbudget_fault_time=time.time() - 1000,
                threshold=3,
                cooldown_seconds=900,
            )
            is False
        )

    def test_threshold_zero_disables_breaker(self) -> None:
        """A threshold of 0 disables the breaker entirely."""
        assert (
            is_model_locally_unservable(
                overbudget_fault_count=99,
                last_overbudget_fault_time=time.time(),
                threshold=0,
                cooldown_seconds=900,
            )
            is False
        )


class TestJobTrackerResourceFaultAccounting:
    """The tracker records the raw per-model and global resource-fault counters the breaker reads."""

    async def test_terminal_overbudget_faults_accumulate_per_model(self, job_tracker: JobTracker) -> None:
        """Each terminal over-budget fault increments the model's streak and the global window count.

        This is the storm reproduced as bounded counters: three jobs for an unservable model each fault
        terminally (mirroring the ~2-3 min live cadence), driving the streak to the breaker threshold.
        """
        job_tracker.set_retry_policy(1)  # one shot, then fault: each fault is terminal
        for _ in range(3):
            resolution = await _terminal_overbudget_fault(job_tracker, _UNSERVABLE_MODEL)
            assert resolution is InferenceFailureResolution.FAULTED

        assert job_tracker.get_model_overbudget_fault_count(_UNSERVABLE_MODEL) == 3
        assert job_tracker.model_last_overbudget_fault_time(_UNSERVABLE_MODEL) is not None
        assert job_tracker.count_recent_resource_faults(window_seconds=600) == 3
        # The streak now satisfies the breaker policy: the worker should stop admitting/popping the model.
        assert (
            is_model_locally_unservable(
                overbudget_fault_count=job_tracker.get_model_overbudget_fault_count(_UNSERVABLE_MODEL),
                last_overbudget_fault_time=job_tracker.model_last_overbudget_fault_time(_UNSERVABLE_MODEL),
                threshold=3,
                cooldown_seconds=900,
            )
            is True
        )

    async def test_successful_generation_resets_streak(self, job_tracker: JobTracker) -> None:
        """A model that later produces a result is no longer unservable: its streak resets to zero.

        Guards against permanently blacklisting a model whose earlier faults were transient (a passing
        storm, a since-evicted competitor): once it generates, the breaker forgets the streak.
        """
        job_tracker.set_retry_policy(1)
        await _terminal_overbudget_fault(job_tracker, _UNSERVABLE_MODEL)
        await _terminal_overbudget_fault(job_tracker, _UNSERVABLE_MODEL)
        assert job_tracker.get_model_overbudget_fault_count(_UNSERVABLE_MODEL) == 2

        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)
        job_info = Mock()
        job_info.sdk_api_job_info = job
        await job_tracker.queue_for_safety(job_info)

        assert job_tracker.get_model_overbudget_fault_count(_UNSERVABLE_MODEL) == 0

    async def test_non_resource_fault_does_not_count(self, job_tracker: JobTracker) -> None:
        """An ordinary (non-resource) terminal fault does not feed the resource-fault counters.

        Only capacity faults should drive the breaker/self-throttle; a transient crash unrelated to VRAM
        must not flag a model unservable.
        """
        job_tracker.set_retry_policy(1)
        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)
        # No over-budget tag and no explicit resource signal: an ordinary fault.
        slot = make_mock_process_info(1, model_name=_UNSERVABLE_MODEL)
        resolution = job_tracker.handle_job_fault_now(faulted_job=job, process_info=slot)  # pyrefly: ignore

        assert resolution is InferenceFailureResolution.FAULTED
        assert job_tracker.get_model_overbudget_fault_count(_UNSERVABLE_MODEL) == 0
        assert job_tracker.count_recent_resource_faults(window_seconds=600) == 0


class TestPopperHoldsBackUnservableModels:
    """THE FIX (popper half): a locally-unservable model is not advertised for pop.

    With the model held back, the queue stops refilling with jobs the worker can only drop.
    """

    async def test_unservable_model_excluded_from_pop(self, job_tracker: JobTracker) -> None:
        """Once a model crosses the breaker threshold the popper drops it from the request, keeping others.

        This is what actually stops the server-maintenance storm: no new Flux jobs are popped during the
        cooldown, so the worker stops dropping them while still serving everything else.
        """
        job_tracker.set_retry_policy(1)
        for _ in range(3):
            await _terminal_overbudget_fault(job_tracker, _UNSERVABLE_MODEL)

        bridge_data = make_mock_bridge_data(
            image_models_to_load=[_UNSERVABLE_MODEL, "WAI-NSFW-illustrious-SDXL"],
            unservable_model_fault_threshold=3,
            unservable_model_cooldown_seconds=900,
        )
        models = _select_models_for_pop(
            bridge_data,
            ProcessMap({}),
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert models is not None
        assert _UNSERVABLE_MODEL not in models
        assert "WAI-NSFW-illustrious-SDXL" in models

    async def test_servable_models_still_popped_below_threshold(self, job_tracker: JobTracker) -> None:
        """CONTROL: the same model below the threshold is still popped, isolating the streak as the cause."""
        job_tracker.set_retry_policy(1)
        await _terminal_overbudget_fault(job_tracker, _UNSERVABLE_MODEL)  # 1 < threshold

        bridge_data = make_mock_bridge_data(
            image_models_to_load=[_UNSERVABLE_MODEL],
            unservable_model_fault_threshold=3,
            unservable_model_cooldown_seconds=900,
        )
        models = _select_models_for_pop(
            bridge_data,
            ProcessMap({}),
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert models is not None
        assert _UNSERVABLE_MODEL in models


class TestExclusiveRunSuppressesConcurrency:
    """THE FIX (exclusive-first half): an over-budget admit runs with the device to itself.

    The live storm came from a *second* process loading another model while the over-budget Flux job
    sampled, pushing free VRAM to ~0 and spilling Flux's weights to system RAM. Marking the job exclusive
    blocks any additional concurrent dispatch for its duration.
    """

    async def test_exclusive_job_caps_dispatch_to_itself(self, job_tracker: JobTracker) -> None:
        """With an exclusive job in progress, the concurrent-dispatch cap collapses to the running job."""
        process_map = ProcessMap({1: make_mock_process_info(1), 2: make_mock_process_info(2)})
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2, overbudget_exclusive_mode=True),
            max_concurrent=2,
            max_inference=2,
        )

        # Baseline: two concurrent slots permitted.
        assert scheduler._max_jobs_in_progress_allowed() == 2

        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)
        job_tracker.mark_admitted_exclusive(job)

        assert job_tracker.has_exclusive_job_in_progress() is True
        # No additional dispatch alongside the exclusive job.
        assert scheduler._max_jobs_in_progress_allowed() == 1

    async def test_no_exclusive_flag_keeps_normal_concurrency(self, job_tracker: JobTracker) -> None:
        """CONTROL: the same in-progress job *without* the exclusive flag keeps normal concurrency."""
        process_map = ProcessMap({1: make_mock_process_info(1), 2: make_mock_process_info(2)})
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2, overbudget_exclusive_mode=True),
            max_concurrent=2,
            max_inference=2,
        )
        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        assert job_tracker.has_exclusive_job_in_progress() is False
        assert scheduler._max_jobs_in_progress_allowed() == 2


class TestOverBudgetStepGrace:
    """THE FIX (don't kill a slow-but-progressing heavy job): an over-budget job earns a per-step grace."""

    async def test_overbudget_job_widens_step_timeout(self, job_tracker: JobTracker) -> None:
        """A slot running an over-budget job uses ``overbudget_step_timeout``, not ``inference_step_timeout``.

        At 83 s/it (the live Flux rate) the 20 s step timeout kills the slot; the widened grace lets the
        legitimately-slow step complete. Exercises the lifecycle's timeout-selection in isolation.
        """
        bridge_data = make_mock_bridge_data(inference_step_timeout=20, overbudget_step_timeout=120)
        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        await job_tracker.record_popped_job(job)
        proc = make_mock_process_info(1, model_name=_UNSERVABLE_MODEL)
        proc.last_job_referenced = job

        stub = Mock()
        stub._job_tracker = job_tracker

        # Ordinary job: the tight per-step timeout.
        assert ProcessLifecycleManager._effective_inference_step_timeout(stub, bridge_data, proc) == 20

        # Over-budget job: the widened grace (floored at the per-step timeout).
        job_tracker.mark_admitted_over_budget(job)
        assert ProcessLifecycleManager._effective_inference_step_timeout(stub, bridge_data, proc) == 120


class TestSelfThrottleBackstop:
    """THE FIX (backstop): the worker locally pauses pops before the horde forces server maintenance."""

    def test_engages_after_threshold_and_resumes_after_cooldown(self) -> None:
        """Crossing the resource-fault threshold pauses pops locally; the cooldown then auto-resumes them.

        This is the guarantee the worker never trips *server* maintenance: it throttles itself first.
        """
        manager = make_testable_process_manager(
            self_maintenance_fault_threshold=3,
            self_maintenance_window_seconds=600,
            self_maintenance_cooldown_seconds=300,
        )

        # Below threshold: no throttle.
        manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is False

        # Crossing the threshold engages the local pop-pause.
        manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True

        # Still within cooldown: stays paused.
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True

        # Cooldown elapsed: auto-resume.
        manager._state.self_throttle_paused_until = time.time() - 1
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is False

    def test_disabled_when_threshold_zero(self) -> None:
        """A threshold of 0 disables the backstop."""
        manager = make_testable_process_manager(self_maintenance_fault_threshold=0)
        for _ in range(10):
            manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is False


class TestVramMeasurementFixes:
    """The measurement half: account for the load peak, and freshen the free-VRAM signal."""

    def test_predict_uses_max_of_steady_and_load_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The VRAM prediction is the max of steady burden and the baseline load peak.

        The steady estimate (~13 GB) understates the transient text-encoder+transformer load peak (~16 GB)
        that drove device free VRAM to 52 MB in the live run; the max captures it.
        """
        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda j, b: Mock(vram_mb=13000, ram_mb=1000))
        monkeypatch.setattr(resource_budget, "_baseline_load_peak_mb", lambda b: 16000.0)
        assert predict_job_vram_mb(job, "flux_1") == 16000.0

    def test_predict_falls_back_to_steady_without_load_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a load-peak figure, the steady burden stands (unchanged behavior)."""
        job = make_job_pop_response(model=_UNSERVABLE_MODEL)
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda j, b: Mock(vram_mb=13000, ram_mb=1000))
        monkeypatch.setattr(resource_budget, "_baseline_load_peak_mb", lambda b: None)
        assert predict_job_vram_mb(job, "flux_1") == 13000.0

    def test_residency_snapshot_names_models_and_device_free(self) -> None:
        """The residency snapshot names per-slot models and the single device-wide free VRAM.

        Reproduces the over-commit signature a future live log will show: two heavy models resident with
        device free VRAM at ~52 MB.
        """
        proc_flux = make_mock_process_info(1, model_name=_UNSERVABLE_MODEL)
        proc_sdxl = make_mock_process_info(2, model_name="WAI-NSFW-illustrious-SDXL")
        for proc in (proc_flux, proc_sdxl):
            proc.total_vram_mb = 16375
            proc.vram_usage_mb = 16375 - 52  # device-wide free of 52 MB, the live load-peak figure
        snapshot = ProcessMap({1: proc_flux, 2: proc_sdxl}).residency_snapshot()

        assert _UNSERVABLE_MODEL in snapshot
        assert "WAI-NSFW-illustrious-SDXL" in snapshot
        assert "device_free_vram=52MB" in snapshot
