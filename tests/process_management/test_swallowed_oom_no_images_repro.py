"""Reproduction and fix for the *swallowed-OOM* crash storm that neither resource-fault backstop sees.

Failure mode:
    Several inference processes are co-resident on one GPU, each pinning a resident model plus a
    per-process CUDA context, so free VRAM collapses to a sliver. A job then tries to allocate a few MiB
    and the driver returns CUDA OOM. ComfyUI *catches* the OOM internally ("Got an OOM, unloading
    all loaded models") and the pipeline produces no output node, so hordelib raises the generic

        RuntimeError: Pipeline failed to run - no images were produced. Model: unknown, Steps: unknown, Resolution: ?x?

    The inference process surfaces that text verbatim as the faulted result's ``info``. Because ComfyUI
    swallowed the OOM, the *torch* "CUDA out of memory" wording never reaches the worker: the only signal it
    sees is "no images were produced", which carries no OOM substring.

Why both designed backstops stayed blind:
    - ``is_resource_failure("...no images were produced...")`` returned False, so the dispatcher routed the
      fault as a *generic* inference failure: an ordinary requeue, not the degraded/isolated retry that
      clears the device.
    - A generic (non-resource) terminal fault never calls ``JobTracker._record_resource_fault``, so it feeds
      neither the per-model circuit-breaker streak nor the global self-throttle window.
    - Net effect across a storm of "no images produced" faults: ``Self-throttle engaged`` never fires
      and no model is ever ``held back as locally unservable``. The storm is instead absorbed only by the
      far more disruptive save-our-ship soft reset (rebuild every pool, limp by), which churns the whole
      backlog.

The fix pinned here:
    Recognize the swallowed-OOM surface form ("no images were produced" / "pipeline failed to run") as a
    resource-class failure in :mod:`failure_classification`. A pipeline that yields no output node is, in
    practice on a multi-process worker, the visible end of an OOM ComfyUI handled internally; treating it as
    a resource fault earns the device-clearing degraded retry and, on repetition, feeds the breaker and
    self-throttle so the gentle backstops engage before the supervisor has to soft-reset. A genuinely
    deterministic non-resource fault still faults terminally after its one isolated retry, and the per-model
    streak only trips when a model produces no images *every* attempt (a success resets it), so a model that
    works for other jobs is never held back.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.failure_classification import is_resource_failure
from horde_worker_regen.process_management.job_tracker import InferenceFailureResolution, JobTracker

from .conftest import make_job_pop_response, make_mock_process_info

# The exact faulted-result ``info`` the inference process emits for a swallowed-OOM pipeline failure:
# ``f"{type(e).__name__}: {e}"`` over hordelib's ``run_image_pipeline`` RuntimeError, which surfaces with the
# resolution fields unresolved ("Model: unknown, Steps: unknown, Resolution: ?x?") when no output node ran.
_NO_IMAGES_INFO = (
    "RuntimeError: Pipeline failed to run - no images were produced. "
    "Model: unknown, Steps: unknown, Resolution: ?x?"
)
_STORM_MODEL = "AlbedoBase XL (SDXL)"


class TestSwallowedOomIsRecognizedAsResource:
    """The classifier must treat the no-images surface form as a recoverable resource failure."""

    @pytest.mark.parametrize(
        "info",
        [
            _NO_IMAGES_INFO,
            "RuntimeError: Pipeline failed to run - no images were produced. "
            "Model: AlbedoBase XL (SDXL), Steps: 30, Resolution: 1024x1024",
            "Pipeline failed to run - no images were produced.",
        ],
    )
    def test_no_images_produced_classifies_as_resource(self, info: str) -> None:
        """A swallowed-OOM 'no images were produced' fault is a resource failure, not a generic one."""
        assert is_resource_failure(info) is True

    @pytest.mark.parametrize(
        "info",
        [
            "ValueError: bad prompt",
            "RuntimeError: model graph node 'KSampler' is missing an input",
            "12.3 it/s",
            None,
        ],
    )
    def test_genuine_non_resource_faults_still_not_resource(self, info: str | None) -> None:
        """The fix stays surgical: ordinary input/graph faults are not misclassified as resource."""
        assert is_resource_failure(info) is False


async def _terminal_no_images_fault(job_tracker: JobTracker, model: str, info: str) -> InferenceFailureResolution:
    """Drive one terminal fault exactly as the dispatcher does for a faulted inference *result* message.

    The dispatcher computes ``is_resource_failure(message.info)`` and passes it to ``handle_job_fault``; the
    job is *not* tagged ``admitted_over_budget`` because it OOM'd from sibling over-commit, not its own
    over-budget admit. So the resource classification rides entirely on the ``info`` string, which is the
    exact coupling this storm exposed.
    """
    job = make_job_pop_response(model=model)
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    slot = make_mock_process_info(1, model_name=model)
    return job_tracker.handle_job_fault_now(
        faulted_job=job,
        process_info=slot,
        is_resource_failure=is_resource_failure(info),
    )  # pyrefly: ignore


class TestSwallowedOomFeedsResourceBackstops:
    """A repeated no-images storm must feed the per-model breaker and the self-throttle window."""

    async def test_no_images_storm_drives_the_breaker_streak(self, job_tracker: JobTracker) -> None:
        """Three terminal no-images faults accumulate the model's resource streak and the global window.

        When the no-images info is classified generic these counters stay at zero, so neither backstop
        engages. With the fix the dispatcher's resource flag is True, so the same faults feed both backstops,
        mirroring the over-budget path that ``test_overbudget_unservable_storm_repro`` already pins.
        """
        job_tracker.set_retry_policy(1)  # one shot, then fault: every fault is terminal

        for _ in range(3):
            resolution = await _terminal_no_images_fault(job_tracker, _STORM_MODEL, _NO_IMAGES_INFO)
            assert resolution is InferenceFailureResolution.FAULTED

        assert job_tracker.get_model_overbudget_fault_count(_STORM_MODEL) == 3
        assert job_tracker.model_last_overbudget_fault_time(_STORM_MODEL) is not None
        assert job_tracker.count_recent_resource_faults(window_seconds=600) == 3

    async def test_no_images_fault_earns_a_degraded_isolated_retry(self, job_tracker: JobTracker) -> None:
        """The first no-images fault is requeued *degraded* so the retry runs with the device cleared.

        The generic-fault path requeues onto another equally over-committed slot, which simply OOMs again;
        the degraded retry is what isolates the job so the second attempt has room to succeed.
        """
        job_tracker.set_retry_policy(2)  # allow one retry before the terminal fault

        job = make_job_pop_response(model=_STORM_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)
        slot = make_mock_process_info(1, model_name=_STORM_MODEL)
        resolution = job_tracker.handle_job_fault_now(
            faulted_job=job,
            process_info=slot,
            is_resource_failure=is_resource_failure(_NO_IMAGES_INFO),
        )  # pyrefly: ignore

        assert resolution is InferenceFailureResolution.RETRY_DEGRADED

    async def test_genuine_non_resource_storm_still_ignored_by_backstops(self, job_tracker: JobTracker) -> None:
        """A deterministic non-resource fault must not feed the resource counters even when it repeats."""
        job_tracker.set_retry_policy(1)

        for _ in range(3):
            resolution = await _terminal_no_images_fault(job_tracker, _STORM_MODEL, "ValueError: bad prompt")
            assert resolution is InferenceFailureResolution.FAULTED

        assert job_tracker.get_model_overbudget_fault_count(_STORM_MODEL) == 0
        assert job_tracker.count_recent_resource_faults(window_seconds=600) == 0
