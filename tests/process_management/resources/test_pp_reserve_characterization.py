"""Characterization (golden-master) pins for the post-processing VRAM-reserve interactions.

The upcoming post-processing-reclaim fix adds a *passive layer*: it will charge a job's own imminent
post-processing peak into the forecast's ``post_processing_reserve_mb`` and the scheduler's committed
reserve. The existing suite (``test_resource_budget.py``) already covers the ``post_processing_reserve_mb
== 0`` cases (``TestUpscaleDoesNotDriveResidency``, ``TestCommittedPostProcessingReserve``,
``TestPostProcessingOverlapGate``). What is NOT yet pinned is what happens once that reserve is *non-zero*
-- exactly the regime the fix moves into. These tests pin today's behavior there so the fix's effect is a
visible, intended diff and any accidental conflation (a PP reserve flipping a moderate model into claiming
the whole card) is caught.

None of these assert desired-but-absent behavior; they pin what the current tree does. The desired
post-fix behavior lives in ``test_post_processing_reclaim.py`` (the RED suite).
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling import inference_scheduler as scheduler_module
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# A moderate SDXL job on a 16 GB card: weights ~4900 MB, a sampling-phase peak that co-resides with room
# to spare, and the per-process/overhead figures observed live (bridge.log: overhead/proc 1288-1354 MB,
# total 16375 MB, ~15 GB free alone). Values mirror test_resource_budget's
# ``test_forecast_uses_sampling_not_combined_peak`` so the two pins read against the same shape.
_SDXL_WEIGHTS_MB = 4900.0
_SDXL_SAMPLING_PEAK_MB = 6948.0
_BASE_RESERVE_MB = 2048.0
_TOTAL_VRAM_MB = 16375.0
_FREE_NOW_MB = 15005.0
_PER_PROCESS_OVERHEAD_MB = 1288.0
# A 4x upscale + face-fixer post-processing peak, as hordelib sizes it for this job shape.
_PP_PEAK_MB = 8533.0


def _sdxl_forecast(
    monkeypatch: pytest.MonkeyPatch, *, post_processing_reserve_mb: float
) -> resource_budget.StreamForecast:
    """Build the moderate-SDXL stream forecast with a controllable post-processing reserve.

    The weight, sampling-peak, and base-reserve seeds are pinned so the residency arithmetic is
    deterministic and independent of the live hordelib estimate; only ``post_processing_reserve_mb`` varies.
    """
    monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _SDXL_WEIGHTS_MB)
    monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: _SDXL_SAMPLING_PEAK_MB)
    monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *args, **kwargs: _BASE_RESERVE_MB)
    return resource_budget.forecast_weight_streaming(
        make_job_pop_response("stable_diffusion_xl"),
        "stable_diffusion_xl",
        free_now_mb=_FREE_NOW_MB,
        total_vram_mb=_TOTAL_VRAM_MB,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        num_inference_processes=1,
        configured_reserve_floor_mb=_BASE_RESERVE_MB,
        post_processing_reserve_mb=post_processing_reserve_mb,
    )


class TestPostProcessingReserveFoldsIntoPeakNotWeight:
    """A non-zero PP reserve folds into the activation-inclusive peak reserve, not the weight reserve.

    That separation is the structural reason it can drive eviction/teardown but never whole-card residency.
    """

    def test_reserve_folds_into_peak_reserve_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The PP reserve adds to ``reserve_mb`` (the peak) while ``base_reserve_mb`` (the weight floor) is held."""
        plain = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=0.0)
        with_pp = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=_PP_PEAK_MB)

        assert with_pp.reserve_mb == plain.reserve_mb + _PP_PEAK_MB
        # The bounded weight reserve is unchanged: weight-residency decisions stay PP-independent.
        assert with_pp.base_reserve_mb == plain.base_reserve_mb == _BASE_RESERVE_MB


class TestPostProcessingReserveNeverTriggersWholeCard:
    """The critical regression guard: a PP reserve must never flip a moderate SDXL into claiming the card.

    Whole-card residency (``needs_exclusive_residency``) and the weight-keyed verdicts are gated on the
    *bounded weight reserve*, which the PP reserve does not touch, so they are invariant to it. This is the
    deliberate separation (resource_budget.py:683-708) the fix's passive layer must preserve.
    """

    def test_whole_card_verdict_invariant_to_pp_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even an 8.5 GB PP reserve leaves the whole-card verdict False, as with no reserve at all."""
        plain = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=0.0)
        with_pp = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=_PP_PEAK_MB)

        assert plain.needs_exclusive_residency is False
        assert with_pp.needs_exclusive_residency is False

    def test_weight_keyed_verdicts_invariant_to_pp_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The weight-footprint verdicts (fit-alone, streams-unavoidably, context-reduction) ignore the PP reserve."""
        plain = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=0.0)
        with_pp = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=_PP_PEAK_MB)

        assert plain.fits_alone is True and with_pp.fits_alone is True
        assert plain.streams_unavoidably is False and with_pp.streams_unavoidably is False
        assert plain.needs_process_count_reduction is with_pp.needs_process_count_reduction is False


class TestPostProcessingReserveDrivesPeakPath:
    """A large PP reserve *does* engage the activation-peak path (co-residency closes, teardown demanded).

    This is the mechanism by which the passive layer will (correctly) stop a sibling co-residing into the
    headroom a job's own upscaler is about to need. The DANGER it pins: with a single inference process and
    no sibling to tear down, ``requires_sibling_teardown`` going True is unsatisfiable: the wedge the
    active reclaim ladder must resolve rather than letting the head park.
    """

    def test_peak_path_engages_under_large_pp_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Co-residency closes and a sibling teardown is demanded once the PP reserve overflows the headroom."""
        plain = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=0.0)
        with_pp = _sdxl_forecast(monkeypatch, post_processing_reserve_mb=_PP_PEAK_MB)

        assert plain.fits_coresident is True and with_pp.fits_coresident is False
        assert plain.fits_after_model_evict is True and with_pp.fits_after_model_evict is False
        assert plain.requires_sibling_teardown is False and with_pp.requires_sibling_teardown is True


def _committed_reserve_bridge_data() -> object:
    """Bridge data with the VRAM budget and the post-processing reserve feature both active."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2000,
        ram_reserve_mb=4096,
        post_processing_budget_reserve_enabled=True,
        image_models_to_load=["model_pp"],
    )


class TestCommittedReserveChargesOnlyActivePostProcessing:
    """The committed reserve charges a job only once it is *actually* in the post-processing phase.

    These pin the seam the passive layer will change: today a job's own imminent post-processing peak is
    not reserved while it samples (it counts only once the process reports ``INFERENCE_POST_PROCESSING``),
    and a process whose referenced job has already left flight contributes nothing (the self-healing
    stale-reference skip).
    """

    async def test_sampling_job_own_pp_peak_not_charged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job mid-inference (not yet post-processing) does not charge its own upscaler peak: the gap."""
        monkeypatch.setattr(scheduler_module, "predict_job_post_processing_vram_mb", lambda job, baseline: 1500.0)

        job_tracker = JobTracker()
        sampling_job = make_job_pop_response("model_pp")
        await track_popped_job_async(job_tracker, sampling_job)
        await job_tracker.mark_inference_started(sampling_job)

        sampling_proc = make_mock_process_info(
            0,
            model_name="model_pp",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        sampling_proc.last_job_referenced = sampling_job  # pyrefly: ignore - referenced job for the reserve lookup

        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: sampling_proc}),
            job_tracker=job_tracker,
            bridge_data=_committed_reserve_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler._committed_post_processing_reserve_mb() == 0.0

    async def test_stale_post_processing_reference_not_charged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A process in the PP phase whose job is no longer in flight contributes nothing (stale-ref skip)."""
        monkeypatch.setattr(scheduler_module, "predict_job_post_processing_vram_mb", lambda job, baseline: 1500.0)

        stale_job = make_job_pop_response("model_pp")  # never tracked -> not in jobs_in_progress
        pp_proc = make_mock_process_info(
            0,
            model_name="model_pp",
            state=HordeProcessState.INFERENCE_POST_PROCESSING,
        )
        pp_proc.last_job_referenced = stale_job  # pyrefly: ignore - referenced job for the reserve lookup

        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: pp_proc}),
            job_tracker=JobTracker(),
            bridge_data=_committed_reserve_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler._committed_post_processing_reserve_mb() == 0.0


@pytest.mark.skip(
    reason="Documents a hordelib/ComfyUI behavior not unit-testable in the worker repo; see docstring.",
)
def test_retention_is_advisory_when_an_upscale_needs_room() -> None:
    """DOCUMENTATION: ``keep_model_resident_after`` is advisory and an upscale can override it.

    ComfyUI's ``ImageUpscaleWithModel.upscale`` calls ``model_management.free_memory(memory_required,
    device)`` with no ``keep_loaded`` argument before the tiled upscale. ``free_memory`` will therefore
    evict *any* in-process model (including one the worker asked to keep resident via
    ``keep_model_resident_after`` / ``defer_vram_unload``) if the upscaler needs the room. The eviction is
    in-process only: it cannot touch sibling worker processes' models or CUDA contexts (those live in other
    processes), which is why the over-commit fix must reclaim cross-process VRAM in the orchestrator.

    The expectation to preserve: inter-job VRAM retention is a best-effort throughput optimization, never a
    correctness guarantee, and an imminent post-processing peak legitimately overrides it. This is asserted
    in hordelib's own test suite, not here; the worker has no in-process hook to observe ComfyUI's
    ``current_loaded_models`` to pin it as a unit test.
    """
