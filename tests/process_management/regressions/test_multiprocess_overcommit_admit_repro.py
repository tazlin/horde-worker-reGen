"""Reproduction of the *admission-side* root cause behind the swallowed-OOM storm.

Companion to ``test_swallowed_oom_no_images_repro`` (which fixed how the *resulting* fault is classified).
This module pins why the worker drove four/five SDXL processes onto one 24 GB card in the first place.

The forecast for a head-of-queue SDXL job, with four inference processes alive, reads:

    Stream forecast for WAI-NSFW-illustrious-SDXL: weights ~4900 MB + 6275 MB reserve exceed 16140 MB free
    (after model evict: 3794 MB, alone: 20018 MB) ... [free_now=16140, after_model_evict=3794, alone=20018,
    live_procs=4, overhead/proc=4056MB] -> coresident=False, needs_exclusive=False, needs_teardown=False,
    streams_unavoidably=False

Read the numbers: with four ~4056 MB process contexts resident, evicting every sibling *model* still leaves
only 3794 MB free, which does not even hold the 4900 MB of weights, let alone the activation reserve. The
only thing that can make room is stopping a sibling *process* (a CUDA context is reclaimed only by the
process exiting). The model also fits with the card to itself (alone=20018). So the physically-correct
remedy is a partial sibling-process teardown -- exactly what ``requires_sibling_teardown`` exists to signal.

The blind spot:
    ``requires_sibling_teardown`` (and ``needs_exclusive_residency``) are gated behind ``_weights_dominant``,
    which is computed against a *fixed two-context* ceiling (``total_vram - 2 * per_process_overhead``: self
    plus one sibling). A moderate-weight SDXL job fits under that two-context ceiling, so it reads
    "not weight-dominant" and the teardown remedy is suppressed -- even though there are *four* live contexts,
    not two, and model eviction provably cannot free enough. With no structural remedy the head is deferred
    every tick until the starvation backstop force-admits it onto a card whose contexts already consume most
    of the VRAM, and it OOMs (the "no images were produced" fault). The two-context assumption underestimates
    contention whenever more than two inference processes are live.

This module reproduces that forecast contradiction directly. The fix makes the teardown signal topology-aware
(keyed on the live-context ``free_after_model_evict`` floor the forecast already computes) instead of the
fixed two-context dominance heuristic.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async
from tests.process_management.gpu.test_whole_card_residency_repro import _build_context_overcommit_scheduler

# Exact figures from the soak's forecast line (24 GB card, four live inference processes).
_DEVICE_TOTAL_VRAM_MB = 24074.0
_PER_PROCESS_OVERHEAD_MB = 4056.0
_SDXL_WEIGHTS_MB = 4900.0
_SDXL_RESERVE_MB = 6275.0  # activation-inclusive sampling peak headroom
_FREE_NOW_MB = 16140.0
_FREE_AFTER_MODEL_EVICT_MB = 3794.0  # four contexts resident, every sibling model evicted
_FREE_IF_ALONE_MB = 20018.0  # sole residency: one context remains


def _sdxl_four_process_forecast() -> StreamForecast:
    """The head-of-queue SDXL forecast with four sibling contexts over-committing a 24 GB card."""
    return StreamForecast(
        weights_mb=_SDXL_WEIGHTS_MB,
        reserve_mb=_SDXL_RESERVE_MB,
        free_now_mb=_FREE_NOW_MB,
        free_if_alone_mb=_FREE_IF_ALONE_MB,
        free_after_model_evict_mb=_FREE_AFTER_MODEL_EVICT_MB,
        total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
    )


class TestMultiProcessOvercommitForecast:
    """The forecast must call for a sibling-process teardown when the live contexts over-commit the card."""

    def test_model_eviction_is_structurally_insufficient(self) -> None:
        """Evicting every sibling *model* cannot free enough: even the bare weights do not fit the floor.

        This is the signature of a context (process-count) over-commit rather than a model over-commit: the
        remedy is stopping a sibling process, not unloading a sibling model.
        """
        forecast = _sdxl_four_process_forecast()
        assert forecast.fits_coresident is False
        assert forecast.fits_after_model_evict is False
        # 3794 MB free after model eviction does not even hold the 4900 MB of weights.
        assert _FREE_AFTER_MODEL_EVICT_MB < _SDXL_WEIGHTS_MB
        assert forecast.fits_alone is True  # the card to itself fits the weights

    def test_overcommit_needs_process_count_reduction(self) -> None:
        """THE BUG: with model eviction insufficient and sole residency fitting, a teardown must be signalled.

        The weight-dominant gates miss it: ``_weights_dominant`` (and so ``needs_exclusive_residency`` and
        ``requires_sibling_teardown``) judge the moderate-weight SDXL "not card-filling" under the fixed
        two-context ceiling, so no structural remedy fires from them and the head would be force-admitted into
        an OOM. The topology-aware ``needs_process_count_reduction`` catches it: the live four contexts
        squeeze the bounded weights off the card, but the model co-resides once the process count is reduced.
        """
        forecast = _sdxl_four_process_forecast()
        # The weight-dominant gates stay False -- this model does not need *sole* residency, only fewer procs.
        assert forecast.needs_exclusive_residency is False
        assert forecast.requires_sibling_teardown is False
        # The new, topology-aware signal fires: stop a sibling process so the weights fit.
        assert forecast.needs_process_count_reduction is True

    def test_partial_teardown_target_is_sized_not_sole_residency(self) -> None:
        """The teardown reduces the process count to what fits, not all the way to one process.

        budget = total - weights - reserve = 24074 - 4900 - 6275 = 12899; 12899 // 4056 = 3. So three
        contexts fit: the worker drops 4 -> 3 processes (a concurrency reduction), it does not serialize to
        sole residency. This is why the case is distinct from a weight-dominant whole-card (Flux) model.
        """
        forecast = _sdxl_four_process_forecast()
        assert forecast.max_resident_processes() == 3

    def test_not_misread_as_unavoidable_streaming(self) -> None:
        """The model fits alone, so it is servable via teardown -- never flagged unavoidably-streaming."""
        forecast = _sdxl_four_process_forecast()
        assert forecast.streams_unavoidably is False


class TestSchedulerActuatesProcessReduction:
    """The admission gate must act on ``needs_process_count_reduction``: reduce processes, not force-admit.

    Drives the *real* forecast end to end through the scheduler. The 16 GB scaffolding's small per-process
    overhead means a *heavy* checkpoint is what squeezes its weights off the card with four contexts live --
    the same forecast condition (``needs_process_count_reduction``) the soak's SDXL hit via large contexts on
    a 24 GB card. The behavioral point is identical: the gate stops a sibling process so the weights fit,
    rather than deferring until the starvation backstop force-admits the head into an OOM.
    """

    async def test_context_overcommit_head_reduces_process_count_then_defers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A head whose weights are squeezed off the card by live contexts triggers a partial teardown.

        Weights ~9500 MB: with four ~1288 MB contexts live the model-evicted floor (~11223 MB) no longer
        holds the weights + reserve, but the card fits it with fewer processes (max_resident == 3). Before the
        fix the moderate (not weight-dominant) head fell through to deferral; now the scheduler reduces the
        process count to what fits and defers one tick while the freed VRAM drains.
        """
        # A heavy checkpoint whose weights overflow the four-context floor but fit with fewer processes.
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: 9500.0)
        # Modest activation working set so the model reads activation-light (not weight-dominant), exercising
        # the needs_process_count_reduction path rather than needs_exclusive_residency.
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 9500.0 + 600)

        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=4)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=3)
        # Prevent the real psutil RAM reading from spuriously tripping the RAM danger floor gate
        # when system available memory is low (common in large combined test runs).
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)

        head_job = make_job_pop_response("CyberRealistic Pony")
        await track_popped_job_async(job_tracker, head_job)

        # Sanity: the forecast takes the new path, not the weight-dominant sole-residency one.
        forecast = scheduler._forecast_streaming(head_job, "stable_diffusion_xl")
        assert forecast.needs_process_count_reduction is True
        assert forecast.needs_exclusive_residency is False
        assert forecast.max_resident_processes() == 3

        admitted = scheduler.preload_models()

        assert admitted is False, "the head must defer while the card is cleared, not force-admit into an OOM"
        # The remedy: a sibling process is stopped down to the fitting count (3), not all the way to one.
        scheduler._process_lifecycle.scale_inference_processes.assert_called_once_with(3, device_index=None)
        assert scheduler._sibling_teardown_for_model == "CyberRealistic Pony"
