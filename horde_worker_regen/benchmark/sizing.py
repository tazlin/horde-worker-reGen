"""Resolution sizing for the post-processing sweep.

Post-processing VRAM cost scales with the *output* megapixels (the upscaler and face-fixer
activations dominate), so the resolution that most stresses post-processing is the largest one the
GPU can hold. This module derives that ceiling from the hordelib burden registry rather than a hard
coded guess, so it tracks the registry's per-baseline and per-feature estimates. Pure and
table-testable: no torch, no NVML.
"""

from __future__ import annotations

DEFAULT_VRAM_RESERVE_MB = 1500
"""VRAM kept free of the job footprint, matching ``LevelCriteria.min_vram_headroom_mb``."""

_CANDIDATE_SQUARE_EDGES = (2048, 1536, 1280, 1024, 768, 512)
"""Square output edges probed largest-first; the first that fits within the VRAM budget wins."""

_FALLBACK_NATIVE_MULTIPLE = 2
"""When VRAM is unknown, post-process at this multiple of the baseline's native edge (capped)."""

_MAX_FALLBACK_EDGE = 1024
"""Cap for the VRAM-unknown fallback, so a fake/CI run never asks for an absurd resolution."""


def max_post_processing_resolution(
    *,
    baseline: str,
    total_vram_mb: int | None,
    reserve_vram_mb: int = DEFAULT_VRAM_RESERVE_MB,
) -> int:
    """Return the largest square edge (px) a post-processed single-image job fits within VRAM.

    Probes a descending ladder of standard edges and returns the first whose estimated burden
    (baseline + an upscaler + a face-fixer at that output resolution) leaves at least
    ``reserve_vram_mb`` free. When VRAM or the baseline is unknown, falls back to a multiple of the
    baseline's native edge so callers always get a usable, bounded value; the per-level VRAM
    pre-flight remains the real backstop on undersized GPUs.

    Args:
        baseline: A ``KNOWN_IMAGE_GENERATION_BASELINE`` value (e.g. ``stable_diffusion_xl``).
        total_vram_mb: The GPU's total VRAM in MB, or None when it could not be detected.
        reserve_vram_mb: VRAM to keep free of the job footprint.

    Returns:
        The chosen square edge in pixels.
    """
    from hordelib.feature_impact import FEATURE_KIND, estimate_job_burden, get_baseline_burden

    burden = get_baseline_burden(baseline)
    native_edge = burden.native_resolution[0] if burden is not None else _MAX_FALLBACK_EDGE

    if total_vram_mb is None:
        return min(_MAX_FALLBACK_EDGE, native_edge * _FALLBACK_NATIVE_MULTIPLE)

    budget_mb = total_vram_mb - reserve_vram_mb
    post_processing_features = [FEATURE_KIND.post_processing_upscale, FEATURE_KIND.post_processing_facefix]

    for edge in _CANDIDATE_SQUARE_EDGES:
        if edge < native_edge:
            break
        estimate = estimate_job_burden(
            baseline=baseline,
            width=edge,
            height=edge,
            batch=1,
            features=post_processing_features,
        )
        if estimate.vram_mb <= budget_mb:
            return edge

    return native_edge


__all__ = ["DEFAULT_VRAM_RESERVE_MB", "max_post_processing_resolution"]
