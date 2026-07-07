"""Model-level VRAM serviceability arithmetic for offer shaping and dispatch guards."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb

_SMALLEST_LEGAL_IMAGE_SIDE = 512
"""Smallest image side (px) accepted for Horde image jobs and used for the model minimum footprint."""


@dataclass(frozen=True)
class ModelFootprintFigures:
    """The model footprint terms used by the serviceability inequality.

    ``weights_mb`` is the resident model-weight seed. ``minimum_activation_mb`` is the smallest legal job's
    sampling activation above those weights. Their sum is the minimum device footprint the model class must
    host before any child process can usefully touch VRAM for that model.
    """

    weights_mb: float
    minimum_activation_mb: float

    @property
    def minimum_footprint_mb(self) -> float:
        """Return ``weights + minimum activation`` in MB."""
        return self.weights_mb + self.minimum_activation_mb


@dataclass(frozen=True)
class ModelServiceabilityVerdict:
    """The result of checking a model's minimum footprint against one card."""

    serviceable: bool
    total_vram_mb: float | None
    baseline_mb: float
    noise_buffer_mb: float
    figures: ModelFootprintFigures | None

    @property
    def capacity_mb(self) -> float | None:
        """Return ``total - baseline - noise`` in MB, or None when total is unknown."""
        if self.total_vram_mb is None:
            return None
        return (self.total_vram_mb - self.baseline_mb) - self.noise_buffer_mb

    def reason(self) -> str:
        """Render the checked arithmetic for logs and fault diagnostics."""
        if self.total_vram_mb is None:
            return "device total is unknown; serviceability check abstains"
        if self.figures is None:
            return "model footprint is unknown; serviceability check abstains"
        verb = "fits" if self.serviceable else "does NOT fit"
        return (
            f"minimum footprint weights {self.figures.weights_mb:.0f} + activation "
            f"{self.figures.minimum_activation_mb:.0f} = {self.figures.minimum_footprint_mb:.0f} MB vs "
            f"capacity total {self.total_vram_mb:.0f} - baseline {self.baseline_mb:.0f} - noise "
            f"{self.noise_buffer_mb:.0f} = {self.capacity_mb:.0f} MB: {verb}"
        )


def assess_model_serviceability(
    *,
    total_vram_mb: float | None,
    baseline_mb: float,
    noise_buffer_mb: float | None,
    figures: ModelFootprintFigures | None,
) -> ModelServiceabilityVerdict:
    """Return whether a model's minimum footprint can ever fit one card.

    Unknown capacity or unknown footprint figures abstain as serviceable: the worker must not de-list a model
    on missing metadata. When both sides are known, serviceability is exactly
    ``weights + minimum_activation <= total - baseline - noise``. The baseline is the shared device load the
    worker cannot reclaim; the noise buffer is the same admission slack used by runtime VRAM admission.
    """
    resolved_noise_mb = noise_buffer_mb if noise_buffer_mb is not None else admission_noise_buffer_mb(total_vram_mb)
    if total_vram_mb is None or total_vram_mb <= 0 or figures is None:
        return ModelServiceabilityVerdict(
            serviceable=True,
            total_vram_mb=None if total_vram_mb is None or total_vram_mb <= 0 else total_vram_mb,
            baseline_mb=max(0.0, baseline_mb),
            noise_buffer_mb=resolved_noise_mb,
            figures=figures,
        )
    capacity_mb = (float(total_vram_mb) - max(0.0, baseline_mb)) - max(0.0, resolved_noise_mb)
    return ModelServiceabilityVerdict(
        serviceable=figures.minimum_footprint_mb <= capacity_mb,
        total_vram_mb=float(total_vram_mb),
        baseline_mb=max(0.0, baseline_mb),
        noise_buffer_mb=max(0.0, resolved_noise_mb),
        figures=figures,
    )


def model_footprint_figures_for_baseline(baseline: str | None) -> ModelFootprintFigures | None:
    """Return footprint figures for a baseline using torch-free hordelib seeds, or None when unavailable."""
    if baseline is None:
        return None
    try:
        from hordelib.feature_impact import estimate_job_burden, get_baseline_burden

        burden = get_baseline_burden(str(baseline))
        if burden is None:
            return None
        weights_mb = float(burden.resident_weight_estimate_mb())
        minimum = estimate_job_burden(
            baseline=str(baseline),
            width=_SMALLEST_LEGAL_IMAGE_SIDE,
            height=_SMALLEST_LEGAL_IMAGE_SIDE,
            batch=1,
        )
        minimum_sampling_mb = float(minimum.vram_sampling_mb)
        return ModelFootprintFigures(
            weights_mb=weights_mb,
            minimum_activation_mb=max(0.0, minimum_sampling_mb - weights_mb),
        )
    except Exception as e:
        logger.debug(f"Model serviceability footprint lookup failed for {baseline!r}: {type(e).__name__} {e}")
        return None
