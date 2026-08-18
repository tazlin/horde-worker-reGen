"""Model-level VRAM serviceability arithmetic for offer shaping and dispatch guards.

A model is judged against one card in three tiers. ``UNSERVICEABLE`` means the smallest legal job cannot
fit: the model is never offered and a job for it is faulted before any child touches VRAM. ``CONSTRAINED``
means the smallest job fits but the operator's ``max_power`` job does not: the model is offered with the
pop's ``max_power`` lowered to the largest size that fits, and the operator is told. ``SERVICEABLE`` means
the ``max_power`` job fits outright.

The footprint the tiers use is the disaggregated sampler charge: resident core (diffusion) weights plus the
sampling activation scaled by megapixels. That is what a card must host at sample time; support weights
(text encoders, VAE) are loaded around the sampler rather than beside it, and the decode transient is bounded
by tiling and covered by the admission noise buffer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from loguru import logger

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb

_SMALLEST_LEGAL_IMAGE_SIDE = 512
"""Smallest image side (px) accepted for Horde image jobs and used for the model minimum footprint."""

PIXELS_PER_MAX_POWER = 8 * 64 * 64
"""The horde's ``max_pixels`` per unit of ``max_power``."""


def max_power_to_pixels(max_power: int) -> int:
    """Return the ``max_pixels`` cap the horde derives from a ``max_power`` setting."""
    return max_power * PIXELS_PER_MAX_POWER


class ModelServiceabilityTier(enum.StrEnum):
    """How much of the operator's configured job range a model can run on a card."""

    SERVICEABLE = "serviceable"
    """The ``max_power`` job fits."""
    CONSTRAINED = "constrained"
    """The smallest legal job fits; the ``max_power`` job does not."""
    UNSERVICEABLE = "unserviceable"
    """Not even the smallest legal job fits."""


@dataclass(frozen=True)
class ModelFootprintFigures:
    """The model footprint terms used by the serviceability inequality.

    ``weights_mb`` is the resident core-weight seed. ``activation_per_megapixel_mb`` is the sampling
    activation above those weights per output megapixel (batch 1). ``minimum_activation_mb`` is the smallest
    legal job's activation, kept as its own figure so a caller can supply a floor the per-megapixel line does
    not capture; when it is zero the smallest legal job's activation is read off the per-megapixel line.
    """

    weights_mb: float
    activation_per_megapixel_mb: float = 0.0
    minimum_activation_mb: float = 0.0

    def activation_mb(self, pixels: int) -> float:
        """Return the sampling activation (MB) for a batch-1 job of ``pixels`` output pixels."""
        return max(self.minimum_activation_mb, self.activation_per_megapixel_mb * pixels / 1_000_000)

    def footprint_mb(self, pixels: int) -> float:
        """Return ``weights + activation`` (MB) for a batch-1 job of ``pixels`` output pixels."""
        return self.weights_mb + self.activation_mb(pixels)

    @property
    def minimum_footprint_mb(self) -> float:
        """Return the smallest legal job's footprint (MB)."""
        return self.footprint_mb(_SMALLEST_LEGAL_IMAGE_SIDE**2)

    def largest_fitting_pixels(self, capacity_mb: float) -> int | None:
        """Return the largest batch-1 pixel count that fits ``capacity_mb``, or None when the minimum does not."""
        if self.minimum_footprint_mb > capacity_mb:
            return None
        if self.activation_per_megapixel_mb <= 0:
            return None if capacity_mb < self.weights_mb else 2**62
        return int((capacity_mb - self.weights_mb) * 1_000_000 / self.activation_per_megapixel_mb)


@dataclass(frozen=True)
class ModelServiceabilityVerdict:
    """The result of checking a model's footprint against one card."""

    tier: ModelServiceabilityTier
    total_vram_mb: float | None
    baseline_mb: float
    noise_buffer_mb: float
    figures: ModelFootprintFigures | None
    max_pixels: int | None = None
    """The operator's ``max_power`` job in pixels, or None when the check covered the minimum job only."""

    @property
    def serviceable(self) -> bool:
        """Whether the smallest legal job fits (the model may be offered and jobs for it may be dispatched)."""
        return self.tier is not ModelServiceabilityTier.UNSERVICEABLE

    @property
    def capacity_mb(self) -> float | None:
        """Return ``total - baseline - noise`` in MB, or None when total is unknown."""
        if self.total_vram_mb is None:
            return None
        return (self.total_vram_mb - self.baseline_mb) - self.noise_buffer_mb

    @property
    def largest_fitting_pixels(self) -> int | None:
        """The largest batch-1 pixel count that fits this card, or None when unknown or nothing fits."""
        if self.capacity_mb is None or self.figures is None:
            return None
        return self.figures.largest_fitting_pixels(self.capacity_mb)

    @property
    def largest_fitting_max_power(self) -> int | None:
        """The largest ``max_power`` whose job fits this card, or None when unknown or nothing fits."""
        pixels = self.largest_fitting_pixels
        if pixels is None:
            return None
        return max(1, min(pixels // PIXELS_PER_MAX_POWER, 2**31))

    def reason(self) -> str:
        """Render the checked arithmetic for logs and fault diagnostics."""
        if self.total_vram_mb is None:
            return "device total is unknown; serviceability check abstains"
        if self.figures is None or self.capacity_mb is None:
            return "model footprint is unknown; serviceability check abstains"
        capacity = (
            f"capacity total {self.total_vram_mb:.0f} - baseline {self.baseline_mb:.0f} - noise "
            f"{self.noise_buffer_mb:.0f} = {self.capacity_mb:.0f} MB"
        )
        minimum = self.figures.minimum_footprint_mb
        if self.tier is ModelServiceabilityTier.UNSERVICEABLE:
            return (
                f"minimum footprint weights {self.figures.weights_mb:.0f} + activation "
                f"{self.figures.activation_mb(_SMALLEST_LEGAL_IMAGE_SIDE**2):.0f} = {minimum:.0f} MB vs "
                f"{capacity}: does NOT fit"
            )
        if self.max_pixels is None:
            return f"minimum footprint {minimum:.0f} MB vs {capacity}: fits"
        at_max = self.figures.footprint_mb(self.max_pixels)
        max_power = self.max_pixels // PIXELS_PER_MAX_POWER
        if self.tier is ModelServiceabilityTier.SERVICEABLE:
            return f"footprint at max_power {max_power} is {at_max:.0f} MB vs {capacity}: fits"
        return (
            f"footprint at max_power {max_power} is {at_max:.0f} MB vs {capacity}: does NOT fit; "
            f"largest fitting max_power is {self.largest_fitting_max_power}"
        )


def assess_model_serviceability(
    *,
    total_vram_mb: float | None,
    baseline_mb: float,
    noise_buffer_mb: float | None,
    figures: ModelFootprintFigures | None,
    max_pixels: int | None = None,
) -> ModelServiceabilityVerdict:
    """Return which tier a model falls in on one card.

    Unknown capacity or unknown footprint figures abstain as serviceable: the worker must not de-list a model
    on missing metadata. When both sides are known, the smallest legal job's ``weights + activation`` is
    compared with ``total - baseline - noise``; when ``max_pixels`` is given, the job at that size is compared
    too, and a model whose minimum fits but whose ``max_pixels`` job does not is ``CONSTRAINED``. The baseline
    is the shared device load the worker cannot reclaim; the noise buffer is the same admission slack used by
    runtime VRAM admission.
    """
    resolved_noise_mb = noise_buffer_mb if noise_buffer_mb is not None else admission_noise_buffer_mb(total_vram_mb)
    if total_vram_mb is None or total_vram_mb <= 0 or figures is None:
        return ModelServiceabilityVerdict(
            tier=ModelServiceabilityTier.SERVICEABLE,
            total_vram_mb=None if total_vram_mb is None or total_vram_mb <= 0 else total_vram_mb,
            baseline_mb=max(0.0, baseline_mb),
            noise_buffer_mb=resolved_noise_mb,
            figures=figures,
            max_pixels=max_pixels,
        )
    capacity_mb = (float(total_vram_mb) - max(0.0, baseline_mb)) - max(0.0, resolved_noise_mb)
    if figures.minimum_footprint_mb > capacity_mb:
        tier = ModelServiceabilityTier.UNSERVICEABLE
    elif max_pixels is not None and figures.footprint_mb(max_pixels) > capacity_mb:
        tier = ModelServiceabilityTier.CONSTRAINED
    else:
        tier = ModelServiceabilityTier.SERVICEABLE
    return ModelServiceabilityVerdict(
        tier=tier,
        total_vram_mb=float(total_vram_mb),
        baseline_mb=max(0.0, baseline_mb),
        noise_buffer_mb=max(0.0, resolved_noise_mb),
        figures=figures,
        max_pixels=max_pixels,
    )


def model_footprint_figures_for_baseline(baseline: str | None) -> ModelFootprintFigures | None:
    """Return footprint figures for a baseline using torch-free hordelib seeds, or None when unavailable.

    The figures are the disaggregated sampler charge: hordelib's resident core-weight seed plus its
    per-megapixel sampling activation. The whole-job seed (``vram_base_mb``) is not used because it bundles
    support weights and decode headroom that a small card serves around the sampler rather than beside it.
    """
    if baseline is None:
        return None
    try:
        from hordelib.feature_impact import get_baseline_burden

        burden = get_baseline_burden(str(baseline))
        if burden is None:
            return None
        return ModelFootprintFigures(
            weights_mb=float(burden.resident_weight_estimate_mb()),
            activation_per_megapixel_mb=float(burden.vram_per_megapixel_mb),
        )
    except Exception as e:
        logger.debug(f"Model serviceability footprint lookup failed for {baseline!r}: {type(e).__name__} {e}")
        return None
