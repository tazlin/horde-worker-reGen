"""Post-processing load model: empirically grounded per-op costs, job classes, and traffic scenarios.

The dedicated post-processing lane runs ESRGAN-family upscalers, GFPGAN/CodeFormer face-fixers, and
background stripping as chained per-image operations. Their true resource behavior, measured on a
representative 16 GB consumer card with the lane's exact hordelib bring-up, has a shape that any
admission policy must respect:

- Upscaler VRAM peaks are effectively *flat* with respect to image size (the backend tiles the
  activation), so a 4x-family upscale costs roughly the same few GB at any generation resolution,
  while its wall time scales with *output* megapixels.
- Face-fixer VRAM peaks scale with *input* megapixels (whole-image face detection), and because a
  face-fixer is chained after an upscaler its input is the upscaled image: a 4x chain multiplies the
  face-fixer's input megapixels by 16.
- A chain's VRAM peak is the maximum of its ops' peaks (weights release between graph runs), never
  the sum; a chain's wall time is the sum.
- When an op is dispatched into VRAM it does not comfortably fit (device over-subscription across
  processes), the driver demand-pages rather than failing: the op still completes but one to two
  orders of magnitude slower, silently. A watchdog that equates silence with death then reaps live
  work. This "thrash regime" is a first-class simulation outcome, not an error.

This module gives tests and the harness a shared, deterministic stand-in for those dynamics: a cost
model (:class:`PostProcessLoadModel`), canonical job classes spanning cheap to pathological, card
profiles from low-end to high-end, and named traffic scenarios that avoid a combinatorial matrix while
still covering the load structures that matter.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

_UPSCALE_FACTORS: dict[str, float] = {
    "RealESRGAN_x2plus": 2.0,
    "2xModernSpanimationV1": 2.0,
    "RealESRGAN_x4plus": 4.0,
    "RealESRGAN_x4plus_anime_6B": 4.0,
    "NMKD_Siax": 4.0,
    "4x_AnimeSharp": 4.0,
    "4xNomos8kSC": 4.0,
    "4xLSDIRplus": 4.0,
    "4xNomosWebPhoto_RealPLKSR": 4.0,
    "4xNomos2_realplksr_dysample": 4.0,
    "4xNomos2_hq_dat2": 4.0,
}

_FACEFIXERS = frozenset({"GFPGAN", "GFPGANv1.3", "CodeFormers", "RestoreFormer"})

# Measured envelope, uncontended. Upscaler peaks are flat (tiled activation); wall time scales with
# output megapixels. Face-fixer peaks scale with input megapixels; wall time does too, mildly.
_UPSCALE_4X_PEAK_MB = 3600.0
_UPSCALE_2X_PEAK_MB = 1000.0
_UPSCALE_WALL_S_PER_OUTPUT_MEGAPIXEL = 0.40
_UPSCALE_WALL_BASE_S = 0.6
_FACEFIX_PEAK_BASE_MB = 1000.0
_FACEFIX_PEAK_PER_INPUT_MEGAPIXEL_MB = 220.0
_FACEFIX_WALL_BASE_S = 1.8
_FACEFIX_WALL_S_PER_INPUT_MEGAPIXEL = 0.42
_STRIP_BACKGROUND_PEAK_MB = 1200.0
_STRIP_BACKGROUND_WALL_S = 2.5
_UNKNOWN_OP_PEAK_MB = 3600.0
_UNKNOWN_OP_WALL_S = 8.0

THRASH_GUARD_BAND_MB = 1500.0
"""Free-VRAM margin below which a dispatched op enters the demand-paging regime.

Per-process free-VRAM readings are soft on WDDM-style drivers (another process's allocations can be
paged out to satisfy this one), so an op whose peak lands within this band of the measured free figure
is not guaranteed in-VRAM execution even though it nominally "fits".
"""

THRASH_SLOWDOWN_FACTOR = 20.0
"""Wall-time multiplier for ops run in the demand-paging regime (measured 20x to 50x; the optimistic
end keeps simulated timelines short while still dwarfing every watchdog and patience threshold)."""

CONTENTION_SLOWDOWN_FACTOR = 1.5
"""Wall-time multiplier for ops that fit comfortably but share the device with saturating compute."""


@dataclass(frozen=True)
class OpCost:
    """The resource cost of one post-processing operation on one image."""

    peak_vram_mb: float
    wall_s: float


class PostProcessLoadModel:
    """Deterministic cost model for post-processing chains, mirroring the measured envelope."""

    def op_cost(self, operation: str, input_megapixels: float) -> OpCost:
        """Return the cost of one operation applied to an image of ``input_megapixels``.

        Args:
            operation: The post-processor name as it appears in a job's ``post_processing`` list.
            input_megapixels: Megapixels of the image handed to this op (after any prior chained op).
        """
        factor = _UPSCALE_FACTORS.get(operation)
        if factor is not None:
            output_megapixels = input_megapixels * factor**2
            peak = _UPSCALE_2X_PEAK_MB if factor <= 2.0 else _UPSCALE_4X_PEAK_MB
            return OpCost(
                peak_vram_mb=peak,
                wall_s=_UPSCALE_WALL_BASE_S + _UPSCALE_WALL_S_PER_OUTPUT_MEGAPIXEL * output_megapixels,
            )
        if operation in _FACEFIXERS:
            return OpCost(
                peak_vram_mb=_FACEFIX_PEAK_BASE_MB + _FACEFIX_PEAK_PER_INPUT_MEGAPIXEL_MB * input_megapixels,
                wall_s=_FACEFIX_WALL_BASE_S + _FACEFIX_WALL_S_PER_INPUT_MEGAPIXEL * input_megapixels,
            )
        if operation == "strip_background":
            return OpCost(peak_vram_mb=_STRIP_BACKGROUND_PEAK_MB, wall_s=_STRIP_BACKGROUND_WALL_S)
        return OpCost(peak_vram_mb=_UNKNOWN_OP_PEAK_MB, wall_s=_UNKNOWN_OP_WALL_S)

    def chain_cost(self, post_processing: list[str], generation_megapixels: float, num_images: int = 1) -> OpCost:
        """Return the cost of a full chain over ``num_images`` images.

        Upscalers run before face-fixers (the lane's execution order), so a face-fixer's input
        megapixels are the generation megapixels multiplied by the squared upscale factor of any
        upscaler in the chain. The chain peak is the max of the op peaks; wall time is the per-image
        sum multiplied by the image count.
        """
        upscale_factor = max(
            (_UPSCALE_FACTORS.get(op, 1.0) for op in post_processing),
            default=1.0,
        )
        peak = 0.0
        wall = 0.0
        for op in post_processing:
            megapixels = generation_megapixels
            if op in _FACEFIXERS or op == "strip_background":
                megapixels = generation_megapixels * upscale_factor**2
            cost = self.op_cost(op, megapixels)
            peak = max(peak, cost.peak_vram_mb)
            wall += cost.wall_s
        return OpCost(peak_vram_mb=peak, wall_s=wall * max(1, num_images))

    def estimate_for_job(self, job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
        """Return the chain VRAM peak (MB) for a popped job, with the production estimator's signature.

        A drop-in for ``predict_job_post_processing_vram_mb`` so simulations can decouple dispatch-policy
        behavior from the accuracy of the seed-based estimator: the policy is exercised against the
        measured envelope. Returns 0.0 for a job with no post-processing.
        """
        post_processing = list(job.payload.post_processing or [])
        if not post_processing:
            return 0.0
        megapixels = (job.payload.width * job.payload.height) / 1_000_000
        return self.chain_cost(post_processing, megapixels, max(1, job.payload.n_iter)).peak_vram_mb


class PostProcessJobClass(enum.StrEnum):
    """Canonical post-processing job shapes spanning the cheap-to-pathological cost range."""

    X2_ONLY = "x2_only"
    X4_ONLY = "x4_only"
    X4_FACEFIX = "x4_facefix"
    FACEFIX_ONLY = "facefix_only"
    STRIP_BACKGROUND = "strip_background"
    X4_FACEFIX_LARGE = "x4_facefix_large"


@dataclass(frozen=True)
class JobShape:
    """The pop-visible shape of a post-processing job class."""

    post_processing: list[str]
    width: int
    height: int
    n_iter: int = 1


JOB_SHAPES: dict[PostProcessJobClass, JobShape] = {
    PostProcessJobClass.X2_ONLY: JobShape(["RealESRGAN_x2plus"], 1024, 1024),
    PostProcessJobClass.X4_ONLY: JobShape(["RealESRGAN_x4plus"], 1024, 1024),
    PostProcessJobClass.X4_FACEFIX: JobShape(["RealESRGAN_x4plus", "GFPGAN"], 1024, 1024),
    PostProcessJobClass.FACEFIX_ONLY: JobShape(["CodeFormers"], 1024, 1024),
    PostProcessJobClass.STRIP_BACKGROUND: JobShape(["strip_background"], 1024, 1024),
    PostProcessJobClass.X4_FACEFIX_LARGE: JobShape(["RealESRGAN_x4plus", "GFPGAN"], 1536, 1536),
}


@dataclass(frozen=True)
class CardProfile:
    """A card's steady free-VRAM behavior as the lane sees it while inference runs beside it.

    ``free_vram_mb`` is the measured device-wide free figure during normal inference residency;
    ``free_vram_dip_mb``/``dip_period_s``/``dip_duty`` describe periodic troughs (checkpoint loads,
    sampling activation spikes). A zero duty models a steady card.
    """

    name: str
    total_vram_mb: float
    free_vram_mb: float
    free_vram_dip_mb: float = 0.0
    dip_period_s: float = 30.0
    dip_duty: float = 0.0

    def free_at(self, now_s: float) -> float:
        """Return the free VRAM the lane's card reports at simulated time ``now_s``."""
        if self.dip_duty <= 0.0:
            return self.free_vram_mb
        phase = (now_s % self.dip_period_s) / self.dip_period_s
        return self.free_vram_dip_mb if phase < self.dip_duty else self.free_vram_mb


LOW_END_CARD = CardProfile(name="low_end_8gb", total_vram_mb=8192, free_vram_mb=2500)
"""An 8 GB card with an SD15-class checkpoint resident: barely 2.5 GB to spare for transient peaks."""

AVERAGE_CARD = CardProfile(
    name="average_16gb",
    total_vram_mb=16384,
    free_vram_mb=6500,
    free_vram_dip_mb=2500,
    dip_duty=0.25,
)
"""A 16 GB card running SDXL with several worker contexts: ~6.5 GB free, dipping under load spikes."""

HIGH_END_CARD = CardProfile(name="high_end_24gb", total_vram_mb=24576, free_vram_mb=12000)
"""A 24 GB card with comfortable steady headroom for any measured chain peak."""


@dataclass(frozen=True)
class ArrivalPattern:
    """When post-processing-eligible jobs finish inference and enter the lane's queue."""

    interval_s: float
    burst_size: int = 1


@dataclass(frozen=True)
class PostProcessLoadScenario:
    """A named traffic scenario: which job classes arrive, how often, onto which card."""

    name: str
    card: CardProfile
    arrivals: ArrivalPattern
    job_sequence: list[PostProcessJobClass]
    duration_s: float = 600.0


def _cycle(classes: list[PostProcessJobClass], count: int) -> list[PostProcessJobClass]:
    return [classes[i % len(classes)] for i in range(count)]


LIGHT_MIX = [
    PostProcessJobClass.X2_ONLY,
    PostProcessJobClass.FACEFIX_ONLY,
    PostProcessJobClass.X2_ONLY,
    PostProcessJobClass.STRIP_BACKGROUND,
]
"""Cheap-op traffic: the profile of a worker whose users mostly request 2x upscales and face fixes."""

TYPICAL_MIX = [
    PostProcessJobClass.X2_ONLY,
    PostProcessJobClass.X4_ONLY,
    PostProcessJobClass.X4_FACEFIX,
    PostProcessJobClass.X2_ONLY,
    PostProcessJobClass.FACEFIX_ONLY,
    PostProcessJobClass.X4_ONLY,
]
"""Mixed traffic matching observed live proportions: roughly a third heavy 4x work."""

HEAVY_MIX = [
    PostProcessJobClass.X4_FACEFIX,
    PostProcessJobClass.X4_ONLY,
    PostProcessJobClass.X4_FACEFIX_LARGE,
    PostProcessJobClass.X4_FACEFIX,
]
"""Worst-case traffic: dominated by 4x chains, including large-generation chains."""


def canned_scenarios() -> list[PostProcessLoadScenario]:
    """Return the canonical scenario set: card tiers crossed with the mixes that stress each tier.

    Deliberately not a full Cartesian product: each entry exists because it isolates one dynamic
    (cheap traffic must never queue, typical traffic must survive dips, heavy traffic must degrade
    gracefully on cards that structurally cannot host its peaks).
    """
    return [
        PostProcessLoadScenario(
            name="high_end_typical",
            card=HIGH_END_CARD,
            arrivals=ArrivalPattern(interval_s=12.0),
            job_sequence=_cycle(TYPICAL_MIX, 20),
        ),
        # Burst spacing sits just above the measured serial service time of one burst, so the queue
        # spikes and drains rather than growing without bound (an overloaded lane starves under any
        # policy and would only measure the scenario, not the dispatcher).
        PostProcessLoadScenario(
            name="high_end_heavy_burst",
            card=HIGH_END_CARD,
            arrivals=ArrivalPattern(interval_s=120.0, burst_size=4),
            job_sequence=_cycle(HEAVY_MIX, 12),
        ),
        PostProcessLoadScenario(
            name="average_light",
            card=AVERAGE_CARD,
            arrivals=ArrivalPattern(interval_s=15.0),
            job_sequence=_cycle(LIGHT_MIX, 16),
        ),
        PostProcessLoadScenario(
            name="average_typical",
            card=AVERAGE_CARD,
            arrivals=ArrivalPattern(interval_s=12.0),
            job_sequence=_cycle(TYPICAL_MIX, 18),
        ),
        PostProcessLoadScenario(
            name="average_heavy",
            card=AVERAGE_CARD,
            arrivals=ArrivalPattern(interval_s=20.0),
            job_sequence=_cycle(HEAVY_MIX, 10),
        ),
        PostProcessLoadScenario(
            name="low_end_light",
            card=LOW_END_CARD,
            arrivals=ArrivalPattern(interval_s=20.0),
            job_sequence=_cycle(LIGHT_MIX, 12),
        ),
        PostProcessLoadScenario(
            name="low_end_typical",
            card=LOW_END_CARD,
            arrivals=ArrivalPattern(interval_s=20.0),
            job_sequence=_cycle(TYPICAL_MIX, 12),
        ),
    ]
