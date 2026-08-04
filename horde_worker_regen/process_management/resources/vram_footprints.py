"""Learned VRAM footprint store: an in-memory, per-run record of measured device-memory peaks.

The static per-model VRAM seeds the scheduler charges (from the model reference) systematically
undershoot what a stage actually reserves at its activation peak: calibration on a 16GB card measured a
sampler holding ~11GB against a static seed of 6158MB. This store observes the real peaks children
report and offers them back as an estimate that can only ever *raise* the static seed, never lower it,
so a consumer sizing a request never plans below what the hardware has already demonstrated it needs.

Admission pricing of a job's sampling peak reads :meth:`LearnedFootprintStore.estimate_mb` with the static
per-model predictor as the seed: a measured watermark for the job's (baseline, resolution, platform, stage)
raises the priced peak above the seed and never below it. A whole-job monolithic peak and a disaggregated
UNet-only sampler peak are physically different quantities and are kept under distinct stages
(:attr:`FootprintStage.SAMPLE` vs :attr:`FootprintStage.SAMPLE_ISOLATED`) so a monolithic peak never
over-prices an isolated sampler (mixed operation is designed: a stage fault re-routes a disaggregated job
monolithic). Monolithic peaks are observed from child memory reports; isolated-sampler peaks are observed
from the disaggregation orchestrator at sample completion.

Not every stage is an activation peak. :attr:`FootprintStage.RESIDENT` and :attr:`FootprintStage.SAFETY`
record steady device charges (a loaded checkpoint's weights, the safety process's residency) observed while
nothing is running, and they live in the same store under the same raise-only watermark contract: a consumer
pricing "what does this already cost the card" reads them exactly as it reads a sampling peak.

Thread-safety: the store is written and read from the parent's single-threaded control loop (the same
loop that drains child memory reports), so no locking is required. It holds nothing across runs; every
worker start begins with cold keys that fall back to the static seed until a peak is observed.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field


class ResolutionBucket(enum.StrEnum):
    """A coarse image-resolution band used to key learned footprints.

    Bucketing is by the request's *maximum* dimension (a 1024x512 job and a 512x1024 job land in the
    same band): the activation peak tracks the larger side, and collapsing both orientations keeps the
    key space small. Batch size is deliberately NOT folded into the bucket: peaks are observed per
    request exactly as the hardware reported them, so a batched request's larger peak is recorded
    against the same key as a single image and naturally raises the learned watermark for that band.
    """

    LE_512 = "le_512"
    """Maximum dimension at or below 512 px."""
    LE_768 = "le_768"
    """Maximum dimension above 512 and at or below 768 px."""
    LE_1024 = "le_1024"
    """Maximum dimension above 768 and at or below 1024 px."""
    GT_1024 = "gt_1024"
    """Maximum dimension above 1024 px."""

    @classmethod
    def from_dimensions(cls, width: int, height: int, batch: int = 1) -> ResolutionBucket:
        """Classify a request into a bucket by its maximum dimension.

        Args:
            width (int): The request width in pixels.
            height (int): The request height in pixels.
            batch (int, optional): The batch size (``n_iter``). Accepted for call-site clarity but NOT
                folded into the bucket: peaks are observed per request as-is. Defaults to 1.

        Returns:
            ResolutionBucket: The band for the larger of ``width``/``height``.
        """
        _ = batch  # documented no-op: batch is not part of the bucket key
        largest = max(width, height)
        if largest <= 512:
            return cls.LE_512
        if largest <= 768:
            return cls.LE_768
        if largest <= 1024:
            return cls.LE_1024
        return cls.GT_1024


class FootprintStage(enum.StrEnum):
    """The pipeline stage a footprint peak is attributed to.

    The future VRAM arbiter's request kinds map onto these: a monolithic inference process's whole-job
    peak is attributed to :attr:`SAMPLE` (the dominant activation term), while the disaggregated lanes
    map to their respective stages.
    """

    SAMPLE = "sample"
    """The whole-job sampling stage (dominant activation peak of a monolithic job: UNet plus the text-encoder
    and VAE weights co-resident in the same process)."""
    SAMPLE_ISOLATED = "sample_isolated"
    """A disaggregated UNet-only sampler process's sampling peak: the text-encode, VAE, and post-processing run
    in other processes, so this holds only the core diffusion weights plus sampling activation. Kept distinct
    from :attr:`SAMPLE` because the two are physically different quantities (a whole-pipeline peak is far larger
    than an isolated sampler's), and watermarks are raise-only: folding a monolithic whole-job peak into the
    isolated key would permanently over-price a disaggregated sampler and deny the second concurrent sampler the
    card physically holds."""
    DECODE = "decode"
    """The VAE-decode stage."""
    ENCODE = "encode"
    """The text-encode (or VAE-encode) stage."""
    POST_PROCESS = "post_process"
    """The post-processing stage (upscale/face-fix)."""
    RESIDENT = "resident"
    """A loaded checkpoint's resident device footprint: the weights (plus whatever support components the
    checkpoint force-loads) a slot holds while it is idle, with no activation on the card. Physically distinct
    from the sampling stages for the same reason :attr:`SAMPLE_ISOLATED` is distinct from :attr:`SAMPLE`: a
    sampling peak includes transient activation that is released between jobs, so folding one into the resident
    key would permanently over-price what a merely-loaded slot costs and deny co-residency the card physically
    holds. Keyed per checkpoint (see :attr:`FootprintKey.checkpoint`), since weights are a property of the
    specific file rather than of the architecture."""
    SAFETY = "safety"
    """The safety process's device footprint (its resident classifier weights plus its CUDA context).

    Not tied to any generation model or request, so its key carries :data:`SAFETY_PROCESS_BASELINE` and no
    resolution band. What a consumer prices when deciding whether safety can sit on the card beside a job."""


SAFETY_PROCESS_BASELINE = "safety_process"
"""The :attr:`FootprintKey.model_baseline` token used for :attr:`FootprintStage.SAFETY` observations.

The safety process holds classifier weights that belong to no generation baseline, so its footprint is
recorded under this fixed token rather than against whatever model happens to be loaded elsewhere."""


class FootprintKey(BaseModel):
    """The identity a learned footprint is recorded under.

    Frozen so an instance is hashable and usable as a dict key. Two observations sharing every field are
    treated as the same footprint population.

    Which fields carry information depends on the stage, because the quantities scale with different things:

    - Activation stages (:attr:`FootprintStage.SAMPLE` and the disaggregated lane stages) are keyed by
      baseline and resolution band, and leave ``checkpoint`` unset: the activation peak scales with the
      architecture and the request size, so every checkpoint of a baseline shares one population and a
      per-checkpoint split would only fragment it.
    - :attr:`FootprintStage.RESIDENT` carries the checkpoint name and no resolution band: resident weights
      are a property of the specific file (two SDXL checkpoints differ by gigabytes) and do not move with
      the request size.
    - :attr:`FootprintStage.SAFETY` carries neither, keyed under :data:`SAFETY_PROCESS_BASELINE`.
    """

    model_config = ConfigDict(frozen=True)

    model_baseline: str
    """The model's baseline category (e.g. ``stable_diffusion_xl``); peaks vary sharply by architecture."""
    resolution_bucket: ResolutionBucket | None
    """The resolution band (by maximum dimension); the activation peak scales with it.

    None for a stage whose footprint does not scale with the request size (resident weights, safety)."""
    platform: str
    """The host platform token (``win32`` / ``linux`` from ``sys.platform``).

    Peaks are keyed by platform because the measured device-memory high-water differs by driver model:
    Windows/WDDM demand-pages and reports peaks unlike native Linux, so a peak learned on one platform
    is not a valid prior for the other."""
    stage: FootprintStage
    """The pipeline stage the peak was observed for."""
    checkpoint: str | None = None
    """The specific checkpoint the footprint belongs to, or None when the stage is baseline-keyed.

    Defaulted so a key that predates the distinction (every activation stage) is written unchanged."""


class _FootprintObservation(BaseModel):
    """The running statistics kept for one :class:`FootprintKey`."""

    ewma_mb: float
    """Exponentially-weighted moving average of observed peaks (smoothed central tendency, observability
    only: the estimate is watermark-driven so a transient dip never lowers the offered figure)."""
    watermark_mb: float
    """The maximum peak ever observed for this key (the undershoot-proof figure the estimate returns)."""
    observation_count: int = Field(default=0)
    """How many peaks have been folded in (diagnostics)."""


_EWMA_ALPHA = 0.3
"""Weight given to the newest observation in the EWMA (``new = alpha*sample + (1-alpha)*prev``).

A moderate value tracks a genuine shift over a handful of jobs without letting a single spike dominate
the smoothed average. The estimate does not depend on it (it uses the watermark); the EWMA is retained
for calibration visibility only."""


class LearnedFootprintStore:
    """An in-memory store of measured VRAM peaks keyed by (baseline, resolution, platform, stage).

    Single-threaded use only (the parent control loop). Not persisted: cold at every worker start.
    """

    def __init__(self) -> None:
        """Initialise an empty store."""
        self._observations: dict[FootprintKey, _FootprintObservation] = {}

    def observe_peak(self, key: FootprintKey, peak_reserved_mb: float) -> None:
        """Fold one observed device-memory figure into the running statistics for ``key``.

        Updates both an EWMA (alpha ``0.3``, smoothed central tendency for observability) and a
        max-watermark (the estimate's undershoot-proof basis). A non-positive figure is ignored: a zero or
        negative reading carries no footprint information and would only pollute the average.

        Args:
            key (FootprintKey): The footprint identity to record under.
            peak_reserved_mb (float): The device memory (MB) observed for this key: the reserved peak for an
                activation stage, the steady reserved figure for a resident stage.
        """
        if peak_reserved_mb <= 0:
            return

        existing = self._observations.get(key)
        if existing is None:
            self._observations[key] = _FootprintObservation(
                ewma_mb=peak_reserved_mb,
                watermark_mb=peak_reserved_mb,
                observation_count=1,
            )
            return

        self._observations[key] = _FootprintObservation(
            ewma_mb=(_EWMA_ALPHA * peak_reserved_mb) + ((1.0 - _EWMA_ALPHA) * existing.ewma_mb),
            watermark_mb=max(existing.watermark_mb, peak_reserved_mb),
            observation_count=existing.observation_count + 1,
        )

    def estimate_mb(self, key: FootprintKey, *, static_seed_mb: float) -> float:
        """Return the footprint estimate for ``key``: the static seed raised by any learned watermark.

        The learned overlay can only RAISE the seed, never lower it: measured peaks routinely exceed the
        static seed (calibration saw ~11GB measured against a 6158MB seed), and undershooting the true
        footprint is the failure this store exists to prevent, so the estimate is
        ``max(static_seed_mb, watermark)``. A cold key (never observed) returns the seed unchanged.

        Args:
            key (FootprintKey): The footprint identity to estimate for.
            static_seed_mb (float): The static per-model seed the caller would otherwise use as the floor.

        Returns:
            float: ``max(static_seed_mb, learned_watermark)`` (the seed for a cold key).
        """
        observation = self._observations.get(key)
        if observation is None:
            return static_seed_mb
        return max(static_seed_mb, observation.watermark_mb)

    def get_observation(self, key: FootprintKey) -> _FootprintObservation | None:
        """Return the raw running statistics for ``key`` (EWMA, watermark, count), or None if cold.

        Diagnostics/observability accessor: the decision surface is :meth:`estimate_mb`.
        """
        return self._observations.get(key)

    def __len__(self) -> int:
        """Return how many distinct keys have at least one observation."""
        return len(self._observations)
