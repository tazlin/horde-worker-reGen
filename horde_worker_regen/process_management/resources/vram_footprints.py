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

Two estimate policies live here and are deliberately kept apart. :meth:`LearnedFootprintStore.estimate_mb`
is the raise-only overlay described above, used wherever undershooting is the only failure that matters.
:meth:`LearnedFootprintStore.measured_estimate_mb` is bidirectional: once a key carries at least
:data:`_MIN_OBSERVATIONS_FOR_MEASURED` observations it answers from the measurements alone, so a seed that
over-states the hardware (a Flux fp8 seed of 16.4 GB against a measured 13.5 GB device-used median) stops
denying co-residency the card physically holds. Its conservatism is one explicit knob,
:data:`_MEASURED_ESTIMATE_MARGIN` plus the platform context floor, rather than being smeared across the
seeds: Linux OOM-kills and WDDM paging both punish an under-estimate far harder than an over-estimate, so
the margin exists, but it is one number a reader can find and change.

Thread-safety: the store is written and read from the parent's single-threaded control loop (the same
loop that drains child memory reports), so no locking is required. Given a path it persists its
observations to a schema-versioned JSON file beside the performance model, so calibration survives a
restart; without one (or when the file is missing or corrupt) every worker start begins with cold keys
that fall back to the static seed until a peak is observed.
"""

from __future__ import annotations

import contextlib
import enum
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.resource_budget import platform_context_constant_mb


def plausible_activation_ceiling_mb(total_vram_mb: float | None) -> float | None:
    """Return the largest activation peak (MB) one process can plausibly reach on a card, or None.

    This is the card total minus the admission noise buffer, the margin admission always keeps free. A
    process only reads above it when it has overflowed (WDDM pages to host RAM rather than failing the
    allocation) or when it is caching other checkpoints alongside the job. Returns None when the card total
    is not known yet, so the caller applies no bound rather than a guessed one.
    """
    if total_vram_mb is None or total_vram_mb <= 0:
        return None
    return total_vram_mb - admission_noise_buffer_mb(total_vram_mb)


class MeasuredJobFootprint(Protocol):
    """The measured per-job device footprint this store can learn from.

    Structural rather than an import of ``hordelib.metrics.JobVramFootprint`` because the worker supports a
    range of backend versions and the footprint is a newer addition: a backend that does not report one
    simply never reaches this seam. Read-only properties, since the store only ever reads.
    """

    @property
    def peak_resident_weights_mb(self) -> float | None:
        """Largest the on-device weight set got at one instant during the run."""

    @property
    def peak_device_used_mb(self) -> float | None:
        """Device-wide used high-water during the run (weights plus transient activation)."""

    @property
    def resident_weights_after_job_mb(self) -> float | None:
        """Weights still on the device when the run finished."""

    @property
    def model_name(self) -> str | None:
        """The horde model identifier the run used."""

    @property
    def baseline(self) -> str | None:
        """The model's baseline family, when the backend resolved one."""

    @property
    def width(self) -> int | None:
        """Request width in pixels."""

    @property
    def height(self) -> int | None:
        """Request height in pixels."""

    @property
    def batch_size(self) -> int | None:
        """Request batch size."""

    @property
    def stage(self) -> str | None:
        """``whole_job`` for a monolithic run, ``sample_stage`` for a disaggregated sampler stage."""


FOOTPRINT_STORE_SCHEMA_VERSION = 2
"""Bumped when the file format changes, or when observations written by an older build can no longer be trusted.
A file with another version is discarded on read. Version 2 drops files written before observations were
bounds-checked (see :meth:`LearnedFootprintStore.observe_peak`): those can hold activation watermarks at the
size of the card and resident figures smaller than the checkpoint's weights, and since watermarks only ever
rise, the store would price from them forever."""

FOOTPRINT_STORE_FILENAME = "vram_footprints.json"
"""Name of the persisted store inside the worker's app-state directory (beside ``perf_model.json``)."""


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
    recent_mb: list[float] = Field(default_factory=list)
    """The most recent observations, oldest first, capped at :data:`_RECENT_WINDOW_SIZE`.

    The basis of :meth:`LearnedFootprintStore.measured_estimate_mb`. A bounded window rather than the
    all-time watermark because the measured estimate is allowed to fall: a driver update, a quantisation
    change, or a different checkpoint of the same baseline can genuinely lower what the hardware needs,
    and an all-time maximum would hold the old figure forever."""


_RECENT_WINDOW_SIZE = 20
"""How many recent observations back the measured estimate.

Wide enough that one anomalous job cannot dominate the window's maximum for long, narrow enough that a
genuine change in what a key costs works its way through within a few minutes of ordinary traffic."""

_MIN_OBSERVATIONS_FOR_MEASURED = 5
"""Observations a key needs before :meth:`LearnedFootprintStore.measured_estimate_mb` answers at all.

Below this the key keeps the raise-only :meth:`LearnedFootprintStore.estimate_mb` contract, so a single
unrepresentative run (a job that faulted mid-sample, a slot that never finished loading) can never talk a
consumer into planning below the static seed."""

_MEASURED_ESTIMATE_MARGIN = 1.10
"""The single conservatism knob applied to a measured estimate.

The measurements carry no safety margin of their own (a footprint is one observation of real hardware, not a
budget), and the two failure modes of an under-estimate are severe and asymmetric: on
Linux the OOM killer takes the process, on Windows/WDDM the driver pages to host RAM and the step rate
collapses. Ten percent over the recent watermark buys headroom for the run-to-run variation the window
already shows, without re-introducing the seed's multi-gigabyte over-statement. It is deliberately one
constant applied at one seam rather than a fudge folded into every seed."""

_PERSIST_EVERY_N_OBSERVATIONS = 10
"""Throttle disk writes: persist after this many new observations (plus an explicit save on shutdown).

Mirrors the performance model's cadence, for the same reason: the store is written from the control loop,
so a write per observation would put a disk sync on the hot path."""

_EWMA_ALPHA = 0.3
"""Weight given to the newest observation in the EWMA (``new = alpha*sample + (1-alpha)*prev``).

A moderate value tracks a genuine shift over a handful of jobs without letting a single spike dominate
the smoothed average. The estimate does not depend on it (it uses the watermark); the EWMA is retained
for calibration visibility only."""


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to ``path`` atomically: a temp file in the same directory, then ``os.replace``.

    Mirrors the performance model's persistence so a half-written store can never be read back; kept
    local rather than shared because ``resources`` must not depend on the scheduling package.
    """
    handle, temp_path_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    temp_path = Path(temp_path_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


class LearnedFootprintStore:
    """A store of measured VRAM peaks keyed by (baseline, resolution, platform, stage).

    Single-threaded use only (the parent control loop). Given a path it loads any previously persisted
    observations at construction and writes them back on a debounce, so a restart keeps its calibration;
    without one it is purely in memory and cold at every worker start.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        """Initialise the store, loading any persisted observations from ``path``.

        Args:
            path: Where to persist observations; ``None`` keeps the store purely in memory (tests).
        """
        self._observations: dict[FootprintKey, _FootprintObservation] = {}
        self._path = path
        self._observations_since_save = 0
        self._file_disabled = False
        self._load()

    def observe_peak(
        self,
        key: FootprintKey,
        peak_reserved_mb: float,
        *,
        plausible_min_mb: float | None = None,
        plausible_max_mb: float | None = None,
    ) -> None:
        """Fold one observed device-memory figure into the running statistics for ``key``.

        Updates both an EWMA (alpha ``0.3``, smoothed central tendency for observability) and a
        max-watermark (the estimate's undershoot-proof basis). A non-positive figure is ignored: a zero or
        negative reading carries no footprint information and would only pollute the average.

        A reading outside the caller's bounds is dropped. The bounds are facts the feeder knows and the store
        does not: a checkpoint cannot be resident in less than its own weight bytes, and a job cannot peak
        above what the card can give one process. Feeders read a per-process allocator counter, and that
        counter can include things the key does not describe: an idle slot with its weights staged in system
        RAM reports a nearly empty allocator against a resident key, and a process whose allocator grew to the
        size of the card reports that against a per-job activation key. Because the watermark only rises and
        is persisted, one such reading misprices its key for every later admission. Dropping it here keeps
        every entry describing the key it is filed under.

        Args:
            key (FootprintKey): The footprint identity to record under.
            peak_reserved_mb (float): The device memory (MB) observed for this key: the reserved peak for an
                activation stage, the steady reserved figure for a resident stage.
            plausible_min_mb (float | None): The smallest reading that can describe this key (a resident
                stage's weight bytes), or None for no floor.
            plausible_max_mb (float | None): The largest reading that can describe this key (the card total
                minus its noise buffer), or None for no ceiling.
        """
        if peak_reserved_mb <= 0:
            return
        if plausible_min_mb is not None and peak_reserved_mb < plausible_min_mb:
            logger.trace(
                f"Learned footprint {key.stage}/{key.checkpoint or key.model_baseline}: dropping "
                f"{peak_reserved_mb:.0f} MB below the plausible floor {plausible_min_mb:.0f} MB.",
            )
            return
        if plausible_max_mb is not None and peak_reserved_mb > plausible_max_mb:
            logger.trace(
                f"Learned footprint {key.stage}/{key.checkpoint or key.model_baseline}: dropping "
                f"{peak_reserved_mb:.0f} MB above the plausible ceiling {plausible_max_mb:.0f} MB.",
            )
            return

        existing = self._observations.get(key)
        if existing is None:
            self._observations[key] = _FootprintObservation(
                ewma_mb=peak_reserved_mb,
                watermark_mb=peak_reserved_mb,
                observation_count=1,
                recent_mb=[peak_reserved_mb],
            )
        else:
            self._observations[key] = _FootprintObservation(
                ewma_mb=(_EWMA_ALPHA * peak_reserved_mb) + ((1.0 - _EWMA_ALPHA) * existing.ewma_mb),
                watermark_mb=max(existing.watermark_mb, peak_reserved_mb),
                observation_count=existing.observation_count + 1,
                recent_mb=[*existing.recent_mb, peak_reserved_mb][-_RECENT_WINDOW_SIZE:],
            )

        self._observations_since_save += 1
        if self._observations_since_save >= _PERSIST_EVERY_N_OBSERVATIONS:
            self.save()

    def observe_job_footprint(
        self,
        footprint: MeasuredJobFootprint,
        *,
        baseline: str | None,
        platform: str,
        context_constant_mb: float = 0.0,
        resident_floor_mb: float | None = None,
    ) -> list[FootprintKey]:
        """Fold a child's measured per-job footprint into every key it can be attributed to.

        A footprint is one run measured against device truth, which is the only evidence the worker has that
        its seeds are wrong in either direction. One key comes out of it: the resident weight set
        (``peak_resident_weights_mb``, else ``resident_weights_after_job_mb``) is recorded under
        :attr:`FootprintStage.RESIDENT` for this checkpoint, with ``context_constant_mb`` added so it is in the
        whole-device terms the resident population is already kept in (the caller passes the same platform
        context charge the reserve uses; the backend measures weights alone). ``resident_floor_mb`` is the
        checkpoint's weight bytes. A resident figure below it is dropped: a run that block-swapped or offloaded
        most of the file measured only the part that fit, and a model too heavy to sit whole on the card must
        keep its seed price rather than the price of whatever fraction was loaded at one instant.

        The footprint's ``peak_device_used_mb`` is deliberately not folded into the activation keys. It is a
        device-wide high-water, so on a card holding other resident models it carries their weights too; fed
        into a per-job SAMPLE key that admission then prices from, it makes an ordinary SDXL preload look like
        it needs the whole card and defers it against room that is really there. The activation keys keep
        their process-local source (the child's own reserved-peak report).

        Anything the footprint cannot key (no baseline, a non-positive figure) is skipped rather than
        guessed. Returns the keys actually written, for the caller's logging.

        Args:
            footprint: The measured footprint the child reported.
            baseline: The model's baseline, resolved by the caller when the footprint does not carry one.
            platform: The host platform token the observation belongs to (``sys.platform``).
            context_constant_mb: The per-process CUDA-context charge to add to the resident figure.
            resident_floor_mb: The checkpoint's known weight bytes (MB); a resident figure below it is dropped.

        Returns:
            list[FootprintKey]: Every key this footprint was recorded under.
        """
        resolved_baseline = footprint.baseline or baseline
        if resolved_baseline is None:
            return []

        written: list[FootprintKey] = []

        resident_mb = footprint.peak_resident_weights_mb
        if resident_mb is None:
            resident_mb = footprint.resident_weights_after_job_mb
        if resident_mb is not None and resident_mb > 0 and footprint.model_name is not None:
            resident_key = FootprintKey(
                model_baseline=str(resolved_baseline),
                resolution_bucket=None,
                platform=platform,
                stage=FootprintStage.RESIDENT,
                checkpoint=footprint.model_name,
            )
            before = self.observation_count(resident_key)
            self.observe_peak(
                resident_key,
                float(resident_mb) + max(0.0, context_constant_mb),
                plausible_min_mb=resident_floor_mb,
            )
            if self.observation_count(resident_key) > before:
                written.append(resident_key)

        return written

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

    def measured_estimate_mb(self, key: FootprintKey) -> float | None:
        """Return what the measurements alone say ``key`` costs, or None while it is under-observed.

        This is the *bidirectional* counterpart to :meth:`estimate_mb`, and the two policies are kept apart
        on purpose. ``estimate_mb`` answers "never plan below what the hardware has demonstrated"; this
        answers "what does the hardware actually need", which a consumer uses to stop honouring a seed the
        measurements have disproved in the other direction. A Flux fp8 seed of 16.4 GB against a measured
        13.5 GB device-used median is the case in point: the seed reserved the whole card and bought
        nothing.

        The figure is the maximum of the recent window (:data:`_RECENT_WINDOW_SIZE` observations, so a
        genuine downward shift eventually lands while one high job still counts), times
        :data:`_MEASURED_ESTIMATE_MARGIN`, plus the platform's per-process context charge. The margin and
        the floor are the whole of the conservatism, held here rather than folded into the estimate's
        basis, so a reader can see and change what the worker is paying for safety. The floor is the same
        charge the reserve prices a CUDA context at: a consumer sizing a device against this figure must
        cover the context of the process holding it, and where an observation already carries its context
        it is additional margin in the direction this store exists to fail in.

        Returns None below :data:`_MIN_OBSERVATIONS_FOR_MEASURED` observations, which leaves the caller on
        the raise-only path with the static seed intact.

        Args:
            key (FootprintKey): The footprint identity to estimate for.

        Returns:
            float | None: The margined measured estimate (MB), or None for an under-observed key.
        """
        observation = self._observations.get(key)
        if observation is None or observation.observation_count < _MIN_OBSERVATIONS_FOR_MEASURED:
            return None
        if not observation.recent_mb:
            return None
        return (max(observation.recent_mb) * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(
            platform=key.platform,
        )

    def get_observation(self, key: FootprintKey) -> _FootprintObservation | None:
        """Return the raw running statistics for ``key`` (EWMA, watermark, count), or None if cold.

        Diagnostics/observability accessor: the decision surface is :meth:`estimate_mb`.
        """
        return self._observations.get(key)

    def observation_count(self, key: FootprintKey) -> int:
        """Return how many observations ``key`` carries (0 when cold), for logging a measured verdict."""
        observation = self._observations.get(key)
        return observation.observation_count if observation is not None else 0

    def __len__(self) -> int:
        """Return how many distinct keys have at least one observation."""
        return len(self._observations)

    # region persistence

    def save(self) -> None:
        """Persist the observations atomically. No-op without a path; never raises.

        A footprint-store write must never take the worker down, so a failed write degrades the store to
        memory-only for the rest of the run rather than retrying every debounce.
        """
        self._observations_since_save = 0
        if self._path is None or self._file_disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": FOOTPRINT_STORE_SCHEMA_VERSION,
                "observations": [
                    {"key": key.model_dump(mode="json"), "observation": observation.model_dump(mode="json")}
                    for key, observation in self._observations.items()
                ],
            }
            _atomic_write_text(self._path, json.dumps(payload))
        except OSError as write_error:
            logger.debug(
                f"Could not persist learned VRAM footprints to {self._path} ({write_error}); continuing in memory.",
            )
            self._file_disabled = True

    def _load(self) -> None:
        """Load previously persisted observations, tolerating a missing, unreadable, or corrupt file.

        A file the current build cannot parse is discarded rather than repaired: the store re-learns from
        live traffic within a handful of jobs, so nothing is worth failing a worker start over.
        """
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as read_error:
            logger.debug(f"Could not read learned VRAM footprints at {self._path} ({read_error}); starting cold.")
            return

        if not isinstance(raw, dict) or raw.get("schema_version") != FOOTPRINT_STORE_SCHEMA_VERSION:
            return
        entries = raw.get("observations")
        if not isinstance(entries, list):
            return

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                key = FootprintKey.model_validate(entry.get("key"))
                observation = _FootprintObservation.model_validate(entry.get("observation"))
            except ValueError:
                continue
            self._observations[key] = observation

    # endregion
