"""A live session's job stream and learned footprints, fed to the dispatch world as its inputs.

The fixture under ``replay/`` is one session's job stream (``session_24gb_two_lane.jsonl``, one row per finished
job) and the learned footprint store the worker held (``footprints_24gb_two_lane.json``, every key's platform
rewritten to ``<platform>``), the two files ``horde-log replay-fixture`` writes from a support bundle, renamed.
This module turns them into the world's inputs, so a replay drives the real scheduler with that stream over that
prior:

- :func:`load_session_jobs` reads the stream as :class:`ReplayJob` rows, each with the pop response the world is
  offered.
- :func:`load_footprint_store` builds a store from a platform-rewritten copy of the footprint file.
- :func:`model_classes` and :func:`true_footprint` give the world one class per served model and the footprint
  each job really costs the card.
- :func:`sample_seconds_by_job`, :func:`decode_seconds_by_job` and the session medians state the stream's
  measured durations.
- :func:`pop_driver` offers the stream to the world on its own clock, gated by the room the popper would have.

Known mismatches between a replay and the live session:

- Loads run at one rate, :func:`calibrated_load_seconds_per_gb`, where a live load varies with page-cache state
  and checkpoint. Over the stream's 1,017 jobs that paid a load, ``ram_to_vram_s`` over the weights and support
  components the world moves has a median of 0.15 s/GB, about a sixth of the world's default 0.96 s/GB; per
  gigabyte of resident weights alone the medians run from 0.15 (SD 1.5, Anima) through 0.21 (SDXL) and 0.35
  (Flux) to 0.62 (Krea2). The replay passes the calibrated figure through the world's ``load_seconds_per_gb``.
- Context charges are pinned at the world's 1354 MB first context and 384 MB per further process.
- Safety checks and post-processing chains run whole ticks, at the session's median duration, and every chain
  holds the session's median measured chain peak; the world has no per-job chain figures.
- The replay has no crashes, lane replacements, LoRA download waits or submit latency. A row that faulted live
  is replayed as a job like any other.
- The admission noise buffer is the world's WDDM-conservative figure, where the session ran on Linux.
- Host RAM per checkpoint comes from the world's per-baseline profile, which sizes Z-Image, Krea2, Anima and Qwen
  as SD 1.5 does.
- A model the world's default pool already serves (Flux) resolves to the pool's class for host-RAM staging,
  while its device charge and load time come from :func:`true_footprint`.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry
from hordelib.feature_impact import get_baseline_burden
from pydantic import BaseModel, ConfigDict

from horde_worker_regen.analysis.replay_fixture import PLATFORM_PLACEHOLDER, ReplayJobRow, read_replay_job_rows
from horde_worker_regen.process_management.resources.resource_budget import predict_job_sampling_vram_mb
from horde_worker_regen.process_management.resources.vram_footprints import (
    POST_PROCESS_CHAIN_BASELINE,
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    job_class_key,
    sampling_footprint_key,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import _DispatchWorld, _ModelClass, _support_component_mb

REPLAY_FIXTURE_DIRECTORY = Path(__file__).parent / "replay"
"""Where the committed session fixture lives."""
SESSION_JOBS_PATH = REPLAY_FIXTURE_DIRECTORY / "session_24gb_two_lane.jsonl"
"""The job stream of one 191-minute session on a 24 GB card."""
SESSION_FOOTPRINTS_PATH = REPLAY_FIXTURE_DIRECTORY / "footprints_24gb_two_lane.json"
"""The learned footprint store the same worker held, with every key's platform replaced by a placeholder."""

_REPLAY_CLASS_LABEL_PREFIX = "replay_"
"""Keeps a replayed class's label apart from the world's own pool labels."""


@dataclass(frozen=True)
class ReplayJob:
    """Represents one finished job of the live session and the pop response the replay offers in its place."""

    offset_seconds: float
    """When the job was popped, in seconds from the session's start."""
    dispatch_offset_seconds: float | None
    """When the job was dispatched live, in seconds from the session's start."""
    finalized_offset_seconds: float | None
    """When the job was finalized live, in seconds from the session's start."""
    live_job_id: str
    model: str
    baseline: str
    width: int
    height: int
    steps: int
    batch: int
    sampler: str
    hires_fix: bool
    lora_count: int
    textual_inversion_count: int
    post_processing: tuple[str, ...]
    sampling_seconds: float | None
    decode_seconds: float | None
    encode_seconds: float | None
    ram_to_vram_seconds: float | None
    chain_seconds: float | None
    safety_seconds: float | None
    kudos: float | None
    peak_device_used_mb: float | None
    peak_resident_weights_mb: float | None
    vram_used_high_water_mb: float | None
    faulted: bool
    """Whether the job faulted live. The replay does not reproduce faults, so such a job runs like any other."""
    pop_response: ImageGenerateJobPopResponse
    """The job as the world is offered it. Its id is generated, so the per-job maps key on it."""


def _required[T](row: ReplayJobRow, column: str, value: T | None) -> T:
    """Return a column a replayed job cannot run without, raising when the row carries none.

    Raises:
        ValueError: The row has no value in ``column``.
    """
    if value is None:
        raise ValueError(f"fixture row {row.job_id} has no {column}, which a replayed job needs")
    return value


def _replay_job(row: ReplayJobRow) -> ReplayJob:
    """Create the replayed job for one fixture row, with the pop response carrying every request field it records.

    Raises:
        ValueError: The row lacks the model, baseline, resolution, steps or sampler a pop response needs.
    """
    model = _required(row, "model", row.model)
    width = _required(row, "width", row.width)
    height = _required(row, "height", row.height)
    steps = _required(row, "steps", row.steps)
    sampler = _required(row, "sampler", row.sampler)
    pop_response = make_job_pop_response(
        model,
        width=width,
        height=height,
        ddim_steps=steps,
        n_iter=row.batch,
        sampler_name=sampler,
        loras=[LorasPayloadEntry(name=f"replay-lora-{index}") for index in range(row.loras)] or None,
        tis=[TIPayloadEntry(name=f"replay-ti-{index}") for index in range(row.tis)] or None,
        post_processing=list(row.post_processing) or None,
        hires_fix=row.hires_fix,
    )
    return ReplayJob(
        offset_seconds=row.offset_s,
        dispatch_offset_seconds=row.dispatch_offset_s,
        finalized_offset_seconds=row.finalized_offset_s,
        live_job_id=row.job_id,
        model=model,
        baseline=_required(row, "baseline", row.baseline),
        width=width,
        height=height,
        steps=steps,
        batch=row.batch,
        sampler=sampler,
        hires_fix=row.hires_fix,
        lora_count=row.loras,
        textual_inversion_count=row.tis,
        post_processing=row.post_processing,
        sampling_seconds=row.sampling_s,
        decode_seconds=row.vae_decode_s,
        encode_seconds=row.clip_encode_s,
        ram_to_vram_seconds=row.ram_to_vram_s,
        chain_seconds=row.post_processing_chain_s,
        safety_seconds=row.safety_s,
        kudos=row.kudos,
        peak_device_used_mb=row.peak_device_used_mb,
        peak_resident_weights_mb=row.peak_resident_weights_mb,
        vram_used_high_water_mb=row.vram_used_high_water_mb,
        faulted=row.faulted,
        pop_response=pop_response,
    )


def load_session_jobs(path: Path) -> list[ReplayJob]:
    """Parse the session fixture at ``path`` into jobs, in pop order.

    Args:
        path: A job stream written by ``horde-log replay-fixture``.

    Returns:
        list[ReplayJob]: One job per row, sorted by pop offset.
    """
    jobs = [_replay_job(row) for row in read_replay_job_rows(path)]
    jobs.sort(key=lambda job: job.offset_seconds)
    return jobs


def load_footprint_store(path: Path, *, scratch_directory: Path) -> LearnedFootprintStore:
    """Build a learned footprint store from the fixture at ``path``, keyed for this host's platform.

    The fixture's keys carry :data:`PLATFORM_PLACEHOLDER`, and the store only prices keys for ``sys.platform``, so
    the placeholder is rewritten into a copy under ``scratch_directory`` and the store is built from the copy. A
    store built with a path writes back to it as it learns, which is why the committed fixture is never the path.

    Args:
        path: The platform-neutral footprint fixture.
        scratch_directory: A directory the run owns, where the rewritten copy is written and later saved over.

    Returns:
        LearnedFootprintStore: The store, loaded with every observation of the fixture.
    """
    scratch_directory.mkdir(parents=True, exist_ok=True)
    copy_path = scratch_directory / path.name
    copy_path.write_text(_rewritten_fixture_text(path), encoding="utf-8")
    return LearnedFootprintStore(path=copy_path)


class _FixtureStoreEntry(BaseModel):
    """Represents one persisted observation's key in the footprint fixture."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    key: FootprintKey


class _FixtureStoreFile(BaseModel):
    """Represents the footprint fixture's observation list, read for its keys alone."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    observations: list[_FixtureStoreEntry]


def load_footprint_keys(path: Path) -> tuple[FootprintKey, ...]:
    """Return every key the footprint fixture at ``path`` holds, keyed for this host's platform.

    The store exposes lookups by key and no listing, so the keys a replay reads by stage (a baseline's resident
    checkpoints, the post-processing chains) are taken from the fixture itself.
    """
    stored = _FixtureStoreFile.model_validate_json(_rewritten_fixture_text(path))
    return tuple(entry.key for entry in stored.observations)


def _rewritten_fixture_text(path: Path) -> str:
    """Return the footprint fixture's text with :data:`PLATFORM_PLACEHOLDER` replaced by ``sys.platform``."""
    return path.read_text(encoding="utf-8").replace(f'"{PLATFORM_PLACEHOLDER}"', f'"{sys.platform}"')


def _resident_key(model: str, baseline: str) -> FootprintKey:
    """Return the at-rest key a checkpoint's resident weights are learned under."""
    return FootprintKey(
        model_baseline=baseline,
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.RESIDENT,
        checkpoint=model,
    )


def _priced_mb(store: LearnedFootprintStore, key: FootprintKey, class_key: FootprintKey | None = None) -> float | None:
    """Return the lower of a key's watermark and its measured price net of context, or None for a cold key.

    The watermark is the raise-only figure and the measurement the window's margined price, so the lower of the
    two is what a parent that trusts its measurement charges. A job-class key prices from its own window once that
    is observed, else from the pooled key's, as the parent prices a job.
    """
    observation = store.get_observation(key)
    if observation is None:
        return None
    if class_key is None:
        measured_mb = store.measured_estimate_net_of_context_mb(key)
    else:
        measured_mb = store.measured_job_estimate_net_of_context_mb(key, class_key)
    if measured_mb is None:
        return observation.watermark_mb
    return min(observation.watermark_mb, measured_mb)


def _resident_weights_mb(
    store: LearnedFootprintStore,
    keys: Sequence[FootprintKey],
    *,
    model: str,
    baseline: str,
) -> float:
    """Return a checkpoint's resident weights: its own key, else its baseline's other checkpoints, else hordelib's.

    The store has no baseline-wide resident key, so a checkpoint the worker never learned takes the median of
    the learned checkpoints of its baseline, and a baseline with none takes hordelib's weight seed.
    """
    own_mb = _priced_mb(store, _resident_key(model, baseline))
    if own_mb is not None:
        return own_mb
    sibling_figures = [
        priced_mb
        for sibling_key in keys
        if sibling_key.stage is FootprintStage.RESIDENT and sibling_key.model_baseline == baseline
        if (priced_mb := _priced_mb(store, sibling_key)) is not None
    ]
    if sibling_figures:
        return statistics.median(sibling_figures)
    burden = get_baseline_burden(baseline, model)
    if burden is None:
        raise ValueError(f"no resident figure for {model} ({baseline}): no learned key and no hordelib burden")
    return float(burden.vram_weights_mb)


def _class_label(model: str) -> str:
    """Return a label for a replayed model, unique among the world's classes."""
    return _REPLAY_CLASS_LABEL_PREFIX + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def model_classes(
    jobs: Sequence[ReplayJob],
    store: LearnedFootprintStore,
    keys: Sequence[FootprintKey],
) -> list[_ModelClass]:
    """Return one world class per distinct model in the stream, weighted by its learned resident footprint.

    Args:
        jobs: The replayed stream.
        store: The learned store the weights are read from, unaltered by any run.
        keys: The store's keys, from :func:`load_footprint_keys`.

    Returns:
        list[_ModelClass]: The classes, in order of first appearance.
    """
    classes: dict[str, _ModelClass] = {}
    for job in jobs:
        if job.model in classes:
            continue
        classes[job.model] = _ModelClass(
            _class_label(job.model),
            job.model,
            KNOWN_IMAGE_GENERATION_BASELINE(job.baseline),
            _resident_weights_mb(store, keys, model=job.model, baseline=job.baseline),
        )
    return list(classes.values())


def true_footprint(
    store: LearnedFootprintStore,
    classes: Sequence[_ModelClass],
) -> Callable[[ImageGenerateJobPopResponse], tuple[float, float]]:
    """Return the world's ``true_footprint_mb`` hook, stating what each job cost the live card.

    The resident weights are the class's figure (its checkpoint's resident key). The sampling transient is the
    job's sample key, matched on baseline, resolution band, batch and hires fix as the worker keys it, priced as
    the lower of watermark and measurement, less those weights. A job whose band was never learned takes the
    static sampling predictor in place of the measurement.

    The store must be one no run feeds. A replay with ``learned_footprints`` teaches its own store from the
    world's reports, and a truth read from that store would follow what the world charged.

    Args:
        store: A learned store loaded from the fixture and kept apart from the world's.
        classes: The replayed classes, from :func:`model_classes`.

    Returns:
        Callable[[ImageGenerateJobPopResponse], tuple[float, float]]: The hook, as (weights MB, transient MB).
    """
    classes_by_name = {model_class.name: model_class for model_class in classes}
    figures: dict[tuple[str, FootprintKey | None], tuple[float, float]] = {}

    def true_footprint_mb(job: ImageGenerateJobPopResponse) -> tuple[float, float]:
        model_class = classes_by_name[job.model or ""]
        baseline = str(model_class.baseline)
        pooled_key = sampling_footprint_key(job, baseline, stage=FootprintStage.SAMPLE)
        class_key = job_class_key(pooled_key, job) if pooled_key is not None else None
        cache_key = (model_class.name, class_key)
        cached = figures.get(cache_key)
        if cached is not None:
            return cached
        sample_mb = _priced_mb(store, pooled_key, class_key) if pooled_key is not None else None
        if sample_mb is None:
            sample_mb = predict_job_sampling_vram_mb(job, baseline) or model_class.weights_mb
        figure = (model_class.weights_mb, max(0.0, sample_mb - model_class.weights_mb))
        figures[cache_key] = figure
        return figure

    return true_footprint_mb


def sample_seconds_by_job(jobs: Sequence[ReplayJob]) -> dict[str, float]:
    """Return each job's live sampling seconds, keyed by its replayed id. A job with none is left to the rates."""
    return {
        str(job.pop_response.id_): job.sampling_seconds
        for job in jobs
        if job.sampling_seconds is not None and job.pop_response.id_ is not None
    }


def decode_seconds_by_job(jobs: Sequence[ReplayJob]) -> dict[str, float]:
    """Return each job's live VAE decode seconds, keyed by its replayed id. A job with none is left to the rates."""
    return {
        str(job.pop_response.id_): job.decode_seconds
        for job in jobs
        if job.decode_seconds is not None and job.pop_response.id_ is not None
    }


def calibrated_load_seconds_per_gb(jobs: Sequence[ReplayJob], classes: Sequence[_ModelClass]) -> float:
    """Return the load rate that makes the world's load time the stream's median ``ram_to_vram_s``.

    The world charges a cold monolithic load its weights plus its support components at one rate, so each job
    that paid a load gives the rate its own seconds imply over that same figure, and the median is taken. Jobs
    with no load (their model was already on the card) say nothing about the rate.

    Raises:
        ValueError: The stream has no job that paid a load.
    """
    classes_by_name = {model_class.name: model_class for model_class in classes}
    rates: list[float] = []
    for job in jobs:
        if job.ram_to_vram_seconds is None or job.ram_to_vram_seconds <= 0.0:
            continue
        model_class = classes_by_name[job.model]
        moved_gb = (model_class.weights_mb + _support_component_mb(model_class)) / 1024.0
        rates.append(job.ram_to_vram_seconds / moved_gb)
    if not rates:
        raise ValueError("no job in the stream paid a host-to-device load")
    return statistics.median(rates)


def median_seconds_in_ticks(seconds: Sequence[float], *, tick_seconds: float) -> int:
    """Return the median of ``seconds`` as a whole number of ticks, at least one."""
    return max(1, math.ceil(statistics.median(seconds) / tick_seconds))


def chain_seconds(jobs: Sequence[ReplayJob]) -> list[float]:
    """Return every measured post-processing chain duration in the stream."""
    return [job.chain_seconds for job in jobs if job.chain_seconds is not None]


def safety_seconds(jobs: Sequence[ReplayJob]) -> list[float]:
    """Return every measured safety-check duration in the stream."""
    return [job.safety_seconds for job in jobs if job.safety_seconds is not None]


def median_chain_peak_mb(store: LearnedFootprintStore, keys: Sequence[FootprintKey]) -> float:
    """Return the median measured chain peak over every post-processing key with a measured price.

    Raises:
        ValueError: No chain key has enough observations to be priced.
    """
    figures = [
        measured_mb
        for key in keys
        if key.stage is FootprintStage.POST_PROCESS
        and key.model_baseline == POST_PROCESS_CHAIN_BASELINE
        and (measured_mb := store.measured_estimate_net_of_context_mb(key)) is not None
    ]
    if not figures:
        raise ValueError("the store prices no post-processing chain")
    return statistics.median(figures)


def safety_check_growth_mb(store: LearnedFootprintStore) -> float:
    """Return how far a safety check's measured footprint rises above safety's measured at-rest footprint.

    Raises:
        ValueError: The store prices either footprint from too few readings.
    """
    at_rest_mb = store.measured_estimate_net_of_context_mb(_safety_key(FootprintStage.SAFETY))
    checking_mb = store.measured_estimate_net_of_context_mb(_safety_key(FootprintStage.SAFETY_CHECK))
    if at_rest_mb is None or checking_mb is None:
        raise ValueError("the store does not price the safety process at rest and while checking")
    return max(0.0, checking_mb - at_rest_mb)


def _safety_key(stage: FootprintStage) -> FootprintKey:
    """Return the safety process's key for ``stage``."""
    return FootprintKey(
        model_baseline=SAFETY_PROCESS_BASELINE,
        resolution_bucket=None,
        platform=sys.platform,
        stage=stage,
    )


@dataclass(frozen=True)
class PopDriverRecord:
    """Represents what the pop driver offered the world over a run."""

    offered: int
    """Jobs the world took."""
    remaining: int
    """Jobs of the stream never offered, because their offset had not passed or the queue had no room."""
    refused_offers: int
    """Offers the world refused (its claim narrowed the offer, or host RAM held pops), each retried next tick."""


async def pop_driver(
    world: _DispatchWorld,
    jobs: Sequence[ReplayJob],
    *,
    queue_size: int,
    start_offset_seconds: float,
    seconds: float,
) -> PopDriverRecord:
    """Step ``world`` for ``seconds``, offering the stream's jobs as their offsets pass and the queue has room.

    Each tick, before the world steps, the next job is offered while its offset (from ``start_offset_seconds``)
    has passed on the world clock and the tracker holds fewer than ``queue_size`` jobs pending or in progress. A
    refused offer stops the tick's offers and is retried next tick, as a live pop under the same narrowing would
    be taken again on the next pop cycle.

    Args:
        world: A world built from this module's inputs.
        jobs: The stream to offer, in pop order.
        queue_size: How many jobs the popper keeps pending or in progress, its ``queue_size`` plus
            ``max_threads`` budget on one card.
        start_offset_seconds: The stream offset the run's first tick stands for.
        seconds: How much world time to run.

    Returns:
        PopDriverRecord: What was offered and refused.
    """
    began_at = world.now
    next_index = 0
    refused_offers = 0
    for _ in range(math.ceil(seconds / world.tick_seconds)):
        elapsed = world.now - began_at
        while (
            next_index < len(jobs)
            and jobs[next_index].offset_seconds - start_offset_seconds <= elapsed
            and len(world.job_tracker.jobs_pending_inference) < queue_size
        ):
            if not await world.offer_job(jobs[next_index].pop_response):
                refused_offers += 1
                break
            next_index += 1
        await world.step()
    return PopDriverRecord(offered=next_index, remaining=len(jobs) - next_index, refused_offers=refused_offers)
