"""A LoRA-carrying mono-model queue must not starve the GPU while its head downloads auxiliaries.

When the head-of-queue job's model is resident on an inference lane that is fetching that job's LoRAs,
the lane reads as busy-but-not-sampling (``DOWNLOADING_AUX_MODEL``): it holds no denoise slot, yet the
card produces nothing until the download finishes. To keep the GPU fed, the scheduler line-skips a
different pending job onto an idle sibling lane via ``_select_line_skip_candidate``.

That selector historically refused any LoRA-carrying candidate outright, on the standing contract that
filling the idle lane with another download would only trade one blocking download for another. The
refusal is correct while the candidate's LoRAs are absent, but on a queue where every job carries a LoRA
(the ordinary illustrious/pony/SDXL production mix) it left both lanes idle for the whole download even
when the queued LoRAs were already on disk from an earlier job.

The refinement keeps the no-trade contract exactly: a LoRA candidate may line-skip only when every LoRA
it needs is already cached, so the skip starts sampling immediately and introduces no fresh download. The
cached set is learned as jobs complete their aux downloads and is not trusted while the worker purges
LoRAs on download or its LoRA disk is exhausted (either can evict a file after it was recorded), so those
modes fall back to the original refusal.

These tests pin both halves: an uncached LoRA candidate is still refused (contract intact), a cached one
feeds the idle lane (wedge resolved), and a completed download lets the next job needing the same LoRAs
skip. GREEN controls (no-LoRA backfill) bound the mechanism; the fence guards cover the purge and
disk-exhausted modes and all-or-nothing cache coverage.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl

# Model names mirror the production mono-model SDXL/illustrious/pony mix that exhibits the wedge.
_ILLUSTRIOUS = "WAI-NSFW-illustrious-SDXL"
_PONY = "Nova Furry Pony"
_ANI_PONY = "WAI-ANI-NSFW-PONYXL"
_CYBER = "CyberRealistic Pony"


def _lora_entries(names: Iterable[str]) -> list[LorasPayloadEntry]:
    """LoRA payload entries for the given references (all version-pinned, as production LoRAs usually are)."""
    return [LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=True) for name in names]


def _small_job(model: str, *, lora_names: Iterable[str] = ()) -> ImageGenerateJobPopResponse:
    """A small (512x512, 20-step) job, well under any performance-mode line-skip eMPS ceiling."""
    entries = _lora_entries(lora_names)
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=entries or None)


def _make_scheduler(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    reference: dict[str, object],
    max_concurrent: int = 1,
    high_performance_mode: bool = False,
    moderate_performance_mode: bool = False,
    purge_loras_on_download: bool = False,
    state: WorkerState | None = None,
) -> InferenceScheduler:
    """Build a scheduler over an explicit process map, model reference, and job tracker.

    The device-free provider returns an ample truthful reading so measured-truth admission never withholds
    a dispatch for a missing reading: these scenarios isolate the line-skip candidate filter, not the VRAM
    gate.
    """
    bridge_data = make_mock_bridge_data(
        max_threads=max_concurrent,
        high_performance_mode=high_performance_mode,
        moderate_performance_mode=moderate_performance_mode,
        purge_loras_on_download=purge_loras_on_download,
    )
    scheduler = InferenceScheduler(
        state=state if state is not None else WorkerState(),
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker,
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
            aux_download_deadline_for_dispatch=Mock(return_value=120.0),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(reference),
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=max(2, len(process_map)),
        lru=LRUCache(max(2, len(process_map))),
    )
    scheduler.set_device_free_mb_provider(lambda _device_index: 24000.0)
    return scheduler


async def _dispatch(scheduler: InferenceScheduler) -> object | None:
    """One real dispatch decision (the value ``start_inference`` acts on)."""
    return await scheduler.get_next_job_and_process()


class TestLoraAuxDownloadLaneStarvation:
    """The head's aux download must feed an idle sibling only from work that needs no fresh download."""

    async def test_nonlora_resident_backfill_feeds_idle_lane(self) -> None:
        """Control: a no-LoRA candidate whose model is resident on the idle lane is line-skipped in."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY)

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        result = await _dispatch(scheduler)
        assert result is not None
        assert result.next_job is backfill
        assert result.process_with_model.process_id == 1
        assert result.line_skip is not None and result.line_skip.displaced_job is head

    async def test_uncached_lora_backfill_is_refused(self) -> None:
        """Fence intact: a resident-base LoRA candidate whose LoRAs are not on disk is still refused.

        Skipping it would move the blocking download to the idle lane, trading one for another; the card
        holds rather than doing that.
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["uncached-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        result = await _dispatch(scheduler)
        assert result is None, (
            "an uncached LoRA candidate must not skip in and trade one blocking download for another"
        )

    async def test_cached_lora_backfill_feeds_idle_lane(self) -> None:
        """The fix: once the candidate's LoRAs are cached, it feeds the idle lane instead of the card idling."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        job_tracker.mark_job_loras_cached(backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        result = await _dispatch(scheduler)
        assert result is not None, (
            "a resident-base candidate whose LoRAs are already on disk should feed the idle lane"
        )
        assert result.next_job is backfill
        assert result.process_with_model.process_id == 1
        assert result.line_skip is not None and result.line_skip.displaced_job is head

    async def test_completed_download_enables_next_same_lora_job_to_skip(self) -> None:
        """The production win: job A downloading LoRA X lets a queued job B needing X skip once X lands.

        Before A's download completes, B is refused (X is not yet on disk). After the dispatcher records
        A's aux-download completion, B's LoRA is cached and B skips onto the idle sibling.
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["shared-lora"])
        follower = _small_job(_PONY, lora_names=["shared-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, follower)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        assert await _dispatch(scheduler) is None, "the follower's LoRA is not on disk yet"

        # What the message dispatcher does on the head's DOWNLOAD_AUX_COMPLETE.
        job_tracker.mark_job_loras_cached(head)

        result = await _dispatch(scheduler)
        assert result is not None, "with the shared LoRA now cached, the follower should skip onto the idle lane"
        assert result.next_job is follower
        assert result.process_with_model.process_id == 1

    async def test_mono_model_cached_lora_second_job_is_rescued(self) -> None:
        """A second same-model job with cached LoRAs skips onto a sibling copy of the model."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        second = _small_job(_ILLUSTRIOUS, lora_names=["cached-a", "cached-b"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_ILLUSTRIOUS, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, second)
        job_tracker.mark_job_loras_cached(second)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={_ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL)},
        )

        result = await _dispatch(scheduler)
        assert result is not None
        assert result.next_job is second
        assert result.process_with_model.process_id == 1

    async def test_full_production_queue_shape_wedges_on_nonresident(self) -> None:
        """Even with LoRAs cached, a queue whose base models are not resident on the idle lane cannot skip.

        Residency, not the LoRA gate, is the binding constraint here: nothing safe can run without first
        loading a model. This pins the lever that lives outside the line-skip filter (resolving a
        resident-base candidate before the head is dispatched).
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        queued_b = _small_job(_PONY, lora_names=["b-lora"])
        queued_c = _small_job(_ANI_PONY, lora_names=["c-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        for job in (head, queued_b, queued_c):
            await track_popped_job_async(job_tracker, job)
            job_tracker.mark_job_loras_cached(job)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
                _ANI_PONY: make_mock_model_reference_record(_ANI_PONY, baseline=_SDXL),
            },
        )

        result = await _dispatch(scheduler)
        assert result is None, "no queued job's base model is resident on the idle lane, so nothing can skip in"


class TestCachedLoraLineSkipFenceGuards:
    """The cached-LoRA skip is refused whenever the cache cannot be trusted, and only when fully covered."""

    async def test_partial_lora_coverage_is_refused(self) -> None:
        """A job needing two LoRAs, only one cached, is refused: the other would still block on a download."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        partial = _small_job(_PONY, lora_names=["cached-one", "still-missing"])
        cached_only = _small_job(_PONY, lora_names=["cached-one"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, partial)
        job_tracker.mark_job_loras_cached(cached_only)  # only "cached-one" is on disk

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        result = await _dispatch(scheduler)
        assert result is None, "a candidate with any uncached LoRA must be refused"

    async def test_cached_skip_refused_when_purge_loras_on_download(self) -> None:
        """With aggressive LoRA purging on, a recorded cache entry may already be gone: refuse the skip."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        job_tracker.mark_job_loras_cached(backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            purge_loras_on_download=True,
        )

        result = await _dispatch(scheduler)
        assert result is None
        # The untrustworthy cache is dropped so no stale entry survives into a later trusted moment.
        assert job_tracker.are_all_job_loras_cached(backfill) is False

    async def test_cached_skip_refused_when_lora_disk_exhausted(self) -> None:
        """A full LoRA disk evicts files under the worker: a recorded cache entry is no longer trustworthy."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        job_tracker.mark_job_loras_cached(backfill)

        state = WorkerState()
        state.lora_disk_exhausted = True
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            state=state,
        )

        result = await _dispatch(scheduler)
        assert result is None

    async def test_cached_lora_feeds_idle_lane_independent_of_performance_mode(self) -> None:
        """The cached-LoRA skip works the same under a widened performance-mode size ceiling."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        downloading = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        job_tracker.mark_job_loras_cached(backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            high_performance_mode=True,
        )

        result = await _dispatch(scheduler)
        assert result is not None
        assert result.next_job is backfill

    async def test_cap_reached_branch_feeds_cached_lora_backfill(self) -> None:
        """With another lane sampling (card at cap), a cached-LoRA backfill still reaches the idle lane.

        A distinct job samples on one lane (the card is at its ``max_threads`` cap), the head's model is
        resident on a second lane downloading its LoRAs, and a third lane is idle holding the backfill's
        base model. The cap-bypass path for an aux-download-blocked head reaches the same candidate filter,
        which now admits the cached-LoRA backfill.
        """
        sampling_job = _small_job(_CYBER)
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        sampling = make_mock_process_info(0, model_name=_CYBER, state=HordeProcessState.INFERENCE_STARTING)
        sampling.last_job_referenced = sampling_job
        downloading = make_mock_process_info(1, model_name=_ILLUSTRIOUS, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        downloading.last_job_referenced = head
        idle = make_mock_process_info(2, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, sampling_job)
        await mark_job_in_progress_async(job_tracker, sampling_job)
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        job_tracker.mark_job_loras_cached(backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: sampling, 1: downloading, 2: idle}),
            job_tracker=job_tracker,
            reference={
                _CYBER: make_mock_model_reference_record(_CYBER, baseline=_SDXL),
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            max_concurrent=1,
        )

        result = await _dispatch(scheduler)
        assert result is not None
        assert result.next_job is backfill
        assert result.process_with_model.process_id == 2
