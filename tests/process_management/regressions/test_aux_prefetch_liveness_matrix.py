"""Bounded-idleness liveness for the pop-time auxiliary (LoRA/TI) prefetch pipeline, over a variation matrix.

A job whose auxiliary files are not yet on disk is invisible to both dispatch and preload: it holds no
sampling lane and no VRAM reservation, and nothing prices around it while the dedicated download process
places its files. This module proves the campaign's core liveness claims over a parametrized matrix rather
than re-checking any single mechanism, driving the real scheduler, the real pop-time prefetch coordinator,
and the real job tracker together across scheduling cycles with a hand-advanced clock and a scripted
download process.

Three contracts are held:

- Bounded idleness: while a pending job waits on auxiliary prefetch, any dispatchable work that physically
  fits reaches sampling within a bounded number of scheduling cycles, and once the waiting job's prefetch
  completes that job itself reaches sampling within a bound (no permanent shadow). When the prefetch fails or
  its deadline expires the job faults and the queue keeps draining, its sibling work unaffected.
- Admit implies dispatchable: a job the scheduler admits to a lane is one it actually dispatches to sampling
  in the same act. The scheduler never selects a job it cannot dispatch while another job's outstanding
  demand is priced in, so no job is admitted into a lane it can never be released from.
- Cold-path no-reservation: a cold-model job with uncached LoRAs holds neither a lane nor a VRAM reservation
  while its files download; a fitting sibling samples throughout that window; and when the prefetch completes
  the head is preloaded and sampled.

The capacity axis is expressed through the enforced simultaneous-sampling limit (the concurrency cap): a
two-lane pool with ample VRAM lets both jobs reach sampling, while a single-lane pool serializes them so the
sibling samples first and the head samples once the lane frees. VRAM-arbiter eviction fidelity (making room
for a head that does not fit alongside an idle resident) is proven separately by the head-of-queue make-room
tests in the scheduling suite; here the concurrency cap is the liveness-relevant capacity knob.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
    HordeControlFlag,
    HordeImageResult,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
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
    track_popped_job_async,
)

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl

# A resident sibling (or a prepared head) reaches sampling within this many scheduling cycles of becoming
# dispatchable. The driver frees a lane at each cycle start, then preloads, then dispatches; a resident job
# needs one cycle, a cold job needs a preload-then-materialize cycle, so a small constant covers both.
_SAMPLING_BOUND = 4

# A hand-advanced clock ticks one second per scheduling cycle, so a per-job download deadline expressed in
# seconds maps directly onto a cycle count.
_CYCLE_SECONDS = 1.0

# An ample per-card free-VRAM reading, sized to fit two concurrent SDXL sampling peaks plus their
# reservations, so the measured-truth VRAM gate never withholds a dispatch these scenarios are not about.
# The capacity axis is carried by the concurrency cap, not a starved card: head-of-queue VRAM protection and
# its eviction-based resolution are a separate subsystem this harness does not model (it never evicts), so a
# tight two-peak regime would misattribute that hold to the auxiliary-preparation gate under test.
_AMPLE_FREE_MB = 48_000.0


def _loras(*names: str) -> list[LorasPayloadEntry]:
    """Version-pinned LoRA payload entries for the given references."""
    return [LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=True) for name in names]


def _small_job(model: str, *, lora_names: tuple[str, ...] = ()) -> ImageGenerateJobPopResponse:
    """A small (512x512, 20-step) job, light enough that no size gate interferes with these scenarios."""
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=_loras(*lora_names) or None)


@dataclass
class _PrefetchScript:
    """How a job's auxiliary prefetch resolves, from the perspective of the scripted download process.

    Attributes:
        on_disk: The files are already cached at pop, so the coordinator short-circuits with no request and
            the job is prepared immediately (its completion cycle is zero).
        resolve_cycle: The scheduling cycle at which a success (or failure) outcome is delivered for a job
            whose files were not cached at pop. Ignored when ``on_disk`` is set or when the job is left to its
            download deadline.
        succeeds: Whether the delivered outcome reports the files placed (True) or a download failure (False).
            A reported failure is terminal and serves the job without the file (it becomes dispatchable at
            ``resolve_cycle``), so only a deadline case leaves the job undispatchable.
        by_deadline: No outcome is ever delivered; the job is left to its per-job download deadline, which
            faults it from the periodic deadline scan.
    """

    on_disk: bool = False
    resolve_cycle: int | None = None
    succeeds: bool = True
    by_deadline: bool = False

    @property
    def completion_cycle(self) -> int | None:
        """The cycle by which the job becomes dispatchable, or ``None`` if it never does (the deadline case).

        A reported download failure is served without the file rather than faulted, so a failed prefetch makes
        the job dispatchable at the cycle its outcome lands, exactly as a success does. Only the deadline case,
        where no outcome is ever delivered, leaves the job to fault undispatchable.
        """
        if self.on_disk:
            return 0
        if self.by_deadline:
            return None
        return self.resolve_cycle


class _AuxLivenessWorld:
    """Drives the real scheduler, prefetch coordinator, and job tracker across scheduling cycles.

    Each cycle mirrors the control loop's ordering at the grain these liveness claims live at: a lane freed by
    a completed job becomes available, the scripted download process delivers any due prefetch outcome (or a
    deadline faults an unresolved job), one preload pass may bring a cold model resident, and dispatch feeds
    sampling. A model preloaded in one cycle materialises (becomes resident) at the start of the next, modelling
    the child's load latency. Every dispatch is checked for the admit-implies-dispatchable invariant.
    """

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        model_map: HordeModelMap,
        reference: dict[str, object],
        max_threads: int,
        download_timeout: float,
    ) -> None:
        self.now = 1_000.0
        self.cycle = 0
        self._process_map = process_map
        self._model_map = model_map
        # The tracker shares the world clock so a time-scoped auxiliary skip verdict (a served-without-file head)
        # is evaluated against the same synthetic time the coordinator stamped it with, not the wall clock.
        self._job_tracker = JobTracker(clock=lambda: self.now)
        bridge_data = make_mock_bridge_data(max_threads=max_threads, download_timeout=int(download_timeout))
        self._scheduler = InferenceScheduler(
            state=WorkerState(),
            process_map=process_map,
            horde_model_map=model_map,
            job_tracker=self._job_tracker,
            process_lifecycle=Mock(
                get_processes_with_model_for_queued_job=Mock(return_value=[]),
                is_model_load_quarantined=Mock(return_value=False),
            ),
            runtime_config=make_test_runtime_config(bridge_data=bridge_data),
            model_metadata=make_test_model_metadata(reference),
            max_concurrent_inference_processes=max_threads,
            max_inference_processes=max(2, len(process_map)),
            lru=LRUCache(max(2, len(process_map))),
        )
        self._scheduler.set_device_free_mb_provider(lambda _device_index: _AMPLE_FREE_MB)
        self._coordinator = AuxPrefetchCoordinator(
            job_tracker=self._job_tracker,
            state=WorkerState(),
            prefetch_sender=lambda _entries, _pins: None,
            download_timeout_provider=lambda: download_timeout,
            pin_sender=lambda _pins: None,
            clock=lambda: self.now,
        )
        # Per-job scripted outcome delivery and readiness bookkeeping.
        self._scripts: dict[GenerationID, _PrefetchScript] = {}
        self._delivered: set[GenerationID] = set()
        # Observability: the first cycle each job entered sampling, and every job that ever sampled.
        self.first_sampled: dict[GenerationID, int] = {}
        self._in_progress_since: dict[GenerationID, int] = {}
        self._lane_of: dict[GenerationID, int] = {}

    async def pop(self, job: ImageGenerateJobPopResponse, *, script: _PrefetchScript) -> None:
        """Record a popped job and drive its pop-time prefetch exactly as the control loop would.

        For an ``on_disk`` script the referenced files are seeded into the session cache before the coordinator
        sees the job, so it short-circuits (no request) and the job is prepared at pop.
        """
        assert job.id_ is not None
        if script.on_disk:
            for lora in job.payload.loras or []:
                self._job_tracker.mark_aux_prefetched(lora.name, is_version=bool(lora.is_version), is_ti=False)
            for ti in job.payload.tis or []:
                self._job_tracker.mark_aux_prefetched(ti.name, is_version=False, is_ti=True)
        await track_popped_job_async(self._job_tracker, job)
        self._scripts[job.id_] = script
        self._coordinator.on_job_popped(job)

    def _materialise_preloads(self) -> None:
        """Bring any model preloaded in the previous cycle resident, modelling the child's load completing."""
        for name, info in list(self._model_map.root.items()):
            if info.horde_model_load_state == ModelLoadState.LOADING and info.process_id is not None:
                process = self._process_by_id(info.process_id)
                if process is None:
                    continue
                process.last_process_state = HordeProcessState.PRELOADED_MODEL
                process.loaded_horde_model_name = name
                self._model_map.update_entry(name, load_state=ModelLoadState.LOADED_IN_RAM, process_id=info.process_id)

    def _mark_loading_lanes_busy(self) -> None:
        """Keep a lane that a preload just started off dispatch until its model materialises next cycle.

        A real child moves to ``PRELOADING_MODEL`` the moment it is told to load and only becomes dispatchable
        once it reports the model ready. The mock lane never transitions on its own, so without this it would
        look idle-with-model and wrongly accept a dispatch of a model whose weights are not actually resident.
        """
        for info in self._model_map.root.values():
            if info.horde_model_load_state == ModelLoadState.LOADING and info.process_id is not None:
                process = self._process_by_id(info.process_id)
                if process is not None and process.last_process_state != HordeProcessState.PRELOADING_MODEL:
                    process.last_process_state = HordeProcessState.PRELOADING_MODEL

    def _process_by_id(self, process_id: int):  # noqa: ANN202 - HordeProcessInfo, kept local to avoid an import
        for process in self._process_map.values():
            if process.process_id == process_id:
                return process
        return None

    def _deliver_due_outcomes(self) -> None:
        """Deliver any scripted success/failure outcome whose cycle has arrived to the coordinator."""
        for tracked in list(self._job_tracker.tracked_jobs()):
            job = tracked.sdk_api_job_info
            job_id = job.id_
            if job_id is None or job_id in self._delivered:
                continue
            script = self._scripts.get(job_id)
            if script is None or script.on_disk or script.by_deadline or script.resolve_cycle is None:
                continue
            if self.cycle < script.resolve_cycle:
                continue
            self._delivered.add(job_id)
            outcomes = [
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name=lora.name,
                    is_version=bool(lora.is_version),
                    ok=script.succeeds,
                    retryable=False,
                    requesting_job_ids=[job_id],
                )
                for lora in job.payload.loras or []
            ]
            self._coordinator.on_prefetch_result(
                HordeAuxPrefetchResultMessage(
                    process_id=9_000,
                    process_launch_identifier=1,
                    info="scripted prefetch outcome",
                    outcomes=outcomes,
                ),
            )

    async def _complete_finished_samplers(self) -> None:
        """Move each job that sampled on an earlier cycle out of progress, freeing its lane for fresh work."""
        for job in list(self._job_tracker.jobs_in_progress):
            job_id = job.id_
            if job_id is None or self._in_progress_since.get(job_id, self.cycle) >= self.cycle:
                continue
            job_info = HordeJobInfo(
                sdk_api_job_info=job,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
                state=GENERATION_STATE.ok,
                censored=False,
                time_popped=time.time(),
            )
            await self._job_tracker.queue_for_safety(job_info)
            self._in_progress_since.pop(job_id, None)
            # A real child returns its lane to a resident, accepting state once the result lands; mirror that so
            # a shared lane can take the next same-model job rather than staying stuck mid-sample.
            lane_id = self._lane_of.pop(job_id, None)
            if lane_id is not None:
                lane = self._process_by_id(lane_id)
                if lane is not None and lane.loaded_horde_model_name is not None:
                    lane.last_process_state = HordeProcessState.PRELOADED_MODEL

    async def _dispatch_until_full(self, max_threads: int) -> None:
        """Dispatch pending work onto free lanes, asserting admit-implies-dispatchable on every attempt."""
        for _ in range(max_threads):
            before = {j.id_ for j in self._job_tracker.jobs_in_progress}
            started = await self._scheduler.start_inference()
            newly_admitted = [job for job in self._job_tracker.jobs_in_progress if job.id_ not in before]

            if not started:
                assert newly_admitted == [], "start_inference returned False yet a job entered progress"
                break

            # Admit implies dispatchable: exactly one job was admitted, and a lane holding that job's model
            # received the dispatch, so selection and dispatch agree rather than admitting a job into a lane
            # that never runs it. Entering progress happens only through the dispatch path, so a lone new
            # in-progress job is itself the dispatch; the lane check confirms it landed on a real lane.
            assert len(newly_admitted) == 1, "a successful dispatch must admit exactly one job"
            admitted = newly_admitted[0]
            assert admitted.id_ is not None
            dispatched = [
                process.process_id
                for process in self._process_map.values()
                if process.loaded_horde_model_name == admitted.model
                and process.last_control_flag == HordeControlFlag.START_INFERENCE
            ]
            assert dispatched, "an admitted job must have been dispatched onto a lane holding its model"
            self._in_progress_since[admitted.id_] = self.cycle
            self._lane_of[admitted.id_] = dispatched[0]
            self.first_sampled.setdefault(admitted.id_, self.cycle)

    async def step(self) -> None:
        """Advance one scheduling cycle."""
        self.cycle += 1
        self.now += _CYCLE_SECONDS
        self._materialise_preloads()
        await self._complete_finished_samplers()
        self._deliver_due_outcomes()
        self._coordinator.scan_deadlines()
        self._coordinator.reconcile_and_refresh_pins()
        self._scheduler.preload_models()
        self._mark_loading_lanes_busy()
        max_threads = self._runtime_max_threads()
        await self._dispatch_until_full(max_threads)

    def _runtime_max_threads(self) -> int:
        return int(self._scheduler._runtime_config.bridge_data.max_threads)

    async def run(self, cycles: int) -> None:
        """Advance ``cycles`` scheduling cycles."""
        for _ in range(cycles):
            await self.step()

    def stage(self, job: ImageGenerateJobPopResponse) -> JobStage | None:
        assert job.id_ is not None
        return self._job_tracker.get_stage(job.id_)

    def sampled(self, job: ImageGenerateJobPopResponse) -> bool:
        return job.id_ in self.first_sampled

    def first_sampled_cycle(self, job: ImageGenerateJobPopResponse) -> int | None:
        return self.first_sampled.get(job.id_) if job.id_ is not None else None

    @property
    def scheduler(self) -> InferenceScheduler:
        return self._scheduler

    @property
    def job_tracker(self) -> JobTracker:
        return self._job_tracker


# --- The variation matrix -------------------------------------------------------------------------------- #


@dataclass
class _Cell:
    """One point in the variation matrix.

    Attributes:
        id: A stable identifier for the parametrized case.
        head_cold: Whether the head's base model must be preloaded (True) or is already resident (False).
        head_script: How the head's auxiliary prefetch resolves.
        sibling_kind: The sibling's shape: ``none``, ``non_lora``, ``lora_prepared`` (files cached at pop),
            or ``lora_slow`` (an uncached LoRA that resolves during the run).
        sibling_same_model: Whether the sibling shares the head's base model.
        sibling_resolve_cycle: For a ``lora_slow`` sibling, the cycle its prefetch completes.
        lanes: The concurrency cap and the pool size, so ``1`` is the only-one-fits/single-lane capacity and
            ``2`` is the both-fit/two-lane capacity.
        run_cycles: How many scheduling cycles to drive before asserting.
    """

    id: str
    head_cold: bool
    head_script: _PrefetchScript
    sibling_kind: str
    sibling_same_model: bool
    lanes: int
    run_cycles: int = 12
    sibling_resolve_cycle: int = 2


_HEAD_MODEL = "head-model"
_SIBLING_MODEL = "sibling-model"


_MATRIX: tuple[_Cell, ...] = (
    _Cell(
        id="warm_head_ondisk_loras_nonlora_sibling",
        head_cold=False,
        head_script=_PrefetchScript(on_disk=True),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
    _Cell(
        id="warm_head_slow_loras_nonlora_sibling",
        head_cold=False,
        head_script=_PrefetchScript(resolve_cycle=3),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
    _Cell(
        id="warm_head_slow_loras_nonlora_same_model_single_lane",
        head_cold=False,
        head_script=_PrefetchScript(resolve_cycle=3),
        sibling_kind="non_lora",
        sibling_same_model=True,
        lanes=1,
    ),
    _Cell(
        id="cold_head_slow_loras_resident_nonlora_sibling",
        head_cold=True,
        head_script=_PrefetchScript(resolve_cycle=3),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
    _Cell(
        id="cold_head_ondisk_loras_resident_nonlora_sibling",
        head_cold=True,
        head_script=_PrefetchScript(on_disk=True),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
    _Cell(
        id="warm_head_slow_loras_prepared_lora_sibling_same_model",
        head_cold=False,
        head_script=_PrefetchScript(resolve_cycle=3),
        sibling_kind="lora_prepared",
        sibling_same_model=True,
        lanes=2,
    ),
    _Cell(
        id="warm_head_slow_loras_unprepared_lora_sibling",
        head_cold=False,
        head_script=_PrefetchScript(resolve_cycle=4),
        sibling_kind="lora_slow",
        sibling_same_model=False,
        sibling_resolve_cycle=2,
        lanes=2,
    ),
    _Cell(
        id="warm_head_failed_loras_nonlora_sibling",
        head_cold=False,
        head_script=_PrefetchScript(resolve_cycle=3, succeeds=False),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
    _Cell(
        id="warm_head_deadline_loras_nonlora_sibling",
        head_cold=False,
        head_script=_PrefetchScript(by_deadline=True),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=2,
    ),
)


async def _build_world(
    cell: _Cell,
) -> tuple[_AuxLivenessWorld, ImageGenerateJobPopResponse, ImageGenerateJobPopResponse | None]:
    """Construct a world, its process/model layout, and the head (and optional sibling) jobs for a cell."""
    sibling_model = _HEAD_MODEL if cell.sibling_same_model else _SIBLING_MODEL

    reference: dict[str, object] = {_HEAD_MODEL: make_mock_model_reference_record(_HEAD_MODEL, baseline=_SDXL)}
    if sibling_model != _HEAD_MODEL:
        reference[sibling_model] = make_mock_model_reference_record(sibling_model, baseline=_SDXL)

    processes: dict[int, object] = {}
    model_map = HordeModelMap(root={})

    # The head's base model is resident on its own lane when warm; when cold it must be preloaded, so an idle
    # slot stands in for the lane it will land on.
    if not cell.head_cold:
        head_lane = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.total_vram_mb = 24_000
        head_lane.process_reserved_mb = 1_372
        processes[0] = head_lane
        model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)

    # A distinct-model sibling gets its own resident lane; a same-model sibling shares the head's resident lane
    # (and requires the head to be warm to have one). A cold head with a same-model sibling would leave the
    # sibling nowhere resident, which no production shape reaches, so the matrix never pairs them.
    needs_idle_slot = cell.head_cold
    if cell.sibling_kind != "none" and sibling_model != _HEAD_MODEL:
        sibling_lane = make_mock_process_info(1, model_name=sibling_model, state=HordeProcessState.PRELOADED_MODEL)
        sibling_lane.total_vram_mb = 24_000
        sibling_lane.process_reserved_mb = 1_372
        processes[1] = sibling_lane
        model_map.update_entry(sibling_model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)

    if needs_idle_slot:
        idle = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        idle.total_vram_mb = 24_000
        processes[2] = idle

    # Guarantee at least ``lanes`` inference slots so the concurrency cap, not a slot shortage, sets capacity.
    while len(processes) < cell.lanes:
        pid = max(processes) + 1 if processes else 0
        spare = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 24_000
        processes[pid] = spare

    world = _AuxLivenessWorld(
        process_map=ProcessMap(processes),  # type: ignore[arg-type]
        model_map=model_map,
        reference=reference,
        max_threads=cell.lanes,
        download_timeout=3.0 if cell.head_script.by_deadline else 120.0,
    )

    head = _small_job(_HEAD_MODEL, lora_names=("head-lora",))
    await world.pop(head, script=cell.head_script)

    sibling: ImageGenerateJobPopResponse | None = None
    if cell.sibling_kind == "non_lora":
        sibling = _small_job(sibling_model)
        await world.pop(sibling, script=_PrefetchScript(on_disk=True))
    elif cell.sibling_kind == "lora_prepared":
        sibling = _small_job(sibling_model, lora_names=("sibling-lora",))
        await world.pop(sibling, script=_PrefetchScript(on_disk=True))
    elif cell.sibling_kind == "lora_slow":
        sibling = _small_job(sibling_model, lora_names=("sibling-lora",))
        await world.pop(sibling, script=_PrefetchScript(resolve_cycle=cell.sibling_resolve_cycle))

    return world, head, sibling


@pytest.mark.parametrize("cell", _MATRIX, ids=[cell.id for cell in _MATRIX])
async def test_bounded_idleness_across_matrix(cell: _Cell) -> None:
    """Waiting on prefetch never freezes the queue: fitting work samples, and the head resolves or drains.

    Across every matrix cell the sibling (when one exists and can fit) reaches sampling within a bound while
    the head is gated; the gated head never seizes a lane before its files are on disk; and once its prefetch
    resolves the head reaches sampling within a bound. A prefetch that fails serves the head without the file,
    so it still reaches sampling; only when its deadline expires does the head fault terminally, and either way
    its sibling still samples, so the queue keeps draining.
    """
    world, head, sibling = await _build_world(cell)

    await world.run(cell.run_cycles)

    # A dispatchable sibling must have reached sampling while the head was gated, keeping the card fed.
    if sibling is not None:
        assert world.sampled(sibling), f"{cell.id}: the fitting sibling never reached sampling"
        if cell.sibling_kind == "lora_slow":
            # An unprepared sibling must not sample before its own files land.
            assert (world.first_sampled_cycle(sibling) or 0) >= cell.sibling_resolve_cycle, (
                f"{cell.id}: the unprepared sibling sampled before its prefetch completed"
            )

    completion_cycle = cell.head_script.completion_cycle
    if completion_cycle is None:
        # Deadline cell: no outcome is ever delivered, so the head faults from the deadline backstop and leaves
        # the pending-inference queue; it must never have sampled, and its sibling work is unaffected.
        assert not world.sampled(head), f"{cell.id}: a head left to its deadline must never sample"
        assert world.stage(head) == JobStage.PENDING_SUBMIT, f"{cell.id}: the faulted head did not drain to submit"
        assert head not in world.job_tracker.jobs_in_progress
        return

    # Resolved cells (files placed, or a failure served without them): the head reaches sampling, never before
    # its outcome landed.
    assert world.sampled(head), f"{cell.id}: the head never reached sampling after its prefetch resolved"
    first = world.first_sampled_cycle(head)
    assert first is not None
    assert first >= max(1, completion_cycle), f"{cell.id}: the head sampled before its prefetch completed"
    assert first <= completion_cycle + _SAMPLING_BOUND, (
        f"{cell.id}: the head reached sampling {first - completion_cycle} cycles after preparation, "
        f"exceeding the bound of {_SAMPLING_BOUND}"
    )


_GATED_CELLS = [c for c in _MATRIX if c.head_script.completion_cycle and c.sibling_kind != "none"]


@pytest.mark.parametrize("cell", _GATED_CELLS, ids=[c.id for c in _GATED_CELLS])
async def test_gated_head_never_admitted_while_sibling_cycles(cell: _Cell) -> None:
    """No admitted job is held every tick until the head resolves, and the head is never admitted while gated.

    The per-cycle admit-implies-dispatchable invariant is enforced inside the driver: any job the scheduler
    admits is one it dispatches to a lane in the same act. This case additionally proves the defect it guards
    against cannot recur: while the head is gated across many cycles the sibling reaches sampling, and the
    gated head is never once observed in progress before its files are on disk.
    """
    world, head, sibling = await _build_world(cell)
    assert sibling is not None

    resolve_cycle = cell.head_script.resolve_cycle or 0
    # Every cycle strictly before the head's files land: the head is still gated, so it holds no lane.
    for _ in range(max(0, resolve_cycle - 1)):
        await world.step()
        assert head not in world.job_tracker.jobs_in_progress, (
            f"{cell.id}: the gated head seized a lane before its prefetch completed"
        )

    assert world.sampled(sibling), f"{cell.id}: the sibling was starved behind the gated head"

    await world.run(_SAMPLING_BOUND + 2)
    assert world.sampled(head), f"{cell.id}: the head never dispatched after its gate cleared"


async def test_cold_uncached_head_holds_no_reservation_while_sibling_samples() -> None:
    """A cold head with uncached LoRAs holds no lane and no VRAM reservation while its files download.

    The tightest cell: a cold base model whose LoRAs are not on disk, a single sampling lane so only one
    job's peak fits at a time, and a slow download. Throughout the download the head must own neither a lane
    nor a VRAM reservation, a resident sibling must keep the card sampling, and once the prefetch completes
    the head must be preloaded and sampled.
    """
    resolve_cycle = 4
    cell = _Cell(
        id="cold_uncached_head_single_lane",
        head_cold=True,
        head_script=_PrefetchScript(resolve_cycle=resolve_cycle),
        sibling_kind="non_lora",
        sibling_same_model=False,
        lanes=1,
        run_cycles=resolve_cycle + _SAMPLING_BOUND + 1,
    )
    world, head, sibling = await _build_world(cell)
    assert sibling is not None
    assert head.id_ is not None

    # Drive the download window (strictly before the files land): the head is gated the whole time.
    for _ in range(resolve_cycle - 1):
        await world.step()

        # No lane holds the head's cold base model, and no preload for it is in flight, so it reserves no
        # VRAM by any path: a VRAM reservation would require the head to have been admitted to preload
        # (its model LOADING/resident) or to dispatch (in progress), and it is none of these.
        assert all(
            process.loaded_horde_model_name != _HEAD_MODEL for process in world.scheduler._process_map.values()
        ), "the cold head's model became resident while its LoRAs were still downloading"
        assert not world.scheduler._horde_model_map.is_model_loading(_HEAD_MODEL)
        assert _HEAD_MODEL not in world.scheduler._horde_model_map.root
        assert head not in world.job_tracker.jobs_in_progress
        assert world.job_tracker.get_stage(head.id_) == JobStage.PENDING_INFERENCE

    # The resident sibling kept the single lane fed throughout the head's wait.
    assert world.sampled(sibling), "the resident sibling never sampled during the head's download window"

    # Once the files land, the cold head is preloaded and sampled within the bound; no permanent shadow.
    await world.run(_SAMPLING_BOUND + 1)
    assert world.sampled(head), "the head never sampled after its prefetch completed"
    first = world.first_sampled_cycle(head)
    assert first is not None and first > resolve_cycle, "the head sampled before its LoRAs were on disk"
