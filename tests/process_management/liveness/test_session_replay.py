"""A live session's job stream replayed through the dispatch world, read back as the live duty report reads it.

The fixture is one 191-minute session on a 24 GB card (see ``_session_replay.py``). Each row builds the world
with the session's configuration, seeds the parent's learned footprints from the session's store, offers the
session's jobs as they were popped, and reads back the slot-duty sampling share, the sampling concurrency, the
hold buckets and the handoff gaps. The live figures calibrate the replay. They say whether the world reproduces
the session's duty shape, and a scheduling change is judged by how it moves the replay against the replay at
the change's parent, never by a live figure.

Values chosen without a live figure: the host's foreign RAM, the service-lane exit time, the safety placement
readiness (the world's default of zero), and the jobs in flight at a window's start, which are not replayed. The
scheduler's performance model starts unseeded and learns sampling rates and dispatch-to-first-step overheads
from the replay's jobs, where the live worker carried a model learned over earlier sessions.
"""

from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.process_management.resources.vram_footprints import (
    _MEASURED_ESTIMATE_MARGIN,
    FootprintKey,
    FootprintStage,
    ResolutionBucket,
    job_class_key,
    sampling_footprint_key,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _FLUX,
    _KREA2,
    _SD15,
    _DispatchWorld,
    _ModelClass,
    _support_component_mb,
)
from tests.process_management.liveness._host_ram import HostRamLedger
from tests.process_management.liveness._session_replay import (
    SESSION_FOOTPRINTS_PATH,
    SESSION_JOBS_PATH,
    PopDriverRecord,
    ReplayJob,
    _at_rest_holds_support,
    _measured_mb,
    _resident_key,
    calibrated_load_seconds_per_gb,
    chain_seconds,
    decode_seconds_by_job,
    kudos_by_job,
    load_footprint_keys,
    load_footprint_store,
    load_session_jobs,
    median_chain_peak_mb,
    median_seconds_in_ticks,
    model_classes,
    pop_driver,
    safety_check_growth_mb,
    safety_seconds,
    sample_seconds_by_job,
    true_footprint,
)
from tests.process_management.liveness._world_assertions import (
    SAMPLING_CONCURRENCY_TOP,
    HandoffGapSegment,
    ModelLoadTotals,
    concurrency_shares,
    handoff_gap_segments,
    kudos_per_simulated_hour,
    load_totals_by_model,
    pop_to_start_seconds,
    slot_duty_shares,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 1.0
"""A second per tick, so a phase of a few seconds is observed at the stats stream's own cadence."""

_LANE_COUNT = 4
"""The session's inference lanes, process ids 3 to 6, which served inference for the whole session."""
_MAX_THREADS = 2
"""The session's ``max_threads``."""
_QUEUE_SIZE = 2
"""The session's ``queue_size``."""
_POP_BUDGET = _QUEUE_SIZE + _MAX_THREADS
"""How many jobs the popper keeps pending or in progress on one card: ``queue_size`` plus ``max_threads``."""
_LEASE_SLOTS = 3
"""The session's ``gpu_sampling_lease_slots``."""

_HOST_RAM_MB = 65_536.0
"""The session host's physical RAM."""
_FOREIGN_RAM_MB = 8192.0
"""Host RAM held outside the lanes: the parent, the service lanes' host pages and the operating system.

Chosen, since the recorded session has no host RAM reading."""
_RAM_RESERVE_MB = 0.0
"""The session's ``ram_reserve_mb``."""
_RAM_PAUSE_PERCENT = 95.0
"""The session's ``ram_pressure_pause_percent``."""
_FOREIGN_VRAM_MB = 3300.0
"""Device memory on the session's card that no lane's report accounts for, held for the whole session."""
_SERVICE_LANE_EXIT_SECONDS = 1.0
"""How long a paused service lane's process takes to exit. Chosen as one tick, with no measured figure recorded."""
_LANE_START_SECONDS = 0.0
"""How long a regrown inference lane takes to start. Zero, the world's default, since the session records no
inference start against which a figure could be chosen."""
_PHASE_RESIDENT_COMPONENTS = True
"""The replay charges a monolithic job's text encoder and VAE as a lane tenant, so the truth hook leaves them out."""

_SESSION_START_CLOCK_SECONDS = 13 * 3600 + 49 * 60 + 37.58
"""The session's start on the host's local clock (13:49:37.58), from which the fixture's offsets run."""
_W3_START_CLOCK_SECONDS = 15 * 3600 + 5 * 60
"""The start of the session's weakest middle stretch, 15:05 on the host's local clock."""
_W3_START_OFFSET_SECONDS = _W3_START_CLOCK_SECONDS - _SESSION_START_CLOCK_SECONDS
_W3_SECONDS = 1200.0
"""The stretch runs 15:05 to 15:25."""
_W3_RESIDENTS: tuple[tuple[str, bool] | None, ...] = (
    ("CyberRealistic Pony", True),
    ("Flux.1-Schnell fp8 (Compact)", False),
    ("Anima-Turbo-v1.1", True),
    None,
)
"""What each inference lane held at 15:05, in lane order (process ids 3 to 6), and whether it was on the card.

Process 3 and 5 were sampling CyberRealistic Pony and Anima, process 4 had just moved Flux to RAM and process 6
had just dropped its model to disk."""

_LIVE_SLOT_DUTY_SAMPLING_SHARE = 0.39
"""The session's slot-duty sampling share over its whole length."""
_SAMPLING_SHARE_BAND = 0.15
"""How far the replay's sampling share may sit from the live share and still count as the same duty shape.

Wide because the replay omits what the module's known mismatches list, each of which moves the share by some
points. A replay outside it is reproducing a different session."""
_TWO_SAMPLER_SHARE_CEILING = 0.30
"""The share of ticks with two lanes sampling the replay stays below.

The session sat at two samplers for 13.7% of its samples. Above twice that, the world is overlapping work the
session's scheduler could not, a fidelity gap in the world."""
_ADMISSION_NOISE_MB = admission_margin_mb(_CARD_24GB.total_mb, platform="linux")
"""The admission margin the session ran under on Linux, where it is the device-wide fraction of the card."""


@dataclass(frozen=True)
class _ReplayFigures:
    """Represents a finished replay's figures, read as the live duty report reads a session."""

    sampling_share: float
    concurrency: dict[int, float]
    slot_duty: dict[str, float]
    gap_segments: dict[str, float]
    pop_to_start_median_seconds: float | None
    completed_jobs: int
    overcommit_ticks: int
    overcommit_max_mb: float
    driver: PopDriverRecord
    kudos_per_hour: float
    """The live kudos of the jobs the replay completed, per simulated hour."""
    load_totals: list[ModelLoadTotals]
    """Each model's device loads, the model with the most load seconds first."""

    @property
    def partial_loads(self) -> int:
        """Return how many device loads left part of their weights in host RAM."""
        return sum(totals.partial_loads for totals in self.load_totals)

    @property
    def complete_loads(self) -> int:
        """Return how many device loads put all of their weights on the card."""
        return sum(totals.loads - totals.partial_loads for totals in self.load_totals)

    def describe(self) -> str:
        """Return the figures as text a failing row prints, so a regression shows where the slot time went."""
        concurrency = ", ".join(
            f"{lanes}{'+' if lanes == SAMPLING_CONCURRENCY_TOP else ''}: {share:.1%}"
            for lanes, share in self.concurrency.items()
        )
        holds = ", ".join(
            f"{bucket} {share:.1%}" for bucket, share in sorted(self.slot_duty.items(), key=lambda item: -item[1])
        )
        gaps = ", ".join(f"{segment} {seconds:.0f} s" for segment, seconds in self.gap_segments.items())
        pop_to_start = (
            "none" if self.pop_to_start_median_seconds is None else f"{self.pop_to_start_median_seconds:.1f} s"
        )
        loads = ", ".join(
            f"{totals.model} {totals.load_seconds:.0f} s over {totals.loads} ({totals.moved_mb:.0f} MB moved)"
            for totals in self.load_totals
        )
        return (
            f"sampling share {self.sampling_share:.1%}; lanes sampling {concurrency}; slot duty {holds}; "
            f"handoff gaps {gaps}; median pop to start {pop_to_start}; {self.completed_jobs} jobs completed; "
            f"{self.overcommit_ticks} over-committed ticks (at most {self.overcommit_max_mb:.0f} MB); "
            f"offered {self.driver.offered}, "
            f"left {self.driver.remaining}, refused offers {self.driver.refused_offers}, "
            f"taken out of order {self.driver.taken_out_of_order}; {self.kudos_per_hour:.0f} kudos per hour; "
            f"loads {self.complete_loads} complete and {self.partial_loads} partial; ram_to_vram {loads}"
        )


def _replay_world(jobs: list[ReplayJob], *, scratch_directory: Path) -> tuple[_DispatchWorld, list[_ModelClass]]:
    """Build the session's world: its card, lanes and flags, its learned store, and the stream's measured figures.

    ``jobs`` is the whole session's stream, whatever part of it a row offers, so the world serves every model the
    session served and the load rate and the median durations are the session's. The world learns into its own
    store. The truth hook reads a second store loaded from the same fixture, so what a job really costs does not
    follow what the world taught its parent.
    """
    keys = load_footprint_keys(SESSION_FOOTPRINTS_PATH)
    truth_store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=scratch_directory / "truth")
    parent_store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=scratch_directory / "parent")
    classes = model_classes(jobs, truth_store, keys)
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=_LANE_COUNT,
        max_threads=_MAX_THREADS,
        queue_depth=_QUEUE_SIZE,
        gpu_sampling_lease_slots=_LEASE_SLOTS,
        tick_seconds=_TICK_SECONDS,
        high_performance_mode=True,
        closed_loop=True,
        clearance_lease=True,
        two_phase=True,
        tail_overlap=True,
        staged_lane_prefetch=True,
        idle_lane_warm=True,
        performance_model=True,
        service_contexts=True,
        service_lane_exit_seconds=_SERVICE_LANE_EXIT_SECONDS,
        safety_off_gpu_allowed=True,
        safety_check_growth_mb=safety_check_growth_mb(truth_store),
        safety_check_ticks=median_seconds_in_ticks(safety_seconds(jobs), tick_seconds=_TICK_SECONDS),
        safety_cpu_check_ticks=median_seconds_in_ticks(safety_seconds(jobs), tick_seconds=_TICK_SECONDS),
        post_process_chain_peak_mb=median_chain_peak_mb(truth_store, keys),
        post_process_chain_hold_ticks=median_seconds_in_ticks(chain_seconds(jobs), tick_seconds=_TICK_SECONDS),
        post_process_chain_measured_admission=True,
        learned_footprints=True,
        footprint_store=parent_store,
        phase_resident_components=_PHASE_RESIDENT_COMPONENTS,
        partial_loads=True,
        text_encoder_released_before_wait=True,
        lane_start_seconds=_LANE_START_SECONDS,
        lane_reports_carry_context=True,
        admission_noise_mb=_ADMISSION_NOISE_MB,
        host_ram=HostRamLedger(_HOST_RAM_MB, _FOREIGN_RAM_MB),
        ram_reserve_mb=_RAM_RESERVE_MB,
        ram_pause_percent=_RAM_PAUSE_PERCENT,
        foreign_vram_mb_by_tick={1: _FOREIGN_VRAM_MB},
        extra_model_classes=tuple(classes),
        true_footprint_mb=true_footprint(truth_store, classes, phase_resident_components=_PHASE_RESIDENT_COMPONENTS),
        sample_seconds_by_job=sample_seconds_by_job(jobs),
        decode_seconds_by_job=decode_seconds_by_job(jobs),
        load_seconds_per_gb=calibrated_load_seconds_per_gb(jobs, classes),
    )
    # The parent's foreign floor ages its readings on its own clock. A replay runs far faster than the world
    # clock, so the floor is put on the world clock for its window to mean simulated seconds.
    world.scheduler._foreign_floor_clock = lambda: world.now
    return world, classes


def _figures(world: _DispatchWorld, driver: PopDriverRecord, jobs: list[ReplayJob]) -> _ReplayFigures:
    """Read a finished replay's figures, pricing completed jobs at the kudos ``jobs`` recorded for them live."""
    slot_duty = slot_duty_shares(world)
    waits = pop_to_start_seconds(world)
    return _ReplayFigures(
        sampling_share=slot_duty.get("sampling", 0.0),
        concurrency=concurrency_shares(world),
        slot_duty=slot_duty,
        gap_segments={str(segment): seconds for segment, seconds in handoff_gap_segments(world).items()},
        pop_to_start_median_seconds=statistics.median(waits) if waits else None,
        completed_jobs=world.completed_jobs,
        overcommit_ticks=len({tick for tick, _card, _over_mb in world.card_overcommits}),
        overcommit_max_mb=max((over_mb for _tick, _card, over_mb in world.card_overcommits), default=0.0),
        driver=driver,
        kudos_per_hour=kudos_per_simulated_hour(world, kudos_by_job(jobs)),
        load_totals=load_totals_by_model(world),
    )


async def _replay_w3_window(scratch_directory: Path, *, on_demand: bool = False) -> _ReplayFigures:
    """Replay 15:05 to 15:25 from the residents the lanes held at 15:05, and read the figures back."""
    _world, figures = await _run_w3_window(scratch_directory, on_demand=on_demand)
    return figures


async def _run_w3_window(scratch_directory: Path, *, on_demand: bool = False) -> tuple[_DispatchWorld, _ReplayFigures]:
    """Replay 15:05 to 15:25 from the residents the lanes held at 15:05, returning the finished world and figures.

    At the recorded offsets the stream is the jobs popped in the stretch. On demand it is every job dispatched from
    15:05 on, offered as fast as the world asks (:func:`pop_driver`), so the stretch's twenty minutes are bounded by
    the card and not by what the live worker popped in them.
    """
    window_end = _W3_START_OFFSET_SECONDS + _W3_SECONDS
    session_jobs = load_session_jobs(SESSION_JOBS_PATH)
    jobs = [
        job
        for job in session_jobs
        if (on_demand or job.offset_seconds < window_end)
        and (job.dispatch_offset_seconds or job.offset_seconds) >= _W3_START_OFFSET_SECONDS
    ]
    world, classes = _replay_world(session_jobs, scratch_directory=scratch_directory)
    classes_by_name = {model_class.name: model_class for model_class in classes}
    for lane_id, resident in zip(world.inference_lane_ids(), _W3_RESIDENTS, strict=True):
        if resident is None:
            continue
        model_name, on_card = resident
        world.seed_resident(lane_id, classes_by_name[model_name], in_vram=on_card)
    driver = await pop_driver(
        world,
        jobs,
        queue_size=_POP_BUDGET,
        start_offset_seconds=_W3_START_OFFSET_SECONDS,
        seconds=_W3_SECONDS,
        on_demand=on_demand,
    )
    return world, _figures(world, driver, jobs)


async def _replay_whole_session(scratch_directory: Path) -> _ReplayFigures:
    """Replay the whole stream from a cold pool, for as long as the session ran, and read the figures back."""
    jobs = load_session_jobs(SESSION_JOBS_PATH)
    world, _classes = _replay_world(jobs, scratch_directory=scratch_directory)
    session_seconds = math.ceil(max(job.finalized_offset_seconds or job.offset_seconds for job in jobs))
    driver = await pop_driver(
        world,
        jobs,
        queue_size=_POP_BUDGET,
        start_offset_seconds=0.0,
        seconds=float(session_seconds),
    )
    return _figures(world, driver, jobs)


def _assert_live_duty_shape(figures: _ReplayFigures, *, context: str) -> None:
    """Assert the replay's duty shape sits inside the calibration band around the live session's."""
    description = figures.describe()
    # An over-commit inside the admission margin is what the margin exists to absorb, and the live session's
    # decode tails ran there too. One beyond it is a pricing decision that admitted room the card did not have.
    assert figures.overcommit_max_mb <= _ADMISSION_NOISE_MB, (
        f"{context}: the card was over-committed beyond the admission margin ({_ADMISSION_NOISE_MB:.0f} MB). "
        f"{description}"
    )
    assert abs(figures.sampling_share - _LIVE_SLOT_DUTY_SAMPLING_SHARE) <= _SAMPLING_SHARE_BAND, (
        f"{context}: sampling share {figures.sampling_share:.1%} is more than {_SAMPLING_SHARE_BAND:.0%} from the "
        f"live {_LIVE_SLOT_DUTY_SAMPLING_SHARE:.0%}. {description}"
    )
    two_sampler_share = figures.concurrency[SAMPLING_CONCURRENCY_TOP]
    assert two_sampler_share < _TWO_SAMPLER_SHARE_CEILING, (
        f"{context}: two lanes sampled on {two_sampler_share:.1%} of ticks, at or above "
        f"{_TWO_SAMPLER_SHARE_CEILING:.0%}. {description}"
    )


async def test_replay_w3_window_reproduces_the_live_duty_shape(tmp_path: Path) -> None:
    """The 15:05 to 15:25 stretch, replayed, keeps the live session's duty shape and never over-commits the card.

    The stretch is the session's weakest middle span. Flux and Z-Image held most of the in-flight seconds and the
    clearance hold took 854 of its 1,200 seconds. The band is the calibration this replay is held to: a sampling
    share within 15 points of the session's 39%, fewer than 30% of ticks with two lanes sampling against the
    session's 13.7%, and no tick over the card by more than the admission margin. The share is the session's
    because the stretch alone has too few jobs to fix a figure. The band is wide for the same reason and for the
    mismatches the fixture module lists.
    """
    figures = await _replay_w3_window(tmp_path)
    _assert_live_duty_shape(figures, context="15:05 to 15:25 window")


_SPARSE_STREAM_ROWS = 8
"""Rows of the sparse stream the on-demand row offers, more than the pop budget holds at once."""
_SPARSE_STREAM_SPACING_SECONDS = 10_000.0
"""How far apart the sparse stream's recorded offsets sit, far longer than the run, so offsets alone feed one row."""
_SPARSE_STREAM_SECONDS = 240.0
"""How long the on-demand row runs: long enough for the first jobs to finish and their successors to be offered."""


async def _drive_sparse_stream(tmp_path: Path, *, on_demand: bool) -> tuple[_DispatchWorld, PopDriverRecord]:
    """Offer the session's first SDXL rows, their offsets spread far apart, to a fresh replay world."""
    session_jobs = load_session_jobs(SESSION_JOBS_PATH)
    world, _classes = _replay_world(session_jobs, scratch_directory=tmp_path)
    sdxl_rows = [job for job in session_jobs if job.baseline == "stable_diffusion_xl"][:_SPARSE_STREAM_ROWS]
    sparse = [
        replace(job, offset_seconds=index * _SPARSE_STREAM_SPACING_SECONDS) for index, job in enumerate(sdxl_rows)
    ]
    driver = await pop_driver(
        world,
        sparse,
        queue_size=_POP_BUDGET,
        start_offset_seconds=0.0,
        seconds=_SPARSE_STREAM_SECONDS,
        on_demand=on_demand,
    )
    return world, driver


async def test_on_demand_supply_feeds_a_world_that_asks_faster_than_the_recorded_offsets(tmp_path: Path) -> None:
    """On demand, every row the queue has room for is offered at once, so no handoff waits on an unpopped successor.

    The same stream at its recorded offsets offers one row in the run, which is the calibration model starving a
    world that runs ahead of the live worker.
    """
    _calibrated_world, calibrated = await _drive_sparse_stream(tmp_path / "calibrated", on_demand=False)
    world, driver = await _drive_sparse_stream(tmp_path / "on_demand", on_demand=True)

    assert calibrated.offered == 1
    assert driver.offered > _POP_BUDGET, (
        f"on demand the world took {driver.offered} rows in {_SPARSE_STREAM_SECONDS:.0f} s, no more than one pop "
        f"budget ({_POP_BUDGET}), so no row was offered as an earlier one finished"
    )
    assert driver.refused_offers == 0 and driver.taken_out_of_order == 0
    figures = _figures(world, driver, [])
    assert figures.card_busy_share == pytest.approx(
        sum(share for lanes, share in figures.concurrency.items() if lanes > 0)
    ), f"the card-busy share is not the share of ticks with a lane sampling. {figures.describe()}"
    not_popped_seconds = handoff_gap_segments(world).get(HandoffGapSegment.NOT_POPPED, 0.0)
    assert not_popped_seconds == 0.0, (
        f"handoffs waited {not_popped_seconds:.0f} s on a successor not yet popped while {driver.remaining} rows "
        "were left to offer"
    )


_DRAINED_STREAM_SECONDS = 4_000.0
"""A run length far beyond what the sparse stream takes to serve on demand, so only the drain can end the run."""


async def test_an_on_demand_run_ends_when_its_last_job_completes(tmp_path: Path) -> None:
    """Fed on demand, the run stops on the tick its last offered job completes, so its figures carry no dry tail.

    The duty shares and the kudos per hour read over the world clock the run advanced, which is then the span the
    stream was served.
    """
    session_jobs = load_session_jobs(SESSION_JOBS_PATH)
    world, _classes = _replay_world(session_jobs, scratch_directory=tmp_path)
    sdxl_rows = [job for job in session_jobs if job.baseline == "stable_diffusion_xl"][:_SPARSE_STREAM_ROWS]
    completed_after_step: list[int] = []
    world_step = world.step

    async def recording_step() -> None:
        await world_step()
        completed_after_step.append(world.completed_jobs)

    world.step = recording_step  # type: ignore[method-assign]

    driver = await pop_driver(
        world,
        sdxl_rows,
        queue_size=_POP_BUDGET,
        start_offset_seconds=0.0,
        seconds=_DRAINED_STREAM_SECONDS,
        on_demand=True,
    )

    assert driver.offered == len(sdxl_rows) and driver.remaining == 0
    assert world.completed_jobs == len(sdxl_rows), f"{world.completed_jobs} of {len(sdxl_rows)} jobs completed"
    assert world.now - world.started_at < _DRAINED_STREAM_SECONDS, "the run went on to its full length"
    first_drained_step = completed_after_step.index(len(sdxl_rows))
    assert first_drained_step == len(completed_after_step) - 1, (
        f"the run stepped {len(completed_after_step) - 1 - first_drained_step} ticks after its last job completed"
    )


@pytest.mark.slow
async def test_replay_whole_session(tmp_path: Path) -> None:
    """The whole session, replayed from a cold pool, keeps the same duty shape as its middle stretch.

    It runs for the session's whole length and is held to the window row's band.
    """
    figures = await _replay_whole_session(tmp_path)
    _assert_live_duty_shape(figures, context="whole session")


_PEAK_PARTIAL_MODEL = _FLUX.name
"""A replayed checkpoint whose whole-job peak was taken partially loaded, below its weights and support seed."""


def test_the_truth_hook_leaves_the_encoder_and_vae_to_the_support_tenant(tmp_path: Path) -> None:
    """With the components charged as a tenant, a job's transient leaves them out of its peak's share.

    The sample key is the whole job's peak with the text encoder and VAE in it, and the world charges those
    again as the lane's support tenant, so a transient read off the sample key less the weights charges them
    twice. A peak below the weights and the seed was taken partially loaded, and a peak of a class whose at-rest
    key held its support components carries an unknown share of an encoder the fixture's child partially unloaded,
    so either one's transient is the class's measured activation instead, where one is measured. Every replayed job
    is checked against the branch its figures put it in.
    """
    jobs = load_session_jobs(SESSION_JOBS_PATH)
    store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path)
    classes = model_classes(jobs, store, load_footprint_keys(SESSION_FOOTPRINTS_PATH))
    phased = true_footprint(store, classes, phase_resident_components=True)
    whole = true_footprint(store, classes, phase_resident_components=False)

    classes_by_name = {model_class.name: model_class for model_class in classes}
    decomposed_models: set[str] = set()
    measured_activation_models: set[str] = set()
    for replay_job in jobs:
        job = replay_job.pop_response
        model_class = classes_by_name[replay_job.model]
        weights_mb, transient_mb = phased(job)
        _, peak_share_mb = whole(job)
        support_mb = _support_component_mb(model_class)
        assert weights_mb == model_class.weights_mb
        activation_key = sampling_footprint_key(job, str(model_class.baseline), stage=FootprintStage.SAMPLE_ACTIVATION)
        assert activation_key is not None
        measured_activation_mb = store.measured_job_estimate_net_of_context_mb(
            activation_key, job_class_key(activation_key, job)
        )
        holds_residue = _at_rest_holds_support(store, model_class)
        decomposes = peak_share_mb > support_mb and not (holds_residue and measured_activation_mb is not None)
        if decomposes:
            decomposed_models.add(replay_job.model)
            assert transient_mb == pytest.approx(peak_share_mb - support_mb), (
                f"{replay_job.model}: the transient is {transient_mb:.0f} MB, not its {peak_share_mb:.0f} MB peak "
                f"share less the {support_mb:.0f} MB the support tenant already charges"
            )
            continue
        measured_activation_models.add(replay_job.model)
        expected_mb = 0.0 if measured_activation_mb is None else measured_activation_mb / _MEASURED_ESTIMATE_MARGIN
        assert transient_mb == pytest.approx(expected_mb), (
            f"{replay_job.model}: a peak that does not decompose gave a {transient_mb:.0f} MB transient, not the "
            f"{expected_mb:.0f} MB measured activation"
        )
    assert _PEAK_PARTIAL_MODEL in measured_activation_models, (
        "the Flux peak decomposes, so the measured activation branch is untested"
    )
    assert decomposed_models, "no replayed class decomposes, so the double count branch is untested"


def test_a_resident_key_holding_the_support_components_charges_the_diffusion_weights_alone(tmp_path: Path) -> None:
    """A class whose at-rest key covers hordelib's core and support seeds is charged that key less the support seed.

    The support components are the world's support tenant, which a child that releases its encoder before the
    wait does not hold while it samples, so the class's weights are the diffusion weights alone.
    """
    jobs = load_session_jobs(SESSION_JOBS_PATH)
    store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path)
    classes = {
        model_class.name: model_class
        for model_class in model_classes(jobs, store, load_footprint_keys(SESSION_FOOTPRINTS_PATH))
    }
    flux_class = classes[_FLUX.name]
    assert _at_rest_holds_support(store, flux_class), "the fixture's Flux key reads as weights alone"
    at_rest_mb = _measured_mb(store, _resident_key(flux_class.name, str(flux_class.baseline)))
    assert at_rest_mb is not None
    assert flux_class.weights_mb == pytest.approx(at_rest_mb - _support_component_mb(flux_class))


_FLUX_SAMPLE_KEY = FootprintKey(
    model_baseline=str(_FLUX.baseline),
    resolution_bucket=ResolutionBucket.LE_1024,
    platform=sys.platform,
    stage=FootprintStage.SAMPLE,
)
"""The Flux sampling key most of the window's Flux jobs are learned under."""


async def test_the_w3_window_teaches_the_flux_sample_figure_a_released_child_reports(tmp_path: Path) -> None:
    """Over the window the parent's measured Flux sampling figure never climbs and settles at the child's sample.

    The fixture's figure was learned from a child that kept its text encoder on the card, so a child that releases
    it reports less, and the parent's figure moves down toward the diffusion weights and activation the world
    charges, priced through the store's margin. A figure that rises is teaching the parent figures no child
    reported. Each of the world's reports carries the job's sampling peak without the context share or the decode
    spike. Truth figures are read without the store's margin, which the parent adds when it learns from them.
    """
    start_store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path / "start")
    start_mb = start_store.measured_estimate_net_of_context_mb(_FLUX_SAMPLE_KEY)
    assert start_mb is not None, "the fixture's Flux key is under-observed, so the row is moot"
    jobs = load_session_jobs(SESSION_JOBS_PATH)
    truth_store = load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path / "truth")
    classes = model_classes(jobs, truth_store, load_footprint_keys(SESSION_FOOTPRINTS_PATH))
    weights_mb, activation_mb = true_footprint(truth_store, classes, phase_resident_components=True)(
        make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=4)
    )
    released_sample_mb = (weights_mb + activation_mb) * _MEASURED_ESTIMATE_MARGIN

    world, figures = await _run_w3_window(tmp_path / "replay")

    end_mb = world.footprint_store.measured_estimate_net_of_context_mb(_FLUX_SAMPLE_KEY)
    assert end_mb is not None
    tolerance_mb = start_mb * (_MEASURED_ESTIMATE_MARGIN - 1.0)
    assert end_mb <= start_mb + tolerance_mb, (
        f"the learned Flux le_1024 figure climbed from {start_mb:.0f} MB to {end_mb:.0f} MB over the window, beyond "
        f"the store's {tolerance_mb:.0f} MB margin. {figures.describe()}"
    )
    assert abs(end_mb - released_sample_mb) <= tolerance_mb, (
        f"the learned Flux le_1024 figure ended at {end_mb:.0f} MB, more than the store's {tolerance_mb:.0f} MB "
        f"margin from the {released_sample_mb:.0f} MB a released child's weights and activation price at. "
        f"{figures.describe()}"
    )


def _stream_job(model: _ModelClass, *, offset_seconds: float) -> ReplayJob:
    """Create a recorded job for ``model`` popped at ``offset_seconds``, with no measured figures."""
    return ReplayJob(
        offset_seconds=offset_seconds,
        dispatch_offset_seconds=None,
        finalized_offset_seconds=None,
        live_job_id=f"recorded-{model.label}",
        model=model.name,
        baseline=str(model.baseline),
        width=1024,
        height=1024,
        steps=20,
        batch=1,
        sampler="k_euler",
        hires_fix=False,
        lora_count=0,
        textual_inversion_count=0,
        post_processing=(),
        sampling_seconds=None,
        decode_seconds=None,
        encode_seconds=None,
        ram_to_vram_seconds=None,
        chain_seconds=None,
        safety_seconds=None,
        kudos=None,
        peak_device_used_mb=None,
        peak_resident_weights_mb=None,
        vram_used_high_water_mb=None,
        faulted=False,
        pop_response=make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=20),
    )


def _throttled_world(*, switch_min_seconds: float) -> _DispatchWorld:
    """Build a two-lane 24 GB world serving Flux and Krea2, both very large, under the popper's switch throttle."""
    return _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=1,
        queue_depth=2,
        whole_card_enabled=False,
        extra_model_classes=(_KREA2,),
        large_model_switch_min_seconds=switch_min_seconds,
    )


async def test_the_switch_throttle_narrows_the_world_offer_while_a_very_large_model_is_queued() -> None:
    """A queued Flux job withholds Krea2 from the offer under the throttle, and the off default withholds nothing.

    The world runs the popper's large-model governor ahead of the whole-card claim, so a replay with the
    throttle or the re-entry cooldown configured is offered what the live popper would have asked for.
    """
    unthrottled = _throttled_world(switch_min_seconds=0.0)
    assert await unthrottled.offer_job(make_job_pop_response(_FLUX.name, width=1024, height=1024))
    assert _KREA2.name in unthrottled.advertised_models()
    assert not unthrottled.large_model_withholds

    world = _throttled_world(switch_min_seconds=600.0)
    assert await world.offer_job(make_job_pop_response(_FLUX.name, width=1024, height=1024)), (
        "an idle worker withholds nothing, so the first very-large job is taken"
    )
    advertised = world.advertised_models()
    assert _FLUX.name in advertised
    assert _SD15.name in advertised
    assert _KREA2.name not in advertised
    assert not await world.offer_job(make_job_pop_response(_KREA2.name, width=1024, height=1024))
    assert {decision.reason for decision in world.large_model_withholds.values()} == {"switch throttle"}


async def test_the_pop_driver_looks_past_a_refused_row_and_keeps_it_for_later() -> None:
    """With a look-ahead, a row the offer refuses holds back none of the rows behind it.

    The stream is Flux, Krea2 and SD 1.5, all due at once, under the switch throttle. Without the look-ahead the
    refused Krea2 row stalls SD 1.5 behind it. With it, SD 1.5 is taken out of order and Krea2 stays in the stream.
    """
    stream = [
        _stream_job(_FLUX, offset_seconds=0.0),
        _stream_job(_KREA2, offset_seconds=0.0),
        _stream_job(_SD15, offset_seconds=0.0),
    ]

    stalled = await pop_driver(
        _throttled_world(switch_min_seconds=600.0),
        stream,
        queue_size=_POP_BUDGET,
        start_offset_seconds=0.0,
        seconds=1.0,
    )
    assert (stalled.offered, stalled.remaining, stalled.taken_out_of_order) == (1, 2, 0)

    world = _throttled_world(switch_min_seconds=600.0)
    looked_ahead = await pop_driver(
        world,
        stream,
        queue_size=_POP_BUDGET,
        start_offset_seconds=0.0,
        seconds=1.0,
        look_ahead_rows=2,
    )
    assert (looked_ahead.offered, looked_ahead.remaining, looked_ahead.taken_out_of_order) == (2, 1, 1)
    assert sorted(world.popped_at) == sorted(str(job.pop_response.id_) for job in (stream[0], stream[2]))
