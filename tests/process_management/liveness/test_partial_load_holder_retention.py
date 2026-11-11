"""A whole-card holder served as a partial load keeps its weights between its own jobs.

An extra-large checkpoint whose sampling peak exceeds a 16 GB card takes the card on rung 2 and samples with part
of its weights in host RAM. Its jobs run back to back on one lane, and the card does not hold anything else. The
static fit prices the job's whole sampling peak against the card's total, and such a checkpoint's peak never fits
that total. A rung-2 holder with a job of its own pending is priced at its partial-load seat instead and passes the
governor gate, so the child keeps the weights and the next job returns only what was paged out of them.

The rows run the clearance lease in two phases, with partial loads and phase-resident components. The checkpoint's
true footprint is its class weights and an activation that pushes about 2.9 GB of them back to host RAM while it
samples. That leaves the 9.7 GB share held by a live partial load of this class. The load rate is the live upload
rate for that share, and the encode requests the same free memory as a live encode of this class.

A copy kept between jobs loses part of its weights to host RAM through the child's shortfall relief. The encode at
dispatch, the sampling activation and the decode spike can each move part of it, and the next load returns what
they moved. The world does not charge time for a job-end unload, so it understates what keeping the copy saves.
It dispatches a successor on the tick of its predecessor's completion, one governor sample after the card freed.
The successor's load lands before a second sample, so the debounced governor stays at PRESSURE for the whole
streak.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.scheduling.governance.whole_card import DominanceRung
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.retention import RetentionFit
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _CHILD_FREE_MARGIN_MB,
    _KREA2,
    _SD15,
    _DispatchWorld,
    _JobWindow,
)
from tests.process_management.liveness._world_assertions import handoff_gaps_by_card, load_totals_by_model

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 0.1
"""Fine enough that a handoff gap of a few seconds is read to a tenth of a second."""
_MAX_TICKS = 2000
"""Ample for the streak when every job uploads its weights, about six seconds per job."""
_STREAK_JOBS = 8
"""Jobs in the streak: one cold load and seven successors whose loads differ between the two gates."""
_PENDING_SUPPLY = 2
"""Jobs kept popped and waiting, so a successor is always pending when its predecessor ends."""
_JOB_SHAPE = (768, 1344)
"""A portrait megapixel, whose static sampling peak sits above the 16 GB card's total less its context."""
_JOB_STEPS = 8

_ACTIVATION_MB = 3900.0
"""The job's true sampling activation. Funding it moves about 2.9 GB of the 12.6 GB checkpoint off the card."""
_LIVE_PARTIAL_LOAD_MB = 9720.0
_LIVE_PARTIAL_LOAD_SECONDS = 2.16
_LOAD_SECONDS_PER_GB = _LIVE_PARTIAL_LOAD_SECONDS / (_LIVE_PARTIAL_LOAD_MB / 1024.0)
"""The upload rate of a live partial load of this class: 9720 MB in 2.16 s."""
_LIVE_ENCODE_REQUEST_MB = 7752.0
"""The free memory requested from ComfyUI by a live encode of this class before it loaded its text encoder."""
_ENCODE_CHARGE_MB = _LIVE_ENCODE_REQUEST_MB - _CHILD_FREE_MARGIN_MB
"""The encode charge committed by the world at dispatch. The child relieves it with its margin on top, so the kept
copy is paged out for a request equal to the live one."""

_OTHER_ACTIVATION_MB = 1000.0
"""The true sampling activation of the small model whose job queues behind the holder in the wrong-guess row."""
_WRONG_GUESS_HOLDER_JOBS = 3
"""Holder jobs queued ahead of the other model's job. The last one does not have a job of its own behind it."""

_KEEP_RESIDENT = InferenceScheduler._should_keep_model_resident
"""The retention gate as shipped. The recording wrapper calls this reference, so a second patch does not wrap the
first."""


@dataclass(frozen=True)
class _RetentionVerdict:
    """One dispatch's retention decision, with the static fit and the governor state at the time of the decision."""

    job_id: str
    granted: bool
    governor: GovernorState
    holder_rung: DominanceRung | None
    """The rung of the card's residency when it is held for the job's model, else None."""
    static_fit: RetentionFit | None


@dataclass
class _Streak:
    """The finished world, each job's window in pop order, and every retention verdict in dispatch order."""

    world: _DispatchWorld
    windows: list[_JobWindow]
    verdicts: list[_RetentionVerdict]

    @property
    def gaps(self) -> list[float]:
        """The card's handoff gaps, one per successor."""
        return handoff_gaps_by_card(self.world).get(0, [])


def _true_footprint_mb(job: ImageGenerateJobPopResponse) -> tuple[float, float]:
    """A job's real cost to the card in resident weights and sampling activation (MB), by its model."""
    if job.model == _KREA2.name:
        return _KREA2.weights_mb, _ACTIVATION_MB
    return _SD15.weights_mb, _OTHER_ACTIVATION_MB


def _holder_world(*, queue_depth: int = _PENDING_SUPPLY) -> _DispatchWorld:
    """Two lanes on a 16 GB card with the post-processing lane and safety, serving the extra-large class."""
    return _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=queue_depth,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
        safety_off_gpu_allowed=True,
        clearance_lease=True,
        two_phase=True,
        partial_loads=True,
        phase_resident_components=True,
        true_footprint_mb=_true_footprint_mb,
        load_seconds_per_gb=_LOAD_SECONDS_PER_GB,
        extra_model_classes=(_KREA2,),
        staged_encode_mb_by_model={_KREA2.name: _ENCODE_CHARGE_MB},
    )


def _record_verdicts(
    monkeypatch: pytest.MonkeyPatch,
    verdicts: list[_RetentionVerdict],
    *,
    holder_rule: bool,
) -> None:
    """Record every retention verdict. Without ``holder_rule`` the gate does not treat the rung-2 holder separately.

    With the rule switched off, the holder is judged like any other model, on the governor and the static fit alone.
    """
    if not holder_rule:
        monkeypatch.setattr(
            InferenceScheduler,
            "_rung_two_holder_has_next_job",
            lambda _self, _model, *, device_index, dispatched_job: False,
        )

    def recording_keep_resident(
        self: InferenceScheduler,
        dispatched_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> bool:
        governor = self.governor_state(device_index)
        residency = self._whole_card_ledger.get(device_index)
        holder_rung = residency.rung if residency is not None and residency.model == dispatched_job.model else None
        fit = self._retention_fit(
            dispatched_job,
            target=process_with_model,
            device_index=device_index,
            include_target_retained=True,
        )
        granted = _KEEP_RESIDENT(
            self,
            dispatched_job,
            process_with_model=process_with_model,
            device_index=device_index,
        )
        verdicts.append(
            _RetentionVerdict(
                job_id=str(dispatched_job.id_),
                granted=granted,
                governor=governor,
                holder_rung=holder_rung,
                static_fit=fit if isinstance(fit, RetentionFit) else None,
            ),
        )
        return granted

    monkeypatch.setattr(InferenceScheduler, "_should_keep_model_resident", recording_keep_resident)


async def _serve_streak(world: _DispatchWorld, verdicts: list[_RetentionVerdict]) -> _Streak:
    """Keep jobs pending until the streak is popped, and step until every job has left its lane."""
    width, height = _JOB_SHAPE
    jobs = [
        make_job_pop_response(_KREA2.name, width=width, height=height, ddim_steps=_JOB_STEPS)
        for _ in range(_STREAK_JOBS)
    ]
    popped = 0
    for _ in range(_MAX_TICKS):
        pending = len(world._job_tracker.jobs_pending_inference)
        while pending < _PENDING_SUPPLY and popped < len(jobs):
            await world.pop(jobs[popped])
            popped += 1
            pending += 1
        await world.step()
        windows = [world.job_windows.get(str(job.id_)) for job in jobs]
        finished = [window for window in windows if window is not None and window.retention_granted is not None]
        if len(finished) == len(jobs):
            return _Streak(world=world, windows=finished, verdicts=verdicts)
    raise AssertionError(f"the streak did not finish within {_MAX_TICKS} ticks. {world.state_dump()}")


async def _run_streak(*, holder_rule: bool) -> _Streak:
    """Serve the streak on a fresh world, recording verdicts, with or without the gate's rung-2 holder rule."""
    verdicts: list[_RetentionVerdict] = []
    with pytest.MonkeyPatch.context() as patch:
        _record_verdicts(patch, verdicts, holder_rule=holder_rule)
        return await _serve_streak(_holder_world(), verdicts)


def _describe(streak: _Streak) -> str:
    """The streak's loads, gaps and verdicts, for an assertion message."""
    loads = [
        (window.loaded_now, round(window.load_moved_mb), round(window.load_returned_mb), round(window.load_seconds, 2))
        for window in streak.windows
    ]
    verdicts = [
        (verdict.granted, verdict.governor.value, verdict.static_fit.describe() if verdict.static_fit else None)
        for verdict in streak.verdicts
    ]
    gaps = [round(gap, 2) for gap in streak.gaps]
    return (
        f"loads (fresh, moved, returned, seconds): {loads}; gaps: {gaps}; verdicts: {verdicts}. "
        f"{streak.world.state_dump()}"
    )


def _mean(values: list[float]) -> float:
    """The arithmetic mean of a non-empty list."""
    assert values, "the list to average is empty"
    return sum(values) / len(values)


def _successors(streak: _Streak) -> list[_JobWindow]:
    """Every job after the streak's first."""
    return streak.windows[1:]


async def test_without_the_holder_rule_a_partial_load_holder_uploads_again_after_every_job() -> None:
    """When the holder is judged like any other model, its weights leave the card after every job.

    At every dispatch the card's rung-2 residency is held for the job's model and the static fit denies, since it
    prices the whole sampling peak against the card total. Every job ends without a grant and loads the whole
    checkpoint afresh from host RAM. This is the gate replaced by the holder rule, and the baseline for the next
    row's handoff gap.
    """
    streak = await _run_streak(holder_rule=False)
    detail = _describe(streak)

    assert len(streak.verdicts) == _STREAK_JOBS, f"a dispatch was not judged for retention. {detail}"
    for verdict in streak.verdicts:
        assert verdict.holder_rung is DominanceRung.CANNOT_CO_RESIDE, (
            f"precondition: job {verdict.job_id} was dispatched without its model holding the card on rung 2. {detail}"
        )
        assert verdict.static_fit is not None and not verdict.static_fit.granted, (
            f"the static fit did not deny job {verdict.job_id}. {detail}"
        )
        assert not verdict.granted, f"job {verdict.job_id} was granted retention. {detail}"
    for window in streak.windows:
        assert window.loaded_now, f"job {window.job_id} did not load its weights afresh. {detail}"
        assert window.load_moved_mb == pytest.approx(_KREA2.weights_mb), (
            f"job {window.job_id} moved {window.load_moved_mb} MB, not the whole checkpoint. {detail}"
        )
    (totals,) = load_totals_by_model(streak.world)
    assert totals.loads == _STREAK_JOBS, f"{totals.loads} of {_STREAK_JOBS} jobs paid a load. {detail}"


async def test_a_partial_load_holder_keeps_its_weights_between_its_own_jobs() -> None:
    """Each successor starts on its predecessor's kept copy and moves only what the world paged out of it.

    Paged out means what the encode, the sampling activation and the decode spike moved to host RAM, which the next
    load returns. Every job ends under a grant though the static fit denies each one and the governor reads
    PRESSURE. A successor does not load the checkpoint afresh. Each handoff gap is shorter than the gap without the
    holder rule by the load seconds saved by the kept copy, within a tick.
    """
    baseline = await _run_streak(holder_rule=False)
    streak = await _run_streak(holder_rule=True)
    detail = f"with the holder rule: {_describe(streak)} without it: {_describe(baseline)}"

    for verdict in streak.verdicts:
        assert verdict.static_fit is not None and not verdict.static_fit.granted, (
            f"precondition: the static fit granted job {verdict.job_id}, so the row does not exercise the seat. "
            f"{detail}"
        )
        assert verdict.granted, f"job {verdict.job_id} was denied retention. {detail}"
    for earlier, later in zip(streak.windows, _successors(streak), strict=False):
        assert earlier.retention_granted, f"job {earlier.job_id} ended without a grant. {detail}"
        assert not later.loaded_now, f"job {later.job_id} uploaded the checkpoint again. {detail}"
        assert 0.0 < later.load_returned_mb < _KREA2.weights_mb, (
            f"job {later.job_id} returned {later.load_returned_mb} MB, which is not a part of the kept copy. {detail}"
        )
        assert later.load_moved_mb == pytest.approx(later.load_returned_mb), (
            f"job {later.job_id} moved {later.load_moved_mb} MB, beyond the {later.load_returned_mb} MB paged out of "
            f"the kept copy. {detail}"
        )
    assert len(baseline.gaps) == len(streak.gaps) == _STREAK_JOBS - 1, detail
    baseline_load_seconds = _mean([window.load_seconds for window in _successors(baseline)])
    kept_load_seconds = _mean([window.load_seconds for window in _successors(streak)])
    saved_seconds = _mean(baseline.gaps) - _mean(streak.gaps)
    assert saved_seconds == pytest.approx(baseline_load_seconds - kept_load_seconds, abs=_TICK_SECONDS), (
        f"keeping the copy saved {saved_seconds:.2f} s per handoff, not the "
        f"{baseline_load_seconds - kept_load_seconds:.2f} s of load it removed. {detail}"
    )


@dataclass
class _WrongGuess:
    """The finished world after the holder's last job was kept on a claim, and another holder job did not arrive."""

    world: _DispatchWorld
    holder_windows: list[_JobWindow]
    other_window: _JobWindow
    resident_while_other_sampled: frozenset[str]
    """The models on the card on the tick the other model's job first sampled."""


async def _serve_wrong_guess(*, holder_rule: bool) -> _WrongGuess:
    """Queue the holder's jobs and then one job of another model, and step until that job samples."""
    width, height = _JOB_SHAPE
    holder_jobs = [
        make_job_pop_response(_KREA2.name, width=width, height=height, ddim_steps=_JOB_STEPS)
        for _ in range(_WRONG_GUESS_HOLDER_JOBS)
    ]
    other_job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=_JOB_STEPS)
    verdicts: list[_RetentionVerdict] = []
    with pytest.MonkeyPatch.context() as patch:
        _record_verdicts(patch, verdicts, holder_rule=holder_rule)
        world = _holder_world(queue_depth=_WRONG_GUESS_HOLDER_JOBS + 1)
        for job in (*holder_jobs, other_job):
            await world.pop(job)
        for _ in range(_MAX_TICKS):
            await world.step()
            other_window = world.job_windows.get(str(other_job.id_))
            if other_window is not None and other_window.sample_from is not None:
                holder_windows = [world.job_windows[str(job.id_)] for job in holder_jobs]
                resident = frozenset(world.card_resident_models(0).values())
                return _WrongGuess(world, holder_windows, other_window, resident)
    raise AssertionError(f"the other model's job never sampled. {world.state_dump()}")


async def test_a_holder_kept_for_a_claim_that_brings_no_job_of_its_own_does_not_delay_the_next_model() -> None:
    """The holder's last job is kept on the standing claim, and the other model's job does not sample any later.

    While the claim stands the worker asks for the holder's model alone, so the next job popped is expected to be
    the holder's. Here it does not arrive. The kept copy is evicted for the queued job of the other model, and that
    job does not start sampling later than it does when the holder's copy is not kept.
    """
    kept = await _serve_wrong_guess(holder_rule=True)
    unkept = await _serve_wrong_guess(holder_rule=False)
    detail = f"kept: {kept.world.state_dump()} not kept: {unkept.world.state_dump()}"

    last_holder = kept.holder_windows[-1]
    assert last_holder.retention_granted, f"precondition: the holder's last job was not kept. {detail}"
    assert _KREA2.name not in kept.resident_while_other_sampled, (
        f"the kept copy was still on the card when the other model's job sampled "
        f"({sorted(kept.resident_while_other_sampled)}). {detail}"
    )
    assert kept.other_window.sample_from is not None and unkept.other_window.sample_from is not None
    assert kept.other_window.sample_from <= unkept.other_window.sample_from, (
        f"the other model's job sampled at {kept.other_window.sample_from:.1f} with the copy kept, after "
        f"{unkept.other_window.sample_from:.1f} without it. {detail}"
    )
