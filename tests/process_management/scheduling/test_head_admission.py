"""The head-of-queue admission ledger, stated directly: every clock is bounded and every step is reproducible.

The scheduler's own tests cover the log lines, ledger events and actuations these steps drive; these rows pin the
state transitions themselves, so a change to a bound or a reset rule is caught where it lives. The scheduler
classes here pin the card inputs the scheduler feeds the starvation clock and the preloads that restart it.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.admission.preload import PreloadGatePlan, cards_able_to_seat
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.head_admission import (
    DISPATCH_STALL_LOG_INTERVAL_SECONDS,
    HEAD_RAM_DEFER_BARRIER_CAP_SECONDS,
    HEAD_RAM_DEFER_BARRIER_SECONDS,
    SAFETY_RECOVERY_HOLD_TTL_SECONDS,
    STAGING_DEFER_REPEAT_SECONDS,
    HeadAdmissionLedger,
    HeadRamDeferStep,
    SafetyRecoveryHoldStep,
    StagingDeferReason,
    format_staging_defer_tally,
    head_waits_behind_work,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _ledger() -> tuple[HeadAdmissionLedger, _Clock]:
    clock = _Clock()
    return HeadAdmissionLedger(clock), clock


class TestHeadStarvation:
    """The starvation clock times one head on an idle device and nothing else."""

    def test_times_the_head_only_while_the_device_is_idle(self) -> None:
        """Live work zeroes the clock; an idle device starts it at the head's first observation."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("head", work_in_progress=True)
        assert ledger.starved_seconds("head") == 0.0
        ledger.track_head_starvation("head", work_in_progress=False)
        clock.now += 30.0
        assert ledger.starved_seconds("head") == 30.0
        assert ledger.starved_seconds("other") == 0.0
        assert ledger.starved_seconds(None) == 0.0

    def test_a_new_head_restarts_the_clock(self) -> None:
        """The clock measures this head's wait, not the queue's age."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("first", work_in_progress=False)
        clock.now += 30.0
        ledger.track_head_starvation("second", work_in_progress=False)
        assert ledger.starved_seconds("first") == 0.0
        assert ledger.starved_seconds("second") == 0.0
        clock.now += 5.0
        assert ledger.starved_seconds("second") == 5.0

    def test_clear_and_no_head(self) -> None:
        """A dispatch or an empty queue stops the clock outright."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("head", work_in_progress=False)
        ledger.clear_head_starvation()
        assert ledger.starvation_job_id is None and ledger.starvation_since == 0.0
        ledger.track_head_starvation("head", work_in_progress=False)
        ledger.track_head_starvation(None, work_in_progress=False)
        assert ledger.starvation_job_id is None

    def test_reconcile_stops_the_clock_only_for_a_job_no_longer_waiting(self) -> None:
        """A timed job still waiting keeps its clock; one that left the queue, by any path, stops it."""
        ledger, clock = _ledger()
        assert ledger.reconcile_head_starvation(set()) is False
        ledger.track_head_starvation("head", work_in_progress=False)
        clock.now += 30.0
        assert ledger.reconcile_head_starvation({"head", "other"}) is False
        assert ledger.starved_seconds("head") == 30.0
        assert ledger.reconcile_head_starvation({"other"}) is True
        assert ledger.starvation_job_id is None and ledger.starvation_since == 0.0
        assert ledger.reconcile_head_starvation(set()) is False

    @pytest.mark.parametrize(
        ("eligible", "with_work", "waits"),
        [
            ({0, 1}, {0}, False),
            ({0, 1}, {1}, False),
            ({0, 1}, {0, 1}, True),
            ({0}, {0}, True),
            ({0}, set(), False),
            (set(), set(), True),
        ],
    )
    def test_a_head_waits_behind_work_only_when_every_card_it_may_use_runs_a_job(
        self,
        eligible: set[int],
        with_work: set[int],
        waits: bool,
    ) -> None:
        """An idle card the head may use leaves its clock running; a busy card it may not use does not."""
        assert head_waits_behind_work(eligible, with_work) is waits


class TestSchedulerStarvationCards:
    """The scheduler stops the clock per card on a multi-GPU worker and worker-wide on one card."""

    @staticmethod
    async def _scheduler(
        *,
        card_count: int,
        busy_cards: tuple[int, ...],
        unowned_job_in_progress: bool = False,
    ) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
        """One idle lane per card, a job sampling on each of ``busy_cards``, and a queued head."""
        bridge_data = make_mock_bridge_data()
        tracker = JobTracker()
        slots: dict[int, object] = {}
        running: list[tuple[object, ImageGenerateJobPopResponse]] = []
        for card in range(card_count):
            busy = card in busy_cards
            lane = make_mock_process_info(
                card,
                model_name="sd",
                device_index=card,
                state=HordeProcessState.INFERENCE_STARTING if busy else HordeProcessState.WAITING_FOR_JOB,
            )
            slots[card] = lane
            if busy:
                job = make_job_pop_response(model="sd")
                await track_popped_job_async(tracker, job)
                await mark_job_in_progress_async(tracker, job)
                running.append((lane, job))
        if unowned_job_in_progress:
            job = make_job_pop_response(model="sd")
            await track_popped_job_async(tracker, job)
            await mark_job_in_progress_async(tracker, job)
        head = make_job_pop_response(model="flux")
        await track_popped_job_async(tracker, head)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap(slots),  # type: ignore[arg-type]
            job_tracker=tracker,
            bridge_data=bridge_data,
            max_inference=card_count,
            card_runtimes=make_test_card_runtimes(device_indices=tuple(range(card_count)), config=bridge_data),
        )
        for lane, job in running:
            lane.record_inference_ownership(job, attempt_ordinal=0)  # type: ignore[attr-defined]
        return scheduler, head

    @staticmethod
    def _clock_runs(scheduler: InferenceScheduler, head: ImageGenerateJobPopResponse) -> bool:
        scheduler._update_head_starvation_timer(head, scheduler.snapshot())
        return scheduler.head_admission.starvation_job_id == str(head.id_)

    @pytest.mark.parametrize("busy_card", [0, 1])
    async def test_a_head_eligible_on_both_cards_starts_its_clock_when_either_card_is_idle(
        self,
        busy_card: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job on one card frees no room on the other, so the head is starved while that card idles."""
        scheduler, head = await self._scheduler(card_count=2, busy_cards=(busy_card,))
        monkeypatch.setattr(scheduler, "_eligible_card_indices", lambda _job: {0, 1})

        assert self._clock_runs(scheduler, head) is True

    async def test_a_head_eligible_on_both_cards_waits_while_both_run_jobs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every card it may use busy, the head is queued behind work that can free room."""
        scheduler, head = await self._scheduler(card_count=2, busy_cards=(0, 1))
        monkeypatch.setattr(scheduler, "_eligible_card_indices", lambda _job: {0, 1})

        assert self._clock_runs(scheduler, head) is False

    async def test_a_head_eligible_only_on_the_busy_card_does_not_start_its_clock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An idle card the head may not use does not make its wait a starvation."""
        scheduler, head = await self._scheduler(card_count=2, busy_cards=(0,))
        monkeypatch.setattr(scheduler, "_eligible_card_indices", lambda _job: {0})

        assert self._clock_runs(scheduler, head) is False

    async def test_one_card_starts_the_clock_only_with_no_job_in_progress(self) -> None:
        """The single-card rule is the worker-wide one, read straight off the tracker."""
        idle, idle_head = await self._scheduler(card_count=1, busy_cards=())
        busy, busy_head = await self._scheduler(card_count=1, busy_cards=(0,))

        assert self._clock_runs(idle, idle_head) is True
        assert self._clock_runs(busy, busy_head) is False

    async def test_one_card_counts_an_in_progress_job_with_no_identified_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defect reinjection: the per-card rule on one card would run the clock beside this job.

        The per-card count omits a job whose running lane cannot be identified, so applying it on one card would
        start the clock while the tracker still holds work in progress.
        """
        scheduler, head = await self._scheduler(card_count=1, busy_cards=(), unowned_job_in_progress=True)
        monkeypatch.setattr(scheduler, "_eligible_card_indices", lambda _job: {0})

        assert self._clock_runs(scheduler, head) is False


class TestPreloadRestartsHeadStarvation:
    """A sent preload restarts the head's clock only when it is for the head or lands on a card the head may use."""

    _WAITED_SECONDS = 60.0

    @staticmethod
    async def _scheduler(
        *,
        card_count: int,
        head_cards: set[int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[InferenceScheduler, _Clock, ImageGenerateJobPopResponse, ImageGenerateJobPopResponse]:
        """One idle lane per card, a queued head the scheduler may place on ``head_cards``, and a job behind it."""
        clock = _Clock()
        bridge_data = make_mock_bridge_data()
        tracker = JobTracker()
        lanes = {
            card: make_mock_process_info(
                card,
                model_name=None,
                device_index=card,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
            for card in range(card_count)
        }
        head = make_job_pop_response(model="flux")
        other = make_job_pop_response(model="sd")
        await track_popped_job_async(tracker, head)
        await track_popped_job_async(tracker, other)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap(lanes),  # type: ignore[arg-type]
            job_tracker=tracker,
            bridge_data=bridge_data,
            max_inference=card_count,
            card_runtimes=make_test_card_runtimes(device_indices=tuple(range(card_count)), config=bridge_data),
            clock=clock,
        )
        all_cards = set(range(card_count))
        monkeypatch.setattr(
            scheduler,
            "_eligible_card_indices",
            lambda job: set(head_cards) if job.id_ == head.id_ else set(all_cards),
        )
        monkeypatch.setattr(scheduler, "_admit_preload_under_budget", lambda *_args, **_kwargs: True)
        scheduler._update_head_starvation_timer(head, scheduler.snapshot())
        clock.now += TestPreloadRestartsHeadStarvation._WAITED_SECONDS
        return scheduler, clock, head, other

    @staticmethod
    def _preload(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse, process_id: int) -> None:
        """Send ``job``'s preload to ``process_id`` through the admitting act site the preload pass uses."""
        lane = scheduler._process_map.get(process_id)
        assert lane is not None
        plan = PreloadGatePlan(
            decision=AdmissionDecision.ADMIT,
            job_id=str(job.id_),
            reason="admitted",
            target_process_id=process_id,
        )
        decision, _, _ = scheduler._admit_and_send_preload(job, lane, plan, scheduler.snapshot())
        assert decision is AdmissionDecision.ADMIT

    async def test_another_jobs_preload_on_a_card_the_head_may_not_use_leaves_the_clock_running(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The head is as starved as before, so its clock keeps running."""
        scheduler, clock, head, other = await self._scheduler(card_count=2, head_cards={0}, monkeypatch=monkeypatch)

        self._preload(scheduler, other, process_id=1)

        assert scheduler._head_starved_seconds(head) == self._WAITED_SECONDS
        clock.now += 30.0
        assert scheduler._head_starved_seconds(head) == self._WAITED_SECONDS + 30.0

    async def test_the_heads_own_preload_restarts_the_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A preload for the head is progress for it wherever it lands."""
        scheduler, _, head, _ = await self._scheduler(card_count=2, head_cards={0}, monkeypatch=monkeypatch)

        self._preload(scheduler, head, process_id=1)

        assert scheduler._head_starved_seconds(head) == 0.0
        assert scheduler.head_admission.starvation_job_id is None

    async def test_another_jobs_preload_on_a_card_the_head_may_use_restarts_the_clock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A load on the head's card changes what that card can seat, so the head's wait starts over."""
        scheduler, _, head, other = await self._scheduler(card_count=2, head_cards={0}, monkeypatch=monkeypatch)

        self._preload(scheduler, other, process_id=0)

        assert scheduler._head_starved_seconds(head) == 0.0
        assert scheduler.head_admission.starvation_job_id is None

    async def test_one_card_restarts_the_clock_on_every_preload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On one card every preload is on the head's card, so the reset fires as it always has."""
        scheduler, _, head, other = await self._scheduler(card_count=1, head_cards={0}, monkeypatch=monkeypatch)

        self._preload(scheduler, other, process_id=0)

        assert scheduler._head_starved_seconds(head) == 0.0


class TestHeadStarvationCardsAreTheCardsThatCanSeatIt:
    """A card eligible for the head that cannot seat it neither stops the head's clock nor restarts it."""

    _WAITED_SECONDS = 60.0

    @staticmethod
    async def _scheduler(
        *,
        unseatable_cards: set[int],
        busy_cards: tuple[int, ...],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[InferenceScheduler, _Clock, ImageGenerateJobPopResponse, ImageGenerateJobPopResponse]:
        """Two cards with one lane each, a job sampling on each of ``busy_cards``, a queued head and a job behind it.

        The head is eligible on both cards by configuration, and neither busy nor free can a card in
        ``unseatable_cards`` seat it.
        """
        clock = _Clock()
        bridge_data = make_mock_bridge_data()
        tracker = JobTracker()
        lanes: dict[int, object] = {}
        running: list[tuple[object, ImageGenerateJobPopResponse]] = []
        for card in range(2):
            busy = card in busy_cards
            lane = make_mock_process_info(
                card,
                model_name="sd" if busy else None,
                device_index=card,
                state=HordeProcessState.INFERENCE_STARTING if busy else HordeProcessState.WAITING_FOR_JOB,
            )
            lanes[card] = lane
            if busy:
                job = make_job_pop_response(model="sd")
                await track_popped_job_async(tracker, job)
                await mark_job_in_progress_async(tracker, job)
                running.append((lane, job))
        head = make_job_pop_response(model="flux")
        other = make_job_pop_response(model="sd15")
        await track_popped_job_async(tracker, head)
        await track_popped_job_async(tracker, other)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap(lanes),  # type: ignore[arg-type]
            job_tracker=tracker,
            bridge_data=bridge_data,
            max_inference=2,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), config=bridge_data),
            clock=clock,
        )
        for lane, job in running:
            lane.record_inference_ownership(job, attempt_ordinal=0)  # type: ignore[attr-defined]
        monkeypatch.setattr(scheduler, "_eligible_card_indices", lambda _job: {0, 1})
        monkeypatch.setattr(scheduler, "_admit_preload_under_budget", lambda *_args, **_kwargs: True)
        head_id = str(head.id_)

        def can_seat(
            snapshot: SchedulingSnapshot,
            job_id: str,
            process_id: int,
            *,
            when_free: bool = False,
        ) -> bool:
            del when_free
            return job_id != head_id or snapshot.slots[process_id].device_index not in unseatable_cards

        monkeypatch.setattr(inference_scheduler_module, "preload_slot_can_seat", can_seat)
        return scheduler, clock, head, other

    @staticmethod
    def _clock_runs(scheduler: InferenceScheduler, head: ImageGenerateJobPopResponse) -> bool:
        scheduler._update_head_starvation_timer(head, scheduler.snapshot())
        return scheduler.head_admission.starvation_job_id == str(head.id_)

    async def test_the_head_can_use_only_the_cards_that_can_seat_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The one definition drops the eligible card that cannot seat the head while another card can."""
        scheduler, _, head, other = await self._scheduler(
            unseatable_cards={1},
            busy_cards=(),
            monkeypatch=monkeypatch,
        )
        snapshot = scheduler.snapshot()

        assert cards_able_to_seat(snapshot, snapshot.queue.jobs[str(head.id_)]) == frozenset({0})
        assert cards_able_to_seat(snapshot, snapshot.queue.jobs[str(other.id_)]) == frozenset({0, 1})

    async def test_another_jobs_preload_on_a_card_that_cannot_seat_the_head_leaves_the_clock_running(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A load on a card the head can never take is no progress for the head."""
        scheduler, clock, head, other = await self._scheduler(
            unseatable_cards={1},
            busy_cards=(),
            monkeypatch=monkeypatch,
        )
        assert self._clock_runs(scheduler, head) is True
        clock.now += self._WAITED_SECONDS

        TestPreloadRestartsHeadStarvation._preload(scheduler, other, process_id=1)

        assert scheduler._head_starved_seconds(head) == self._WAITED_SECONDS

    async def test_a_job_on_a_card_that_cannot_seat_the_head_does_not_stop_the_clock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The card that can seat the head is idle, so the head is starved whatever runs on the other card."""
        scheduler, _, head, _ = await self._scheduler(unseatable_cards={1}, busy_cards=(1,), monkeypatch=monkeypatch)

        assert self._clock_runs(scheduler, head) is True

    async def test_a_job_on_the_card_that_can_seat_the_head_stops_the_clock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every card that can seat the head runs a job, so the head waits behind work that can free room."""
        scheduler, _, head, _ = await self._scheduler(unseatable_cards={1}, busy_cards=(0,), monkeypatch=monkeypatch)

        assert self._clock_runs(scheduler, head) is False

    async def test_a_head_no_card_can_seat_keeps_its_eligible_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no card able to seat the head, both eligible cards stop the clock and restart it, as before."""
        scheduler, clock, head, other = await self._scheduler(
            unseatable_cards={0, 1},
            busy_cards=(0,),
            monkeypatch=monkeypatch,
        )
        snapshot = scheduler.snapshot()
        assert cards_able_to_seat(snapshot, snapshot.queue.jobs[str(head.id_)]) == frozenset({0, 1})
        assert self._clock_runs(scheduler, head) is True
        clock.now += self._WAITED_SECONDS

        TestPreloadRestartsHeadStarvation._preload(scheduler, other, process_id=1)

        assert scheduler._head_starved_seconds(head) == 0.0


class TestRamDeferBarrier:
    """The RAM-defer clock latches the barrier past its bound and declines the head past the cap."""

    def test_first_defer_starts_the_clock_without_a_step(self) -> None:
        """A head seen deferring for the first time is ordinarily queued."""
        ledger, _ = _ledger()
        step = ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        assert step is HeadRamDeferStep.NONE
        assert ledger.ram_defer_job_id == "head"

    def test_engages_past_the_bound_unless_reclaim_grace_holds(self) -> None:
        """Past the bound with nothing freed the barrier engages; a reclaim cycle in its grace suppresses it."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS - 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        )
        clock.now += 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=True)
            is HeadRamDeferStep.NONE
        )
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.ENGAGE_BARRIER
        )

    def test_progress_restarts_the_clock_and_releases_a_held_barrier(self) -> None:
        """Reclaim progress means the head is not starving; a barrier held for it is released."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS
        assert ledger.engage_barrier("head") is True
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=True, reclaim_grace_active=False)
            is HeadRamDeferStep.RELEASE_BARRIER
        )
        assert ledger.ram_defer_since == clock.now
        assert ledger.release_barrier() == "head", "the release itself is the scheduler's actuation"
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=True, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        ), "progress with no barrier held asks for nothing"

    def test_a_new_head_restarts_the_clock(self) -> None:
        """A different head at the front is a fresh defer episode."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("first", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS * 2
        assert (
            ledger.govern_ram_defer("second", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        )
        assert ledger.ram_defer_since == clock.now

    def test_declines_past_the_cap(self) -> None:
        """A barrier that has held past its cap without admitting the head declines it."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        ledger.engage_barrier("head")
        clock.now += HEAD_RAM_DEFER_BARRIER_CAP_SECONDS - 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.ENGAGE_BARRIER
        )
        clock.now += 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.DECLINE_HEAD
        )

    def test_barrier_withholds_every_job_but_the_head(self) -> None:
        """While held, only the barred head may dispatch; engaging twice is a no-op; release reports the head."""
        ledger, _ = _ledger()
        assert ledger.barrier_withholds("sibling") is False
        assert ledger.engage_barrier("head") is True
        assert ledger.engage_barrier("head") is False
        assert ledger.barrier_withholds("sibling") is True
        assert ledger.barrier_withholds("head") is False
        assert ledger.release_barrier() == "head"
        assert ledger.release_barrier() is None
        assert ledger.barrier_withholds("sibling") is False

    def test_resolve_and_reconcile(self) -> None:
        """An admitted head clears its clock and reports its barrier; a departed head releases its barrier."""
        ledger, _ = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        ledger.engage_barrier("head")
        assert ledger.reconcile_to_head("head") is False
        assert ledger.resolve_ram_defer("other") is False
        assert ledger.ram_defer_job_id == "head"
        assert ledger.resolve_ram_defer("head") is True
        assert ledger.ram_defer_job_id is None
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        assert ledger.reconcile_to_head("next") is True
        assert ledger.ram_defer_job_id is None


class TestSafetyRecoveryHold:
    """The hold engages once, holds to its TTL, expires with a latch, and releases when the pool recovers."""

    def test_engage_hold_expire_and_latch(self) -> None:
        """One query engages, later ones hold, the TTL expires and latches the episode."""
        ledger, clock = _ledger()
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.ENGAGED
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.HOLDING
        clock.now += SAFETY_RECOVERY_HOLD_TTL_SECONDS
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.EXPIRED
        assert ledger.recovery_hold_since == 0.0 and ledger.recovery_hold_expired is True
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING

    def test_recovery_releases_and_clears_the_latch(self) -> None:
        """The pool recovering releases an engaged hold; the next episode may hold again."""
        ledger, clock = _ledger()
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.RELEASED
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        clock.now += SAFETY_RECOVERY_HOLD_TTL_SECONDS
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_expired is False
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.ENGAGED

    def test_another_card_neither_engages_nor_releases(self) -> None:
        """A query for a card that is not waiting on the safety start leaves the hold as it is."""
        ledger, _ = _ledger()
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=False) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_since == 0.0
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=False) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_since != 0.0


class TestRecordsAndThrottles:
    """Admission records, staging deferrals, the missing-model latch, the stall diagnostic and the heavy head."""

    def test_preload_admission_record_and_overlay(self) -> None:
        """The last decision is kept with the clock's stamp; the overlay reads worker-wide under key 0."""
        ledger, clock = _ledger()
        ledger.record_preload_admission(AdmissionDecision.ADMIT, job_id="j", model="m", process_id=3, reason="fits")
        record = ledger.last_preload_admission
        assert record is not None
        assert (record.decision, record.model, record.process_id, record.reason, record.timestamp) == (
            AdmissionDecision.ADMIT,
            "m",
            3,
            "fits",
            clock.now,
        )
        assert ledger.admission_headroom_mb(None) is None
        ledger.note_admission_headroom(None, 512.0)
        assert ledger.admission_headroom_mb(0) == 512.0
        assert ledger.admission_denials(None) == 0

    def test_a_later_jobs_decision_leaves_the_heads_decision_readable(self) -> None:
        """A later job's decision replaces the last record and leaves the head's decision readable by job id."""
        ledger, _clock = _ledger()
        ledger.begin_preload_pass()
        ledger.record_preload_admission(
            AdmissionDecision.EXCLUSIVE_IN_PROGRESS,
            job_id="head",
            model="m",
            process_id=0,
            reason="card 0 is held for an exclusive admit",
        )
        ledger.record_preload_admission(AdmissionDecision.ADMIT, job_id="next", model="m", process_id=1, reason="fits")

        head_record = ledger.preload_admission_for("head")
        assert head_record is not None
        assert head_record.decision is AdmissionDecision.EXCLUSIVE_IN_PROGRESS
        assert ledger.last_preload_admission is not None
        assert ledger.last_preload_admission.decision is AdmissionDecision.ADMIT

    def test_a_new_pass_forgets_the_previous_passes_decisions(self) -> None:
        """Beginning a pass drops the previous pass's decisions, so an earlier decision for a job is not readable."""
        ledger, _clock = _ledger()
        ledger.begin_preload_pass()
        ledger.record_preload_admission(
            AdmissionDecision.DEFER_BUDGET,
            job_id="head",
            model="m",
            process_id=0,
            reason="does not fit",
        )

        ledger.begin_preload_pass()

        assert ledger.preload_admission_for("head") is None
        assert ledger.last_preload_admission is not None, "the last record is kept for the harness's budget reason"

    def test_staging_defers_tally_and_throttle(self) -> None:
        """Every deferral is tallied; an unchanged reason is restated only after the repeat window."""
        ledger, clock = _ledger()
        assert ledger.note_staging_defer(StagingDeferReason.MEASUREMENT_UNREAD) == 0
        assert ledger.note_staging_defer(StagingDeferReason.MEASUREMENT_UNREAD) is None
        assert ledger.note_staging_defer(StagingDeferReason.ENCODE_HEADROOM_SHORT) == 1
        clock.now += STAGING_DEFER_REPEAT_SECONDS
        assert ledger.note_staging_defer(StagingDeferReason.ENCODE_HEADROOM_SHORT) == 0
        assert ledger.staging_defers == {
            StagingDeferReason.MEASUREMENT_UNREAD: 2,
            StagingDeferReason.ENCODE_HEADROOM_SHORT: 2,
        }
        assert format_staging_defer_tally(ledger.staging_defers) == "staging deferred: 4 (headroom 50%, unread 50%)"

    def test_missing_model_latch_expires_on_the_budget(self) -> None:
        """The latch holds inside the preload budget, expires past it, and a dispatch clears it early."""
        ledger, clock = _ledger()
        assert ledger.missing_model_latched(60.0) is False
        ledger.latch_missing_model()
        clock.now += 59.0
        assert ledger.missing_model_latched(60.0) is True
        clock.now += 1.0
        assert ledger.missing_model_latched(60.0) is False
        ledger.latch_missing_model()
        ledger.clear_missing_model()
        assert ledger.missing_model_latched(60.0) is False

    def test_stall_diagnostic_throttles_on_the_injected_clock(self) -> None:
        """A changed reason logs at once; an unchanged one waits out the interval on the ledger's clock."""
        ledger, clock = _ledger()
        assert ledger.note_stall("cap") is True
        assert ledger.note_stall("cap") is False
        assert ledger.stall_reason == "cap"
        assert ledger.note_stall("overlap") is True
        clock.now += DISPATCH_STALL_LOG_INTERVAL_SECONDS - 1.0
        assert ledger.note_stall("overlap") is False
        clock.now += 1.0
        assert ledger.note_stall("overlap") is True
        ledger.clear_stall()
        assert ledger.stall_reason is None

    def test_heavy_head_grace_is_bounded(self) -> None:
        """No admission means no grace; an admission opens it for exactly the window."""
        ledger, clock = _ledger()
        assert ledger.heavy_head_load_grace_active(30.0) is False
        ledger.note_heavy_head_admitted()
        clock.now += 29.0
        assert ledger.heavy_head_load_grace_active(30.0) is True
        clock.now += 1.0
        assert ledger.heavy_head_load_grace_active(30.0) is False


class TestMeasuredFloorDenials:
    """The run-metrics denial figure counts measured refusals of statically admitted candidates, per card."""

    def test_denials_count_per_card_and_read_zero_elsewhere(self) -> None:
        """Each note adds one under the card's key, the worker-wide view under key 0, and an unseen card reads 0."""
        ledger, _clock = _ledger()
        assert ledger.admission_denials(None) == 0
        ledger.note_measured_floor_denial(None)
        ledger.note_measured_floor_denial(1)
        ledger.note_measured_floor_denial(1)
        assert ledger.admission_denials(None) == 1
        assert ledger.admission_denials(0) == 1
        assert ledger.admission_denials(1) == 2
        assert ledger.admission_denials(2) == 0
