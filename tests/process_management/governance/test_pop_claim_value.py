"""The pop claim's value verdict, the policy that withholds a claim on it, and their disclosure.

The verdict compares the claimed model's kudos per card second with the rest of its card's, each with its 95%
band. Under the default policy a verdict of below the rest withholds the claim, so these tests pin what is
read, what is decided, which readers follow the decision and what is reported.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger

from horde_worker_regen.bridge_data.data_model import ModelPoolConfig, PinnedModelEntry
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StatsSample,
    WholeCardResidencyStatus,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.card_earning import (
    CARD_EARNING_BUCKET_SECONDS,
    CARD_EARNING_READ_REUSE_SECONDS,
    CardEarningAccumulator,
    CardEarningRate,
    CardTick,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    POP_CLAIM_VALUE_MIN_JOBS,
    DominanceRung,
    PopClaimExemption,
    PopClaimHold,
    PopClaimHoldDecision,
    PopClaimHoldReason,
    PopClaimValue,
    PopClaimValueVerdict,
    WholeCardClaimPolicy,
    WholeCardPopClaim,
    WholeCardPopClaimTracker,
    pop_claim_hold_decision,
    pop_claim_value_verdict,
    two_sided_95_band_width,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_admission_pricing import _slot, _worker
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

BUCKET_START = 1000.0 * CARD_EARNING_BUCKET_SECONDS
"""A tick time at a bucket boundary, so each filled bucket is one bucket."""

CLAIMED_MODEL = "Flux.1-Schnell fp8 (Compact)"
OTHER_MODEL = "AlbedoBase XL (SDXL)"

_MANY_BUCKETS = 31
"""Thirty degrees of freedom, where the band is 2.042 standard errors wide."""

_MEASURED_BUCKETS = 8
"""How many time buckets the measured accumulator fills."""


def _rate(
    *,
    model_rate: float | None = 1.0,
    model_error: float | None = 0.1,
    rest_rate: float | None = 1.0,
    rest_error: float | None = 0.1,
    jobs: int = POP_CLAIM_VALUE_MIN_JOBS,
    buckets: int = _MANY_BUCKETS,
) -> CardEarningRate:
    return CardEarningRate(
        model_kudos_per_second=model_rate,
        rest_kudos_per_second=rest_rate,
        jobs=jobs,
        model_seconds=100.0,
        card_wall_seconds=200.0,
        model_standard_error=model_error,
        rest_standard_error=rest_error,
        model_buckets=buckets,
        rest_buckets=buckets,
    )


class TestBandWidth:
    """The band is wide when the standard error comes from few readings and narrows toward the normal value."""

    @pytest.mark.parametrize(
        ("readings", "width"),
        [(2, 12.706), (4, 3.182), (8, 2.365), (31, 2.042), (61, 2.000), (96, 1.985)],
    )
    def test_matches_the_published_values(self, readings: int, width: float) -> None:
        """The listed values and the expansion past them agree with the published table."""
        assert two_sided_95_band_width(readings) == pytest.approx(width, abs=0.003)

    def test_narrows_as_readings_grow(self) -> None:
        """More readings do not widen the band, across the change from listed values to the expansion."""
        widths = [two_sided_95_band_width(readings) for readings in range(2, 120)]
        assert widths == sorted(widths, reverse=True)

    def test_one_reading_is_refused(self) -> None:
        """One reading does not have a standard error."""
        with pytest.raises(ValueError, match="at least two readings"):
            two_sided_95_band_width(1)


class TestVerdict:
    """The verdict reads the job floor, the two rates and their bands."""

    def test_under_the_job_floor_is_unmeasured(self) -> None:
        """A model with too few jobs is unmeasured even when its rate is far below the rest."""
        rate = _rate(model_rate=0.1, rest_rate=5.0, jobs=POP_CLAIM_VALUE_MIN_JOBS - 1)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.UNMEASURED

    def test_one_bucket_is_unmeasured(self) -> None:
        """A rate read from one time bucket does not have a spread, however far apart the rates are."""
        rate = _rate(model_rate=0.1, model_error=0.01, rest_rate=5.0, buckets=1)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.UNMEASURED

    def test_the_same_gap_needs_more_buckets_to_rule_below(self) -> None:
        """With two buckets the band is 12.7 standard errors wide and the bands overlap. With many they do not."""
        few = _rate(model_rate=1.0, model_error=0.1, rest_rate=2.0, rest_error=0.2, buckets=2)
        many = _rate(model_rate=1.0, model_error=0.1, rest_rate=2.0, rest_error=0.2, buckets=_MANY_BUCKETS)
        assert pop_claim_value_verdict(few) is PopClaimValueVerdict.PAYS
        assert pop_claim_value_verdict(many) is PopClaimValueVerdict.BELOW_THE_REST

    @pytest.mark.parametrize(
        ("model_rate", "model_error", "rest_rate", "rest_error"),
        [
            (1.0, None, 1.0, 0.1),
            (1.0, 0.1, 1.0, None),
            (None, None, 1.0, 0.1),
            (1.0, 0.1, None, None),
        ],
    )
    def test_a_missing_rate_or_spread_is_unmeasured(
        self,
        model_rate: float | None,
        model_error: float | None,
        rest_rate: float | None,
        rest_error: float | None,
    ) -> None:
        """A rate or a standard error without a reading leaves the verdict unmeasured."""
        rate = _rate(model_rate=model_rate, model_error=model_error, rest_rate=rest_rate, rest_error=rest_error)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.UNMEASURED

    def test_bands_apart_rule_below_the_rest(self) -> None:
        """The model's band top 1.0 + 2.042 x 0.1 is 1.204, under the rest's band bottom 2.0 - 2.042 x 0.2 at 1.592."""
        rate = _rate(model_rate=1.0, model_error=0.1, rest_rate=2.0, rest_error=0.2)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.BELOW_THE_REST

    def test_overlapping_bands_pay(self) -> None:
        """The model's band top 1.0 + 2.042 x 0.3 is 1.613, over the rest's band bottom 2.0 - 2.042 x 0.2 at 1.592."""
        rate = _rate(model_rate=1.0, model_error=0.3, rest_rate=2.0, rest_error=0.2)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.PAYS

    def test_a_model_rate_above_the_rest_pays(self) -> None:
        """A model that earns more than the rest pays, however narrow both bands are."""
        rate = _rate(model_rate=3.0, model_error=0.0, rest_rate=1.0, rest_error=0.0)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.PAYS


@pytest.fixture
def info_messages() -> Iterator[list[str]]:
    """Collect the INFO-and-above loguru messages written during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _measured_accumulator(device_index: int) -> CardEarningAccumulator:
    """An accumulator with eight time buckets on ``device_index``, the claimed model earning below the other model.

    Each bucket has 20 jobs of the claimed model over 100 s and one job of the other model over 100 s. The
    claimed model earns 0.5 and 0.6 kudos per second in alternate buckets, and the other model 2.0 and 2.2.
    """
    accumulator = CardEarningAccumulator(max_tick_seconds=1000.0, clock=lambda: BUCKET_START)
    for bucket_number in range(_MEASURED_BUCKETS):
        start = BUCKET_START + 900.0 * bucket_number
        claimed_kudos, other_kudos = (50.0, 200.0) if bucket_number % 2 == 0 else (60.0, 220.0)
        accumulator.observe(start, {})
        accumulator.observe(start + 200.0, {device_index: CardTick(in_flight_models=(CLAIMED_MODEL, OTHER_MODEL))})
        for _ in range(POP_CLAIM_VALUE_MIN_JOBS):
            accumulator.credit_finished_job(
                device_index=device_index,
                model=CLAIMED_MODEL,
                kudos=claimed_kudos / POP_CLAIM_VALUE_MIN_JOBS,
            )
        accumulator.credit_finished_job(device_index=device_index, model=OTHER_MODEL, kudos=other_kudos)
    return accumulator


def _scheduler(*, card_index: int, accumulator: CardEarningAccumulator | None) -> InferenceScheduler:
    """A scheduler on a single-card host whose one card has ``card_index``."""
    scheduler = _make_inference_scheduler(card_runtimes=make_test_card_runtimes(device_indices=(card_index,)))
    scheduler._card_earning = accumulator
    return scheduler


def _claim(device_index: int | None) -> WholeCardPopClaim:
    return WholeCardPopClaim(model=CLAIMED_MODEL, device_index=device_index, held_since=100.0, expires_at=280.0)


def _engage_line(scheduler: InferenceScheduler, claim: WholeCardPopClaim, info_messages: list[str]) -> str:
    WholeCardPopClaimTracker().disclose_edge(claim, now=100.0, value_of=scheduler._pop_claim_value)
    engage_lines = [message for message in info_messages if "Whole-card pop claim engaged" in message]
    assert len(engage_lines) == 1
    return engage_lines[0]


class TestEngageLine:
    """The engage line states both rates, the job count and the verdict, read on the claim's card."""

    @pytest.mark.parametrize("card_index", [0, 1])
    def test_a_single_card_claim_reads_the_lane_card(self, card_index: int, info_messages: list[str]) -> None:
        """The ledger keys the claim by None, and the line reads the earning store under the card's index."""
        scheduler = _scheduler(card_index=card_index, accumulator=_measured_accumulator(card_index))

        line = _engage_line(scheduler, _claim(None), info_messages)

        assert f"Earning on card {card_index}, in kudos per card second: 0.550 for the claimed model" in line
        assert "2.100 for the rest of the card" in line
        assert f"over {_MEASURED_BUCKETS * POP_CLAIM_VALUE_MIN_JOBS} jobs of the claimed model" in line
        assert line.endswith("The claimed model earns below the rest of the card.")

    def test_a_claim_keyed_by_a_card_reads_that_card(self, info_messages: list[str]) -> None:
        """A claim with a card index reads that card, not the first card runtime."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(2))

        line = _engage_line(scheduler, _claim(2), info_messages)

        assert "Earning on card 2, in kudos per card second: 0.550" in line

    def test_a_claim_under_the_job_floor_is_unmeasured(self, info_messages: list[str]) -> None:
        """An unmeasured verdict states the jobs counted and the floor it waits for."""
        accumulator = CardEarningAccumulator(max_tick_seconds=1000.0, clock=lambda: BUCKET_START)
        accumulator.observe(BUCKET_START, {})
        accumulator.observe(BUCKET_START + 50.0, {0: CardTick(in_flight_models=(CLAIMED_MODEL,))})
        accumulator.credit_finished_job(device_index=0, model=CLAIMED_MODEL, kudos=25.0)
        scheduler = _scheduler(card_index=0, accumulator=accumulator)

        line = _engage_line(scheduler, _claim(None), info_messages)

        assert "0.500 for the claimed model, unmeasured for the rest of the card, over 1 job of" in line
        assert line.endswith(f"The claim's value is unmeasured until {POP_CLAIM_VALUE_MIN_JOBS} jobs are counted.")

    def test_without_an_accumulator_the_claim_is_unmeasured(self, info_messages: list[str]) -> None:
        """A worker that does not measure card earning says so on the engage line."""
        scheduler = _scheduler(card_index=0, accumulator=None)

        line = _engage_line(scheduler, _claim(None), info_messages)

        assert scheduler._pop_claim_value(_claim(None)).verdict is PopClaimValueVerdict.UNMEASURED
        assert line.endswith("Card earning is not measured on this worker. The claim's value is unmeasured.")


class TestStatsSample:
    """The stats sample records the standing claim with its verdict, and nothing without one."""

    def test_the_residency_state_includes_the_standing_claim_verdict(self) -> None:
        """A held grant claims the offer, and the status reports the claim's card and verdict.

        The policy is ``always``, so the below-the-rest claim stands and its verdict is reported beside it.
        """
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        scheduler._runtime_config.bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS
        now = scheduler._clock()
        scheduler._whole_card_ledger.record_grant(
            None,
            model=CLAIMED_MODEL,
            forecast=None,
            cooldown_until=now + 300.0,
            now=now,
        )

        state = scheduler.whole_card_residency_state()

        assert state.pop_claim_model == CLAIMED_MODEL
        assert state.pop_claim_card_index == 0
        assert state.pop_claim_verdict == PopClaimValueVerdict.BELOW_THE_REST.value
        assert state.pop_claim_hold_reason == PopClaimHoldReason.KEPT_BY_POLICY.value
        assert state.pop_claim_withheld_model is None

    def test_the_residency_state_reports_a_withheld_claim_and_why(self) -> None:
        """Under the default policy the below-the-rest claim is withheld, and the status names the model and why."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        _hold_the_card(scheduler)

        state = scheduler.whole_card_residency_state()

        assert state.pop_claim_model is None
        assert state.pop_claim_withheld_model == CLAIMED_MODEL
        assert state.pop_claim_hold_reason == PopClaimHoldReason.WITHHELD_BELOW_THE_REST.value
        assert state.pop_claim_card_index == 0
        assert state.pop_claim_verdict == PopClaimValueVerdict.BELOW_THE_REST.value
        assert state.pop_claim_release is None
        residency = WholeCardResidencyStatus(
            pop_claim_withheld_model=state.pop_claim_withheld_model,
            pop_claim_card_index=state.pop_claim_card_index,
            pop_claim_verdict=state.pop_claim_verdict,
        )
        assert HordeWorkerProcessManager._standing_pop_claim_samples(residency) == [], (
            "a withheld claim does not stand, so the stats sample does not list it"
        )

    def test_a_standing_claim_is_written_to_the_sample(self) -> None:
        """The sample has one entry with the claim's card, model and verdict."""
        residency = WholeCardResidencyStatus(
            pop_claim_model=CLAIMED_MODEL,
            pop_claim_card_index=0,
            pop_claim_verdict=PopClaimValueVerdict.PAYS.value,
        )

        sample = StatsSample(standing_pop_claims=HordeWorkerProcessManager._standing_pop_claim_samples(residency))

        dumped = sample.model_dump(mode="json")["standing_pop_claims"]
        assert dumped == [{"device_index": 0, "model": CLAIMED_MODEL, "verdict": "pays"}]

    def test_no_claim_leaves_the_sample_empty(self) -> None:
        """Without a standing claim the sample's claim list is empty."""
        residency = WholeCardResidencyStatus(pop_claim_release="maximum_hold")

        assert HordeWorkerProcessManager._standing_pop_claim_samples(residency) == []
        assert StatsSample().standing_pop_claims == []


def _hold_the_card(scheduler: InferenceScheduler, *, rung: DominanceRung = DominanceRung.CANNOT_CO_RESIDE) -> None:
    """Record a grant for the claimed model on the single card at the scheduler's clock."""
    now = scheduler._clock()
    scheduler._whole_card_ledger.record_grant(
        None,
        model=CLAIMED_MODEL,
        forecast=None,
        cooldown_until=now + 300.0,
        now=now,
        rung=rung,
    )


def _no_exemption() -> PopClaimExemption | None:
    return None


class TestHoldDecision:
    """Only a below-the-rest verdict withholds a claim, and the policy and the exemptions keep it."""

    @pytest.mark.parametrize("verdict", [PopClaimValueVerdict.UNMEASURED, PopClaimValueVerdict.PAYS])
    @pytest.mark.parametrize("policy", list(WholeCardClaimPolicy))
    def test_a_verdict_that_is_not_below_keeps_the_claim(
        self,
        verdict: PopClaimValueVerdict,
        policy: WholeCardClaimPolicy,
    ) -> None:
        """An unmeasured or paying model keeps its claim under either policy, and no exemption is read."""

        def exemption() -> PopClaimExemption | None:
            raise AssertionError("an exemption is read only for a model that earns below the rest")

        decision = pop_claim_hold_decision(policy=policy, verdict=verdict, exemption=exemption)

        assert decision == PopClaimHoldDecision(reason=PopClaimHoldReason.NOT_WITHHELD)
        assert decision.withheld is False

    def test_below_the_rest_is_withheld_under_the_default_policy(self) -> None:
        """A below-the-rest model without an exemption does not get a claim."""
        decision = pop_claim_hold_decision(
            policy=WholeCardClaimPolicy.WHEN_IT_PAYS,
            verdict=PopClaimValueVerdict.BELOW_THE_REST,
            exemption=_no_exemption,
        )

        assert decision.reason is PopClaimHoldReason.WITHHELD_BELOW_THE_REST
        assert decision.withheld is True

    def test_the_always_policy_keeps_a_below_rest_claim(self) -> None:
        """The policy ``always`` keeps the claim and is named as the reason."""
        decision = pop_claim_hold_decision(
            policy=WholeCardClaimPolicy.ALWAYS,
            verdict=PopClaimValueVerdict.BELOW_THE_REST,
            exemption=lambda: PopClaimExemption.LISTED,
        )

        assert decision == PopClaimHoldDecision(reason=PopClaimHoldReason.KEPT_BY_POLICY)

    @pytest.mark.parametrize("exemption", list(PopClaimExemption))
    def test_an_exemption_keeps_a_below_rest_claim(self, exemption: PopClaimExemption) -> None:
        """Each exemption keeps the claim and is named in the decision."""
        decision = pop_claim_hold_decision(
            policy=WholeCardClaimPolicy.WHEN_IT_PAYS,
            verdict=PopClaimValueVerdict.BELOW_THE_REST,
            exemption=lambda: exemption,
        )

        assert decision == PopClaimHoldDecision(reason=PopClaimHoldReason.KEPT_BY_EXEMPTION, exemption=exemption)
        assert decision.withheld is False


class TestSchedulerExemptions:
    """The scheduler reads each exemption from what already says the model is served on purpose."""

    def test_a_listed_model_is_exempt(self) -> None:
        """A model in ``whole_card_claim_always_models`` keeps its claim."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        scheduler._runtime_config.bridge_data.whole_card_claim_always_models = [CLAIMED_MODEL]
        _hold_the_card(scheduler)

        assert scheduler.whole_card_pop_claim() is not None
        assert scheduler._pop_claim_exemption(CLAIMED_MODEL) is PopClaimExemption.LISTED

    def test_a_pin_counts_only_while_the_pool_is_enabled(self) -> None:
        """A pinned model keeps its claim while the model pool is on, and not while it is off."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        pool = ModelPoolConfig(enabled=False, pinned=[PinnedModelEntry(name=CLAIMED_MODEL)])
        scheduler._runtime_config.bridge_data.model_pool = pool
        _hold_the_card(scheduler)

        assert scheduler.whole_card_pop_claim() is None, "a pin of a disabled pool does not seat the model"

        scheduler._runtime_config.bridge_data.model_pool = pool.model_copy(update={"enabled": True})

        assert scheduler.whole_card_pop_claim() is not None
        assert scheduler._pop_claim_exemption(CLAIMED_MODEL) is PopClaimExemption.POOL_PIN

    def test_a_model_the_card_fits_only_at_a_reduced_size_is_exempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The constrained lane's own test exempts a model the card fits only at a reduced size."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        _hold_the_card(scheduler)
        assert scheduler.whole_card_pop_claim() is None, "precondition, the claim is withheld without the lane"

        monkeypatch.setattr(
            "horde_worker_regen.process_management.scheduling.inference_scheduler.constrained_max_power",
            lambda _verdicts: 256,
        )

        assert scheduler.whole_card_pop_claim() is not None
        assert scheduler._pop_claim_exemption(CLAIMED_MODEL) is PopClaimExemption.CONSTRAINED_LANE


class TestReadersFollowTheWithheldClaim:
    """The offer, the resident bypass and the run copy's retention all read the claim at one point."""

    async def _bypass_scheduler(self, *, accumulator: CardEarningAccumulator) -> tuple[InferenceScheduler, object]:
        """A cold head, a resident job behind it, and a run held for the claimed model on the card."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({1: holder}), job_tracker=job_tracker)
        scheduler._card_earning = accumulator
        _hold_the_card(scheduler, rung=DominanceRung.CANNOT_CO_SAMPLE)
        return scheduler, bypass_job

    async def test_the_resident_bypass_follows_a_withheld_claim(self) -> None:
        """A standing claim lets only its model take the resident bypass. A withheld one lets another model pass.

        The run is held for the claimed model while a job of another resident model waits behind a cold head.
        Under the always policy the claim stands and the resident job is held back. Under the default policy the
        claim is withheld, and the same job passes the head.
        """
        standing, _job = await self._bypass_scheduler(accumulator=_measured_accumulator(0))
        standing._runtime_config.bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS
        assert standing.whole_card_pop_claim() is not None
        assert await standing.get_next_job_and_process() is None

        withheld, bypass_job = await self._bypass_scheduler(accumulator=_measured_accumulator(0))
        assert withheld.whole_card_pop_claim() is None

        result = await withheld.get_next_job_and_process()

        assert result is not None
        assert result.next_job is bypass_job
        assert result.line_skip is not None
        assert result.line_skip.reason == "resident_bypass"

    async def test_the_run_copy_follows_a_withheld_claim(self) -> None:
        """Without a queued job, the run's copy is spared by a standing claim and not by a withheld one."""
        run_lane = _slot(1, model=CLAIMED_MODEL)
        scheduler, _jobs = await _worker(slots={0: _slot(0, model=None), 1: run_lane}, pending=["other"])
        scheduler._card_earning = _measured_accumulator(0)
        _hold_the_card(scheduler, rung=DominanceRung.CANNOT_CO_SAMPLE)

        scheduler._runtime_config.bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS
        standing = scheduler.snapshot()
        assert standing.pop_claim_model == CLAIMED_MODEL
        assert pricing.copy_belongs_to_dominant_run(standing, CLAIMED_MODEL, None) is True

        scheduler._runtime_config.bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.WHEN_IT_PAYS
        withheld = scheduler.snapshot()
        assert withheld.pop_claim_model is None
        assert pricing.copy_belongs_to_dominant_run(withheld, CLAIMED_MODEL, None) is False


def _below_hold() -> PopClaimHold:
    rate = _measured_accumulator(0).rate(model=CLAIMED_MODEL, device_index=0)
    return PopClaimHold(
        claim=_claim(None),
        value=PopClaimValue(card_index=0, rate=rate, verdict=PopClaimValueVerdict.BELOW_THE_REST),
        decision=PopClaimHoldDecision(reason=PopClaimHoldReason.WITHHELD_BELOW_THE_REST),
    )


class TestWithheldDisclosure:
    """The withheld line is logged once per model, and a kept claim says why on its engage line."""

    def test_the_withheld_line_names_both_rates_and_the_settings(self, info_messages: list[str]) -> None:
        """The line states the model, both rates, that the other models are asked for, the return and the settings."""
        tracker = WholeCardPopClaimTracker()

        tracker.disclose_edge(None, now=100.0, withheld=_below_hold(), low_earning_return_seconds=180.0)
        tracker.disclose_edge(None, now=101.0, withheld=_below_hold(), low_earning_return_seconds=180.0)

        assert info_messages == [
            f"Not holding the card for {CLAIMED_MODEL}, since it earns below the rest of the card. On card 0 it "
            f"earns 0.550 kudos per card second, and the rest of the card's work earns 2.100. The worker asks the "
            f"horde for the other models, and asks for {CLAIMED_MODEL} again 180s after its jobs finish "
            f"(whole_card_low_earning_return_seconds). To ask the horde only for {CLAIMED_MODEL} while it holds the "
            "card, set whole_card_claim_policy to always, or add the model to whole_card_claim_always_models."
        ]

    def test_the_withheld_line_needs_the_return_interval(self) -> None:
        """A withheld claim disclosed without the return interval raises, so the line never states a wrong one."""
        with pytest.raises(ValueError, match="low_earning_return_seconds"):
            WholeCardPopClaimTracker().disclose_edge(None, now=100.0, withheld=_below_hold())

    def test_a_standing_claim_withheld_later_is_not_logged_as_released(self, info_messages: list[str]) -> None:
        """A claim that stood and is then withheld for the same model logs the withheld line and no release."""
        tracker = WholeCardPopClaimTracker()
        tracker.disclose_edge(_claim(None), now=100.0)

        tracker.disclose_edge(None, now=150.0, withheld=_below_hold(), low_earning_return_seconds=180.0)

        assert not any("released" in message for message in info_messages)
        assert any(message.startswith("Not holding the card for") for message in info_messages)
        assert tracker.recent_release(150.0) is None

    @pytest.mark.parametrize(
        ("decision", "sentence"),
        [
            (
                PopClaimHoldDecision(reason=PopClaimHoldReason.KEPT_BY_POLICY),
                "The card is held for it because whole_card_claim_policy is always.",
            ),
            (
                PopClaimHoldDecision(reason=PopClaimHoldReason.KEPT_BY_EXEMPTION, exemption=PopClaimExemption.LISTED),
                "The card is held for it because it is listed in whole_card_claim_always_models.",
            ),
            (
                PopClaimHoldDecision(
                    reason=PopClaimHoldReason.KEPT_BY_EXEMPTION,
                    exemption=PopClaimExemption.POOL_PIN,
                ),
                "The card is held for it because it is pinned to a model pool seat.",
            ),
        ],
    )
    def test_a_kept_claim_says_why_on_its_engage_line(
        self,
        decision: PopClaimHoldDecision,
        sentence: str,
        info_messages: list[str],
    ) -> None:
        """A below-the-rest claim kept by the policy or an exemption ends its engage line with the reason."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))

        WholeCardPopClaimTracker().disclose_edge(
            _claim(None),
            now=100.0,
            value_of=scheduler._pop_claim_value,
            decision_of=lambda _claim: decision,
        )

        (line,) = [message for message in info_messages if "Whole-card pop claim engaged" in message]
        assert line.endswith(f"The claimed model earns below the rest of the card. {sentence}")

    def test_the_scheduler_logs_the_withheld_line_at_its_release_pass(self, info_messages: list[str]) -> None:
        """The scheduler's edge disclosure logs the withheld line, not an engage line, for a below-rest grant."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        _hold_the_card(scheduler)

        scheduler._disclose_pop_claim_edge()

        assert not any("Whole-card pop claim engaged" in message for message in info_messages)
        assert [message for message in info_messages if message.startswith("Not holding the card for")]


class TestPacingInputs:
    """The two readings the job popper's large-model governor takes from the scheduler to pace a model."""

    def test_a_withheld_claim_names_its_model(self) -> None:
        """A below-the-rest grant's withheld claim names the model to pace."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        _hold_the_card(scheduler)

        assert scheduler.withheld_pop_claim_model() == CLAIMED_MODEL
        assert scheduler.pop_claim_pacing_exempt(CLAIMED_MODEL) is False

    def test_a_listed_model_and_the_always_policy_are_exempt(self) -> None:
        """A listed model, and any model under policy ``always``, are not paced."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
        _hold_the_card(scheduler)
        bridge_data = scheduler._runtime_config.bridge_data

        bridge_data.whole_card_claim_always_models = [CLAIMED_MODEL]
        assert scheduler.pop_claim_pacing_exempt(CLAIMED_MODEL) is True
        assert scheduler.withheld_pop_claim_model() is None

        bridge_data.whole_card_claim_always_models = []
        bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS
        assert scheduler.pop_claim_pacing_exempt(CLAIMED_MODEL) is True
        assert scheduler.withheld_pop_claim_model() is None


class TestValueReuse:
    """The claim is read several times per tick, so its measured value is reused for a few seconds."""

    def test_a_recent_reading_is_reused_and_an_old_one_is_read_again(self) -> None:
        """A job counted after the first read shows only once the reuse interval has passed."""
        accumulator = _measured_accumulator(0)
        scheduler = _scheduler(card_index=0, accumulator=accumulator)
        clock = [BUCKET_START + 9 * CARD_EARNING_BUCKET_SECONDS]
        scheduler._clock = lambda: clock[0]
        first = scheduler._pop_claim_value(_claim(None)).rate
        assert first is not None
        accumulator.credit_finished_job(device_index=0, model=CLAIMED_MODEL, kudos=1.0)

        reused = scheduler._pop_claim_value(_claim(None)).rate
        clock[0] += CARD_EARNING_READ_REUSE_SECONDS
        read_again = scheduler._pop_claim_value(_claim(None)).rate

        assert reused is not None and reused.jobs == first.jobs
        assert read_again is not None and read_again.jobs == first.jobs + 1
