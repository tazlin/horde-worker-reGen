"""The pop claim's value verdict and its disclosure in the engage line and the stats stream.

The verdict compares the claimed model's kudos per card second with the rest of its card's, each with a band of
two standard errors. It is disclosed only, so these tests pin what is read and what is reported.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StatsSample,
    WholeCardResidencyStatus,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.card_earning import (
    CARD_EARNING_BUCKET_SECONDS,
    CardEarningAccumulator,
    CardEarningRate,
    CardTick,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    POP_CLAIM_VALUE_MIN_BUCKETS,
    POP_CLAIM_VALUE_MIN_JOBS,
    PopClaimValueVerdict,
    WholeCardPopClaim,
    WholeCardPopClaimTracker,
    pop_claim_value_verdict,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_test_card_runtimes
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

BUCKET_START = 1000.0 * CARD_EARNING_BUCKET_SECONDS
"""A tick time at a bucket boundary, so each filled bucket is one bucket."""

CLAIMED_MODEL = "Flux.1-Schnell fp8 (Compact)"
OTHER_MODEL = "AlbedoBase XL (SDXL)"


def _rate(
    *,
    model_rate: float | None = 1.0,
    model_error: float | None = 0.1,
    rest_rate: float | None = 1.0,
    rest_error: float | None = 0.1,
    jobs: int = POP_CLAIM_VALUE_MIN_JOBS,
    buckets: int = POP_CLAIM_VALUE_MIN_BUCKETS,
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


class TestVerdict:
    """The verdict reads the job floor, the two rates and their bands."""

    def test_under_the_job_floor_is_unmeasured(self) -> None:
        """A model with too few jobs is unmeasured even when its rate is far below the rest."""
        rate = _rate(model_rate=0.1, rest_rate=5.0, jobs=POP_CLAIM_VALUE_MIN_JOBS - 1)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.UNMEASURED

    def test_under_the_bucket_floor_is_unmeasured(self) -> None:
        """A spread read from too few time buckets is unmeasured even when the bands are far apart."""
        rate = _rate(model_rate=0.1, model_error=0.01, rest_rate=5.0, buckets=POP_CLAIM_VALUE_MIN_BUCKETS - 1)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.UNMEASURED

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
        """The model's band top 1.0 + 2 x 0.1 = 1.2 is under the rest's band bottom 2.0 - 2 x 0.2 = 1.6."""
        rate = _rate(model_rate=1.0, model_error=0.1, rest_rate=2.0, rest_error=0.2)
        assert pop_claim_value_verdict(rate) is PopClaimValueVerdict.BELOW_THE_REST

    def test_overlapping_bands_pay(self) -> None:
        """The model's band top 1.0 + 2 x 0.3 = 1.6 reaches the rest's band bottom 2.0 - 2 x 0.2 = 1.6."""
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
    """An accumulator with eight buckets on ``device_index``, the claimed model earning below the other model.

    Each bucket has 20 jobs of the claimed model over 100 s and one job of the other model over 100 s. The
    claimed model earns 0.5 and 0.6 kudos per second in alternate buckets, and the other model 2.0 and 2.2.
    """
    accumulator = CardEarningAccumulator(max_tick_seconds=1000.0, clock=lambda: BUCKET_START)
    for bucket_number in range(POP_CLAIM_VALUE_MIN_BUCKETS):
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
        assert f"over {POP_CLAIM_VALUE_MIN_BUCKETS * POP_CLAIM_VALUE_MIN_JOBS} jobs of the claimed model" in line
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
        """A held grant claims the offer, and the status reports the claim's card and verdict."""
        scheduler = _scheduler(card_index=0, accumulator=_measured_accumulator(0))
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
