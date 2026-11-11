"""Card earning: attributed card seconds and kudos per card and per model.

The accumulator's figures rest on two properties pinned here. A card's attributed seconds never exceed its
wall, however many lanes share it, so kudos over attributed seconds is a card rate. And the persisted window
loads back only the entries that could have been written by a healthy worker inside the retention, under the
same capacity and workload settings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CapacityWorkloadSettings,
    capacity_workload_fingerprint,
)
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.card_earning import (
    CARD_EARNING_BUCKET_SECONDS,
    CARD_EARNING_RETENTION_SECONDS,
    CARD_EARNING_SCHEMA_VERSION,
    CardEarningAccumulator,
    CardTick,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

BUCKET_START = 1000.0 * CARD_EARNING_BUCKET_SECONDS
"""A tick time at a bucket boundary, so a worked example stays inside one bucket."""

GENEROUS_TICK_SECONDS = 1000.0
"""A tick clamp above every step the worked examples take, so only the clamp test is clamped."""


def _settings(*, max_threads: int = 1, max_power: int = 8) -> CapacityWorkloadSettings:
    return CapacityWorkloadSettings(
        num_models=3,
        max_power=max_power,
        max_threads=max_threads,
        queue_size=1,
        max_batch=1,
        safety_on_gpu=True,
        allow_img2img=True,
        allow_lora=False,
        allow_controlnet=False,
        allow_sdxl_controlnet=False,
        allow_post_processing=False,
        high_performance_mode=False,
        moderate_performance_mode=False,
        extra_slow_worker=False,
        alchemist=False,
        alchemy_concurrent=False,
        alchemy_max_concurrency=1,
        alchemy_vram_headroom_mb=0,
        alchemy_forms=(),
    )


SETTINGS = _settings()
OTHER_SETTINGS = _settings(max_threads=2)
FINGERPRINT = capacity_workload_fingerprint(SETTINGS)


@pytest.fixture
def info_messages() -> Iterator[list[str]]:
    """Collect the INFO-and-above loguru messages written during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _accumulator(
    *,
    max_tick_seconds: float = GENEROUS_TICK_SECONDS,
    path: Path | None = None,
    settings: CapacityWorkloadSettings = SETTINGS,
) -> CardEarningAccumulator:
    return CardEarningAccumulator(
        max_tick_seconds=max_tick_seconds,
        settings=settings,
        path=path,
        clock=lambda: BUCKET_START,
    )


def _accumulator_with_one_job(*, path: Path | None = None) -> CardEarningAccumulator:
    """An accumulator whose window holds 30 seconds and 12 kudos of model A on card 0."""
    accumulator = _accumulator(path=path)
    accumulator.observe(BUCKET_START, {0: CardTick()})
    accumulator.observe(BUCKET_START + 30.0, {0: CardTick(in_flight_models=("A",))})
    accumulator.credit_finished_job(device_index=0, model="A", kudos=12.0)
    return accumulator


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _lane(process_id: int, *, device_index: int = 0, model: str | None = None) -> HordeProcessInfo:
    """An inference lane on ``device_index``, owning a dispatched job for ``model`` when one is given."""
    process_info = make_mock_process_info(process_id, model_name=model, device_index=device_index)
    if model is not None:
        process_info.record_inference_ownership(make_job_pop_response(model), attempt_ordinal=0)
    return process_info


def _scheduler(
    lanes: list[HordeProcessInfo],
    *,
    clock: _Clock,
    device_indices: tuple[int, ...] = (0,),
) -> tuple[InferenceScheduler, CardEarningAccumulator]:
    accumulator = _accumulator()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap({lane.process_id: lane for lane in lanes}),
        card_runtimes=make_test_card_runtimes(device_indices=device_indices),
        max_concurrent=2,
        clock=clock,
    )
    scheduler._card_earning = accumulator
    return scheduler, accumulator


def _tick_twice(scheduler: InferenceScheduler, clock: _Clock, elapsed: float) -> None:
    """Anchor the accumulator's clock, then let ``elapsed`` seconds pass to the next tick."""
    scheduler.record_slot_duty({})
    clock.now += elapsed
    scheduler.record_slot_duty({})


class TestTickAttribution:
    """The scheduler's tick gives each card's time to the models working on it."""

    def test_two_lanes_with_different_models_split_one_card_tick(self) -> None:
        """Two lanes share the card's tick equally, and the card's wall is the elapsed time once."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1, model="A"), _lane(2, model="B")], clock=clock)

        _tick_twice(scheduler, clock, 4.0)

        rate_a = accumulator.rate(model="A", device_index=0)
        rate_b = accumulator.rate(model="B", device_index=0)
        assert rate_a.model_seconds == pytest.approx(2.0)
        assert rate_b.model_seconds == pytest.approx(2.0)
        assert rate_a.card_wall_seconds == pytest.approx(4.0)

    def test_one_lane_with_a_job_takes_the_whole_tick(self) -> None:
        """An idle sibling lane does not take a share of the card's time."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1, model="A"), _lane(2)], clock=clock)

        _tick_twice(scheduler, clock, 3.0)

        rate = accumulator.rate(model="A", device_index=0)
        assert rate.model_seconds == pytest.approx(3.0)
        assert rate.card_wall_seconds == pytest.approx(3.0)

    def test_a_held_card_with_no_job_in_flight_credits_the_held_model(self) -> None:
        """A whole-card grant on a single-card host is keyed by ``None`` and lands on card 0."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1), _lane(2)], clock=clock)
        scheduler._whole_card_ledger.state_for(None).model = "Flux"

        _tick_twice(scheduler, clock, 5.0)

        rate = accumulator.rate(model="Flux", device_index=0)
        assert rate.model_seconds == pytest.approx(5.0)
        assert rate.card_wall_seconds == pytest.approx(5.0)

    def test_an_idle_card_without_a_grant_grows_only_its_wall(self) -> None:
        """No job and no grant leaves the time in the card's wall alone."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1)], clock=clock)

        _tick_twice(scheduler, clock, 5.0)

        rate = accumulator.rate(model="stable_diffusion", device_index=0)
        assert rate.model_seconds == 0.0
        assert rate.card_wall_seconds == pytest.approx(5.0)
        assert accumulator.session_seconds_by_model() == {}

    def test_a_job_in_flight_outranks_the_grant(self) -> None:
        """While a lane runs a job the card's time goes to that job's model, not to the held model."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1, model="A")], clock=clock)
        scheduler._whole_card_ledger.state_for(None).model = "Flux"

        _tick_twice(scheduler, clock, 2.0)

        assert accumulator.rate(model="A", device_index=0).model_seconds == pytest.approx(2.0)
        assert accumulator.rate(model="Flux", device_index=0).model_seconds == 0.0

    def test_two_cards_accumulate_independently(self) -> None:
        """On a multi-card host each card has its own wall, and the ledger is read by the card's own index."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler(
            [_lane(1, device_index=0, model="A"), _lane(2, device_index=0, model="A"), _lane(3, device_index=1)],
            clock=clock,
            device_indices=(0, 1),
        )
        scheduler._whole_card_ledger.state_for(1).model = "Flux"

        _tick_twice(scheduler, clock, 4.0)

        on_card_0 = accumulator.rate(model="A", device_index=0)
        assert on_card_0.model_seconds == pytest.approx(4.0)
        assert on_card_0.card_wall_seconds == pytest.approx(4.0)
        assert accumulator.rate(model="A", device_index=1).model_seconds == 0.0
        on_card_1 = accumulator.rate(model="Flux", device_index=1)
        assert on_card_1.model_seconds == pytest.approx(4.0)
        assert on_card_1.card_wall_seconds == pytest.approx(4.0)
        assert accumulator.rate(model="Flux", device_index=0).model_seconds == 0.0


class TestAccumulator:
    """The accumulator's arithmetic, fed directly."""

    def test_worked_example_rates(self) -> None:
        """Model and rest rates over a mixed window, against hand-computed figures.

        Card 0 runs A alone for 60 s, A beside B for 40 s, then idles 20 s. Wall 120 s, A 80 s, B 20 s. A
        earns 40 kudos over two jobs and B earns 30 over one, so the card earned 70.
        """
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 60.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.observe(BUCKET_START + 100.0, {0: CardTick(in_flight_models=("A", "B"))})
        accumulator.observe(BUCKET_START + 120.0, {0: CardTick()})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=15.0)
        accumulator.credit_finished_job(device_index=0, model="A", kudos=25.0)
        accumulator.credit_finished_job(device_index=0, model="B", kudos=30.0)

        rate_a = accumulator.rate(model="A", device_index=0)
        assert rate_a.model_kudos_per_second == pytest.approx(40.0 / 80.0)
        assert rate_a.rest_kudos_per_second == pytest.approx(30.0 / 40.0)
        assert rate_a.jobs == 2
        rate_b = accumulator.rate(model="B", device_index=0)
        assert rate_b.model_kudos_per_second == pytest.approx(30.0 / 20.0)
        assert rate_b.rest_kudos_per_second == pytest.approx(40.0 / 100.0)
        assert rate_b.jobs == 1

    def test_rates_are_none_on_a_zero_denominator(self) -> None:
        """A model without seconds has no model rate, and a model that took the whole wall has no rest rate."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(in_flight_models=("A", "A"))})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=5.0)

        rate_a = accumulator.rate(model="A", device_index=0)
        assert rate_a.model_kudos_per_second == pytest.approx(0.5)
        assert rate_a.rest_kudos_per_second is None
        rate_unseen = accumulator.rate(model="B", device_index=0)
        assert rate_unseen.model_kudos_per_second is None
        assert rate_unseen.rest_kudos_per_second == pytest.approx(0.5)
        assert rate_unseen.jobs == 0

    def test_a_faulted_job_adds_seconds_and_no_kudos(self) -> None:
        """A fault counts as a job and its time lowers the model's rate."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=None)

        rate = accumulator.rate(model="A", device_index=0)
        assert rate.model_kudos_per_second == 0.0
        assert rate.jobs == 1
        assert rate.model_seconds == pytest.approx(10.0)

    def test_a_stalled_tick_is_clamped(self) -> None:
        """A gap longer than the tick clamp adds only the clamp, to both the model and the wall."""
        accumulator = _accumulator(max_tick_seconds=5.0)
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 100.0, {0: CardTick(in_flight_models=("A",))})

        rate = accumulator.rate(model="A", device_index=0)
        assert rate.model_seconds == pytest.approx(5.0)
        assert rate.card_wall_seconds == pytest.approx(5.0)

    def test_a_backwards_reading_adds_nothing(self) -> None:
        """A clock that does not advance adds no time."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick()})
        accumulator.observe(BUCKET_START + 5.0, {0: CardTick(in_flight_models=("A",))})

        assert accumulator.rate(model="A", device_index=0).card_wall_seconds == 0.0

    def test_buckets_older_than_the_retention_drop_out_of_the_rates(self) -> None:
        """Figures from a bucket that closed more than the retention ago are not read."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=10.0)
        later = BUCKET_START + CARD_EARNING_RETENTION_SECONDS + CARD_EARNING_BUCKET_SECONDS
        accumulator.observe(later, {0: CardTick()})
        accumulator.observe(later + 2.0, {0: CardTick(in_flight_models=("B",))})

        rate_a = accumulator.rate(model="A", device_index=0)
        assert rate_a.model_seconds == 0.0
        assert rate_a.jobs == 0
        # The jump past the retention is one clamped tick, and the old bucket's 10 s are gone.
        assert rate_a.card_wall_seconds == pytest.approx(GENEROUS_TICK_SECONDS + 2.0)
        assert accumulator.rate(model="B", device_index=0).model_seconds == pytest.approx(2.0)

    def test_session_totals_do_not_age(self) -> None:
        """The per-model session totals for the rollups keep time that the trailing window has dropped."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(in_flight_models=("A",))})
        later = BUCKET_START + CARD_EARNING_RETENTION_SECONDS + CARD_EARNING_BUCKET_SECONDS
        accumulator.observe(later, {0: CardTick()})

        assert accumulator.rate(model="A", device_index=0).model_seconds == 0.0
        assert accumulator.session_seconds_by_model() == {"A": pytest.approx(10.0)}


class TestPersistence:
    """The window persists beside the performance model and loads back sanitised."""

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        """A saved window loads into a new accumulator with the same figures."""
        path = tmp_path / "card_earning.json"
        accumulator = _accumulator(path=path)
        accumulator.observe(BUCKET_START, {0: CardTick(), 1: CardTick()})
        accumulator.observe(BUCKET_START + 30.0, {0: CardTick(in_flight_models=("A", "B")), 1: CardTick()})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=12.0)
        accumulator.save()

        reloaded = _accumulator(path=path)

        for model in ("A", "B"):
            assert reloaded.rate(model=model, device_index=0) == accumulator.rate(model=model, device_index=0)
        assert reloaded.rate(model="A", device_index=1).card_wall_seconds == pytest.approx(30.0)

    def test_load_drops_invalid_and_expired_entries_and_keeps_the_rest(self, tmp_path: Path) -> None:
        """A NaN, a negative, a wrong type and an expired bucket are dropped, and the valid entries are kept."""
        path = tmp_path / "card_earning.json"
        expired_start = BUCKET_START - CARD_EARNING_RETENTION_SECONDS - 2 * CARD_EARNING_BUCKET_SECONDS
        payload = {
            "schema_version": CARD_EARNING_SCHEMA_VERSION,
            "bucket_seconds": CARD_EARNING_BUCKET_SECONDS,
            "config_fingerprint": FINGERPRINT,
            "buckets": [
                {
                    "start": BUCKET_START,
                    "walls": [
                        {"card": 0, "seconds": 100.0},
                        {"card": 1, "seconds": float("nan")},
                        {"card": 2, "seconds": -5.0},
                    ],
                    "tallies": [
                        {"card": 0, "model": "A", "seconds": 40.0, "kudos": 20.0, "jobs": 2},
                        {"card": 0, "model": "B", "seconds": "10", "kudos": 5.0, "jobs": 1},
                        {"card": 0, "model": "C", "seconds": 10.0, "kudos": -1.0, "jobs": 1},
                    ],
                },
                {
                    "start": expired_start,
                    "walls": [{"card": 0, "seconds": 500.0}],
                    "tallies": [{"card": 0, "model": "A", "seconds": 500.0, "kudos": 500.0, "jobs": 9}],
                },
                "not a bucket",
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = _accumulator(path=path)

        rate_a = loaded.rate(model="A", device_index=0)
        assert rate_a.card_wall_seconds == pytest.approx(100.0)
        assert rate_a.model_seconds == pytest.approx(40.0)
        assert rate_a.model_kudos_per_second == pytest.approx(0.5)
        assert rate_a.jobs == 2
        assert loaded.rate(model="B", device_index=0).jobs == 0
        assert loaded.rate(model="C", device_index=0).jobs == 0
        assert loaded.rate(model="A", device_index=1).card_wall_seconds == 0.0
        assert loaded.rate(model="A", device_index=2).card_wall_seconds == 0.0

    def test_a_file_of_another_bucket_width_is_discarded(self, tmp_path: Path) -> None:
        """Buckets of another width cannot merge into this one, so the file loads empty."""
        path = tmp_path / "card_earning.json"
        payload = {
            "schema_version": CARD_EARNING_SCHEMA_VERSION,
            "bucket_seconds": CARD_EARNING_BUCKET_SECONDS * 2,
            "buckets": [{"start": BUCKET_START, "walls": [{"card": 0, "seconds": 100.0}], "tallies": []}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert _accumulator(path=path).rate(model="A", device_index=0).card_wall_seconds == 0.0

    def test_credits_persist_on_the_performance_model_cadence(self, tmp_path: Path) -> None:
        """The window is written after every tenth credited job, without an explicit save."""
        path = tmp_path / "card_earning.json"
        accumulator = _accumulator(path=path)
        accumulator.observe(BUCKET_START, {0: CardTick()})
        for _ in range(9):
            accumulator.credit_finished_job(device_index=0, model="A", kudos=1.0)
        assert not path.exists()

        accumulator.credit_finished_job(device_index=0, model="A", kudos=1.0)

        assert _accumulator(path=path).rate(model="A", device_index=0).jobs == 10


class TestConfigurationWindow:
    """Figures measured under one capacity and workload configuration are not read under another."""

    def test_a_file_loads_intact_under_its_own_fingerprint(self, tmp_path: Path) -> None:
        """The saved fingerprint matches, so the window loads back whole."""
        path = tmp_path / "card_earning.json"
        _accumulator_with_one_job(path=path).save()

        rate = _accumulator(path=path).rate(model="A", device_index=0)

        assert rate.model_seconds == pytest.approx(30.0)
        assert rate.model_kudos_per_second == pytest.approx(12.0 / 30.0)
        assert rate.jobs == 1

    def test_a_file_loads_empty_under_another_fingerprint(self, tmp_path: Path, info_messages: list[str]) -> None:
        """A file written under other settings is discarded, and the restart is logged."""
        path = tmp_path / "card_earning.json"
        _accumulator_with_one_job(path=path).save()

        rate = _accumulator(path=path, settings=OTHER_SETTINGS).rate(model="A", device_index=0)

        assert rate.model_seconds == 0.0
        assert rate.card_wall_seconds == 0.0
        assert rate.jobs == 0
        assert any("Card earning figures restart" in message for message in info_messages)

    def test_a_file_without_a_fingerprint_loads_empty(self, tmp_path: Path) -> None:
        """A file that does not record its settings counts as written under other settings."""
        path = tmp_path / "card_earning.json"
        _accumulator_with_one_job(path=path).save()
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["config_fingerprint"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert _accumulator(path=path).rate(model="A", device_index=0).card_wall_seconds == 0.0

    def test_a_restart_during_a_run_drops_the_window_and_keeps_the_session_totals(
        self, info_messages: list[str]
    ) -> None:
        """The rates read no seconds and no kudos, and the rollups' session totals are kept."""
        accumulator = _accumulator_with_one_job()

        accumulator.restart_window_if_changed(OTHER_SETTINGS)

        rate = accumulator.rate(model="A", device_index=0)
        assert rate.model_seconds == 0.0
        assert rate.card_wall_seconds == 0.0
        assert rate.model_kudos_per_second is None
        assert rate.jobs == 0
        assert accumulator.session_seconds_by_model() == {"A": pytest.approx(30.0)}
        assert any("Card earning figures restart" in message for message in info_messages)

    def test_ticks_after_a_restart_accrue_normally(self) -> None:
        """The clock anchor survives the restart, so the next tick's seconds land in the new window."""
        accumulator = _accumulator_with_one_job()
        accumulator.restart_window_if_changed(OTHER_SETTINGS)

        accumulator.observe(BUCKET_START + 40.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=5.0)

        rate = accumulator.rate(model="A", device_index=0)
        assert rate.model_seconds == pytest.approx(10.0)
        assert rate.model_kudos_per_second == pytest.approx(0.5)
        assert accumulator.session_seconds_by_model() == {"A": pytest.approx(40.0)}

    def test_a_restart_adopts_the_new_fingerprint_in_the_file(self, tmp_path: Path) -> None:
        """After a restart the saved window belongs to the new settings."""
        path = tmp_path / "card_earning.json"
        accumulator = _accumulator_with_one_job(path=path)
        accumulator.restart_window_if_changed(OTHER_SETTINGS)
        accumulator.observe(BUCKET_START + 40.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.save()

        reloaded = _accumulator(path=path, settings=OTHER_SETTINGS)

        assert reloaded.rate(model="A", device_index=0).model_seconds == pytest.approx(10.0)

    def test_the_same_fingerprint_does_not_restart_the_window(self, info_messages: list[str]) -> None:
        """An unchanged fingerprint keeps the window and does not log."""
        accumulator = _accumulator_with_one_job()

        accumulator.restart_window_if_changed(SETTINGS)

        assert accumulator.rate(model="A", device_index=0).model_seconds == pytest.approx(30.0)
        assert not any("Card earning figures restart" in message for message in info_messages)

    def test_a_change_with_an_empty_window_does_not_log(self, info_messages: list[str]) -> None:
        """No figures are dropped, so no restart line is written."""
        accumulator = _accumulator()

        accumulator.restart_window_if_changed(OTHER_SETTINGS)

        assert not any("Card earning figures restart" in message for message in info_messages)


SECOND_BUCKET_START = BUCKET_START + CARD_EARNING_BUCKET_SECONDS


def _fill_bucket(
    accumulator: CardEarningAccumulator,
    *,
    start: float,
    elapsed: float,
    in_flight_models: tuple[str, ...],
    kudos_by_model: dict[str, float],
) -> None:
    """Add ``elapsed`` seconds shared by ``in_flight_models`` to the bucket at ``start``, then credit the kudos."""
    accumulator.observe(start, {})
    accumulator.observe(start + elapsed, {0: CardTick(in_flight_models=in_flight_models)})
    for model, kudos in kudos_by_model.items():
        accumulator.credit_finished_job(device_index=0, model=model, kudos=kudos)


class TestStandardErrors:
    """Each rate's standard error is read across the retained buckets."""

    def test_two_buckets_give_the_hand_computed_errors(self) -> None:
        """The worked example's errors match the ratio formula.

        Bucket one: A and B share 20 s, so 10 s each, with A paid 10 and B paid 5. Bucket two: five lanes share
        50 s, so A has 30 s and B 20 s, with A paid 60 and B paid 30. Model A's rate is 70 / 40 = 1.75, and its
        residuals are 10 - 1.75 x 10 = -7.5 and 60 - 1.75 x 30 = 7.5, so its error is sqrt(2 x 112.5) / 40 =
        0.375. The rest of the card has 10 s and 20 s paid 5 and 30, a rate of 35 / 30 = 7/6, residuals of
        -20/3 and 20/3, and an error of sqrt(2 x 800/9) / 30 = 4/9.
        """
        accumulator = _accumulator()
        _fill_bucket(
            accumulator,
            start=BUCKET_START,
            elapsed=20.0,
            in_flight_models=("A", "B"),
            kudos_by_model={"A": 10.0, "B": 5.0},
        )
        _fill_bucket(
            accumulator,
            start=SECOND_BUCKET_START,
            elapsed=50.0,
            in_flight_models=("A", "A", "A", "B", "B"),
            kudos_by_model={"A": 60.0, "B": 30.0},
        )

        rate = accumulator.rate(model="A", device_index=0)

        assert rate.model_kudos_per_second == pytest.approx(1.75)
        assert rate.model_standard_error == pytest.approx(0.375)
        assert rate.rest_kudos_per_second == pytest.approx(7.0 / 6.0)
        assert rate.rest_standard_error == pytest.approx(4.0 / 9.0)

    def test_one_bucket_has_no_error(self) -> None:
        """A single bucket does not give a spread, so both errors are None."""
        accumulator = _accumulator()
        _fill_bucket(
            accumulator,
            start=BUCKET_START,
            elapsed=20.0,
            in_flight_models=("A", "B"),
            kudos_by_model={"A": 10.0, "B": 5.0},
        )

        rate = accumulator.rate(model="A", device_index=0)

        assert rate.model_kudos_per_second == pytest.approx(1.0)
        assert rate.model_standard_error is None
        assert rate.rest_standard_error is None

    def test_equal_bucket_rates_give_zero_error(self) -> None:
        """Buckets of different sizes with the same rate do not spread the rate."""
        accumulator = _accumulator()
        _fill_bucket(
            accumulator,
            start=BUCKET_START,
            elapsed=20.0,
            in_flight_models=("A", "B"),
            kudos_by_model={"A": 10.0, "B": 5.0},
        )
        _fill_bucket(
            accumulator,
            start=SECOND_BUCKET_START,
            elapsed=60.0,
            in_flight_models=("A", "B"),
            kudos_by_model={"A": 30.0, "B": 15.0},
        )

        rate = accumulator.rate(model="A", device_index=0)

        assert rate.model_standard_error == pytest.approx(0.0)
        assert rate.rest_standard_error == pytest.approx(0.0)


def _two_bucket_two_card_window() -> CardEarningAccumulator:
    """The standard-error worked example on card 0, plus a claimed Flux grant on card 1 in the second bucket."""
    accumulator = _accumulator()
    _fill_bucket(
        accumulator,
        start=BUCKET_START,
        elapsed=20.0,
        in_flight_models=("A", "B"),
        kudos_by_model={"A": 10.0, "B": 5.0},
    )
    accumulator.observe(SECOND_BUCKET_START, {})
    accumulator.observe(
        SECOND_BUCKET_START + 50.0,
        {
            0: CardTick(in_flight_models=("A", "A", "A", "B", "B")),
            1: CardTick(held_model="Flux", claimed_model="Flux"),
        },
    )
    for model, kudos in (("A", 60.0), ("B", 30.0)):
        accumulator.credit_finished_job(device_index=0, model=model, kudos=kudos)
    accumulator.credit_finished_job(device_index=1, model="Flux", kudos=40.0)
    return accumulator


class TestReadWindow:
    """One read returns every pair's figures, each card's time and the window's status."""

    def test_every_pair_matches_its_own_rate_read(self) -> None:
        """Each pair in the all-pairs read equals the single-pair read, standard errors included."""
        accumulator = _two_bucket_two_card_window()

        window = accumulator.read_window()

        assert set(window.rates) == {(0, "A"), (0, "B"), (1, "Flux")}
        for (device_index, model), rate in window.rates.items():
            assert rate == accumulator.rate(model=model, device_index=device_index)
        assert window.rates[(0, "A")].model_standard_error == pytest.approx(0.375)

    def test_each_card_reports_its_wall_and_attributed_seconds(self) -> None:
        """Card 0 idles 20 s of 120, so 20 s are unattributed. Card 1's grant takes its whole wall."""
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick(), 1: CardTick()})
        accumulator.observe(BUCKET_START + 60.0, {0: CardTick(in_flight_models=("A",)), 1: CardTick()})
        accumulator.observe(BUCKET_START + 100.0, {0: CardTick(in_flight_models=("A", "B")), 1: CardTick()})
        accumulator.observe(BUCKET_START + 120.0, {0: CardTick(), 1: CardTick(held_model="Flux")})

        cards = accumulator.read_window().cards

        assert [card.device_index for card in cards] == [0, 1]
        assert cards[0].wall_seconds == pytest.approx(120.0)
        assert cards[0].attributed_seconds == pytest.approx(100.0)
        assert cards[0].unattributed_seconds == pytest.approx(20.0)
        assert cards[1].wall_seconds == pytest.approx(120.0)
        assert cards[1].attributed_seconds == pytest.approx(20.0)
        assert cards[1].unattributed_seconds == pytest.approx(100.0)

    def test_the_window_starts_at_the_oldest_retained_bucket(self) -> None:
        """The start is the oldest bucket's start, and it moves on as the retention drops that bucket."""
        accumulator = _two_bucket_two_card_window()
        assert accumulator.read_window().started_at == BUCKET_START

        later = BUCKET_START + CARD_EARNING_RETENTION_SECONDS + CARD_EARNING_BUCKET_SECONDS + 1.0
        accumulator.observe(later, {0: CardTick()})

        assert accumulator.read_window().started_at == SECOND_BUCKET_START

    def test_an_empty_accumulator_reads_an_empty_window(self) -> None:
        """The pairs and cards are empty, and the start and restart are None."""
        window = _accumulator().read_window()

        assert window.rates == {}
        assert window.cards == []
        assert window.started_at is None
        assert window.last_restart is None


class TestClaimSeconds:
    """The time a whole-card pop claim stands is tallied to the claimed model, apart from the attribution."""

    def test_claim_seconds_accrue_only_while_a_claim_stands_and_to_its_model(self) -> None:
        """Claim seconds accrue to the claimed model only while its claim stands.

        A grant that outlasts its claim adds held seconds and no claim seconds. A job beside a claim takes the
        attributed seconds, and the claim seconds stay with the claimed model.
        """
        accumulator = _accumulator()
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(held_model="Flux", claimed_model="Flux")})
        accumulator.observe(BUCKET_START + 15.0, {0: CardTick(held_model="Flux")})
        accumulator.observe(
            BUCKET_START + 19.0,
            {0: CardTick(in_flight_models=("A",), held_model="Flux", claimed_model="Flux")},
        )
        accumulator.observe(BUCKET_START + 21.0, {0: CardTick()})

        flux = accumulator.rate(model="Flux", device_index=0)
        assert flux.claim_seconds == pytest.approx(14.0)
        assert flux.model_seconds == pytest.approx(15.0)
        assert accumulator.rate(model="A", device_index=0).claim_seconds == 0.0
        assert accumulator.rate(model="A", device_index=0).model_seconds == pytest.approx(4.0)
        assert flux.card_wall_seconds == pytest.approx(21.0)

    def test_claim_seconds_persist(self, tmp_path: Path) -> None:
        """A saved claim tally loads back."""
        path = tmp_path / "card_earning.json"
        accumulator = _accumulator(path=path)
        accumulator.observe(BUCKET_START, {0: CardTick()})
        accumulator.observe(BUCKET_START + 10.0, {0: CardTick(held_model="Flux", claimed_model="Flux")})
        accumulator.save()

        assert _accumulator(path=path).rate(model="Flux", device_index=0).claim_seconds == pytest.approx(10.0)

    def test_a_file_without_the_claim_tally_loads_with_it_at_zero(self, tmp_path: Path) -> None:
        """A tally written before the claim tally existed keeps its seconds, kudos and jobs."""
        path = tmp_path / "card_earning.json"
        payload = {
            "schema_version": CARD_EARNING_SCHEMA_VERSION,
            "bucket_seconds": CARD_EARNING_BUCKET_SECONDS,
            "config_fingerprint": FINGERPRINT,
            "buckets": [
                {
                    "start": BUCKET_START,
                    "walls": [{"card": 0, "seconds": 100.0}],
                    "tallies": [{"card": 0, "model": "A", "seconds": 40.0, "kudos": 20.0, "jobs": 2}],
                },
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        rate = _accumulator(path=path).rate(model="A", device_index=0)

        assert rate.claim_seconds == 0.0
        assert rate.model_seconds == pytest.approx(40.0)
        assert rate.model_kudos_per_second == pytest.approx(0.5)
        assert rate.jobs == 2
        assert rate.card_wall_seconds == pytest.approx(100.0)

    @pytest.mark.parametrize("card_index", [0, 1])
    def test_a_single_card_claim_is_credited_on_the_lane_card(self, card_index: int) -> None:
        """The ledger keys the claim by None on a single-card host, and the tally lands on the lane's card."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler(
            [_lane(1, device_index=card_index)],
            clock=clock,
            device_indices=(card_index,),
        )
        scheduler._whole_card_ledger.record_grant(
            None,
            model="Flux",
            forecast=None,
            cooldown_until=clock.now + 300.0,
            now=clock.now,
        )
        assert scheduler.whole_card_pop_claim() is not None

        _tick_twice(scheduler, clock, 6.0)

        rate = accumulator.rate(model="Flux", device_index=card_index)
        assert rate.claim_seconds == pytest.approx(6.0)
        assert rate.model_seconds == pytest.approx(6.0)

    def test_a_grant_whose_claim_was_released_adds_no_claim_seconds(self) -> None:
        """A grant can outlast its claim. The grant keeps the card's time and the claim tally stays at zero."""
        clock = _Clock(BUCKET_START)
        scheduler, accumulator = _scheduler([_lane(1)], clock=clock)
        scheduler._whole_card_ledger.record_grant(
            None,
            model="Flux",
            forecast=None,
            cooldown_until=clock.now + 300.0,
            now=clock.now,
        )
        state = scheduler._whole_card_ledger.state_for(None)
        state.pop_claim_released_at = clock.now
        assert scheduler.whole_card_pop_claim() is None

        _tick_twice(scheduler, clock, 6.0)

        rate = accumulator.rate(model="Flux", device_index=0)
        assert rate.claim_seconds == 0.0
        assert rate.model_seconds == pytest.approx(6.0)


class TestWindowRestartStatus:
    """The window reports its last restart, naming the changed settings when the run saw them change."""

    def test_a_restart_during_a_run_names_the_changed_setting(self) -> None:
        """Only the live concurrency cap differs between the two settings."""
        accumulator = _accumulator_with_one_job()

        accumulator.restart_window_if_changed(OTHER_SETTINGS)

        restart = accumulator.read_window().last_restart
        assert restart is not None
        assert restart.at == BUCKET_START
        assert restart.changed_settings == ("max_threads",)

    def test_settings_changed_in_two_fields_are_named_in_declaration_order(self) -> None:
        """Both changed fields are named, in the order the settings model declares them."""
        accumulator = _accumulator()

        accumulator.restart_window_if_changed(_settings(max_threads=3, max_power=16))

        restart = accumulator.read_window().last_restart
        assert restart is not None
        assert restart.changed_settings == ("max_power", "max_threads")

    def test_unchanged_settings_do_not_report_a_restart(self) -> None:
        """An unchanged fingerprint does not record a restart."""
        accumulator = _accumulator_with_one_job()

        accumulator.restart_window_if_changed(SETTINGS)

        assert accumulator.read_window().last_restart is None

    def test_a_restart_found_at_load_does_not_name_the_settings(self, tmp_path: Path) -> None:
        """The file holds only the previous fingerprint, so the changed settings are unknown."""
        path = tmp_path / "card_earning.json"
        _accumulator_with_one_job(path=path).save()

        restart = _accumulator(path=path, settings=OTHER_SETTINGS).read_window().last_restart

        assert restart is not None
        assert restart.at == BUCKET_START
        assert restart.changed_settings is None
