"""Tests for A6.3 per-card unservable fault streaks.

A model's over-budget fault streak is keyed by the card it ran on, so a model the small card cannot run does
not stop the big card from being offered it. The popper holds a model back only when *every* card that serves
it has flagged it unservable. A single-GPU host keys streaks under None, identical to the prior model-only
keying.
"""

from __future__ import annotations

from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap

from .conftest import make_job_pop_response, make_mock_bridge_data, make_test_card_runtimes, track_popped_job_async

_THRESHOLD = 3


def _card(*, device_index: int, models: list[str]) -> CardRuntime:
    """A CardRuntime whose effective config serves the given models."""
    config = make_mock_bridge_data(image_models_to_load=models)
    return make_test_card_runtimes(device_indices=(device_index,), config=config)[device_index]


def _build_streak(job_tracker: JobTracker, model: str, *, device_index: int | None, count: int) -> None:
    """Drive a model's over-budget streak on a card to ``count`` consecutive faults."""
    for _ in range(count):
        job_tracker._record_resource_fault(model, device_index=device_index)


class TestPerCardStreakKeying:
    """Fault streaks accrue and clear per card."""

    def test_streak_is_isolated_per_card(self) -> None:
        """Faults on card 1 do not raise card 0's streak; the unfiltered read is the worst card."""
        job_tracker = JobTracker()
        _build_streak(job_tracker, "model", device_index=1, count=3)

        assert job_tracker.get_model_overbudget_fault_count("model", device_index=1) == 3
        assert job_tracker.get_model_overbudget_fault_count("model", device_index=0) == 0
        # The unfiltered (worker-wide) read is the worst card's streak.
        assert job_tracker.get_model_overbudget_fault_count("model") == 3

    def test_success_clears_only_its_card(self) -> None:
        """A result on card 1 clears card 1's streak while card 0's persists."""
        job_tracker = JobTracker()
        _build_streak(job_tracker, "model", device_index=0, count=2)
        _build_streak(job_tracker, "model", device_index=1, count=3)

        job_tracker.record_model_inference_success("model", device_index=1)

        assert job_tracker.get_model_overbudget_fault_count("model", device_index=1) == 0
        assert job_tracker.get_model_overbudget_fault_count("model", device_index=0) == 2

    def test_single_gpu_streak_under_none_key(self) -> None:
        """With no card attribution the streak is worker-wide (the None key), as on a single-GPU host."""
        job_tracker = JobTracker()
        _build_streak(job_tracker, "model", device_index=None, count=3)

        assert job_tracker.get_model_overbudget_fault_count("model") == 3
        assert job_tracker.get_model_overbudget_fault_count("model", device_index=None) == 3
        job_tracker.record_model_inference_success("model")
        assert job_tracker.get_model_overbudget_fault_count("model") == 0


class TestPopperHoldbackAcrossCards:
    """The popper holds a model back only when every serving card finds it unservable."""

    def _select(self, *, job_tracker: JobTracker, card_runtimes: dict[int, CardRuntime], models: set[str]) -> set[str]:
        return (
            _select_models_for_pop(
                make_mock_bridge_data(),
                ProcessMap({}),
                job_tracker,
                max_inference_processes=4,
                last_pop_had_no_jobs=False,
                configured_models=models,
                card_runtimes=card_runtimes,
            )
            or set()
        )

    def test_kept_when_still_servable_on_one_card(self) -> None:
        """A model unservable on card 1 but served by an un-flagged card 0 keeps being advertised."""
        job_tracker = JobTracker()
        card_runtimes = {
            0: _card(device_index=0, models=["shared"]),
            1: _card(device_index=1, models=["shared"]),
        }
        # "shared" has faulted out on card 1 only; card 0 can still run it.
        _build_streak(job_tracker, "shared", device_index=1, count=_THRESHOLD)

        selected = self._select(job_tracker=job_tracker, card_runtimes=card_runtimes, models={"shared"})
        assert selected == {"shared"}

    def test_held_back_when_unservable_on_every_serving_card(self) -> None:
        """A model unservable on the only card that serves it is dropped from the advertised set."""
        job_tracker = JobTracker()
        card_runtimes = {
            0: _card(device_index=0, models=["big", "small"]),
            1: _card(device_index=1, models=["small"]),
        }
        # "big" is served only by card 0, and it has faulted out there.
        _build_streak(job_tracker, "big", device_index=0, count=_THRESHOLD)

        selected = self._select(
            job_tracker=job_tracker,
            card_runtimes=card_runtimes,
            models={"big", "small"},
        )
        assert selected == {"small"}


class TestOrphanPuntDoesNotPoisonStreak:
    """An orphan punt is a scheduling fault, never a per-card "locally unservable" verdict.

    The orphan watchdog punts a stranded in-progress job; that is a host-contention/ownership failure, not
    evidence the model cannot fit the card it was dispatched to. So even when the punted job was admitted
    over budget (which otherwise folds into a resource failure), the punt must not key that card's streak and
    de-list a model a capable card can still run. A genuine resource fault on the same model+card does key it.
    """

    async def test_over_budget_orphan_punt_does_not_key_its_card(self) -> None:
        """An over-budget job punted as a scheduling fault leaves its card's streak at zero."""
        job_tracker = JobTracker()  # one attempt by default, so this punt is terminal
        job = make_job_pop_response(model="big")
        await track_popped_job_async(job_tracker, job)
        await job_tracker.mark_inference_started(job, device_index=1)
        job_tracker.mark_admitted_over_budget(job)

        job_tracker.handle_job_fault_now(faulted_job=job, retryable=True, scheduling_fault=True)

        assert job_tracker.get_model_overbudget_fault_count("big", device_index=1) == 0
        assert job_tracker.get_model_overbudget_fault_count("big") == 0

    async def test_genuine_resource_fault_keys_its_card(self) -> None:
        """The contrast: a real resource fault on the same model+card does feed the per-card streak."""
        job_tracker = JobTracker()
        job = make_job_pop_response(model="big")
        await track_popped_job_async(job_tracker, job)
        await job_tracker.mark_inference_started(job, device_index=1)

        job_tracker.handle_job_fault_now(faulted_job=job, retryable=True, is_resource_failure=True)

        assert job_tracker.get_model_overbudget_fault_count("big", device_index=1) == 1
