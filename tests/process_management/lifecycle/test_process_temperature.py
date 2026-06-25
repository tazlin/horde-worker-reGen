"""Tests for the process-temperature classifier that distinguishes primed slots from idle ones."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.process_temperature import (
    ProcessTemperature,
    classify_process_temperature,
    temperature_phrase,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


def _classify(state: str, loaded_model: str | None, pending: set[str]) -> ProcessTemperature:
    return classify_process_temperature(state=state, loaded_model=loaded_model, pending_models=frozenset(pending))


class TestClassifyProcessTemperature:
    """The classifier separates the materially different slots a single WAITING_FOR_JOB collapses."""

    def test_active_inference_is_hot(self) -> None:
        """A slot sampling on the GPU is hot regardless of what is queued."""
        assert _classify("INFERENCE_STARTING", "model_a", set()) == ProcessTemperature.HOT
        assert _classify("INFERENCE_POST_PROCESSING", "model_a", {"model_a"}) == ProcessTemperature.HOT

    def test_resident_model_a_queued_job_needs_is_next(self) -> None:
        """A ready slot whose resident model a queued job will use fires next."""
        assert _classify("WAITING_FOR_JOB", "model_b", {"model_b"}) == ProcessTemperature.NEXT
        # A preloaded model staged for a pending job is also 'next', not merely idle.
        assert _classify("PRELOADED_MODEL", "model_b", {"model_b"}) == ProcessTemperature.NEXT

    def test_resident_model_nothing_queued_is_warm(self) -> None:
        """A resident model no queued job needs is warm: kept ready, but not firing imminently."""
        assert _classify("WAITING_FOR_JOB", "model_c", set()) == ProcessTemperature.WARM
        assert _classify("WAITING_FOR_JOB", "model_c", {"other_model"}) == ProcessTemperature.WARM

    def test_loading_slot_is_priming(self) -> None:
        """A slot loading or downloading a model is priming, even if a queued job wants that model."""
        assert _classify("PRELOADING_MODEL", "model_d", {"model_d"}) == ProcessTemperature.PRIMING
        assert _classify("DOWNLOADING_MODEL", "model_d", set()) == ProcessTemperature.PRIMING
        assert _classify("PROCESS_STARTING", None, set()) == ProcessTemperature.PRIMING

    def test_empty_slot_is_cold(self) -> None:
        """A ready slot holding no model at all is cold."""
        assert _classify("WAITING_FOR_JOB", None, set()) == ProcessTemperature.COLD

    def test_terminal_slot_is_down(self) -> None:
        """Ended/failed slots are down, distinct from any warmth."""
        assert _classify("PROCESS_ENDED", None, set()) == ProcessTemperature.DOWN
        assert _classify("INFERENCE_FAILED", "model_e", set()) == ProcessTemperature.DOWN


class TestTemperaturePhrase:
    """The folded label reads out activity for hot/priming and readiness for the idle-collapsed buckets."""

    def test_phrases_match_the_folded_label_scheme(self) -> None:
        """Each temperature pairs with the phrase the chosen ``Temp · phrase`` label scheme expects."""
        assert temperature_phrase(ProcessTemperature.HOT, "INFERENCE_STARTING") == "sampling"
        assert temperature_phrase(ProcessTemperature.PRIMING, "PRELOADING_MODEL") == "loading"
        assert temperature_phrase(ProcessTemperature.NEXT, "PRELOADED_MODEL") == "primed"
        assert temperature_phrase(ProcessTemperature.WARM, "WAITING_FOR_JOB") == "ready"
        assert temperature_phrase(ProcessTemperature.COLD, "WAITING_FOR_JOB") == "idle"


class TestStatusLineTemperature:
    """The periodic status line leads each slot with its temperature, keeping the raw state greppable."""

    async def test_primed_slots_read_as_next_or_warm_not_idle(self) -> None:
        """Two slots both reporting WAITING_FOR_JOB read differently by what the queue needs.

        The slot whose resident model a pending job names is ``next``; the slot whose resident model
        nothing queued needs is ``warm``. The raw ``WAITING_FOR_JOB`` is preserved after the colon so log
        greps on state names still match.
        """
        manager = make_testable_process_manager()
        manager._process_map = ProcessMap(
            {
                1: make_mock_process_info(1, model_name="model_next", state=HordeProcessState.WAITING_FOR_JOB),
                2: make_mock_process_info(2, model_name="model_warm", state=HordeProcessState.WAITING_FOR_JOB),
            },
        )
        await track_popped_job_async(manager._job_tracker, make_job_pop_response("model_next"))

        _phase, summary = manager.describe_run_phase()

        assert "inf#1=next:WAITING_FOR_JOB" in summary
        assert "inf#2=warm:WAITING_FOR_JOB" in summary
