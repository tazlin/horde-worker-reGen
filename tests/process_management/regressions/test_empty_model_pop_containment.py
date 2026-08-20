"""A pop carrying no model name must be handed back, not run through the worker as a model identity.

The failure this covers is a chain, not a single bug: an empty model name entering the queue is preloaded as a
literal empty identity, the child's preload raises on it, the exception escapes the control-message handler and
ends an otherwise healthy inference process, and the repeated "load failures" are then counted against the
empty string until it is quarantined as if it were a model. Every link is tested here.
"""

from __future__ import annotations

import queue
from unittest.mock import AsyncMock, Mock

import pytest
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.consts import GENERATION_STATE
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ModelIncidentKind
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_testable_process_manager,
)
from tests.process_management.jobs.test_job_popping import (
    _make_popper,
    _make_process_map_with_available_processes,
)


async def _pop_once(pop_response: object) -> tuple[JobTracker, list[str | None], JobPopper]:
    """Drive one pop that the horde answers with ``pop_response``; return the tracker, observed faults, popper."""
    job_tracker = JobTracker()
    await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop happens
    observed_faults: list[str | None] = []
    job_tracker.set_terminal_fault_observer(observed_faults.append)

    session = Mock()
    session.submit_request = AsyncMock(return_value=pop_response)
    popper = _make_popper(
        job_tracker=job_tracker,
        process_map=_make_process_map_with_available_processes(),
        horde_client_session=session,
    )

    await popper.api_job_pop()

    session.submit_request.assert_awaited_once()
    return job_tracker, observed_faults, popper


class TestPopBoundaryValidation:
    """A popped job with no usable model name never reaches the queue."""

    @pytest.mark.parametrize("model", ["", "   "])
    async def test_blank_model_pop_is_never_queued(self, model: str) -> None:
        """A blank model name is refused at the boundary rather than queued for dispatch."""
        job_tracker, _faults, _popper = await _pop_once(make_job_pop_response(model=model))

        assert list(job_tracker.jobs_pending_inference) == []

    async def test_blank_model_pop_is_faulted_for_reissue(self) -> None:
        """The refused job is faulted terminally, so the horde reissues it instead of waiting on its ttl."""
        job_tracker, _faults, _popper = await _pop_once(make_job_pop_response(model=""))

        tracked_jobs = job_tracker.tracked_jobs()
        assert len(tracked_jobs) == 1
        assert tracked_jobs[0].stage is JobStage.PENDING_SUBMIT
        assert tracked_jobs[0].job_info is not None
        assert tracked_jobs[0].job_info.state is GENERATION_STATE.faulted

    async def test_blank_model_fault_is_attributed_to_the_pop_not_a_model(self) -> None:
        """The fault carries the malformed-pop origin, keeping it out of the generation-verdict pause."""
        job_tracker, _faults, _popper = await _pop_once(make_job_pop_response(model=""))

        tracked = job_tracker.tracked_jobs()[0]
        assert tracked.fault_origin is JobFaultOrigin.MALFORMED_POP
        assert job_tracker.was_faulted_by_non_generation_action(tracked.job_id) is True

    async def test_blank_model_fault_is_kept_out_of_the_fault_rate_breaker(self) -> None:
        """The rejection consumes no slot, so it must not feed the breaker whose pause idles every card.

        The breaker exists to stem fault streams the worker's own generation produces; a malformed pop is the
        horde's defect, its rate is bounded by the pop cadence, and the boundary answers it with the pop error
        backoff instead. Feeding it to the breaker starves all cards over pops that cost nothing to reject.
        """
        _job_tracker, observed_faults, _popper = await _pop_once(make_job_pop_response(model=""))

        assert observed_faults == []

    async def test_blank_model_pop_engages_the_pop_error_backoff(self) -> None:
        """A malformed pop slows the pop cadence like any other unusable API answer, until a clean pop resets it."""
        _job_tracker, _faults, popper = await _pop_once(make_job_pop_response(model=""))

        assert popper._pop_throttler.is_in_error_backoff is True

    async def test_named_model_pop_is_queued_as_before(self) -> None:
        """A well-formed pop is unaffected: it reaches the pending-inference queue with no backoff engaged."""
        job_tracker, observed_faults, popper = await _pop_once(make_job_pop_response(model="stable_diffusion"))

        assert [job.model for job in job_tracker.jobs_pending_inference] == ["stable_diffusion"]
        assert observed_faults == []
        assert popper._pop_throttler.is_in_error_backoff is False


class TestAdvertisedModelObservability:
    """Each pop request records the offer it actually sent, so a blank entry is attributable."""

    async def test_pop_request_logs_the_advertised_models(self) -> None:
        """The advertised set is logged at DEBUG on the pop request."""
        emitted: list[str] = []
        sink_id = logger.add(lambda message: emitted.append(message.record["message"]), level="DEBUG")
        try:
            job_tracker = JobTracker()
            await job_tracker.increment_jobs_completed()
            session = Mock()
            session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
            popper = _make_popper(
                job_tracker=job_tracker,
                process_map=_make_process_map_with_available_processes(),
                horde_client_session=session,
            )

            await popper.api_job_pop()
        finally:
            logger.remove(sink_id)

        advertised_lines = [line for line in emitted if line.startswith("Advertising models in pop request:")]
        assert len(advertised_lines) == 1
        assert "'stable_diffusion'" in advertised_lines[0]

    async def test_custom_model_without_a_name_is_not_advertised(self) -> None:
        """A custom model entry with a blank name is dropped from the offer instead of being advertised.

        Custom model entries are the one offer source that does not pass through the model reference's
        known-name filter, so this is the worker's own path to an unloadable advertised name.
        """
        job_tracker = JobTracker()
        await job_tracker.increment_jobs_completed()
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        bridge_data = make_mock_bridge_data(
            custom_models=[{"name": "", "baseline": "stable_diffusion_1", "filepath": "a.safetensors"}],
        )
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=bridge_data,
        )

        await popper.api_job_pop()

        request = session.submit_request.call_args.args[0]
        assert "" not in request.models
        assert request.models == ["stable_diffusion"]


class _PreloadStubProcess(HordeInferenceProcess):
    """A HordeInferenceProcess wired for the preload control-message path only.

    Real construction spins up HordeLib and the shared model managers; only the preload handler and the base
    class's control-message loop are reached here, so the backend is a stub while the loop that decides whether
    the process survives stays real.
    """

    @staticmethod
    def build(*, preload_error: Exception | None) -> _PreloadStubProcess:
        """Return a stub whose backend preload raises ``preload_error`` (or succeeds when None)."""
        proc = object.__new__(_PreloadStubProcess)
        proc.process_id = 1
        proc.process_launch_identifier = 0
        proc.process_message_queue = Mock(spec=queue.Queue)  # pyrefly: ignore
        proc._control_inbox = queue.Queue()  # pyrefly: ignore
        proc._active_model_name = None
        proc._is_busy = False
        proc._end_process = False
        proc._dry_run_skip_inference = False
        proc.on_horde_model_state_change = Mock()  # pyrefly: ignore
        proc.send_memory_report_message = Mock(return_value=True)  # pyrefly: ignore
        proc.clear_gc_and_torch_cache = Mock()  # pyrefly: ignore

        backend = Mock()
        if preload_error is not None:
            backend.preload_model = Mock(side_effect=preload_error)
        proc._horde = backend  # pyrefly: ignore
        return proc

    def reported_states(self) -> list[tuple[HordeProcessState, str, ModelLoadState]]:
        """Return the (process state, model name, load state) triples reported to the parent."""
        return [
            (call.kwargs["process_state"], call.kwargs["horde_model_name"], call.kwargs["horde_model_state"])
            for call in self.on_horde_model_state_change.call_args_list  # pyrefly: ignore
        ]


def _preload_message(model: str) -> HordeControlMessage:
    """Build the parent's preload command for ``model``."""
    from horde_worker_regen.process_management.ipc.messages import HordePreloadInferenceModelMessage

    return HordePreloadInferenceModelMessage(
        control_flag=HordeControlFlag.PRELOAD_MODEL,
        horde_model_name=model,
        will_load_loras=False,
        seamless_tiling_enabled=False,
        sdk_api_job_info=make_job_pop_response(model=model),
    )


class TestChildSurvivesABadPreloadArgument:
    """A preload the backend cannot even name a file for must not cost the worker a whole process."""

    def test_blank_model_name_reports_a_load_failure_without_ending_the_process(self) -> None:
        """A blank name is reported as a failed load and the main loop keeps running."""
        proc = _PreloadStubProcess.build(preload_error=None)

        proc._control_inbox.put(_preload_message(""))
        proc.receive_and_handle_control_messages()

        assert proc._end_process is False
        assert proc.reported_states() == [
            (HordeProcessState.PRELOADING_FAILED, "", ModelLoadState.FAILED),
        ]
        proc._horde.preload_model.assert_not_called()  # pyrefly: ignore

    def test_unknown_model_reports_a_load_failure_without_ending_the_process(self) -> None:
        """The backend's refusal to resolve an unknown name is contained the same way."""
        proc = _PreloadStubProcess.build(preload_error=ValueError("Model nonesuch is not available."))

        proc._control_inbox.put(_preload_message("nonesuch"))
        proc.receive_and_handle_control_messages()

        assert proc._end_process is False
        assert (HordeProcessState.PRELOADING_FAILED, "nonesuch", ModelLoadState.FAILED) in proc.reported_states()

    def test_next_job_is_still_served_after_a_refused_preload(self) -> None:
        """Containment is only worth anything if the surviving process goes on to do work."""
        proc = _PreloadStubProcess.build(preload_error=None)
        proc._horde.preload_model = Mock()  # pyrefly: ignore

        proc._control_inbox.put(_preload_message(""))
        proc._control_inbox.put(_preload_message("stable_diffusion"))
        proc.receive_and_handle_control_messages()

        assert proc._end_process is False
        assert proc._active_model_name == "stable_diffusion"
        assert [call.args[0] for call in proc._horde.preload_model.call_args_list] == ["stable_diffusion"]  # pyrefly: ignore
        assert (
            HordeProcessState.PRELOADED_MODEL,
            "stable_diffusion",
            ModelLoadState.LOADED_IN_RAM,
        ) in proc.reported_states()

    def test_a_genuine_preload_failure_still_ends_the_process(self) -> None:
        """Containment is scoped: a failure part-way through a real load still replaces the slot."""
        proc = _PreloadStubProcess.build(preload_error=RuntimeError("CUDA error: an illegal memory access"))

        proc._control_inbox.put(_preload_message("stable_diffusion"))
        proc.receive_and_handle_control_messages()

        assert proc._end_process is True


class TestIdentityHygiene:
    """No blank identity may enter the per-model incident, quarantine, or load-state state."""

    @pytest.mark.parametrize("model_name", ["", "   "])
    def test_incident_against_a_blank_model_is_refused(self, model_name: str) -> None:
        """A blank name is counted against nothing and can never reach the quarantine set."""
        process_manager = make_testable_process_manager()
        lifecycle = process_manager._process_lifecycle

        for _ in range(10):
            quarantined = lifecycle.record_model_incident(model_name, ModelIncidentKind.LOAD_FAILURE)
            assert quarantined is False

        assert lifecycle.quarantined_models() == frozenset()
        assert lifecycle.is_model_load_quarantined(model_name) is False

    @pytest.mark.parametrize("model_name", ["", "   "])
    def test_model_map_refuses_a_blank_identity(self, model_name: str) -> None:
        """A blank name never becomes a map entry, so nothing reports it as loaded or loading."""
        model_map = HordeModelMap(root={})

        model_map.update_entry(model_name, load_state=ModelLoadState.LOADING, process_id=1)

        assert model_map.root == {}
        assert model_map.is_model_loaded(model_name) is False
        assert model_map.is_model_loading(model_name) is False

    def test_blank_identity_refusal_is_surfaced_once(self) -> None:
        """The refusal is worth one warning; a repeating cause must not flood the log."""
        emitted: list[str] = []
        sink_id = logger.add(lambda message: emitted.append(message.record["message"]), level="WARNING")
        try:
            model_map = HordeModelMap(root={})
            for _ in range(5):
                model_map.update_entry("", load_state=ModelLoadState.LOADING, process_id=1)
        finally:
            logger.remove(sink_id)

        assert len([line for line in emitted if "blank model name" in line]) == 1


class TestBudgetDependentFlagDisclosure:
    """Disabling the budget silently disables features that depend on it; say so once at startup."""

    @staticmethod
    def _disclosures(**bridge_overrides: object) -> list[str]:
        """Run the startup budget disclosure and return the warnings it emitted."""
        bridge_data = make_mock_bridge_data(**bridge_overrides)
        bridge_data.model_fields_set = set(bridge_overrides)
        process_manager = make_testable_process_manager(bridge_data=bridge_data)

        emitted: list[str] = []
        sink_id = logger.add(lambda message: emitted.append(message.record["message"]), level="WARNING")
        try:
            process_manager._log_resource_budget_posture()
        finally:
            logger.remove(sink_id)
        return emitted

    def test_inert_flags_are_named(self) -> None:
        """An explicitly enabled budget-dependent flag is named alongside the setting that disables it."""
        emitted = self._disclosures(
            enable_vram_budget=False,
            whole_card_exclusive_residency=True,
            overbudget_exclusive_mode=True,
        )

        inert_lines = [line for line in emitted if "inert while enable_vram_budget is false" in line]
        assert len(inert_lines) == 1
        assert "whole_card_exclusive_residency" in inert_lines[0]
        assert "overbudget_exclusive_mode" in inert_lines[0]

    def test_nothing_is_disclosed_when_no_dependent_flag_was_set(self) -> None:
        """A disabled budget with no explicitly enabled dependent flag says nothing extra."""
        emitted = self._disclosures(enable_vram_budget=False)

        assert [line for line in emitted if "inert while enable_vram_budget is false" in line] == []

    def test_nothing_is_disclosed_while_the_budget_is_active(self) -> None:
        """With the budget on, the dependent flags do what they say and need no disclosure."""
        emitted = self._disclosures(
            enable_vram_budget=True,
            whole_card_exclusive_residency=True,
        )

        assert [line for line in emitted if "inert while enable_vram_budget is false" in line] == []
