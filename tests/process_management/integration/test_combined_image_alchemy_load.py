"""Saturated combined image-generation + alchemy load under the shared resource budget.

These tests pin the behavior the multi-flow resource unification is meant to guarantee: image
generation and alchemy share one :class:`CommittedReserveLedger`, so the two flows account for each
other's in-flight VRAM/RAM and cannot independently admit work against the same free VRAM (the
double-commit each separate gate used to cause alone). They exercise the alchemy admission policy and
the scheduler/alchemy ledger wiring across concurrency, alchemy-limit, and resource-pressure axes,
deterministically and without a GPU.
"""

from __future__ import annotations

from collections import deque

import pytest

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec
from horde_worker_regen.process_management.jobs import alchemy_popper as alchemy_popper_module
from horde_worker_regen.process_management.jobs.alchemy_popper import AlchemyCoordinator, AlchemyHeadroomEstimator
from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.scheduling.workload_flow import FlowCoordinator, WorkloadKind
from tests.process_management.conftest import make_testable_process_manager
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# Flow namespace constants mirrored from the production code so a rename there fails a test here.
_IMAGE_FLOW = "image_post_processing"
_ALCHEMY_FLOW = "alchemy"


class _StubState:
    shutting_down = False
    supervisor_paused = False
    self_throttle_paused = False


class _StubRuntimeConfig:
    def __init__(self, bridge_data: reGenBridgeData) -> None:
        self.bridge_data = bridge_data


class _StubJobTracker:
    def __init__(self, *, pending: int = 0, in_progress: int = 0) -> None:
        self._pending = pending
        self._in_progress = in_progress

    @property
    def jobs_pending_inference(self) -> tuple[int, ...]:
        return tuple(range(self._pending))

    @property
    def jobs_in_progress(self) -> tuple[int, ...]:
        return tuple(range(self._in_progress))


class _IdleLane:
    def can_accept_job(self) -> bool:
        return True


class _StubProcessMap:
    """A process map that reports a fixed free-VRAM figure and a set of idle image lanes."""

    def __init__(self, *, idle_image_lanes: int = 4, free_vram_mb: float | None = 8000.0) -> None:
        self._image_lanes = [_IdleLane() for _ in range(idle_image_lanes)]
        self._free_vram_mb = free_vram_mb

    def get_first_available(self, capability: WorkerCapability) -> object:
        return object()  # a capable process always exists for these admission tests

    def get_capable_processes(self, capability: WorkerCapability) -> list[_IdleLane]:
        return self._image_lanes if capability is WorkerCapability.IMAGE_GEN else []

    def get_free_vram_mb(self) -> float | None:
        return self._free_vram_mb


def _bridge_data(**kwargs: object) -> reGenBridgeData:
    defaults: dict[str, object] = {"api_key": "0" * 22, "alchemist": True, "queue_size": 4}
    defaults.update(kwargs)
    return reGenBridgeData(**defaults)  # type: ignore[arg-type]


def _make_coordinator(
    *,
    bridge_data: reGenBridgeData,
    process_map: _StubProcessMap,
    job_tracker: _StubJobTracker,
    reserve_ledger: CommittedReserveLedger,
    in_flight: int = 0,
) -> AlchemyCoordinator:
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._runtime_config = _StubRuntimeConfig(bridge_data)  # type: ignore[assignment]
    coordinator._state = _StubState()  # type: ignore[assignment]
    coordinator._process_map = process_map  # type: ignore[assignment]
    coordinator._job_tracker = job_tracker  # type: ignore[assignment]
    coordinator._reserve_ledger = reserve_ledger
    coordinator._pending_forms = deque()
    coordinator._in_flight = {f"form-{i}": None for i in range(in_flight)}  # type: ignore[misc]
    coordinator._in_flight_owner = {}
    coordinator._pending_submits = deque()
    coordinator._form_time_popped = {}
    coordinator._estimator = AlchemyHeadroomEstimator()
    coordinator._last_pop_time = 0.0
    coordinator._pop_frequency = 4.0
    coordinator.num_forms_faulted = 0
    return coordinator


class TestSharedBudgetNoDoubleCommit:
    """The headline regression: the two flows subtract one another's committed VRAM."""

    def test_alchemy_defers_when_image_has_committed_the_budget(self) -> None:
        """An image post-processing reserve eats alchemy's headroom; clearing it restores the pop."""
        ledger = CommittedReserveLedger()
        # Free VRAM (3000) covers a 2000-MB alchemy form on its own.
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_vram_headroom_mb=2000),
            process_map=_StubProcessMap(free_vram_mb=3000.0),
            job_tracker=_StubJobTracker(pending=0),
            reserve_ledger=ledger,
        )
        assert coordinator._has_vram_headroom() is True

        # Image generation commits 2000 MB it has not yet allocated: effective free is now 1000, below floor.
        ledger.set(_IMAGE_FLOW, "aggregate", vram_mb=2000.0)
        assert coordinator._has_vram_headroom() is False

        # Once the image post-processing completes and releases its reserve, alchemy fits again.
        ledger.release(_IMAGE_FLOW, "aggregate")
        assert coordinator._has_vram_headroom() is True

    def test_scheduler_sees_alchemy_in_flight_reserve(self) -> None:
        """The image scheduler's combined committed figure includes in-flight alchemy forms."""
        ledger = CommittedReserveLedger()
        scheduler = _make_inference_scheduler()
        scheduler._reserve_ledger = ledger

        # No image post-processing in progress (empty process map), so the combined reserve is alchemy-only.
        assert scheduler._committed_vram_reserve_mb() == 0.0
        ledger.replace_flow(_ALCHEMY_FLOW, vram_mb_by_unit={"form-1": 1500.0, "form-2": 500.0})
        assert scheduler._committed_vram_reserve_mb() == 2000.0

    def test_manager_wires_one_shared_ledger(self) -> None:
        """The scheduler and alchemy coordinator are handed the very same ledger instance."""
        pm = make_testable_process_manager(alchemist=True)
        assert pm._inference_scheduler._reserve_ledger is pm._reserve_ledger
        assert pm._alchemy_coordinator._reserve_ledger is pm._reserve_ledger


class TestAlchemyAdmissionMatrix:
    """Alchemy admission across concurrency, alchemy-limit, and VRAM-pressure axes."""

    @pytest.mark.parametrize("allow_concurrent", [True, False])
    @pytest.mark.parametrize("max_concurrency", [1, 2])
    @pytest.mark.parametrize("committed_mb", [0.0, 5000.0])
    def test_should_pop_under_pressure(
        self,
        allow_concurrent: bool,
        max_concurrency: int,
        committed_mb: float,
    ) -> None:
        """A saturated worker pops alchemy only when the fairness *and* capacity layers both allow it."""
        ledger = CommittedReserveLedger()
        ledger.set(_IMAGE_FLOW, "aggregate", vram_mb=committed_mb)
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(
                alchemy_allow_concurrent=allow_concurrent,
                alchemy_max_concurrency=max_concurrency,
                alchemy_vram_headroom_mb=2000,
            ),
            # Spare image lanes exist; the gating under test is fairness (backfill) and VRAM capacity.
            process_map=_StubProcessMap(idle_image_lanes=4, free_vram_mb=6000.0),
            job_tracker=_StubJobTracker(pending=1, in_progress=0),
            reserve_ledger=ledger,
        )

        should = coordinator._should_pop()

        if not allow_concurrent:
            # Backfill: an image job is queued (pending=1), so alchemy always waits regardless of VRAM.
            assert should is False
            return
        # Concurrent: capacity decides. effective free = 6000 - committed; needs >= 2000 floor.
        assert should is (6000.0 - committed_mb >= 2000.0)

    def test_alchemy_in_flight_cap_blocks_pop(self) -> None:
        """alchemy_max_concurrency bounds in-flight forms even with ample VRAM and spare lanes."""
        ledger = CommittedReserveLedger()
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_max_concurrency=1),
            process_map=_StubProcessMap(idle_image_lanes=4, free_vram_mb=8000.0),
            job_tracker=_StubJobTracker(pending=0),
            reserve_ledger=ledger,
            in_flight=1,
        )
        assert coordinator._should_pop() is False


class TestAlchemyRamHeadroom:
    """The RAM analogue of the VRAM gate: alchemy is held back when effective available RAM is scarce."""

    def test_ample_ram_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With plenty of available RAM the gate does not hold alchemy back."""
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_ram_headroom_mb=4096),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=CommittedReserveLedger(),
        )
        monkeypatch.setattr(alchemy_popper_module.psutil, "virtual_memory", lambda: _FakeVmem(available_mb=8000.0))
        assert coordinator._has_ram_headroom() is True

    def test_defers_under_ram_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Low effective available RAM (after the committed-RAM subtraction) holds alchemy back."""
        ledger = CommittedReserveLedger()
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_ram_headroom_mb=4096),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=ledger,
        )
        monkeypatch.setattr(alchemy_popper_module.psutil, "virtual_memory", lambda: _FakeVmem(available_mb=8000.0))
        assert coordinator._has_ram_headroom() is True

        ledger.set(_IMAGE_FLOW, "aggregate", ram_mb=5000.0)  # effective available 3000 < 4096 floor
        assert coordinator._has_ram_headroom() is False


class TestFlowScaffolding:
    """The alchemy coordinator fits the generic flow contract a future audio/video flow plugs into."""

    def test_alchemy_coordinator_satisfies_flow_protocol(self) -> None:
        """AlchemyCoordinator exposes the FlowCoordinator surface (kind, num_in_flight, run)."""
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=CommittedReserveLedger(),
            in_flight=2,
        )
        assert isinstance(coordinator, FlowCoordinator)
        assert coordinator.kind is WorkloadKind.ALCHEMY
        assert coordinator.num_in_flight == 2


class TestSyncReserveLedger:
    """The alchemy reserve mirrors ``_in_flight`` exactly on every reconcile.

    This pins the reconcile contract in isolation by mutating ``_in_flight`` directly; what actually
    removes a form whose process died (the lost-form reaper) is exercised end to end in
    ``TestLostFormReaping``.
    """

    def test_sync_mirrors_in_flight_and_drops_departed_forms(self) -> None:
        """Each in-flight form is published; once a form leaves ``_in_flight`` its reserve is dropped."""
        ledger = CommittedReserveLedger()
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_vram_headroom_mb=1500),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=ledger,
        )
        # Graph forms (which allocate real VRAM); CLIP-form reserving is covered by TestFormCostClassification.
        coordinator._in_flight = {
            "form-a": AlchemyFormSpec(form_id="form-a", form="RealESRGAN_x4plus", source_image_base64="aGk="),
            "form-b": AlchemyFormSpec(form_id="form-b", form="RealESRGAN_x2plus", source_image_base64="aGk="),
        }
        coordinator._sync_reserve_ledger()
        # Cold-start prediction is the floor (1500) per form.
        assert ledger.total_vram_mb() == 3000.0

        # Once a form has left _in_flight (by result or by the reaper), the next reconcile drops its hold.
        del coordinator._in_flight["form-a"]
        coordinator._sync_reserve_ledger()
        assert ledger.total_vram_mb() == 1500.0


def _spec(form_id: str, form: str) -> AlchemyFormSpec:
    return AlchemyFormSpec(form_id=form_id, form=form, source_image_base64="aGk=")


class TestFormCostClassification:
    """The per-form reserve reflects what the form actually allocates, not a flat graph cost.

    Graph forms (upscalers, facefixers, strip_background) load post-processor weights into VRAM (and RAM)
    on an inference process. CLIP forms (caption, nsfw, interrogation) run on the
    safety process against an already-resident model, so dispatching one adds no not-yet-realised VRAM/RAM
    and must not subtract headroom that image generation could otherwise use.
    """

    def _coordinator(self, ledger: CommittedReserveLedger, **bridge_kwargs: object) -> AlchemyCoordinator:
        return _make_coordinator(
            bridge_data=_bridge_data(alchemy_vram_headroom_mb=2000, alchemy_ram_headroom_mb=3000, **bridge_kwargs),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=ledger,
        )

    def test_clip_form_reserves_no_vram(self) -> None:
        """A lone in-flight CLIP form contributes nothing to the committed VRAM total."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        coordinator._in_flight = {"c1": _spec("c1", "nsfw")}
        coordinator._sync_reserve_ledger()
        assert ledger.total_vram_mb() == 0.0

    def test_graph_form_reserves_the_predicted_cost(self) -> None:
        """A graph form reserves the estimator's prediction (the floor at cold start)."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        coordinator._in_flight = {"g1": _spec("g1", "RealESRGAN_x4plus")}
        coordinator._sync_reserve_ledger()
        assert ledger.total_vram_mb() == 2000.0

    def test_mixed_batch_reserves_only_the_graph_forms(self) -> None:
        """With one graph and two CLIP forms in flight, only the graph form's cost is reserved."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        coordinator._in_flight = {
            "g1": _spec("g1", "CodeFormers"),
            "c1": _spec("c1", "caption"),
            "c2": _spec("c2", "interrogation"),
        }
        coordinator._sync_reserve_ledger()
        assert ledger.total_vram_mb() == 2000.0


class TestAlchemyRamReserve:
    """In-flight graph forms keep weights resident in RAM and commit it to the shared ledger.

    Without this the ledger's RAM total stayed zero, so the committed-RAM subtraction the image scheduler's
    RAM gate and alchemy's own RAM headroom check perform was a no-op despite the docstrings advertising it.
    """

    def _coordinator(self, ledger: CommittedReserveLedger, **bridge_kwargs: object) -> AlchemyCoordinator:
        return _make_coordinator(
            bridge_data=_bridge_data(alchemy_vram_headroom_mb=2000, alchemy_ram_headroom_mb=3000, **bridge_kwargs),
            process_map=_StubProcessMap(),
            job_tracker=_StubJobTracker(),
            reserve_ledger=ledger,
        )

    def test_graph_form_commits_ram(self) -> None:
        """A graph form in flight reserves its RAM footprint (the configured floor) for other flows to see."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        coordinator._in_flight = {"g1": _spec("g1", "RealESRGAN_x4plus")}
        coordinator._sync_reserve_ledger()
        assert ledger.total_ram_mb() == 3000.0

    def test_clip_form_commits_no_ram(self) -> None:
        """A CLIP form runs against the resident safety-process model and commits no RAM."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        coordinator._in_flight = {"c1": _spec("c1", "nsfw")}
        coordinator._sync_reserve_ledger()
        assert ledger.total_ram_mb() == 0.0

    def test_committed_ram_self_gates_next_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A graph form's own committed RAM is subtracted by the RAM headroom check for the next form."""
        ledger = CommittedReserveLedger()
        coordinator = self._coordinator(ledger)
        monkeypatch.setattr(
            alchemy_popper_module.psutil,
            "virtual_memory",
            lambda: _FakeVmem(available_mb=5000.0),
        )
        # No alchemy in flight yet: 5000 available clears the 3000 floor.
        assert coordinator._has_ram_headroom() is True

        # One graph form now holds 3000 MB of RAM: effective available 2000 < 3000 floor.
        coordinator._in_flight = {"g1": _spec("g1", "RealESRGAN_x4plus")}
        coordinator._sync_reserve_ledger()
        assert coordinator._has_ram_headroom() is False


class _DispatchProcessInfo:
    """Stands in for a HordeProcessInfo at the dispatch + launch-liveness seam."""

    def __init__(self, process_id: int, launch: int) -> None:
        self.process_id = process_id
        self.process_launch_identifier = launch
        self.sent: list[object] = []

    def safe_send_message(self, message: object) -> bool:
        self.sent.append(message)
        return True


class _LivenessProcessMap:
    """A process map that routes dispatch by capability and can report a launch as dead.

    Mirrors the real :class:`ProcessMap` contract the coordinator touches: ``get_first_available`` for
    dispatch, ``is_launch_active`` for the lost-form reaper, and ``get_free_vram_mb`` for the headroom
    gate. ``kill`` drops a launch the way crash recovery does (the slot's launch identifier is no longer
    the live one), so a form dispatched to it can never receive a result.
    """

    def __init__(
        self,
        processes: dict[WorkerCapability, _DispatchProcessInfo],
        *,
        free_vram_mb: float = 8000.0,
    ) -> None:
        self._processes = processes
        self._free_vram_mb = free_vram_mb
        self._alive: dict[int, int] = {p.process_id: p.process_launch_identifier for p in processes.values()}

    def get_first_available(self, capability: WorkerCapability) -> _DispatchProcessInfo | None:
        return self._processes.get(capability)

    def is_launch_active(self, process_id: int, process_launch_identifier: int) -> bool:
        return self._alive.get(process_id) == process_launch_identifier

    def kill(self, process_id: int) -> None:
        self._alive.pop(process_id, None)

    def get_free_vram_mb(self) -> float | None:
        return self._free_vram_mb


class TestLostFormReaping:
    """A form dispatched to a process that then dies must not leak its shared-ledger reserve forever.

    This is the realistic counterpart to ``TestSyncReserveLedger``: nothing in production hand-deletes a
    lost form from ``_in_flight``. Only a result message (which a hard-crashed process never sends) or the
    reaper removes it, so without the reaper a single crash permanently subtracts an alchemy reserve from
    every image-generation VRAM admission decision.
    """

    def _dispatch_one(
        self,
        *,
        form_id: str,
        form: str,
        capability: WorkerCapability,
        process: _DispatchProcessInfo,
        ledger: CommittedReserveLedger,
    ) -> tuple[AlchemyCoordinator, _LivenessProcessMap]:
        process_map = _LivenessProcessMap({capability: process})
        coordinator = _make_coordinator(
            bridge_data=_bridge_data(alchemy_vram_headroom_mb=2000),
            process_map=process_map,  # type: ignore[arg-type]
            job_tracker=_StubJobTracker(),
            reserve_ledger=ledger,
        )
        coordinator._pending_forms.append(
            AlchemyFormSpec(form_id=form_id, form=form, source_image_base64="aGk="),
        )
        coordinator.dispatch_pending_forms()
        return coordinator, process_map

    def test_dispatch_records_owning_launch(self) -> None:
        """The real dispatch path records which process launch each in-flight form went to."""
        process = _DispatchProcessInfo(process_id=7, launch=3)
        coordinator, _ = self._dispatch_one(
            form_id="form-a",
            form="nsfw",
            capability=WorkerCapability.ALCHEMY_CLIP,
            process=process,
            ledger=CommittedReserveLedger(),
        )
        assert "form-a" in coordinator._in_flight
        assert coordinator._in_flight_owner["form-a"] == (7, 3)

    def test_reserve_released_when_owning_process_dies(self) -> None:
        """When the owning launch is gone, the reaper drops the form so its reserve self-heals to zero."""
        ledger = CommittedReserveLedger()
        process = _DispatchProcessInfo(process_id=7, launch=3)
        coordinator, process_map = self._dispatch_one(
            form_id="form-a",
            form="RealESRGAN_x4plus",
            capability=WorkerCapability.ALCHEMY_GRAPH,
            process=process,
            ledger=ledger,
        )
        coordinator._sync_reserve_ledger()
        assert ledger.total_vram_mb() == 2000.0  # cold-start floor reserved for the in-flight form

        # The process hard-crashes before sending a result: its launch is no longer active.
        process_map.kill(7)
        coordinator._reap_lost_in_flight_forms()
        coordinator._sync_reserve_ledger()

        assert "form-a" not in coordinator._in_flight
        assert "form-a" not in coordinator._in_flight_owner
        assert coordinator.num_forms_faulted == 1
        assert ledger.total_vram_mb() == 0.0

    def test_live_form_is_not_reaped(self) -> None:
        """A form whose process is still alive keeps its reserve across a reap pass."""
        ledger = CommittedReserveLedger()
        process = _DispatchProcessInfo(process_id=7, launch=3)
        coordinator, _ = self._dispatch_one(
            form_id="form-a",
            form="RealESRGAN_x4plus",
            capability=WorkerCapability.ALCHEMY_GRAPH,
            process=process,
            ledger=ledger,
        )
        coordinator._reap_lost_in_flight_forms()
        coordinator._sync_reserve_ledger()

        assert "form-a" in coordinator._in_flight
        assert coordinator.num_forms_faulted == 0
        assert ledger.total_vram_mb() == 2000.0


class _FakeVmem:
    def __init__(self, *, available_mb: float) -> None:
        self.available = int(available_mb * 1024 * 1024)
