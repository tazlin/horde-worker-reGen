"""Wiring test: the dispatcher decodes each memory report's residency snapshot into the shared map.

Confirms the seam ``MessageDispatcher._handle_memory_report`` feeds the component-residency map when one is
registered, that an older child's report (no ``held_components``) leaves residency untouched, and that a
dispatcher without a map registered is unaffected.
"""

from __future__ import annotations

import queue
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot, HordeProcessMemoryMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.component_residency_map import ComponentResidencyMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
)


def _dispatcher(process_map: ProcessMap, residency: ComponentResidencyMap | None) -> MessageDispatcher:
    dispatcher = MessageDispatcher(
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=JobTracker(),
        process_message_queue=Mock(spec=queue.Queue),
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
        model_metadata=make_test_model_metadata(),
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        on_unload_vram=Mock(),
        state=WorkerState(),
    )
    if residency is not None:
        dispatcher.set_component_residency_map(residency)
    return dispatcher


def _memory_message(
    process_id: int,
    *,
    held: list[HeldComponentSnapshot] | None,
    launch: int = 0,
) -> HordeProcessMemoryMessage:
    return HordeProcessMemoryMessage(
        process_id=process_id,
        process_launch_identifier=launch,
        info="Memory report",
        ram_usage_bytes=1024,
        held_components=held,
    )


def test_report_with_held_components_populates_the_map() -> None:
    """A report carrying residency decodes into the shared map for that process."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name="ModelA")})
    residency = ComponentResidencyMap()
    dispatcher = _dispatcher(process_map, residency)

    dispatcher._handle_memory_report(
        _memory_message(1, held=[HeldComponentSnapshot(kind="checkpoint", identity="ModelA", approx_ram_mb=7000.0)]),
    )

    assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelA"})


def test_old_child_report_without_held_components_leaves_map_untouched() -> None:
    """An older child reports held_components=None, which never creates or clears an entry."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name="ModelA")})
    residency = ComponentResidencyMap()
    dispatcher = _dispatcher(process_map, residency)

    dispatcher._handle_memory_report(_memory_message(1, held=None))

    assert residency.identities_held() == frozenset()


def test_dispatcher_without_a_map_registered_is_unaffected() -> None:
    """A dispatcher with no residency map handles the report exactly as before (no error)."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name="ModelA")})
    dispatcher = _dispatcher(process_map, None)

    # Must not raise even though the message carries residency and no map is registered.
    dispatcher._handle_memory_report(
        _memory_message(1, held=[HeldComponentSnapshot(kind="checkpoint", identity="ModelA", approx_ram_mb=7000.0)]),
    )
