"""Tests for ``HordeWorkerProcessManager._sample_system_memory``'s per-role bucketing of the process map.

Every child on the process map (inference, safety, and the component/VAE/post-processing/utilities lanes)
self-reports its RSS, so the sampler must attribute each type's RSS to its role rather than leaving a lane's
footprint to be mislabelled as other applications. The orchestrator and download RSS come from psutil reads
the tests do not pin.
"""

from __future__ import annotations

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.system_memory import (
    ROLE_COMPONENT,
    ROLE_INFERENCE,
    ROLE_POST_PROCESS,
    ROLE_SAFETY,
    ROLE_UTILITIES,
    ROLE_VAE_LANE,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_GB = 1024**3


def _add_process(process_manager: object, process_id: int, process_type: HordeProcessType, ram_bytes: int) -> None:
    info = make_mock_process_info(process_id, process_type=process_type)
    info.ram_usage_bytes = ram_bytes
    process_manager._process_map[process_id] = info  # type: ignore[attr-defined]


def test_sample_buckets_every_lane_type_into_its_role() -> None:
    """Each mapped child's RSS lands under its role: inference, safety, and the four auxiliary lanes."""
    process_manager = make_testable_process_manager()
    process_manager._process_map.clear()

    _add_process(process_manager, 1, HordeProcessType.INFERENCE, 8 * _GB)
    _add_process(process_manager, 2, HordeProcessType.INFERENCE, 7 * _GB)
    _add_process(process_manager, 3, HordeProcessType.SAFETY, 2 * _GB)
    _add_process(process_manager, 4, HordeProcessType.COMPONENT, 3 * _GB)
    _add_process(process_manager, 5, HordeProcessType.VAE_LANE, 2 * _GB)
    _add_process(process_manager, 6, HordeProcessType.POST_PROCESS, 4 * _GB)
    _add_process(process_manager, 7, HordeProcessType.UTILITIES, 1 * _GB)

    summary = process_manager._sample_system_memory()
    by_role = summary.worker_rss_by_role

    assert by_role[ROLE_INFERENCE] == 15 * _GB
    assert by_role[ROLE_SAFETY] == 2 * _GB
    assert by_role[ROLE_COMPONENT] == 3 * _GB
    assert by_role[ROLE_VAE_LANE] == 2 * _GB
    assert by_role[ROLE_POST_PROCESS] == 4 * _GB
    assert by_role[ROLE_UTILITIES] == 1 * _GB


def test_lane_rss_counts_toward_the_worker_subtotal() -> None:
    """A lane child's RSS raises the worker subtotal, so it is never left in the 'other' remainder."""
    process_manager = make_testable_process_manager()
    process_manager._process_map.clear()
    _add_process(process_manager, 1, HordeProcessType.VAE_LANE, 5 * _GB)

    summary = process_manager._sample_system_memory()

    # The orchestrator/download reads are non-negative, so the subtotal is at least the lane's own RSS.
    assert summary.worker_rss_by_role[ROLE_VAE_LANE] == 5 * _GB
    assert summary.worker_total_bytes >= 5 * _GB
