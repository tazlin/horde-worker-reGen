"""Tests for the per-card multi-GPU snapshot projection (``_build_card_snapshots`` + ``ProcessSnapshot``).

The worker projects per-card VRAM/contexts/residency/fault state onto :class:`CardSnapshot` so the dashboard
can render a per-card view. A single-GPU host reports exactly one (collapsed) card keyed worker-wide; a
multi-GPU host reports one per card with the streak/jobs/residency facts keyed by real device index.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SUPERVISOR_PROTOCOL_VERSION,
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap

from .conftest import (
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_testable_process_manager,
)


def test_protocol_version_is_9() -> None:
    """The supervisor protocol is at v9 (per-card data landed at v8; feature readiness bumped it to v9)."""
    assert SUPERVISOR_PROTOCOL_VERSION == 9
    snapshot = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"))
    assert snapshot.protocol_version == 9
    assert snapshot.per_card == []


def test_process_snapshot_carries_device_index() -> None:
    """A process pinned to card 1 projects ``device_index=1`` so the dashboard can group its slots."""
    proc = make_mock_process_info(2, device_index=1)
    assert ProcessSnapshot.from_process_info(proc).device_index == 1


def test_single_gpu_reports_one_collapsed_card() -> None:
    """A single-GPU host projects exactly one card (the collapsed card) with its real device facts."""
    pm = make_testable_process_manager()

    cards = pm._build_card_snapshots()

    assert len(cards) == 1
    card = cards[0]
    assert card.device_index == 0
    assert card.device_name == "TestGPU"
    assert card.kind == "cuda"
    # No process has reported VRAM, so the total falls back to the card runtime's device-map capacity (8 GiB).
    assert card.total_vram_mb == 8192.0
    assert card.free_vram_mb is None
    assert card.residency_model is None
    assert card.unservable_models == []
    assert card.worst_fault_streak == 0
    assert card.jobs_completed == 0


def test_single_gpu_jobs_completed_counts_under_none_key() -> None:
    """On a single-GPU host the per-card jobs/hr source reads the worker-wide (None-keyed) completion count."""
    pm = make_testable_process_manager()
    pm._job_tracker.note_card_inference_result(None)
    pm._job_tracker.note_card_inference_result(None)

    assert pm._build_card_snapshots()[0].jobs_completed == 2


def test_multi_gpu_projects_each_card_independently() -> None:
    """A two-card host projects per-card VRAM pressure, unservable models, and jobs keyed by device index."""
    pm = make_testable_process_manager()
    config0 = make_mock_bridge_data(image_models_to_load=["m0"])
    config1 = make_mock_bridge_data(image_models_to_load=["m1"])
    pm._card_runtimes = {
        0: make_test_card_runtimes(device_indices=(0,), config=config0, total_vram_mb=24576.0)[0],
        1: make_test_card_runtimes(device_indices=(1,), config=config1, total_vram_mb=12288.0)[1],
    }
    pm._device_map = TorchDeviceMap(
        root={
            0: TorchDeviceInfo(device_name="NVIDIA GeForce RTX 4090", device_index=0, total_memory=24 * 1024**3),
            1: TorchDeviceInfo(device_name="NVIDIA GeForce RTX 3090", device_index=1, total_memory=12 * 1024**3),
        },
    )

    # A busy slot on card 1, reporting almost-full VRAM (free 500 MB -> pressured).
    busy = make_mock_process_info(2, model_name="m1", state=HordeProcessState.INFERENCE_STARTING, device_index=1)
    busy.total_vram_mb = 12000
    busy.vram_usage_mb = 11500
    pm._process_map[2] = busy
    pm._process_map[3] = make_mock_process_info(3, model_name="m0", device_index=0)

    # Card 1's m1 has tripped its over-budget breaker; card 0 has finished a couple of jobs.
    for _ in range(3):
        pm._job_tracker._record_resource_fault("m1", device_index=1)
    pm._job_tracker.note_card_inference_result(0)
    pm._job_tracker.note_card_inference_result(0)

    cards = {card.device_index: card for card in pm._build_card_snapshots()}
    assert set(cards) == {0, 1}

    card0, card1 = cards[0], cards[1]
    assert card0.device_name == "NVIDIA GeForce RTX 4090"
    assert card0.jobs_completed == 2
    assert card0.unservable_models == []
    assert card0.busy_contexts == 0

    assert card1.device_name == "NVIDIA GeForce RTX 3090"
    assert card1.busy_contexts == 1
    assert card1.free_vram_mb == 500.0
    assert card1.is_vram_pressured is True
    assert card1.unservable_models == ["m1"]
    assert card1.worst_fault_streak == 3
    # The streak is keyed to card 1 only: card 0 still serves m1.
    assert card0.worst_fault_streak == 0
