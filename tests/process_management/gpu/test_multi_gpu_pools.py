"""Tests for per-card process pools, semaphores, and masking gating (Phase A3, increment 2).

Built around the GPU-free process-manager harness: a faked multi-card device map drives the per-card
CardRuntime plan without any real GPU. The single-card path is asserted to stay identical to before
multi-GPU existed (one runtime, no masking, the same process count).
"""

from __future__ import annotations

from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from tests.process_management.conftest import make_test_mp_primitives, make_testable_process_manager

_GB = 1024 * 1024 * 1024


def _system_resources(*cards: tuple[int, int, str]) -> SystemResources:
    """Build a SystemResources from (index, total_vram_gb, kind) tuples."""
    return SystemResources(
        total_ram_bytes=64 * _GB,
        device_map=TorchDeviceMap(
            root={
                index: TorchDeviceInfo(
                    device_name=f"GPU{index}",
                    device_index=index,
                    total_memory=vram_gb * _GB,
                    kind=kind,
                )
                for index, vram_gb, kind in cards
            },
        ),
    )


class TestSingleCardInvariance:
    """A single-GPU host behaves exactly as before: one runtime, no masking, same process count."""

    def test_one_card_one_runtime_unmasked(self) -> None:
        """The default single fake card yields one CardRuntime with masking disabled."""
        manager = make_testable_process_manager()
        assert sorted(manager._card_runtimes) == [0]
        assert manager._card_runtimes[0].mask_kind is None  # default single-GPU is never masked
        assert manager._card_runtimes[0].device_index == 0

    def test_explicit_single_selection_is_masked(self) -> None:
        """An explicit gpu_device_indices selection masks even a single card (so it pins the chosen slot)."""
        manager = make_testable_process_manager(gpu_device_indices=[0])
        assert manager._card_runtimes[0].mask_kind == "cuda"


class TestMultiCardPools:
    """Two cards yield two independently-sized, independently-gated pools."""

    def test_two_cards_build_two_runtimes(self) -> None:
        """A heterogeneous 24GB+8GB pair produces a runtime per card, keyed by stable index."""
        manager = make_testable_process_manager(
            system_resources=_system_resources((0, 24, "cuda"), (1, 8, "cuda")),
            mp_primitives=make_test_mp_primitives(device_indices=(0, 1)),
        )
        assert sorted(manager._card_runtimes) == [0, 1]

    def test_process_count_is_summed_across_cards(self) -> None:
        """The total inference process count is the sum of every card's target (1 + 1 here)."""
        manager = make_testable_process_manager(
            system_resources=_system_resources((0, 24, "cuda"), (1, 8, "cuda")),
            mp_primitives=make_test_mp_primitives(device_indices=(0, 1)),
        )
        expected = sum(card.target_process_count for card in manager._card_runtimes.values())
        assert manager.max_inference_processes == expected
        assert expected == 2  # each card: 1 model + max_threads 1 -> the single-model/thread collapse to 1

    def test_each_card_has_its_own_semaphores(self) -> None:
        """The two cards must not share a semaphore, so one card's sampling cannot block the other's."""
        manager = make_testable_process_manager(
            system_resources=_system_resources((0, 24, "cuda"), (1, 8, "cuda")),
            mp_primitives=make_test_mp_primitives(device_indices=(0, 1)),
        )
        card0, card1 = manager._card_runtimes[0], manager._card_runtimes[1]
        assert card0.inference_semaphore is not card1.inference_semaphore
        assert card0.vae_decode_semaphore is not card1.vae_decode_semaphore
        assert card0.gpu_sampling_lease is not card1.gpu_sampling_lease

    def test_multi_card_enables_masking_with_each_card_kind(self) -> None:
        """Driving more than one card masks every process, each to its own backend kind."""
        manager = make_testable_process_manager(
            system_resources=_system_resources((0, 24, "cuda"), (1, 8, "rocm")),
            mp_primitives=make_test_mp_primitives(device_indices=(0, 1)),
        )
        assert manager._card_runtimes[0].mask_kind == "cuda"
        assert manager._card_runtimes[1].mask_kind == "rocm"

    def test_lifecycle_spawn_plan_matches_runtimes(self) -> None:
        """The lifecycle manager's total inference-process target equals the summed per-card plan."""
        manager = make_testable_process_manager(
            system_resources=_system_resources((0, 24, "cuda"), (1, 8, "cuda")),
            mp_primitives=make_test_mp_primitives(device_indices=(0, 1)),
        )
        assert manager._process_lifecycle._max_inference_processes == manager.max_inference_processes
