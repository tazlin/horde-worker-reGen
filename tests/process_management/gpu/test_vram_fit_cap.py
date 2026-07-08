"""Tests for the per-card VRAM-fit inference-process cap (:func:`cap_card_processes_to_vram_fit`).

The resolved per-card plan (``queue_size + ceiling``) can ask a card to spawn more inference contexts than
its VRAM physically holds. This cap bounds each card independently (single-GPU included) to the contexts that
fit ``contexts * idle_overhead + working_set <= total - noise_buffer``, only ever reducing a card and never
below one context. The working set is one typical (SDXL-class) model, not the largest model the worker may
load: very large models never co-sample and their footprint is paid just-in-time by whole-card residency at
dispatch, so they do not raise this spawn bound. It composes with the worker-wide shared-RAM cap and is
exempted for the benchmark's deliberately elevated concurrency ceiling.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    SystemResources,
    _EstimatedContextFootprint,
    cap_card_processes_to_vram_fit,
)
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_sd_reference,
    make_test_mp_primitives,
)

_GB = 1024 * 1024 * 1024
_VRAM_8GB_MB = 8192.0
_VRAM_16GB_MB = 16384.0
_VRAM_24GB_MB = 24576.0

_IDLE = _EstimatedContextFootprint.IDLE_CONTEXT_VRAM_MB
_SDXL = _EstimatedContextFootprint.SDXL_CONTEXT_VRAM_MB
_HEAVY = _EstimatedContextFootprint.HEAVY_CONTEXT_VRAM_MB


def _fit(
    per_card: dict[int, int],
    *,
    vram_by_card: dict[int, float | None],
    idle_mb: float = _IDLE,
    working_set_mb: float = _SDXL,
) -> dict[int, int]:
    return cap_card_processes_to_vram_fit(
        per_card_target_processes=per_card,
        total_vram_mb_by_card=vram_by_card,
        idle_context_overhead_mb=idle_mb,
        working_set_footprint_mb=working_set_mb,
    )


def _expected_max_contexts(total_mb: float, *, idle_mb: float, working_set_mb: float) -> int:
    usable = total_mb - admission_noise_buffer_mb(total_mb) - working_set_mb
    return max(1, int(usable // idle_mb))


class TestVramFitArithmetic:
    """The context ceiling is ``floor((total - noise - working_set) / idle)``, clamped to at least one."""

    def test_small_card_clamps_to_one(self) -> None:
        """An 8GB card cannot host an idle context beside an SDXL working set, so any plan collapses to one."""
        assert _expected_max_contexts(_VRAM_8GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL) == 1
        assert _fit({0: 4}, vram_by_card={0: _VRAM_8GB_MB}) == {0: 1}

    def test_16gb_card_holds_two_contexts(self) -> None:
        """A 16GB card fits two idle SDXL contexts plus one resident working set."""
        assert _expected_max_contexts(_VRAM_16GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL) == 2
        assert _fit({0: 4}, vram_by_card={0: _VRAM_16GB_MB}) == {0: 2}

    def test_big_card_unaffected_within_its_fit(self) -> None:
        """A 24GB card whose plan is already within its VRAM fit is left untouched (the cap only reduces)."""
        ceiling = _expected_max_contexts(_VRAM_24GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL)
        assert _fit({0: 2}, vram_by_card={0: _VRAM_24GB_MB}) == {0: 2}
        assert _fit({0: ceiling + 4}, vram_by_card={0: _VRAM_24GB_MB}) == {0: ceiling}

    def test_larger_working_set_lowers_the_ceiling(self) -> None:
        """A larger charged working set caps a 24GB card harder, but production always charges one SDXL set.

        The heavy-family figure is used here only as an arbitrary larger value to prove the arithmetic responds
        to the working-set term; the sizing path never charges it, since very large models are reclaimed for at
        dispatch, not reserved at spawn.
        """
        heavy_ceiling = _expected_max_contexts(_VRAM_24GB_MB, idle_mb=_IDLE, working_set_mb=_HEAVY)
        assert heavy_ceiling < _expected_max_contexts(_VRAM_24GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL)
        assert _fit({0: 6}, vram_by_card={0: _VRAM_24GB_MB}, working_set_mb=_HEAVY) == {0: heavy_ceiling}

    def test_never_below_one(self) -> None:
        """Even when the working set alone nearly fills the card, a card keeps one context to serve at all."""
        assert _fit({0: 3}, vram_by_card={0: _VRAM_8GB_MB}, working_set_mb=_VRAM_8GB_MB) == {0: 1}

    def test_only_reduces(self) -> None:
        """A card already at one context on a roomy card stays at one (the cap never raises a plan)."""
        assert _fit({0: 1}, vram_by_card={0: _VRAM_24GB_MB}) == {0: 1}

    def test_zero_idle_overhead_abstains(self) -> None:
        """A non-positive idle-overhead estimate cannot form a ceiling, so the plan passes through unchanged."""
        assert _fit({0: 4}, vram_by_card={0: _VRAM_8GB_MB}, idle_mb=0.0) == {0: 4}


class TestVramFitPerCardIndependence:
    """Each card is capped against its own VRAM, unknown-capacity cards abstain."""

    def test_cards_capped_independently(self) -> None:
        """A 24GB and an 8GB card in one call are bounded by their own totals, not a shared figure."""
        capped = _fit({0: 3, 1: 3}, vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_8GB_MB})
        assert capped == {0: 3, 1: 1}

    def test_unknown_capacity_abstains(self) -> None:
        """A card whose VRAM is unknown is not VRAM-capped; the runtime budget still gates its loads."""
        assert _fit({0: 4, 1: 4}, vram_by_card={0: None, 1: _VRAM_8GB_MB}) == {0: 4, 1: 1}


def _single_card_manager(
    *,
    total_vram_gb: int,
    total_ram_gb: int = 64,
    max_threads_ceiling: int | None = None,
    **bridge_overrides: object,
) -> HordeWorkerProcessManager:
    """Build a GPU-free manager on one fake card of the given VRAM/RAM, exercising the real sizing path.

    ``total_ram_gb`` defaults to a roomy 64GB so the VRAM-fit arithmetic is isolated from the shared-RAM cap;
    a test that means to exercise the RAM cap lowers it explicitly.
    """
    resources = SystemResources(
        total_ram_bytes=total_ram_gb * _GB,
        device_map=TorchDeviceMap(
            root={0: TorchDeviceInfo(device_name="FitGPU", device_index=0, total_memory=total_vram_gb * _GB)},
        ),
    )
    return HordeWorkerProcessManager(
        ctx=Mock(),
        bridge_data=make_mock_bridge_data(**bridge_overrides),
        horde_model_reference_manager=Mock(),
        max_safety_processes=1,
        system_resources=resources,
        mp_primitives=make_test_mp_primitives(),
        skip_api_init=True,
        stable_diffusion_reference=make_mock_sd_reference(),
        max_threads_ceiling=max_threads_ceiling,
    )


class TestSingleGpuSizingIntegration:
    """The caps run on a single card through the real ``_build_card_runtimes`` path (the over-config gap)."""

    def test_single_gpu_over_config_is_capped(self) -> None:
        """A single 24GB card configured for eight contexts is right-sized to the four its VRAM fits."""
        manager = _single_card_manager(
            total_vram_gb=24,
            image_models_to_load=["model_a", "model_b"],
            queue_size=6,
            max_threads=2,
        )
        expected = _expected_max_contexts(_VRAM_24GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL)
        assert expected == 4
        assert manager._card_runtimes[0].target_process_count == expected
        assert manager.max_inference_processes == expected

    def test_extra_large_models_in_mix_still_size_to_four(self) -> None:
        """A 24GB/64GB threads=2 queue=2 worker offering a very large model still sizes to four contexts.

        The very large model does not inflate the spawn count: it never co-samples and its footprint is claimed
        just-in-time by whole-card residency, so the card keeps the spare contexts that preload the next model.
        """
        manager = _single_card_manager(
            total_vram_gb=24,
            total_ram_gb=64,
            image_models_to_load=["model_a", "Flux.1-Schnell fp8 (Compact)"],
            queue_size=2,
            max_threads=2,
        )
        assert manager._card_runtimes[0].target_process_count == 4

    def test_single_gpu_ram_bound_caps_below_the_vram_verdict(self) -> None:
        """A 32GB host caps a single 24GB card below the four contexts its VRAM alone would allow."""
        vram_verdict = _expected_max_contexts(_VRAM_24GB_MB, idle_mb=_IDLE, working_set_mb=_SDXL)
        assert vram_verdict == 4
        manager = _single_card_manager(
            total_vram_gb=24,
            total_ram_gb=32,
            image_models_to_load=["model_a", "model_b"],
            queue_size=2,
            max_threads=2,
        )
        # 32GB net of the 9GB overhead holds ~23GB, ~two 9GB resident contexts, below the VRAM verdict of four.
        assert manager._card_runtimes[0].target_process_count == 2
        assert manager._card_runtimes[0].target_process_count < vram_verdict

    def test_single_gpu_within_vram_not_capped(self) -> None:
        """A single 24GB card whose small plan already fits keeps every configured context."""
        manager = _single_card_manager(
            total_vram_gb=24,
            image_models_to_load=["model_a", "model_b"],
            queue_size=1,
            max_threads=1,
        )
        assert manager._card_runtimes[0].target_process_count == 2

    def test_benchmark_ceiling_is_exempt(self) -> None:
        """An elevated max_threads_ceiling (the benchmark's provisioning signal) bypasses both caps."""
        manager = _single_card_manager(
            total_vram_gb=24,
            total_ram_gb=32,
            image_models_to_load=["model_a", "model_b"],
            queue_size=6,
            max_threads=2,
            max_threads_ceiling=8,
        )
        # ceiling 8 leaks into the plan (queue 6 + ceiling 8 = 14) and neither the VRAM nor the RAM cap clamps it.
        assert manager._card_runtimes[0].target_process_count == 14
