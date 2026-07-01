"""Tests for the plan-time per-card inference-process cap (:func:`cap_card_process_counts`).

The resolved per-card plan (``queue_size + ceiling``) is sound per card but, summed across cards,
double-counts the single shared system-RAM pool and ignores the post-processing VRAM peak. The cap lowers
the per-card target so each card keeps post-processing headroom and the worker-wide resident-context count
fits system RAM, never below one context per card and only ever reducing the resolved plan.
"""

from __future__ import annotations

from horde_worker_regen.process_management.process_manager import (
    _EstimatedContextFootprint,
    cap_card_process_counts,
)

_GIB = 1024 * 1024 * 1024
_VRAM_16GB_MB = 16384.0
_VRAM_24GB_MB = 24576.0
# The worker keeps min(total/2, 9GB) free; on a 64GB host that is 9GB, leaving ~55GB for resident contexts.
_OVERHEAD_9GB = 9 * _GIB


def _cap(
    per_card: dict[int, int],
    *,
    total_ram_bytes: int,
    overhead_bytes: int,
    vram_by_card: dict[int, float | None],
    allow_post_processing: bool = True,
    has_vram_heavy_models: bool = False,
) -> dict[int, int]:
    """Invoke the cap with explicit hardware so the expected counts are deterministic on any host."""
    return cap_card_process_counts(
        per_card_target_processes=per_card,
        total_ram_bytes=total_ram_bytes,
        target_ram_overhead_bytes=overhead_bytes,
        total_vram_mb_by_card=vram_by_card,
        allow_post_processing=allow_post_processing,
        has_vram_heavy_models=has_vram_heavy_models,
    )


class TestPerCardVramCap:
    """Each card must keep the post-processing peak free on top of its resident contexts."""

    def test_two_16gb_cards_with_post_processing_cap_to_one_each(self) -> None:
        """A 16GB card cannot hold two SDXL contexts plus the ~8.5GB upscale peak, so it caps to one."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_16GB_MB, 1: _VRAM_16GB_MB},
        )
        assert capped == {0: 1, 1: 1}

    def test_post_processing_disabled_frees_the_reserved_headroom(self) -> None:
        """Without post-processing there is no upscale peak to reserve, so a 16GB card keeps two contexts."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_16GB_MB, 1: _VRAM_16GB_MB},
            allow_post_processing=False,
        )
        assert capped == {0: 2, 1: 2}

    def test_24gb_cards_keep_two_contexts(self) -> None:
        """A 24GB card has room for two SDXL contexts plus the post-processing peak."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert capped == {0: 2, 1: 2}

    def test_heavy_models_raise_the_per_context_estimate(self) -> None:
        """A VRAM-heavy family (Flux/Cascade) makes a context cost more, capping even a 24GB card to one."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
            has_vram_heavy_models=True,
        )
        assert capped == {0: 1, 1: 1}

    def test_unknown_card_capacity_abstains(self) -> None:
        """A card whose VRAM is unknown is not VRAM-capped; only the worker-wide RAM cap can apply."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: None, 1: None},
        )
        assert capped == {0: 2, 1: 2}


class TestWorkerWideRamCap:
    """The resident contexts across every card must fit the single shared system-RAM pool."""

    def test_ram_bound_trims_to_one_per_card(self) -> None:
        """Two 24GB cards fit VRAM-wise, but a 20GB host cannot hold four contexts' RAM, so it trims to one each."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=20 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert capped == {0: 1, 1: 1}

    def test_never_trims_below_one_per_card(self) -> None:
        """Even when RAM cannot hold one context per card, each card keeps one (it must to serve at all)."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=8 * _GIB,
            overhead_bytes=4 * _GIB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert capped == {0: 1, 1: 1}

    def test_trims_the_most_provisioned_card_first(self) -> None:
        """The worker-wide trim sheds the most-provisioned card's discretionary slot before an even card."""
        capped = _cap(
            {0: 3, 1: 1},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=64 * _GIB - int(3 * _EstimatedContextFootprint.CONTEXT_RAM_MB * 1024 * 1024),
            vram_by_card={0: None, 1: None},
        )
        # Usable RAM holds three contexts; the input asks for four, so one is trimmed from the larger card.
        assert capped == {0: 2, 1: 1}


class TestOnlyReduces:
    """The cap is a one-way valve: it never raises a card above its resolved plan."""

    def test_does_not_increase_a_small_plan(self) -> None:
        """A card already at one context on a roomy 24GB host stays at one (the cap only reduces)."""
        capped = _cap(
            {0: 1, 1: 1},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert capped == {0: 1, 1: 1}


class TestCoTenantReserveShrinksTheContextBudget:
    """A worker co-hosting an alchemist and/or scribe must reserve their RAM before sizing image contexts.

    The field OOM ran a dreamer, an alchemist, and a scribe on one 64GB host: sizing the resident-context count
    as if the whole pool were the image worker's over-committed the shared RAM. The co-tenant reserve raises the
    RAM overhead so the worker-wide cap trims contexts to what is actually left.
    """

    def test_image_only_worker_reserves_nothing(self) -> None:
        """An image-only worker keeps the prior overhead (byte-identical sizing)."""
        from unittest.mock import Mock

        from horde_worker_regen.process_management.process_manager import co_tenant_ram_reserve_bytes
        from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

        bridge_data = Mock()
        bridge_data.scribe_name = None
        bridge_data.alchemy_ram_headroom_mb = 2048
        # No alchemy workload enabled.
        import horde_worker_regen.process_management.process_manager as pm

        original = pm.enabled_workloads
        pm.enabled_workloads = lambda _bd: frozenset({WorkloadKind.IMAGE_GENERATION})
        try:
            assert co_tenant_ram_reserve_bytes(bridge_data) == 0
        finally:
            pm.enabled_workloads = original

    def test_alchemist_and_scribe_each_add_a_reserve(self) -> None:
        """An alchemist reserves at least its floor and a configured scribe adds its own floor on top."""
        from unittest.mock import Mock

        import horde_worker_regen.process_management.process_manager as pm
        from horde_worker_regen.process_management.process_manager import (
            _ALCHEMIST_CO_TENANT_RAM_BYTES,
            _SCRIBE_CO_TENANT_RAM_BYTES,
            co_tenant_ram_reserve_bytes,
        )
        from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

        bridge_data = Mock()
        bridge_data.scribe_name = "my-scribe"
        bridge_data.alchemy_ram_headroom_mb = 2048  # below the alchemist floor, so the floor governs

        original = pm.enabled_workloads
        pm.enabled_workloads = lambda _bd: frozenset({WorkloadKind.IMAGE_GENERATION, WorkloadKind.ALCHEMY})
        try:
            reserve = co_tenant_ram_reserve_bytes(bridge_data)
        finally:
            pm.enabled_workloads = original

        assert reserve == _ALCHEMIST_CO_TENANT_RAM_BYTES + _SCRIBE_CO_TENANT_RAM_BYTES

    def test_raised_overhead_trims_a_context(self) -> None:
        """On a host where the RAM pool binds, reserving co-tenant RAM trims the worker-wide context budget.

        The coarse RAM cap barely binds on a very large host (where the per-process ceiling is the live guard),
        so this uses a 40GB host to exercise the sizing effect: the same two 24GB cards that hold three contexts
        between them at the base overhead lose one once the alchemist/scribe RAM is reserved.
        """
        # Baseline: 9GB overhead on 40GB leaves room for three contexts.
        baseline = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=40 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert sum(baseline.values()) == 3
        # With ~17GB reserved (9GB base + 8GB alchemist+scribe), usable RAM holds fewer contexts.
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=40 * _GIB,
            overhead_bytes=_OVERHEAD_9GB + 8 * _GIB,
            vram_by_card={0: _VRAM_24GB_MB, 1: _VRAM_24GB_MB},
        )
        assert sum(capped.values()) < sum(baseline.values()), (
            "reserving co-tenant RAM must trim the worker-wide context budget where the RAM pool binds"
        )
