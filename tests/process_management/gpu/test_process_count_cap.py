"""Tests for the worker-wide shared-RAM inference-process cap (:func:`cap_card_process_counts`).

The resolved per-card plan (``queue_size + ceiling``) is sound per card but, summed across cards, double-counts
the single shared system-RAM pool (a second card doubles VRAM, not RAM), and a single card can over-commit the
pool alone. The cap lowers the per-card target so the worker-wide resident-context count fits system RAM, never
below one context per card and only ever reducing the resolved plan. Per-card VRAM fit is an orthogonal bound
tested in ``test_vram_fit_cap.py``. The transient post-processing peak is not reserved here; it is charged as a
dispatch-time entry cost by the runtime post-processing reclaim machinery.
"""

from __future__ import annotations

from horde_worker_regen.process_management.process_manager import (
    _EstimatedContextFootprint,
    cap_card_process_counts,
)

_GIB = 1024 * 1024 * 1024
# The worker keeps min(total/2, 9GB) free; on a 64GB host that is 9GB, leaving ~55GB for resident contexts.
_OVERHEAD_9GB = 9 * _GIB


def _cap(
    per_card: dict[int, int],
    *,
    total_ram_bytes: int,
    overhead_bytes: int,
) -> dict[int, int]:
    """Invoke the cap with explicit RAM so the expected counts are deterministic on any host."""
    return cap_card_process_counts(
        per_card_target_processes=per_card,
        total_ram_bytes=total_ram_bytes,
        target_ram_overhead_bytes=overhead_bytes,
    )


class TestWorkerWideRamCap:
    """The resident contexts across every card must fit the single shared system-RAM pool."""

    def test_ram_bound_trims_to_one_per_card(self) -> None:
        """A 20GB host cannot hold four resident contexts' RAM, so two cards trim to one each."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=20 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert capped == {0: 1, 1: 1}

    def test_never_trims_below_one_per_card(self) -> None:
        """Even when RAM cannot hold one context per card, each card keeps one (it must to serve at all)."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=8 * _GIB,
            overhead_bytes=4 * _GIB,
        )
        assert capped == {0: 1, 1: 1}

    def test_trims_the_most_provisioned_card_first(self) -> None:
        """The worker-wide trim sheds the most-provisioned card's discretionary slot before an even card."""
        capped = _cap(
            {0: 3, 1: 1},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=64 * _GIB - int(3 * _EstimatedContextFootprint.CONTEXT_RAM_MB * 1024 * 1024),
        )
        # Usable RAM holds three contexts; the input asks for four, so one is trimmed from the larger card.
        assert capped == {0: 2, 1: 1}

    def test_roomy_host_leaves_the_plan_intact(self) -> None:
        """A 64GB host holds the full four-context plan across two cards, so nothing is trimmed."""
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert capped == {0: 2, 1: 2}


class TestSingleCardRamCap:
    """A single card's plan can over-commit the shared RAM pool on its own, so the cap applies to one host too."""

    def test_single_card_ram_bound_trims_the_plan(self) -> None:
        """A 32GB host keeps ~23GB usable, holding two contexts, so a single card's four-context plan trims."""
        capped = _cap(
            {0: 4},
            total_ram_bytes=32 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert capped == {0: 2}

    def test_single_card_roomy_host_untouched(self) -> None:
        """A 64GB host holds a single card's four-context plan, so it is left intact."""
        capped = _cap(
            {0: 4},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert capped == {0: 4}


class TestOnlyReduces:
    """The cap is a one-way valve: it never raises a card above its resolved plan."""

    def test_does_not_increase_a_small_plan(self) -> None:
        """A card already at one context on a roomy host stays at one (the cap only reduces)."""
        capped = _cap(
            {0: 1, 1: 1},
            total_ram_bytes=64 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert capped == {0: 1, 1: 1}


class TestCoTenantReserveShrinksTheContextBudget:
    """A worker co-hosting an alchemist and/or scribe must reserve their RAM before sizing image contexts.

    Sizing the resident-context count as if the whole pool were the image worker's over-commits the shared RAM
    when a dreamer, an alchemist, and a scribe share one host. The co-tenant reserve raises the RAM overhead so
    the worker-wide cap trims contexts to what is actually left.
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
        pm.enabled_workloads = lambda bridge_data: frozenset({WorkloadKind.IMAGE_GENERATION})
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
        pm.enabled_workloads = lambda bridge_data: frozenset({WorkloadKind.IMAGE_GENERATION, WorkloadKind.ALCHEMY})
        try:
            reserve = co_tenant_ram_reserve_bytes(bridge_data)
        finally:
            pm.enabled_workloads = original

        assert reserve == _ALCHEMIST_CO_TENANT_RAM_BYTES + _SCRIBE_CO_TENANT_RAM_BYTES

    def test_raised_overhead_trims_a_context(self) -> None:
        """On a host where the RAM pool binds, reserving co-tenant RAM trims the worker-wide context budget.

        The coarse RAM cap barely binds on a very large host (where the per-process ceiling is the live guard),
        so this uses a 40GB host to exercise the sizing effect: the same two cards that hold three contexts
        between them at the base overhead lose one once the alchemist/scribe RAM is reserved.
        """
        # Baseline: 9GB overhead on 40GB leaves room for three contexts.
        baseline = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=40 * _GIB,
            overhead_bytes=_OVERHEAD_9GB,
        )
        assert sum(baseline.values()) == 3
        # With ~17GB reserved (9GB base + 8GB alchemist+scribe), usable RAM holds fewer contexts.
        capped = _cap(
            {0: 2, 1: 2},
            total_ram_bytes=40 * _GIB,
            overhead_bytes=_OVERHEAD_9GB + 8 * _GIB,
        )
        assert sum(capped.values()) < sum(baseline.values()), (
            "reserving co-tenant RAM must trim the worker-wide context budget where the RAM pool binds"
        )
