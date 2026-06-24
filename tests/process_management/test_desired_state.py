"""Unit tests for the desired-on-disk-set authority (``DesiredState`` / ``reconcile``).

These pin the diff semantics the parent relies on to drive the download process: that the picker and config
share one authoritative desired set (so neither prunes the other), that pruning is queue-only (it reports
in-flight cancellations but never touches ``present``), and that ``configured`` is taken fresh each reconcile.
"""

from __future__ import annotations

import dataclasses

import pytest

from horde_worker_regen.process_management.desired_state import DesiredState, ReconcilePlan


class TestReconcileWithoutPickerAdditions:
    """With no picker additions the desired set is exactly the configured set."""

    def test_to_fetch_is_configured_minus_present(self) -> None:
        """Everything configured but not yet on disk is queued to fetch."""
        plan = DesiredState().reconcile(configured=["a", "b", "c"], present=["a"])
        assert plan.desired == frozenset({"a", "b", "c"})
        assert plan.to_fetch == ("b", "c")
        assert plan.to_cancel == ()
        assert plan.has_work is True

    def test_all_present_is_no_work(self) -> None:
        """When every desired model is present there is nothing to fetch or cancel."""
        plan = DesiredState().reconcile(configured=["a", "b"], present=["a", "b"])
        assert plan.to_fetch == ()
        assert plan.to_cancel == ()
        assert plan.has_work is False

    def test_in_flight_not_desired_is_cancelled(self) -> None:
        """A model still downloading that config no longer wants is reported for cancellation."""
        plan = DesiredState().reconcile(configured=["a"], present=["a"], in_flight=["b"])
        assert plan.to_cancel == ("b",)
        assert plan.to_fetch == ()
        assert plan.has_work is True


class TestReconcileWithPickerAdditions:
    """Picker additions join the desired set, and config can never prune them."""

    def test_picker_models_are_part_of_desired_and_fetched(self) -> None:
        """A picker-added model is part of the desired set and fetched when absent."""
        state = DesiredState()
        state.add_picker_models(["x", "y"])
        plan = state.reconcile(configured=["a"], present=["a"])
        assert plan.desired == frozenset({"a", "x", "y"})
        assert plan.to_fetch == ("x", "y")

    def test_picker_addition_is_not_cancelled_by_a_config_only_reconcile(self) -> None:
        """The divergence bug at the unit level: an in-flight picker model is kept, not cancelled.

        ``x`` is in flight and not configured, but it IS a picker addition, so the desired set keeps it and
        it is never put up for cancellation.
        """
        state = DesiredState()
        state.add_picker_models(["x"])
        plan = state.reconcile(configured=["a"], present=["a"], in_flight=["x"])
        assert "x" in plan.desired
        assert plan.to_cancel == ()

    def test_clearing_a_picker_addition_drops_it_from_desired(self) -> None:
        """Clearing a specific picker addition removes it from the desired set."""
        state = DesiredState()
        state.add_picker_models(["x", "y"])
        state.clear_picker_models(["x"])
        plan = state.reconcile(configured=["a"], present=["a"])
        assert plan.desired == frozenset({"a", "y"})
        assert state.picker_additions == frozenset({"y"})

    def test_clear_all_picker_additions(self) -> None:
        """Clearing with no argument drops every picker addition."""
        state = DesiredState()
        state.add_picker_models(["x", "y"])
        state.clear_picker_models()
        assert state.picker_additions == frozenset()


class TestReconcileTakesConfiguredFresh:
    """``configured`` is supplied per call, so a config change is reflected without re-seeding state."""

    def test_shrinking_configured_drops_a_model_from_desired(self) -> None:
        """A later reconcile with a smaller configured set yields a smaller desired set (no stale entry)."""
        state = DesiredState()
        assert state.reconcile(configured=["a", "b"], present=[]).desired == frozenset({"a", "b"})
        plan = state.reconcile(configured=["a"], present=["a"], in_flight=["b"])
        assert plan.desired == frozenset({"a"})
        assert plan.to_cancel == ("b",)


class TestReconcilePlanShape:
    """``ReconcilePlan`` is an immutable, sorted, deduplicated diff."""

    def test_outputs_are_sorted(self) -> None:
        """``to_fetch`` is returned in sorted order regardless of input order."""
        plan = DesiredState().reconcile(configured=["c", "a", "b"], present=[])
        assert plan.to_fetch == ("a", "b", "c")

    def test_plan_is_frozen(self) -> None:
        """The plan is immutable, so a consumer cannot mutate the diff after the fact."""
        plan = ReconcilePlan(desired=frozenset({"a"}), to_fetch=("a",), to_cancel=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.to_fetch = ("b",)  # type: ignore[misc]
