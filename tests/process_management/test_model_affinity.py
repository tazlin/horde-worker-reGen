"""Table-driven tests for model->process affinity protection."""

from __future__ import annotations

from horde_worker_regen.process_management.model_affinity import (
    affinity_active,
    compute_protected_processes,
)


class TestAffinityActive:
    """Affinity applies only when every model can have its own home process."""

    def test_active_when_models_fit(self) -> None:
        """Fewer (or equal) models than processes => each gets a home, affinity on."""
        assert affinity_active(4, 6) is True
        assert affinity_active(6, 6) is True

    def test_inactive_when_more_models_than_processes(self) -> None:
        """More models than processes must share => pinning would deadlock, affinity off."""
        assert affinity_active(8, 6) is False

    def test_inactive_when_no_models(self) -> None:
        """No models to load => nothing to protect."""
        assert affinity_active(0, 6) is False


class TestComputeProtectedProcesses:
    """The protected set is the last resident copy of each still-wanted model."""

    def test_one_copy_each_all_protected(self) -> None:
        """With one copy of each wanted model, every holding process is protected."""
        process_models = {0: "A", 1: "B", 2: "C", 3: "D", 4: None, 5: None}
        assert compute_protected_processes(process_models, {"A", "B", "C", "D"}) == {0, 1, 2, 3}

    def test_surplus_copy_is_displaceable(self) -> None:
        """A model with two copies keeps only one pinned; the surplus stays displaceable."""
        process_models = {0: "A", 5: "A", 1: "B", 2: "C", 3: "D"}
        protected = compute_protected_processes(process_models, {"A", "B", "C", "D"})
        # Exactly one of A's two processes is protected (the lowest id), the other is free.
        assert 0 in protected
        assert 5 not in protected
        assert {1, 2, 3} <= protected

    def test_unwanted_model_not_protected(self) -> None:
        """A process holding a model no longer in models_to_load is displaceable."""
        process_models = {0: "A", 1: "OLD"}
        assert compute_protected_processes(process_models, {"A"}) == {0}

    def test_empty_processes_not_protected(self) -> None:
        """Processes with no model loaded are never protected."""
        assert compute_protected_processes({0: None, 1: None}, {"A"}) == set()
