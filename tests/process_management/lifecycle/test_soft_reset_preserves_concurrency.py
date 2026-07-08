"""A save-our-ship soft reset must rebuild the pools without shedding a concurrency lane.

Cutting ``effective_max_threads`` on every soft reset let a transient wedge, including one provoked by
aggressive co-sampling tripping a sampler watchdog, ratchet worker throughput down and outlast its cause.
The soft reset still rebuilds the pools (recovery is unchanged) and the escalation policy still counts the
reset toward give-up; only the concurrency reduction is demoted to a warning.
"""

from __future__ import annotations

from unittest.mock import Mock

from tests.process_management.conftest import make_testable_process_manager


class TestSoftResetPreservesConcurrency:
    """The soft reset rebuilds the pools but leaves the configured concurrency cap intact."""

    def test_soft_reset_does_not_reduce_effective_max_threads(self) -> None:
        """The rebuild happens, but the concurrency cap is left at its configured value."""
        pm = make_testable_process_manager(max_threads=3)
        coordinator = pm._recovery_coordinator
        coordinator._process_lifecycle = Mock()
        coordinator._inference_scheduler = Mock()
        before = coordinator._runtime_config.effective_max_threads
        # Headroom to reduce, so an "unchanged" assertion is meaningful rather than floored at 1.
        assert before >= 2

        coordinator.perform_soft_reset()

        assert coordinator._runtime_config.effective_max_threads == before
        coordinator._process_lifecycle.rebuild_inference_pool.assert_called_once()
        coordinator._process_lifecycle.rebuild_safety_pool.assert_called_once()

    def test_repeated_soft_resets_never_ratchet_concurrency_down(self) -> None:
        """Several soft resets in a row still leave the concurrency cap untouched."""
        pm = make_testable_process_manager(max_threads=3)
        coordinator = pm._recovery_coordinator
        coordinator._process_lifecycle = Mock()
        coordinator._inference_scheduler = Mock()
        before = coordinator._runtime_config.effective_max_threads

        for _ in range(3):
            coordinator.perform_soft_reset()

        assert coordinator._runtime_config.effective_max_threads == before
