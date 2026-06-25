"""Tests for runtime concurrency scaling: effective thread cap and SET_CONCURRENCY."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.ipc.supervisor_channel import SupervisorCommand, SupervisorControlMessage
from tests.process_management.conftest import make_mock_bridge_data, make_testable_process_manager


class TestRuntimeConfigConcurrency:
    """The live effective-thread cap and its provisioned ceiling."""

    def test_effective_and_ceiling_default_to_max_threads(self) -> None:
        """Without an explicit ceiling, both the cap and ceiling equal max_threads."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=3))  # type: ignore[arg-type]
        assert rc.effective_max_threads == 3
        assert rc.max_threads_ceiling == 3

    def test_explicit_ceiling_allows_runtime_headroom(self) -> None:
        """An explicit ceiling lets the cap be raised above the initial max_threads at runtime."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=2), max_threads_ceiling=6)  # type: ignore[arg-type]
        assert rc.max_threads_ceiling == 6
        assert rc.effective_max_threads == 2
        assert rc.set_effective_max_threads(5) == 5
        assert rc.effective_max_threads == 5

    def test_set_effective_clamps_to_ceiling(self) -> None:
        """The cap can never exceed the provisioned ceiling."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=2))  # type: ignore[arg-type]
        assert rc.set_effective_max_threads(10) == 2

    def test_set_effective_floor_is_one(self) -> None:
        """The cap can never drop below one."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=2), max_threads_ceiling=4)  # type: ignore[arg-type]
        assert rc.set_effective_max_threads(0) == 1

    def test_update_rederives_effective_from_new_config(self) -> None:
        """A config reload re-derives the cap from the new max_threads."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=2), max_threads_ceiling=4)  # type: ignore[arg-type]
        rc.set_effective_max_threads(4)
        rc.update(make_mock_bridge_data(max_threads=1))  # type: ignore[arg-type]
        assert rc.effective_max_threads == 1

    def test_update_clamps_increase_to_ceiling(self) -> None:
        """A reloaded max_threads above the ceiling is clamped to it."""
        rc = RuntimeConfig(initial=make_mock_bridge_data(max_threads=2))  # type: ignore[arg-type]
        rc.update(make_mock_bridge_data(max_threads=8))  # type: ignore[arg-type]
        assert rc.effective_max_threads == 2


class TestManagerSetConcurrency:
    """The manager's SET_CONCURRENCY handling."""

    def test_apply_set_concurrency_updates_threads_and_scales(self) -> None:
        """Applying SET_CONCURRENCY adjusts the live cap and scales the process count."""
        manager = make_testable_process_manager(max_threads=4)
        manager._process_lifecycle = Mock()
        manager._apply_set_concurrency(target_threads=2, target_processes=3)
        assert manager._runtime_config.effective_max_threads == 2
        assert manager.max_concurrent_inference_processes == 2
        manager._process_lifecycle.scale_inference_processes.assert_called_once_with(3)

    def test_apply_set_concurrency_clamps_to_ceiling(self) -> None:
        """A thread target above the ceiling is clamped."""
        manager = make_testable_process_manager(max_threads=2)
        manager._process_lifecycle = Mock()
        manager._apply_set_concurrency(target_threads=10, target_processes=None)
        assert manager.max_concurrent_inference_processes == 2

    def test_supervisor_set_concurrency_command_routes(self) -> None:
        """The SET_CONCURRENCY supervisor command reaches the concurrency handler."""
        manager = make_testable_process_manager(max_threads=4)
        manager._process_lifecycle = Mock()
        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_CONCURRENCY,
                target_threads=1,
                target_processes=2,
            ),
        )
        assert manager.max_concurrent_inference_processes == 1
        manager._process_lifecycle.scale_inference_processes.assert_called_once_with(2)


class TestInstallBenchmarkScenario:
    """The warm-benchmark per-level scenario swap."""

    def test_install_benchmark_scenario_resets_recovery_counter(self) -> None:
        """Swapping in a new level's scenario zeroes the cumulative recovery counter.

        The warm worker reuses one pool across levels; the counter is otherwise only incremented, so
        without this reset every level after the first genuine recovery would read as having recovered.
        """
        manager = make_testable_process_manager()
        manager._process_lifecycle._num_process_recoveries = 2

        manager.install_benchmark_scenario(jobs=[])

        assert manager._process_lifecycle._num_process_recoveries == 0
        assert manager.get_run_metrics_snapshot().num_process_recoveries == 0
