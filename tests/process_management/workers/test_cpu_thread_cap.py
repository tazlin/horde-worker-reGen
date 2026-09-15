"""Tests for the per-child CPU thread cap applied before a spawned worker imports torch.

Every GPU-bearing child sizes its OpenMP/MKL pools to the whole machine by default, so a multi-process
worker oversubscribes the host: sampling pays for it through the torchsde noise it runs on the CPU, and an
off-GPU safety check is pure CPU work queued behind that. The parent divides the host among the children it
plans to run; an operator who has set either variable is tuning the machine themselves and is never
overruled.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import worker_entry_points
from horde_worker_regen.process_management.worker_entry_points import (
    _MKL_NUM_THREADS_ENV,
    _OMP_NUM_THREADS_ENV,
    _apply_cpu_thread_cap,
)
from tests.process_management.conftest import make_testable_process_manager


@pytest.fixture
def unset_thread_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each case from a host where neither thread variable is exported."""
    monkeypatch.delenv(_OMP_NUM_THREADS_ENV, raising=False)
    monkeypatch.delenv(_MKL_NUM_THREADS_ENV, raising=False)


@pytest.mark.usefixtures("unset_thread_env")
def test_cap_sets_both_thread_pools() -> None:
    """A cap sizes the OpenMP and MKL pools alike; one without the other leaves half the host in play."""
    _apply_cpu_thread_cap(4)

    assert os.environ[_OMP_NUM_THREADS_ENV] == "4"
    assert os.environ[_MKL_NUM_THREADS_ENV] == "4"


@pytest.mark.usefixtures("unset_thread_env")
def test_no_cap_leaves_the_pools_untouched() -> None:
    """A single-inference-slot worker has nothing to divide, so its children keep the stock defaults."""
    _apply_cpu_thread_cap(None)

    assert _OMP_NUM_THREADS_ENV not in os.environ
    assert _MKL_NUM_THREADS_ENV not in os.environ


@pytest.mark.usefixtures("unset_thread_env")
@pytest.mark.parametrize("operator_set_env", [_OMP_NUM_THREADS_ENV, _MKL_NUM_THREADS_ENV])
def test_an_operator_value_is_never_overridden(
    monkeypatch: pytest.MonkeyPatch,
    operator_set_env: str,
) -> None:
    """Whichever variable the operator exported keeps their value; the other still takes the cap."""
    monkeypatch.setenv(operator_set_env, "1")

    _apply_cpu_thread_cap(6)

    assert os.environ[operator_set_env] == "1"
    other_env = _MKL_NUM_THREADS_ENV if operator_set_env == _OMP_NUM_THREADS_ENV else _OMP_NUM_THREADS_ENV
    assert os.environ[other_env] == "6"


class TestLifecycleThreadShare:
    """The lifecycle manager decides the share each child gets from what it plans to run."""

    def test_a_single_inference_slot_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One slot competes with nothing worth dividing the host over, so the defaults stand."""
        lifecycle = make_testable_process_manager()._process_lifecycle
        lifecycle._max_inference_processes = 1
        monkeypatch.setattr(os, "cpu_count", lambda: 32)

        assert lifecycle._compute_cpu_thread_cap() is None

    def test_the_host_is_divided_among_the_planned_children(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The divisor counts the inference slots, the safety process and each enabled lane."""
        process_manager = make_testable_process_manager()
        lifecycle = process_manager._process_lifecycle
        lifecycle._max_inference_processes = 3
        process_manager.bridge_data.post_processing_lane_available = False
        process_manager.bridge_data.enable_pipeline_disaggregation = False
        monkeypatch.setattr(os, "cpu_count", lambda: 16)

        # Three inference slots plus safety.
        assert lifecycle._compute_cpu_thread_cap() == 4

        # The VAE and component lanes both follow disaggregation, so enabling it adds two more children.
        process_manager.bridge_data.enable_pipeline_disaggregation = True
        assert lifecycle._compute_cpu_thread_cap() == 2

    def test_a_small_host_still_leaves_each_child_a_usable_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The floor holds: dividing a small host evenly would otherwise pin children to one thread."""
        process_manager = make_testable_process_manager()
        lifecycle = process_manager._process_lifecycle
        lifecycle._max_inference_processes = 4
        process_manager.bridge_data.post_processing_lane_available = False
        process_manager.bridge_data.enable_pipeline_disaggregation = False
        monkeypatch.setattr(os, "cpu_count", lambda: 4)

        assert lifecycle._compute_cpu_thread_cap() == 2

    def test_the_startup_share_is_cached_across_later_topology_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replacement children keep the startup share already used by their surviving peers."""
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        process_manager = make_testable_process_manager()
        lifecycle = process_manager._process_lifecycle
        startup_cap = lifecycle._child_cpu_thread_cap

        lifecycle._max_inference_processes = 8
        process_manager.bridge_data.post_processing_lane_available = True
        process_manager.bridge_data.enable_pipeline_disaggregation = True

        assert lifecycle._compute_cpu_thread_cap() != startup_cap
        assert lifecycle._child_cpu_thread_cap == startup_cap


class _StopAfterEarlySetup(Exception):
    """Stop a worker entry point after its pre-import environment setup."""


@pytest.mark.parametrize(
    ("entry_point", "args"),
    [
        (
            worker_entry_points.start_inference_process,
            (1, Mock(), Mock(), Mock(), Mock(), Mock(), 0),
        ),
        (
            worker_entry_points.start_safety_process,
            (0, Mock(), Mock(), Mock(), 0, False),
        ),
        (
            worker_entry_points.start_post_process_process,
            (2, Mock(), Mock(), Mock(), 0),
        ),
        (
            worker_entry_points.start_vae_lane_process,
            (3, Mock(), Mock(), Mock(), 0),
        ),
        (
            worker_entry_points.start_component_process,
            (4, Mock(), Mock(), Mock(), 0),
        ),
    ],
)
def test_every_gpu_child_applies_the_thread_cap_before_device_pinning(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: Callable[..., None],
    args: tuple[object, ...],
) -> None:
    """Every torch-bearing entry point sets thread-pool variables before importing through device pinning."""
    setup_order: list[str] = []
    monkeypatch.setattr(worker_entry_points, "_spawn_timing_mark", lambda *_args: None)
    monkeypatch.setattr(worker_entry_points, "_apply_cpu_thread_cap", lambda _cap: setup_order.append("cap"))
    monkeypatch.setattr(worker_entry_points, "_apply_device_pin", lambda **_kwargs: setup_order.append("pin"))
    monkeypatch.setattr(worker_entry_points, "_enable_expandable_segments", lambda **_kwargs: None)
    monkeypatch.setattr(worker_entry_points, "enable_child_faulthandler", lambda _name: None)
    monkeypatch.setattr(worker_entry_points, "neutralize_inherited_argv", lambda: None)
    monkeypatch.setattr(worker_entry_points.logger, "remove", lambda: None)

    def _stop_before_import(*_args: object) -> None:
        raise _StopAfterEarlySetup

    monkeypatch.setattr(worker_entry_points, "maybe_wait_for_process_debugger", _stop_before_import)

    with pytest.raises(_StopAfterEarlySetup):
        entry_point(*args, accelerator_kind="cuda", cpu_thread_cap=4)

    assert setup_order == ["cap", "pin"]
