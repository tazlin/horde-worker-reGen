"""Tests each dry-run bypass independently."""

from __future__ import annotations

from tests.process_management.conftest import make_mock_bridge_data, make_testable_process_manager


def test_skip_inference_only() -> None:
    """Verify that setting dry_run_skip_inference to True enables inference bypass in the popper."""
    bridge_data = make_mock_bridge_data(
        dry_run_skip_inference=True,
        dry_run_skip_safety=False,
        dry_run_skip_api=False,
        dry_run_inference_delay=0.5,
    )
    process_manager = make_testable_process_manager(bridge_data=bridge_data)
    assert process_manager._job_popper._dry_run_skip_api is False
    assert process_manager._job_submitter._dry_run_skip_api is False


def test_skip_safety_only() -> None:
    """Verify that setting dry_run_skip_safety to True enables safety bypass in both the popper and submitter."""
    bridge_data = make_mock_bridge_data(
        dry_run_skip_inference=False,
        dry_run_skip_safety=True,
        dry_run_skip_api=False,
        dry_run_inference_delay=1.0,
    )
    process_manager = make_testable_process_manager(bridge_data=bridge_data)
    assert process_manager._job_popper._dry_run_skip_api is False
    assert process_manager._job_submitter._dry_run_skip_api is False


def test_skip_api_only() -> None:
    """Verify that setting dry_run_skip_api to True enables API bypass in both the popper and submitter."""
    bridge_data = make_mock_bridge_data(
        dry_run_skip_inference=False,
        dry_run_skip_safety=False,
        dry_run_skip_api=True,
        dry_run_inference_delay=1.0,
    )
    process_manager = make_testable_process_manager(bridge_data=bridge_data)
    assert process_manager._job_popper._dry_run_skip_api is True
    assert process_manager._job_submitter._dry_run_skip_api is True


def test_bridge_data_dry_run_fields_default_false() -> None:
    """With no overrides, all dry-run flags should be False."""
    bridge_data = make_mock_bridge_data()
    assert bridge_data.dry_run_skip_inference is False
    assert bridge_data.dry_run_skip_safety is False
    assert bridge_data.dry_run_skip_api is False
    assert bridge_data.dry_run_inference_delay == 1.0
