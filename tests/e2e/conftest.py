"""Shared fixtures for end-to-end dry-run tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_testable_process_manager,
)


@pytest.fixture()
def dry_run_bridge_data() -> Mock:
    """Bridge data with all three dry-run flags enabled."""
    return make_mock_bridge_data(
        dry_run_skip_inference=True,
        dry_run_skip_safety=True,
        dry_run_skip_api=True,
        dry_run_inference_delay=0.0,
    )


@pytest.fixture()
def dry_run_process_manager(dry_run_bridge_data: Mock) -> object:
    """A process manager wired for full dry-run."""
    return make_testable_process_manager(bridge_data=dry_run_bridge_data)
