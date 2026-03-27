"""Tests that the full pipeline can run with all dry-run bypasses enabled."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management._canned_scenarios import (
    SCENARIO_BASIC,
    SCENARIO_TRIVIAL,
    get_dry_run_job,
)
from horde_worker_regen.process_management.messages import (
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
)


def test_canned_scenario_trivial_has_one_job() -> None:
    assert len(SCENARIO_TRIVIAL) == 1
    assert SCENARIO_TRIVIAL[0].model == "Deliberate"


def test_canned_scenario_basic_has_five_jobs() -> None:
    assert len(SCENARIO_BASIC) == 5
    for job in SCENARIO_BASIC:
        assert job.model == "Deliberate"


def test_get_dry_run_job_cycles() -> None:
    """get_dry_run_job should cycle through SCENARIO_BASIC indefinitely."""
    seen_ids = set()
    for _ in range(10):
        job = get_dry_run_job()
        assert job.model == "Deliberate"
        assert job.id_ is not None
        seen_ids.add(job.id_)
    assert len(seen_ids) == 5


def test_dry_run_process_manager_has_dry_run_flags(dry_run_bridge_data: Mock) -> None:
    """The dry-run bridge data fixture should have all flags set."""
    assert dry_run_bridge_data.dry_run_skip_inference is True
    assert dry_run_bridge_data.dry_run_skip_safety is True
    assert dry_run_bridge_data.dry_run_skip_api is True
    assert dry_run_bridge_data.dry_run_inference_delay == 0.0


def test_process_manager_passes_dry_run_to_popper(dry_run_process_manager: object) -> None:
    """The process manager should wire dry_run_skip_api to the job popper."""
    process_manager = dry_run_process_manager
    assert process_manager._job_popper._dry_run_skip_api is True  # type: ignore[attr-defined]


def test_process_manager_passes_dry_run_to_submitter(dry_run_process_manager: object) -> None:
    """The process manager should wire dry_run_skip_api to the job submitter."""
    process_manager = dry_run_process_manager
    assert process_manager._job_submitter._dry_run_skip_api is True  # type: ignore[attr-defined]
