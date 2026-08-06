"""Contracts for precise fake-process fault scripts."""

from horde_worker_regen.process_management.simulation.fault_injection import FaultKind, FaultProfile


def test_targeted_slowness_changes_only_the_requested_process_local_job() -> None:
    """A generated slow-child event does not silently slow every job assigned to the process."""
    profile = FaultProfile(slow_factor=4.0, slow_on_job_n=2)

    assert profile.delay_factor_for_ordinal(1) == 1.0
    assert profile.delay_factor_for_ordinal(2) == 4.0
    assert profile.delay_factor_for_ordinal(3) == 1.0
    assert profile.active_kinds() == {FaultKind.SLOW}


def test_untargeted_slowness_retains_the_existing_all_jobs_behavior() -> None:
    """Profiles without an ordinal continue to apply their multiplier to every work item."""
    profile = FaultProfile(slow_factor=3.0)

    assert [profile.delay_factor_for_ordinal(ordinal) for ordinal in range(1, 4)] == [3.0, 3.0, 3.0]
