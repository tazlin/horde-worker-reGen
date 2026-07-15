"""Table-driven tests for the bounded affinity line-skip budget."""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_BUDGET_MAX_SECONDS,
    _AFFINITY_BUDGET_MIN_SECONDS,
    _AFFINITY_MAX_SKIPS,
    AffinitySkipState,
    affinity_budget_seconds,
    affinity_skip_allowed,
    record_affinity_skip,
)


class TestAffinityBudgetSeconds:
    """The bypass budget is a clamped fraction of the job ttl."""

    def test_default_when_ttl_absent(self) -> None:
        """No horde ttl falls back to the conservative default (0.2 * 150 = 30, inside the clamp)."""
        assert affinity_budget_seconds(None) == 30.0

    def test_mid_ttl_scales_linearly(self) -> None:
        """A ttl whose fifth sits inside the clamp band scales without clamping."""
        assert affinity_budget_seconds(150.0) == 30.0
        assert affinity_budget_seconds(200.0) == 40.0

    def test_short_ttl_clamped_to_floor(self) -> None:
        """A short ttl still yields at least one useful window (floored, not proportionally tiny)."""
        assert affinity_budget_seconds(50.0) == _AFFINITY_BUDGET_MIN_SECONDS
        assert affinity_budget_seconds(0.0) == _AFFINITY_BUDGET_MIN_SECONDS

    def test_long_ttl_clamped_to_ceiling(self) -> None:
        """A long ttl does not let the head sit bypassed for an unbounded stretch (capped)."""
        assert affinity_budget_seconds(600.0) == _AFFINITY_BUDGET_MAX_SECONDS

    def test_clamp_boundaries(self) -> None:
        """The clamp endpoints are hit exactly at the fractions that produce them."""
        assert affinity_budget_seconds(_AFFINITY_BUDGET_MIN_SECONDS / 0.2) == _AFFINITY_BUDGET_MIN_SECONDS
        assert affinity_budget_seconds(_AFFINITY_BUDGET_MAX_SECONDS / 0.2) == _AFFINITY_BUDGET_MAX_SECONDS


class TestAffinitySkipAllowed:
    """A head may be bypassed only while under both the count and the wall-clock bound."""

    def test_fresh_head_allowed(self) -> None:
        """A head the window does not yet track gets a first bypass."""
        assert affinity_skip_allowed(AffinitySkipState(), "head-1", now=100.0, budget_seconds=30.0, max_skips=6)

    def test_none_head_never_allowed(self) -> None:
        """No dispatchable head means nothing to bypass."""
        assert not affinity_skip_allowed(AffinitySkipState(), None, now=100.0, budget_seconds=30.0, max_skips=6)

    def test_zero_max_skips_is_off_switch(self) -> None:
        """A zero ceiling disables affinity skips entirely, even for a fresh head."""
        assert not affinity_skip_allowed(AffinitySkipState(), "head-1", now=100.0, budget_seconds=30.0, max_skips=0)

    def test_non_positive_budget_is_off_switch(self) -> None:
        """A non-positive budget disables affinity skips entirely, even for a fresh head."""
        assert not affinity_skip_allowed(AffinitySkipState(), "head-1", now=100.0, budget_seconds=0.0, max_skips=6)

    def test_within_both_bounds_allowed(self) -> None:
        """A tracked head under the count and inside the window is still bypassable."""
        state = AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=3)
        assert affinity_skip_allowed(state, "head-1", now=110.0, budget_seconds=30.0, max_skips=6)

    def test_count_exhaustion_blocks(self) -> None:
        """At the skip ceiling the head is no longer bypassable, even inside the time window."""
        state = AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=6)
        assert not affinity_skip_allowed(state, "head-1", now=101.0, budget_seconds=30.0, max_skips=6)

    def test_time_exhaustion_blocks(self) -> None:
        """Past the wall-clock window the head is no longer bypassable, even under the skip ceiling."""
        state = AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=2)
        assert not affinity_skip_allowed(state, "head-1", now=135.0, budget_seconds=30.0, max_skips=6)

    def test_different_head_resets_to_allowed(self) -> None:
        """An exhausted window for a prior head does not carry over to a new head."""
        exhausted = AffinitySkipState(head_job_id="old-head", first_skip_time=100.0, skip_count=6)
        assert affinity_skip_allowed(exhausted, "new-head", now=1000.0, budget_seconds=30.0, max_skips=6)


class TestRecordAffinitySkip:
    """Recording a committed skip advances the window and pins the budget start to the first skip."""

    def test_first_skip_starts_window(self) -> None:
        """A skip against an untracked head starts a fresh window (count 1, clock starting now)."""
        advanced = record_affinity_skip(AffinitySkipState(), "head-1", now=100.0)
        assert advanced == AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=1)

    def test_subsequent_skip_increments_and_keeps_start(self) -> None:
        """A skip against the tracked head increments the count but keeps the original budget start."""
        state = AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=1)
        advanced = record_affinity_skip(state, "head-1", now=120.0)
        assert advanced == AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=2)

    def test_head_change_resets_window(self) -> None:
        """A skip against a different head restarts the window from scratch."""
        state = AffinitySkipState(head_job_id="head-1", first_skip_time=100.0, skip_count=4)
        advanced = record_affinity_skip(state, "head-2", now=200.0)
        assert advanced == AffinitySkipState(head_job_id="head-2", first_skip_time=200.0, skip_count=1)

    def test_repeated_skips_reach_the_ceiling(self) -> None:
        """Recording the ceiling number of skips leaves the head no longer bypassable."""
        state = AffinitySkipState()
        now = 100.0
        for _ in range(_AFFINITY_MAX_SKIPS):
            state = record_affinity_skip(state, "head-1", now)
            now += 1.0
        assert state.skip_count == _AFFINITY_MAX_SKIPS
        assert not affinity_skip_allowed(state, "head-1", now=now, budget_seconds=45.0, max_skips=_AFFINITY_MAX_SKIPS)
