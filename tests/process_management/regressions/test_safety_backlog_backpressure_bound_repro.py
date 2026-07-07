"""Regression: two GPU producers feeding a slower CPU safety consumer must not grow the backlog unboundedly.

Shaped on the diagnosed field timeline: on a box where safety runs on CPU (~10s/job) behind two GPU
producers, the in-flight safety backlog (``jobs_being_safety_checked`` plus ``jobs_pending_safety_check``)
climbed monotonically as inference outpaced safety, until jobs aged past their horde ttl. The pop-gate
backpressure exists to bound that: when the backlog reaches the self-tuning cap, intake stops until the
backlog drains below the lower hysteresis bound, so arrivals cannot outrun the slow stage indefinitely.

This drives the real :meth:`JobPopper._is_post_inference_backlogged` gate across a producer/consumer loop
and asserts the backlog settles into a small band around the cap rather than the unbounded climb the ungated
pipeline showed.
"""

from __future__ import annotations

from tests.process_management.jobs.test_job_popping import _FakeBacklogTracker, _make_popper


def test_two_producers_slow_safety_never_grows_backlog_unboundedly() -> None:
    """The backpressure gate bounds the in-flight safety backlog under sustained producer-faster-than-safety load."""
    popper = _make_popper()
    tracker = _FakeBacklogTracker()
    popper._job_tracker = tracker  # pyrefly: ignore - a stub stands in for the job tracker

    cap = 3
    popper._max_safe_safety_backlog = lambda: cap  # pyrefly: ignore - fixing the cap isolates the bounding effect

    producer_rate = 2  # two GPU producers each complete a job into safety on an open tick
    consumer_rate = 1  # a single CPU safety process clears one per tick (strictly slower than the producers)

    backlog = 0
    peak_backlog = 0
    for _ in range(500):
        # The slow safety consumer makes its per-tick progress first.
        backlog = max(0, backlog - consumer_rate)
        tracker.set_backlog(backlog)
        # Intake only admits (and thus feeds more completions into safety) while the gate is open.
        if not popper._is_post_inference_backlogged():
            backlog += producer_rate
        tracker.set_backlog(backlog)
        peak_backlog = max(peak_backlog, backlog)

    # Bounded to the cap plus at most one tick of producer arrivals, never the monotonic climb (12 and rising
    # in the field) an ungated pipeline shows: without the gate the backlog would grow by
    # (producer_rate - consumer_rate) every tick, reaching the hundreds over this many iterations.
    assert peak_backlog <= cap + producer_rate
    assert backlog <= cap + producer_rate
