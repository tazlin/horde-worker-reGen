"""Unit tests for the pure host-aware download admission policy.

These exercise :class:`HostAwareDownloadScheduler` and :func:`download_host_for_url` directly (no
hordelib, no real downloads), so the parallelism rules - per-host serialization, cross-host parallelism,
the global cap, live retuning, and prune/close - are proven in isolation.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.model_download_core import UNKNOWN_DOWNLOAD_HOST, download_host_for_url
from horde_worker_regen.process_management.download_scheduler import (
    DownloadKind,
    DownloadTask,
    HostAwareDownloadScheduler,
)


def _task(model: str, host: str) -> DownloadTask:
    return DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name=model, host=host, feature="image model")


def _exclusive_task(model: str, host: str) -> DownloadTask:
    return DownloadTask(
        kind=DownloadKind.ANNOTATORS,
        model_name=model,
        host=host,
        feature="annotators",
        exclusive=True,
    )


class TestDownloadHostForUrl:
    """The host helper underpins per-host scheduling, so its parsing must be exact and forgiving."""

    def test_extracts_lowercased_host(self) -> None:
        """The hostname is returned lowercased, independent of path/scheme."""
        assert download_host_for_url("https://Civitai.com/api/download/123") == "civitai.com"
        assert download_host_for_url("https://huggingface.co/foo/bar.safetensors") == "huggingface.co"

    def test_strips_port(self) -> None:
        """A port in the authority is not part of the host bucket."""
        assert download_host_for_url("http://example.com:8080/x") == "example.com"

    def test_missing_or_garbage_url_is_unknown(self) -> None:
        """A missing or unparseable URL collapses to the conservative unknown bucket."""
        assert download_host_for_url(None) == UNKNOWN_DOWNLOAD_HOST
        assert download_host_for_url("") == UNKNOWN_DOWNLOAD_HOST
        assert download_host_for_url("not a url") == UNKNOWN_DOWNLOAD_HOST


class TestHostAwareScheduler:
    """The admission policy: dedup, per-host serialization, cross-host parallelism, caps, prune, close."""

    def test_enqueue_dedups(self) -> None:
        """The same task is not queued twice."""
        scheduler = HostAwareDownloadScheduler()
        assert scheduler.enqueue(_task("a", "h1")) is True
        assert scheduler.enqueue(_task("a", "h1")) is False
        assert len(scheduler.pending_snapshot()) == 1

    def test_same_host_serializes_by_default(self) -> None:
        """With per_host_concurrency=1 a second same-host task waits until the first is released."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h1")])

        first = scheduler.acquire(timeout=0.05)
        assert first is not None
        assert scheduler.acquire(timeout=0.05) is None  # second same-host task is blocked

        scheduler.release(first)
        second = scheduler.acquire(timeout=0.05)
        assert second is not None and second.model_name != first.model_name

    def test_different_hosts_run_in_parallel(self) -> None:
        """Two tasks on distinct hosts can both be in flight at once."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h2")])

        first = scheduler.acquire(timeout=0.05)
        second = scheduler.acquire(timeout=0.05)
        assert first is not None and second is not None
        assert {first.host, second.host} == {"h1", "h2"}

    def test_global_cap_limits_total_in_flight(self) -> None:
        """The global ceiling caps total concurrency even across distinct hosts."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=2, per_host_concurrency=1)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h2"), _task("c", "h3")])

        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is None  # third blocked by the global cap of 2

    def test_per_host_concurrency_allows_same_host_parallel(self) -> None:
        """Raising per-host concurrency lets multiple same-host downloads run (the toggle)."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=2)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h1"), _task("c", "h1")])

        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is None  # third same-host blocked at per_host=2

    def test_admissible_task_jumps_ahead_of_blocked_host(self) -> None:
        """A different-host task starts ahead of one queued behind a busy host."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h1"), _task("c", "h2")])

        first = scheduler.acquire(timeout=0.05)
        assert first is not None and first.host == "h1"
        # Next pending is also h1 (blocked); the h2 task should be chosen instead.
        second = scheduler.acquire(timeout=0.05)
        assert second is not None and second.host == "h2"

    def test_raising_limit_unblocks_waiter(self) -> None:
        """Raising the global cap live makes a previously-blocked task admissible."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1, per_host_concurrency=1)
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h2")])
        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is None  # blocked by global cap of 1

        scheduler.set_limits(max_parallel_downloads=2)
        assert scheduler.acquire(timeout=0.05) is not None

    def test_prune_removes_pending_and_reports_removed(self) -> None:
        """Prune drops non-kept pending tasks and returns exactly the removed ones."""
        scheduler = HostAwareDownloadScheduler()
        scheduler.enqueue_many([_task("keep", "h1"), _task("drop", "h2")])

        removed = scheduler.prune(keep=lambda task: task.model_name != "drop")

        assert [task.model_name for task in removed] == ["drop"]
        assert [task.model_name for task in scheduler.pending_snapshot()] == ["keep"]

    def test_close_unblocks_acquire(self) -> None:
        """Closing the scheduler makes a waiting acquire return None for shutdown."""
        scheduler = HostAwareDownloadScheduler()
        scheduler.close()
        assert scheduler.acquire(timeout=0.05) is None

    def test_exclusive_task_waits_for_drain(self) -> None:
        """An exclusive task is not admitted while anything else is in flight."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue(_task("a", "h1"))
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))

        first = scheduler.acquire(timeout=0.05)
        assert first is not None and first.model_name == "a"
        # The exclusive task must wait until "a" releases, even though the global cap has room.
        assert scheduler.acquire(timeout=0.05) is None

        scheduler.release(first)
        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True

    def test_exclusive_in_flight_blocks_others(self) -> None:
        """While an exclusive task runs, no other task is admitted (it runs alone)."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))
        scheduler.enqueue(_task("a", "h1"))

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        # Nothing else may start alongside the exclusive task.
        assert scheduler.acquire(timeout=0.05) is None

        scheduler.release(exclusive)
        other = scheduler.acquire(timeout=0.05)
        assert other is not None and other.model_name == "a"

    def test_stuck_exclusive_stops_blocking_after_the_time_bound(self) -> None:
        """A wedged exclusive task stops starving the queue once it exceeds the exclusivity time bound.

        Guards the annotator-monopoly fix: an un-interruptible annotator preload that hangs must not block
        the image-model downloads a worker needs to serve jobs forever. A zero bound relaxes immediately.
        """
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=4,
            per_host_concurrency=1,
            exclusive_timeout_seconds=0.0,
        )
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))
        scheduler.enqueue(_task("a", "h1"))

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        # The exclusive task is still in flight, but its bound is already exceeded, so "a" is admitted.
        other = scheduler.acquire(timeout=0.05)
        assert other is not None and other.model_name == "a"

    def test_exclusivity_blocks_within_the_bound_then_relaxes_past_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Within the bound the exclusive task blocks others; once the bound elapses they are admitted."""
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "horde_worker_regen.process_management.download_scheduler.time.monotonic",
            lambda: clock["now"],
        )
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=4,
            per_host_concurrency=1,
            exclusive_timeout_seconds=300.0,
        )
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))
        scheduler.enqueue(_task("a", "h1"))

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        # Still within the bound: "a" is held back behind the running exclusive task.
        assert scheduler.acquire(timeout=0.05) is None
        # Advance past the bound: "a" becomes admissible even though the exclusive task is still in flight.
        clock["now"] += 301.0
        other = scheduler.acquire(timeout=0.05)
        assert other is not None and other.model_name == "a"

    def test_has_work_tracks_pending_and_in_flight(self) -> None:
        """has_work is true while anything is queued or in flight, false only when fully drained."""
        scheduler = HostAwareDownloadScheduler()
        assert scheduler.has_work() is False
        scheduler.enqueue(_task("a", "h1"))
        assert scheduler.has_work() is True
        task = scheduler.acquire(timeout=0.05)
        assert task is not None and scheduler.has_work() is True  # in flight
        scheduler.release(task)
        assert scheduler.has_work() is False
