"""Unit tests for the pure host-aware download admission policy.

These exercise :class:`HostAwareDownloadScheduler` and :func:`download_host_for_url` directly (no
hordelib, no real downloads), so the parallelism rules - per-host serialization, cross-host parallelism,
the global cap, live retuning, and prune/close - are proven in isolation.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.model_download_core import UNKNOWN_DOWNLOAD_HOST, download_host_for_url
from horde_worker_regen.process_management.models.download_scheduler import (
    DownloadKind,
    DownloadPriorityPolicy,
    DownloadTask,
    HostAwareDownloadScheduler,
)


def _task(model: str, host: str) -> DownloadTask:
    return DownloadTask(kind=DownloadKind.IMAGE_MODEL, model_name=model, host=host, feature="image model")


def _safety_task(host: str = "github.com") -> DownloadTask:
    return DownloadTask(kind=DownloadKind.SAFETY, model_name="safety models", host=host, feature="safety")


def _feature_task(model: str, host: str) -> DownloadTask:
    return DownloadTask(
        kind=DownloadKind.AUX_MODEL,
        model_name=model,
        host=host,
        feature="ControlNet",
        manager_key="controlnet",
    )


def _default_loras_task(host: str = "civitai.com") -> DownloadTask:
    return DownloadTask(kind=DownloadKind.DEFAULT_LORAS, model_name="default LoRas", host=host, feature="LoRa")


def _drain_in_order(scheduler: HostAwareDownloadScheduler) -> list[str]:
    """Acquire and immediately release every admissible task, returning the names in admission order."""
    names: list[str] = []
    while True:
        task = scheduler.acquire(timeout=0.0)
        if task is None:
            return names
        names.append(task.model_name)
        scheduler.release(task)


def _exclusive_task(model: str, host: str) -> DownloadTask:
    return DownloadTask(
        kind=DownloadKind.ANNOTATOR_VERIFY,
        model_name=model,
        host=host,
        feature="annotators",
        exclusive=True,
    )


def _lora_task(model: str, host: str) -> DownloadTask:
    return DownloadTask(kind=DownloadKind.LORA, model_name=model, host=host, feature="LoRa (job)")


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


class TestServeFirstOrdering:
    """Under ``serve_first`` the queue is drained in the order that gets a worker serving soonest."""

    def test_safety_models_start_before_an_earlier_queued_checkpoint(self) -> None:
        """The safety models gate every job, so they are fetched ahead of checkpoints queued before them."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1)
        scheduler.enqueue_many([_task("checkpoint", "civitai.com"), _safety_task()])

        assert _drain_in_order(scheduler) == ["safety models", "checkpoint"]

    def test_tiers_run_in_order_and_arrival_breaks_ties(self) -> None:
        """Image models come before feature models, which come before the default LoRAs; ties are oldest first."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1)
        scheduler.enqueue_many(
            [
                _default_loras_task(),
                _feature_task("controlnet_canny", "huggingface.co"),
                _task("second checkpoint", "civitai.com"),
                _task("first checkpoint", "civitai.com"),
                _feature_task("controlnet_depth", "huggingface.co"),
            ],
        )
        scheduler.enqueue(_task("third checkpoint", "civitai.com"))

        assert _drain_in_order(scheduler) == [
            "second checkpoint",
            "first checkpoint",
            "third checkpoint",
            "controlnet_canny",
            "controlnet_depth",
            "default LoRas",
        ]

    def test_a_popped_job_s_own_files_come_right_after_safety(self) -> None:
        """A job waiting on its LoRA is served before the next checkpoint, since a card sits idle on it."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1)
        scheduler.enqueue_many([_task("checkpoint", "civitai.com"), _lora_task("job lora", "civitai.com")])

        assert _drain_in_order(scheduler) == ["job lora", "checkpoint"]

    def test_exclusive_verify_still_runs_last(self) -> None:
        """The exclusive annotator verify keeps draining into the idle tail, whatever tier it sits in."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1)
        scheduler.enqueue_many([_exclusive_task("verify", "h"), _default_loras_task()])

        assert _drain_in_order(scheduler) == ["default LoRas", "verify"]

    def test_parallel_policy_is_first_come_first_served(self) -> None:
        """An operator who chooses ``parallel`` gets the queue in arrival order, tiers ignored."""
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=1, priority_policy=DownloadPriorityPolicy.PARALLEL
        )
        scheduler.enqueue_many([_default_loras_task(), _task("checkpoint", "civitai.com"), _safety_task()])

        assert _drain_in_order(scheduler) == ["default LoRas", "checkpoint", "safety models"]

    def test_switching_policy_reorders_what_is_still_pending(self) -> None:
        """Changing the policy mid-queue changes which pending task starts next; nothing is dropped."""
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=1, priority_policy=DownloadPriorityPolicy.PARALLEL
        )
        scheduler.enqueue_many(
            [_default_loras_task(), _feature_task("controlnet_canny", "h"), _task("checkpoint", "c")]
        )

        assert scheduler.set_policy(DownloadPriorityPolicy.SERVE_FIRST) is True
        assert scheduler.set_policy(DownloadPriorityPolicy.SERVE_FIRST) is False
        assert _drain_in_order(scheduler) == ["checkpoint", "controlnet_canny", "default LoRas"]


class TestStartupFocus:
    """Until the worker can serve, only what it needs to serve is downloading."""

    def test_focus_admits_safety_and_one_checkpoint_and_holds_the_rest(self) -> None:
        """With focus on, a second checkpoint and every feature model wait; a job's own file does not."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4)
        scheduler.set_startup_focus(True)
        scheduler.enqueue_many(
            [
                _feature_task("controlnet_canny", "huggingface.co"),
                _task("first checkpoint", "civitai.com"),
                _task("second checkpoint", "huggingface.co"),
                _safety_task("github.com"),
                _lora_task("job lora", "civitai.com"),
            ],
        )

        admitted = {scheduler.acquire(timeout=0.0).model_name for _ in range(3)}  # type: ignore[union-attr]
        assert admitted == {"safety models", "first checkpoint", "job lora"}
        assert scheduler.acquire(timeout=0.0) is None

    def test_focus_lets_the_next_checkpoint_start_once_the_first_lands(self) -> None:
        """One checkpoint at a time: the second starts when the first is released, features still wait."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4)
        scheduler.set_startup_focus(True)
        scheduler.enqueue_many(
            [_task("first", "civitai.com"), _task("second", "huggingface.co"), _feature_task("canny", "h")],
        )
        first = scheduler.acquire(timeout=0.0)
        assert first is not None and first.model_name == "first"
        assert scheduler.acquire(timeout=0.0) is None

        scheduler.release(first)
        second = scheduler.acquire(timeout=0.0)
        assert second is not None and second.model_name == "second"
        assert scheduler.acquire(timeout=0.0) is None

    def test_clearing_focus_releases_the_held_queue_in_tier_order(self) -> None:
        """Once the worker can serve, everything that was held starts in tier order."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=1)
        scheduler.set_startup_focus(True)
        scheduler.enqueue_many([_default_loras_task(), _feature_task("canny", "h"), _task("checkpoint", "c")])
        assert _drain_in_order(scheduler) == ["checkpoint"]  # the features stay held

        assert scheduler.set_startup_focus(False) is True
        assert _drain_in_order(scheduler) == ["canny", "default LoRas"]

    def test_focus_is_ignored_under_the_parallel_policy(self) -> None:
        """``parallel`` never narrows the queue, even when the worker has nothing to serve yet."""
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=4, priority_policy=DownloadPriorityPolicy.PARALLEL
        )
        assert scheduler.set_startup_focus(True) is False
        scheduler.enqueue_many([_feature_task("canny", "h1"), _task("first", "h2"), _task("second", "h3")])

        assert len(_drain_in_order(scheduler)) == 3

    def test_exclusive_verify_waits_out_the_focus(self) -> None:
        """A held feature queue keeps the exclusive verify from running, since work is still pending."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4)
        scheduler.set_startup_focus(True)
        scheduler.enqueue_many([_exclusive_task("verify", "h"), _feature_task("canny", "h")])

        assert scheduler.acquire(timeout=0.0) is None


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

    def test_adhoc_prefetch_bypasses_the_host_limit(self) -> None:
        """A job-driven LoRA prefetch is admitted while an ordinary same-host download holds the host slot.

        A pending job's dispatch gate waits on the prefetch, so it must not queue behind an unrelated slow
        transfer to the same host; the global cap still applies.
        """
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue_many([_task("bulk", "civitai.com"), _lora_task("123", "civitai.com")])

        first = scheduler.acquire(timeout=0.05)
        second = scheduler.acquire(timeout=0.05)
        assert first is not None and second is not None
        assert {first.model_name, second.model_name} == {"bulk", "123"}

    def test_adhoc_prefetch_does_not_consume_the_host_slot(self) -> None:
        """An in-flight LoRA prefetch leaves the host slot free for an ordinary download queued after it."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue(_lora_task("123", "civitai.com"))
        prefetch = scheduler.acquire(timeout=0.05)
        assert prefetch is not None and prefetch.model_name == "123"

        scheduler.enqueue(_task("bulk", "civitai.com"))
        bulk = scheduler.acquire(timeout=0.05)
        assert bulk is not None and bulk.model_name == "bulk"

        # Releasing the exempt task must not corrupt the ordinary task's host accounting: a second bulk
        # download stays blocked until the first is released.
        scheduler.release(prefetch)
        scheduler.enqueue(_task("bulk2", "civitai.com"))
        assert scheduler.acquire(timeout=0.05) is None
        scheduler.release(bulk)
        assert scheduler.acquire(timeout=0.05) is not None

    def test_adhoc_prefetch_still_bounded_by_global_cap(self) -> None:
        """Host exemption does not bypass the global ceiling."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=2, per_host_concurrency=1)
        scheduler.enqueue_many(
            [_lora_task("1", "civitai.com"), _lora_task("2", "civitai.com"), _lora_task("3", "civitai.com")],
        )

        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is not None
        assert scheduler.acquire(timeout=0.05) is None

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

    def test_exclusive_task_runs_last_after_ordinary_downloads_drain(self) -> None:
        """An exclusive task yields to every pending ordinary download and runs only in the idle tail.

        Guards the wedge fix: the annotator preload verify must never jump ahead of the image/aux fetches a
        worker needs to serve jobs, so a momentary lull cannot let it grab exclusivity and block them. Even
        enqueued first, it waits until nothing ordinary is pending and nothing is in flight.
        """
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))
        scheduler.enqueue_many([_task("a", "h1"), _task("b", "h2")])

        first = scheduler.acquire(timeout=0.05)
        second = scheduler.acquire(timeout=0.05)
        assert first is not None and second is not None
        assert {first.model_name, second.model_name} == {"a", "b"}  # ordinary downloads go first

        scheduler.release(first)
        scheduler.release(second)
        # Queue now holds only the exclusive task and nothing is in flight: it is finally admitted.
        admitted = scheduler.acquire(timeout=0.05)
        assert admitted is not None and admitted.exclusive is True

    def test_exclusive_in_flight_blocks_others(self) -> None:
        """While an exclusive task runs, no other task is admitted (it runs alone)."""
        scheduler = HostAwareDownloadScheduler(max_parallel_downloads=4, per_host_concurrency=1)
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        # An ordinary task arriving mid-flight may not start alongside the exclusive task.
        scheduler.enqueue(_task("a", "h1"))
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

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        # The exclusive task is still in flight, but its bound is already exceeded, so a later-arriving
        # image download is admitted instead of being starved.
        scheduler.enqueue(_task("a", "h1"))
        other = scheduler.acquire(timeout=0.05)
        assert other is not None and other.model_name == "a"

    def test_exclusivity_blocks_within_the_bound_then_relaxes_past_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Within the bound the exclusive task blocks others; once the bound elapses they are admitted."""
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "horde_worker_regen.process_management.models.download_scheduler.time.monotonic",
            lambda: clock["now"],
        )
        scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=4,
            per_host_concurrency=1,
            exclusive_timeout_seconds=300.0,
        )
        scheduler.enqueue(_exclusive_task("annotators", "unknown"))

        exclusive = scheduler.acquire(timeout=0.05)
        assert exclusive is not None and exclusive.exclusive is True
        scheduler.enqueue(_task("a", "h1"))
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
