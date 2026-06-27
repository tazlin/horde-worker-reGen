"""RED reproduction: Go live after a benchmark drain does not restore inference capacity.

The new benchmark-over-worker flow frees the GPU *gracefully* instead of stopping the worker: it drains the
queue, enters the download-only hold, and scales the inference processes to zero (``SET_CONCURRENCY`` with
``target_processes=0``). The worker stays alive and is meant to resume serving when the operator presses
"Go live".

But ``_apply_set_concurrency`` marks the download coordinator's inference startup latch as true, and
``leave_downloads_only_hold`` (GO_LIVE) restores inference only through ``maybe_start_inference_processes``,
which short-circuits the moment that flag is set. Nothing else regrows the pool from zero during normal
operation (the only auto-scaling paths shrink under VRAM pressure or restore a whole-card residency they
themselves established). So after a benchmark drain, Go live resumes job popping while the inference pool sits
at zero: the worker accepts jobs it has no process to run.

These tests assert the post-fix behaviour (Go live restores inference capacity), so they fail RED against the
current code.
"""

from __future__ import annotations

from tests.process_management.conftest import make_testable_process_manager


class _CountingInferenceLifecycle:
    """A process-lifecycle double that tracks only the inference-process count the drain/resume flow moves.

    Records enough of the lifecycle surface that ``enter_downloads_only_hold`` / ``_apply_set_concurrency`` /
    ``leave_downloads_only_hold`` exercise: scaling sets the live count, the up-front starter restores it to
    the provisioned size, and the count is readable back. Everything else the flow may touch is a no-op.
    """

    def __init__(self, *, provisioned: int) -> None:
        """Start fully provisioned (the worker was serving before the benchmark asked for the GPU)."""
        self._provisioned = provisioned
        self.inference_count = provisioned
        self.start_inference_calls = 0

    def start_download_process(self) -> None:
        """The hold ensures the download process is up; irrelevant to inference capacity here."""

    def start_safety_processes(self) -> None:
        """Safety is started separately; not what this repro measures."""

    def start_inference_processes(self) -> None:
        """Bring the inference pool back to the provisioned count (the up-front starter)."""
        self.start_inference_calls += 1
        self.inference_count = self._provisioned

    def scale_inference_processes(self, target_count: int, *, device_index: int | None = None) -> int:
        """Move the live inference-process count toward *target_count* and report the result."""
        self.inference_count = max(0, target_count)
        return self.inference_count

    def num_loaded_inference_processes(self, device_index: int | None = None) -> int:
        """Return the current inference-process count."""
        return self.inference_count


def _manager_with_counting_lifecycle(*, provisioned: int) -> tuple[object, _CountingInferenceLifecycle]:
    """A background-download manager that is up and serving, wired to a counting inference lifecycle."""
    manager = make_testable_process_manager()
    manager._enable_background_downloads = True
    manager._download_coordinator._enable_background_downloads = True
    manager._model_availability.update(
        present={"stable_diffusion"},
        currently_downloading=None,
        pending=(),
        failed=(),
        scan_complete=True,
    )
    lifecycle = _CountingInferenceLifecycle(provisioned=provisioned)
    manager._process_lifecycle = lifecycle  # type: ignore[assignment]
    manager._download_coordinator._process_lifecycle = lifecycle  # type: ignore[assignment]
    # The worker was already serving when the benchmark asked for the GPU.
    manager._download_coordinator.inference_processes_started = True
    manager._download_coordinator.safety_processes_started = True
    return manager, lifecycle


def test_go_live_after_drain_restores_inference_capacity() -> None:
    """The full drain -> hold -> scale-to-0 -> Go live sequence must leave the pool able to serve again."""
    manager, lifecycle = _manager_with_counting_lifecycle(provisioned=2)

    # The benchmark frees the GPU: hold, then scale inference to zero.
    manager._download_coordinator.enter_downloads_only_hold()  # type: ignore[attr-defined]
    manager._apply_set_concurrency(target_threads=None, target_processes=0)  # type: ignore[attr-defined]
    assert lifecycle.num_loaded_inference_processes() == 0  # GPU freed for the benchmark

    # The operator presses Go live to resume serving once the benchmark has finished.
    manager._download_coordinator.leave_downloads_only_hold()  # type: ignore[attr-defined]

    assert lifecycle.num_loaded_inference_processes() > 0, (
        "Go live left the inference pool at zero after the benchmark drain scaled it down; the worker resumes "
        "popping jobs but has no process to run them on."
    )


def test_go_live_after_drain_unblocks_the_lazy_starter() -> None:
    """Concretely: the lazy inference starter must be allowed to run again after a scale-to-zero drain.

    ``_apply_set_concurrency`` latches ``inference_processes_started`` True; if Go live does not clear that
    latch (or otherwise rescale up), ``maybe_start_inference_processes`` can never restart the pool, and a
    worker that gave up the GPU for a benchmark never gets it back.
    """
    manager, lifecycle = _manager_with_counting_lifecycle(provisioned=3)

    manager._download_coordinator.enter_downloads_only_hold()  # type: ignore[attr-defined]
    manager._apply_set_concurrency(target_threads=None, target_processes=0)  # type: ignore[attr-defined]

    manager._download_coordinator.leave_downloads_only_hold()  # type: ignore[attr-defined]

    assert lifecycle.start_inference_calls > 0 or lifecycle.inference_count > 0, (
        "after the drain, Go live neither restarted nor rescaled the inference pool; "
        "maybe_start_inference_processes stayed latched off by the scale-to-zero's inference_processes_started."
    )
