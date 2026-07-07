"""Reproductions of the threads=2 co-residence thrash on a large single card.

The failure mode (a 24 GB card, ``max_threads=2``, several SDXL models): the worker spins up more
inference processes than it does at ``max_threads=1``, and each extra CUDA context retains multiple GB
of allocator/runtime VRAM that emptying the cache does not return to the device. Steady-state device-free
therefore sits far below what the streaming forecast predicts is reclaimable, because the forecast sizes
``free_after_model_evict`` from the *probe's* marginal per-context cost, which is measured with a minimal
matmul holder that never allocates a model's worth of cache. The gap between forecast-optimism and
reclaim-reality is harmless at ``max_threads=1`` (only one or two contexts) but at ``max_threads=2`` the
extra contexts make it large enough that every head-of-queue model fails its activation-reserve check,
finds nothing left to evict, and is admitted *exclusive*, evicting every resident model. The next
job then reloads from cold, so the worker churns full reloads and duty cycle collapses below the
``max_threads=1`` baseline instead of rising.

The desired behavior these scenarios assert:

* The per-additional-context cost the forecast uses must reflect the *measured* idle reality once it is
  known, not stay pinned to the optimistic probe figure when the device proves the contexts cost more.
* When the live contexts structurally pin device-free below what co-residence needs, the remedy is to
  reduce the live process count (free a context) so two SDXL models co-reside and pipeline, never to
  evict every resident model and reload each job.
* ``max_threads=1`` (one or two contexts, the optimism harmless) must be unaffected: no spurious teardown.

These are RED against the current scheduler; the fix is to reconcile the forecast's marginal with the
measured idle floor and route a context over-commit into the existing process-count-reduction machinery
instead of the exclusive-evict-all admit.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# Figures representative of the 24 GB card the thrash was observed on.
_DEVICE_TOTAL_VRAM_MB = 24074.0
_PER_PROCESS_OVERHEAD_MB = 4266.0  # first-context cost: one-time CUDA runtime + one context (probe section 0)
_PROBE_MARGINAL_MB = 650.0  # optimistic: the minimal matmul holder never allocates a model's cache
_SDXL_WEIGHTS_MB = 4900.0
_SDXL_RESERVE_MB = 4802.0  # activation-inclusive sampling-peak headroom for a 1024x1024 SDXL job

# With four live contexts the device truly leaves ~9.5 GB free when idle with every model evicted: the
# probe's optimistic marginal predicts ~17.9 GB, the gap being unreclaimable per-context cache.
_LIVE_CONTEXT_COUNT = 4
_MEASURED_IDLE_FREE_MB = 9516.0
_MEASURED_IDLE_USED_MB = _DEVICE_TOTAL_VRAM_MB - _MEASURED_IDLE_FREE_MB  # 14558
# Derived true marginal: (used - first_context_overhead) / (contexts - 1) = (14558 - 4266) / 3 ~= 3431.
_TRUE_MARGINAL_MB = (_MEASURED_IDLE_USED_MB - _PER_PROCESS_OVERHEAD_MB) / (_LIVE_CONTEXT_COUNT - 1)


def _idle_context_map(num_processes: int, *, free_mb: float) -> ProcessMap:
    """A process map of idle, model-free inference contexts reporting a given device-wide free VRAM."""
    procs: dict[int, object] = {}
    used_mb = _DEVICE_TOTAL_VRAM_MB - free_mb
    for pid in range(1, num_processes + 1):
        proc = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = used_mb
        procs[pid] = proc
    return ProcessMap(procs)


def _coreside_scheduler(num_processes: int, *, free_mb: float) -> tuple[object, ProcessMap]:
    """A budget-active scheduler over ``num_processes`` idle contexts pinning device-free to ``free_mb``."""
    process_map = _idle_context_map(num_processes, free_mb=free_mb)
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        max_threads=2,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        bridge_data=bridge_data,
        max_concurrent=2,
        max_inference=num_processes,
    )
    return scheduler, process_map


class TestMeasuredMarginalReconciliation:
    """The forecast's per-context cost must track measured reality, not stay pinned to the probe optimism."""

    def test_measured_idle_floor_supersedes_optimistic_probe_marginal(self) -> None:
        """When idle device-free proves the contexts cost more than the probe figure, use the larger.

        The probe's matmul holder under-measures a real inference context (it never allocates a model's
        worth of cache that emptying the allocator does not return). Once the worker has a clean all-idle,
        all-evicted reading, that measured floor is ground truth: the marginal the forecast uses must rise
        to it so ``free_after_model_evict`` reflects what reclaim can actually achieve.
        """
        scheduler, _process_map = _coreside_scheduler(_LIVE_CONTEXT_COUNT, free_mb=_MEASURED_IDLE_FREE_MB)
        scheduler.set_measured_marginal_overhead_mb(_PROBE_MARGINAL_MB)
        # Feed the clean idle baseline the scheduler would capture from the live reports.
        scheduler._maybe_capture_idle_context_residency()

        marginal = scheduler._marginal_process_overhead_mb()

        assert marginal is not None
        # The measured floor (~3431) must win over the optimistic probe figure (650).
        assert marginal == pytest.approx(_TRUE_MARGINAL_MB, abs=50.0)

    def test_probe_marginal_retained_before_any_idle_baseline(self) -> None:
        """Before a clean idle baseline exists (startup), the probe figure is still used.

        The probe exists precisely to cover the startup window where siblings have not reached idle, so a
        cold scheduler with no measured floor must keep trusting it rather than falling back to the
        first-context overhead.
        """
        scheduler, _process_map = _coreside_scheduler(_LIVE_CONTEXT_COUNT, free_mb=_MEASURED_IDLE_FREE_MB)
        scheduler.set_measured_marginal_overhead_mb(_PROBE_MARGINAL_MB)
        # No idle baseline captured yet.

        assert scheduler._marginal_process_overhead_mb() == pytest.approx(_PROBE_MARGINAL_MB)


class TestForecastReflectsContextOvercommit:
    """Fed the true context cost, the forecast must call for fewer contexts so two SDXL co-reside."""

    @staticmethod
    def _forecast(marginal_mb: float, contexts: int) -> StreamForecast:
        additional = max(0, contexts - 1)
        free_after_model_evict = _DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB - marginal_mb * additional
        return StreamForecast(
            weights_mb=_SDXL_WEIGHTS_MB,
            reserve_mb=_SDXL_RESERVE_MB,
            free_now_mb=_MEASURED_IDLE_FREE_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=free_after_model_evict,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )

    def test_true_marginal_shows_model_eviction_insufficient(self) -> None:
        """With the true per-context cost, evicting sibling models cannot make room across four contexts.

        Four contexts plus the evicted-model floor leave only ~9.5 GB, which does not hold the SDXL weights
        plus their activation reserve, so the model genuinely needs a context freed (fewer processes), not
        just a sibling model evicted. ``fits_alone`` confirms the card has room once contexts are reduced.
        """
        forecast = self._forecast(_TRUE_MARGINAL_MB, _LIVE_CONTEXT_COUNT)
        assert forecast.fits_after_model_evict is False
        assert forecast.fits_alone is True

    def test_optimistic_marginal_wrongly_reads_model_eviction_suffices(self) -> None:
        """The probe optimism is the poisoned input: it believes evicting models leaves ample room.

        This is the reading the live worker acted on: ``free_after_model_evict`` predicts ~17.9 GB, so the
        forecast judges that simply evicting a sibling model would make room, and the scheduler keeps trying
        that (and failing, because the device never returns the VRAM) instead of reducing the context count.
        The same head flips to model-eviction-insufficient once the true marginal is used.
        """
        forecast = self._forecast(_PROBE_MARGINAL_MB, _LIVE_CONTEXT_COUNT)
        assert forecast.fits_after_model_evict is True

    def test_two_contexts_let_sdxl_coreside(self) -> None:
        """Reducing to two live contexts frees enough that an SDXL head co-resides: the target end-state.

        This is what the process-count reduction converges to: two contexts, two SDXL models resident,
        pipelined. The forecast must confirm the head fits once the worker is down to two contexts.
        """
        forecast = self._forecast(_TRUE_MARGINAL_MB, 2)
        assert forecast.fits_after_model_evict is True


class TestNoEvictAllThrash:
    """A head that cannot co-reside across the live contexts must reduce the process count, not evict all."""

    async def test_third_sdxl_head_does_not_force_exclusive_evict_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A budget-deferred SDXL head on a context-pinned idle card must not evict every resident model.

        With four idle contexts pinning device-free, the head's weights-plus-reserve does not fit, gentle
        reclaim frees nothing the device returns, and the scheduler admits the head exclusive
        -- which evicts the very models the next jobs reuse. The desired remedy is a process-count
        reduction (stop an idle sibling so a context's VRAM returns), leaving the other resident models in
        place for the pipeline.
        """
        from tests.process_management.conftest import make_job_pop_response, track_popped_job_async

        scheduler, process_map = _coreside_scheduler(_LIVE_CONTEXT_COUNT, free_mb=_MEASURED_IDLE_FREE_MB)
        scheduler.set_measured_marginal_overhead_mb(_PROBE_MARGINAL_MB)
        scheduler._maybe_capture_idle_context_residency()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.resources.resource_budget.predict_job_weight_mb",
            lambda job, baseline: _SDXL_WEIGHTS_MB,
        )

        head_job = make_job_pop_response("AlbedoBase XL (SDXL)")
        await track_popped_job_async(scheduler._job_tracker, head_job)

        scheduler.preload_models()

        # The head must not have been admitted by evicting every resident model exclusively.
        assert scheduler._job_tracker.is_admitted_over_budget(head_job) is False, (
            "the head was admitted over budget (the evict-all thrash) instead of reducing contexts"
        )

    def test_max_threads_one_two_contexts_no_spurious_teardown(self) -> None:
        """At two contexts the optimism is harmless, so the forecast must not demand a context teardown.

        ``max_threads=1`` ran at a healthy duty cycle precisely because two cheap contexts leave ample
        room; the reconciliation must not regress it into tearing down a context it does not need to.
        """
        forecast = TestForecastReflectsContextOvercommit._forecast(_TRUE_MARGINAL_MB, 2)
        assert forecast.needs_process_count_reduction is False
        assert forecast.fits_after_model_evict is True


class TestConcurrentSlotModelDiversity:
    """A multi-threaded worker must fill its second slot with a distinct-model job, not idle the thread."""

    async def test_distinct_model_job_fills_idle_thread_behind_busy_head(self) -> None:
        """Queue A,A,A,B with one A sampling: the idle thread must run B, not sit idle behind the A head.

        The FIFO head (the next A) wants the process already sampling its model, which cannot accept work.
        A selection that only looks at the head returns nothing, so the second inference process sits idle
        while B, resident on it and ready, waits at the back of the queue. Threading B alongside the
        running A processes a distinct model for free under the A run and avoids loading a duplicate copy of
        A onto a second process; idling the thread leaves that throughput on the table.
        """
        proc_a = make_mock_process_info(1, model_name="model_a", state=HordeProcessState.INFERENCE_STARTING)
        proc_b = make_mock_process_info(2, model_name="model_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: proc_a, 2: proc_b})

        hmm = HordeModelMap(root={})
        hmm.update_entry(horde_model_name="model_a", load_state=ModelLoadState.IN_USE, process_id=1)
        hmm.update_entry(horde_model_name="model_b", load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)

        job_tracker = JobTracker()
        a1 = make_job_pop_response("model_a")
        await mark_job_in_progress_async(job_tracker, a1)
        a2 = make_job_pop_response("model_a")
        a3 = make_job_pop_response("model_a")
        b1 = make_job_pop_response("model_b")
        for job in (a2, a3, b1):
            await track_popped_job_async(job_tracker, job)

        sched = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=hmm,
            job_tracker=job_tracker,
            max_concurrent=2,
            max_inference=2,
        )
        result = await sched.get_next_job_and_process()

        assert result is not None, "the second thread must run B, not sit idle behind the busy same-model head"
        assert result.next_job is b1
        assert result.process_with_model is proc_b
        assert result.line_skip is not None and result.line_skip.displaced_job is a2

    async def test_no_distinct_model_keeps_waiting_for_busy_head(self) -> None:
        """With only same-model jobs queued behind a busy head, there is nothing distinct to thread.

        The diversity bypass must not invent work: when every pending job wants the busy process's model,
        the worker waits for that process rather than loading a duplicate copy onto the idle one.
        """
        proc_a = make_mock_process_info(1, model_name="model_a", state=HordeProcessState.INFERENCE_STARTING)
        proc_idle = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: proc_a, 2: proc_idle})

        hmm = HordeModelMap(root={})
        hmm.update_entry(horde_model_name="model_a", load_state=ModelLoadState.IN_USE, process_id=1)

        job_tracker = JobTracker()
        a1 = make_job_pop_response("model_a")
        await mark_job_in_progress_async(job_tracker, a1)
        for job in (make_job_pop_response("model_a"), make_job_pop_response("model_a")):
            await track_popped_job_async(job_tracker, job)

        sched = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=hmm,
            job_tracker=job_tracker,
            max_concurrent=2,
            max_inference=2,
        )
        assert await sched.get_next_job_and_process() is None
