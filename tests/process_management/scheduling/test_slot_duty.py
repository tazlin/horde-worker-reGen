"""Slot-duty accounting: the capacity-normalized active/idle/gated wall-clock ledger.

The device-utilization telemetry says whether the GPU was busy; the slot-duty ledger says what the
*configured capacity* was doing and, for every empty slot-second, which gate or supply state kept it
empty. Its value rests on two invariants pinned here:

- **Conservation**: every observed second of every slot lands in exactly one bucket, so a window's
  bucket totals sum to ``capacity x elapsed`` and shares are directly comparable across windows.
- **Attribution fidelity**: the empty-slot bucket comes from the same derivation that explains a
  parked head (`InferenceScheduler._classify_dispatch_stall`), so the periodic attribution line, the
  stats stream, and the parked-head log text never name different causes for the same stall.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyAccumulator, SlotDutyBucket
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class TestAccumulatorConservation:
    """Every observed slot-second lands in exactly one bucket."""

    def test_totals_sum_to_capacity_times_elapsed(self) -> None:
        """Across mixed observations, the bucket totals conserve capacity x wall exactly.

        Each interval is priced at its closing observation's state (a one-tick approximation; the
        control loop ticks sub-second, so a transition mis-prices at most one tick).
        """
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=2, busy_slots=0, waiting_jobs=0, hold=None)
        acc.observe(110.0, capacity=2, busy_slots=1, waiting_jobs=0, hold=None)
        acc.observe(125.0, capacity=2, busy_slots=1, waiting_jobs=1, hold=SlotDutyBucket.OVERLAP_HEADWAY)
        acc.observe(130.0, capacity=2, busy_slots=2, waiting_jobs=1, hold=None)

        totals = acc.totals()
        assert sum(totals.values()) == (130.0 - 100.0) * 2
        assert totals[SlotDutyBucket.SAMPLING] == 10.0 + 15.0 + 5.0 * 2
        assert totals[SlotDutyBucket.NO_LOCAL_WORK] == 10.0
        assert totals[SlotDutyBucket.OVERLAP_HEADWAY] == 15.0

    def test_first_observation_only_anchors(self) -> None:
        """The first call attributes nothing (there is no prior interval to price)."""
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=2, busy_slots=2, waiting_jobs=0, hold=None)
        assert acc.totals() == {}

    def test_backwards_or_stalled_clock_contributes_nothing(self) -> None:
        """A non-advancing clock reading never corrupts totals."""
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=2, busy_slots=1, waiting_jobs=0, hold=None)
        acc.observe(99.0, capacity=2, busy_slots=1, waiting_jobs=0, hold=None)
        acc.observe(99.0, capacity=2, busy_slots=1, waiting_jobs=0, hold=None)
        assert acc.totals() == {}

    def test_busy_slots_clamped_to_capacity(self) -> None:
        """An in-flight count above capacity (transient over-admit) never over-credits sampling."""
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=2, busy_slots=0, waiting_jobs=0, hold=None)
        acc.observe(110.0, capacity=2, busy_slots=5, waiting_jobs=0, hold=None)
        assert acc.totals() == {SlotDutyBucket.SAMPLING: 20.0}

    def test_waiting_work_without_named_hold_reads_unexplained(self) -> None:
        """An empty slot with queued work and no named gate is the stall-shaped bucket, not silence."""
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=1, busy_slots=0, waiting_jobs=2, hold=None)
        acc.observe(105.0, capacity=1, busy_slots=0, waiting_jobs=2, hold=None)
        assert acc.totals() == {SlotDutyBucket.UNEXPLAINED: 5.0}

    def test_no_waiting_work_overrides_hold(self) -> None:
        """With nothing queued, the empty slot is supply-side regardless of a stale hold value."""
        acc = SlotDutyAccumulator()
        acc.observe(100.0, capacity=1, busy_slots=0, waiting_jobs=0, hold=SlotDutyBucket.OVERLAP_HEADWAY)
        acc.observe(105.0, capacity=1, busy_slots=0, waiting_jobs=0, hold=SlotDutyBucket.OVERLAP_HEADWAY)
        assert acc.totals() == {SlotDutyBucket.NO_LOCAL_WORK: 5.0}


class TestWindowFormatting:
    """The periodic attribution line is compact, share-based, and leads with the productive bucket."""

    def test_sampling_leads_and_shares_sum(self) -> None:
        """Sampling renders first; remaining buckets follow largest-first."""
        line = SlotDutyAccumulator.format_window(
            {
                SlotDutyBucket.OVERLAP_HEADWAY: 30.0,
                SlotDutyBucket.SAMPLING: 60.0,
                SlotDutyBucket.NO_LOCAL_WORK: 10.0,
            },
            capacity=2,
        )
        assert line is not None
        assert line.startswith("slot attribution (capacity 2): sampling 60%")
        assert line.index("overlap_headway 30%") < line.index("no_local_work 10%")

    def test_empty_window_renders_nothing(self) -> None:
        """A quiet window produces no attribution fragment rather than a zero-division."""
        assert SlotDutyAccumulator.format_window({}, capacity=2) is None


class TestSchedulerClassifierBuckets:
    """The stall classifier's bucket half mirrors its text half for the load-path gates."""

    def _scheduler(self, process_map: ProcessMap, horde_model_map: HordeModelMap, job_tracker: JobTracker):  # noqa: ANN202
        return _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )

    async def test_loading_model_classifies_model_loading(self) -> None:
        """A head whose model is mid-preload prices the empty slot as MODEL_LOADING."""
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADING, process_id=1)
        scheduler = self._scheduler(
            ProcessMap({1: make_mock_process_info(1, model_name=None)}), horde_model_map, job_tracker
        )

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.MODEL_LOADING
        assert "preload is in progress" in text

    async def test_unadmitted_preload_classifies_preload_deferred(self) -> None:
        """A head whose model is neither resident nor loading prices as PRELOAD_DEFERRED."""
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        scheduler = self._scheduler(
            ProcessMap({1: make_mock_process_info(1, model_name=None)}), HordeModelMap(root={}), job_tracker
        )

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.PRELOAD_DEFERRED
        assert "no preload has been admitted" in text

    async def test_busy_resident_slot_classifies_resident_slot_busy(self) -> None:
        """A head whose model is resident only on a busy process prices as RESIDENT_SLOT_BUSY."""
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(1, model_name="model-a", state=HordeProcessState.INFERENCE_STARTING)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        scheduler = self._scheduler(ProcessMap({1: holder}), horde_model_map, job_tracker)

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.RESIDENT_SLOT_BUSY
        assert "that process is busy" in text

    async def test_exclusive_hold_classifies_exclusive_isolation(self) -> None:
        """A cap collapsed by an exclusive admit is attributed to the admit, not the generic cap."""
        job_tracker = JobTracker()
        exclusive = make_job_pop_response(model="model-x")
        await job_tracker.record_popped_job(exclusive)
        await job_tracker.mark_inference_started(exclusive)
        job_tracker.mark_admitted_over_budget(exclusive)
        job_tracker.mark_admitted_exclusive(exclusive)

        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(2, model_name="model-a", state=HordeProcessState.PRELOADED_MODEL)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)
        scheduler = self._scheduler(
            ProcessMap({1: make_mock_process_info(1, model_name="model-x"), 2: holder}), horde_model_map, job_tracker
        )

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.EXCLUSIVE_ISOLATION
        assert "exclusively-admitted over-budget job" in text


class TestRecordSlotDutyIntegration:
    """The per-tick hook feeds the accumulator with the live pool's numbers."""

    async def test_snapshot_reflects_busy_and_hold(self, monkeypatch) -> None:  # noqa: ANN001
        """Two ticks apart, a busy slot accrues SAMPLING and the held slot accrues the named gate."""
        job_tracker = JobTracker()
        running = make_job_pop_response(model="model-x")
        await job_tracker.record_popped_job(running)
        await job_tracker.mark_inference_started(running)
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)

        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({1: make_mock_process_info(1, model_name="model-x")}),
            horde_model_map=HordeModelMap(root={}),
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        clock = iter([1000.0, 1010.0])
        monkeypatch.setattr(
            "horde_worker_regen.process_management.scheduling.inference_scheduler.time.time",
            lambda: next(clock),
        )

        scheduler.record_slot_duty({})
        scheduler.record_slot_duty({})

        totals, capacity, hold = scheduler.slot_duty_snapshot()
        assert capacity == 2
        assert totals[SlotDutyBucket.SAMPLING] == 10.0
        # The head's model is not resident and nothing is loading it: the empty slot is a deferred preload.
        assert totals[SlotDutyBucket.PRELOAD_DEFERRED] == 10.0
        assert hold == str(SlotDutyBucket.PRELOAD_DEFERRED)
