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

import dataclasses
import time
from unittest.mock import Mock, patch

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.dispatch_affinity import AffinitySkipState
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
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
        assert "no preload has been attempted" in text

    async def test_preload_deferred_quotes_the_admission_gates_own_verdict(self) -> None:
        """The stall names why the head was declined, since the defer notice coalesces on an unchanged reason.

        A head declined for the same arithmetic every cycle has no live log line to read, so pointing the
        operator at the budget lines leaves them looking for output that is not there.
        """
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        scheduler = self._scheduler(
            ProcessMap({1: make_mock_process_info(1, model_name=None)}), HordeModelMap(root={}), job_tracker
        )
        scheduler._record_preload_admission(
            AdmissionDecision.DEFER_BUDGET,
            job=head,
            reason="candidate 9576 MB vs available 2472 MB: does NOT fit",
        )

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.PRELOAD_DEFERRED
        assert "candidate 9576 MB vs available 2472 MB: does NOT fit" in text

    async def test_an_admission_record_for_another_model_is_not_quoted(self) -> None:
        """The record holds the latest decision for any job, so it is only quoted when it names this head."""
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        scheduler = self._scheduler(
            ProcessMap({1: make_mock_process_info(1, model_name=None)}), HordeModelMap(root={}), job_tracker
        )
        scheduler._record_preload_admission(
            AdmissionDecision.DEFER_BUDGET,
            job=make_job_pop_response(model="model-b"),
            reason="a decision about a different model",
        )

        _bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert "a different model" not in text

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

    async def test_reconcile_held_head_classifies_residency_reconciliation(self) -> None:
        """A reconcile-held resident head is named RESIDENCY_RECONCILIATION, not gate-less UNEXPLAINED.

        The reconcile gate stamps the held job in ``_dispatch_hold_since`` while it evicts idle VRAM so the
        job's materialisation fits the card. That is a benign, self-clearing swap-churn wait, so the stall
        text must name it rather than reporting the ``no matching gate`` scheduler-bug phrase.
        """
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(1, model_name="model-a", state=HordeProcessState.PRELOADED_MODEL)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        scheduler = self._scheduler(ProcessMap({1: holder}), horde_model_map, job_tracker)
        scheduler._dispatch_hold_since[str(head.id_)] = time.time()

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.RESIDENCY_RECONCILIATION
        assert "reconcile residency" in text
        assert "no matching gate" not in text

    async def test_post_processing_deferred_head_classifies_post_processing_defer(self) -> None:
        """A PP-deferred resident head is named POST_PROCESSING_DEFER, not gate-less UNEXPLAINED.

        The dispatch path records the post-processing co-residency defer verdict in
        ``_post_processing_defer_holds`` on the pass it computes it. The stall classifier reads that hold
        directly, so a head parked while an in-flight post-processing chain holds the card names the gate
        rather than reporting the ``no matching gate`` scheduler-bug phrase.
        """
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(1, model_name="model-a", state=HordeProcessState.PRELOADED_MODEL)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        scheduler = self._scheduler(ProcessMap({1: holder}), horde_model_map, job_tracker)
        scheduler._post_processing_defer_holds.add(str(head.id_))

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.POST_PROCESSING_DEFER
        assert "post-processing chain" in text
        assert "no matching gate" not in text

    async def _retained_sibling_card(self, *, predicted_peak_mb: float, monkeypatch):  # noqa: ANN001, ANN202
        """A resident, idle head on a card where a sibling slot holds another model's weights across jobs.

        16376MB total, less the sibling context (2030) and its retained weights (6800), leaves 7546MB for the
        caller's chosen sampling peak.
        """
        monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(1, model_name="model-a", state=HordeProcessState.PRELOADED_MODEL)
        holder.total_vram_mb = 16376
        retainer = make_mock_process_info(2, model_name="model-b")
        retainer.total_vram_mb = 16376
        retainer.retained_resident_model = "model-b"
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        horde_model_map.update_entry(
            horde_model_name="model-b", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2
        )
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({1: holder, 2: retainer}),
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(
                max_threads=2,
                enable_vram_budget=True,
                vram_reserve_mb=2048,
                ram_reserve_mb=4096,
            ),
            max_concurrent=2,
            max_inference=2,
        )
        scheduler._overhead.set_marginal_overhead_mb(2030.0)
        scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
            return_value=Mock(fits=False, predicted_mb=predicted_peak_mb, reserve_mb=4096.0),
        )
        return scheduler, head, retainer

    async def test_retained_resident_hold_classifies_residency_reconciliation(self, monkeypatch) -> None:  # noqa: ANN001
        """A head waiting on a sibling's retained weights is named as residency reconciliation.

        The wait is a self-clearing swap of weights held across jobs, not the gate-less scheduler stall the
        fall-through reports, so it must not read as ``no matching gate``.
        """
        scheduler, head, _retainer = await self._retained_sibling_card(
            predicted_peak_mb=8258.0, monkeypatch=monkeypatch
        )

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.RESIDENCY_RECONCILIATION
        assert "retained weights" in text
        assert "no matching gate" not in text

    async def test_naming_the_retained_resident_hold_evicts_nothing(self, monkeypatch) -> None:  # noqa: ANN001
        """Classifying a stall is a read: the retained weights it names are still held afterwards.

        The classifier runs every tick; if naming this wait actuated, the diagnostic itself would be
        reclaiming the card behind the gate that owns that decision.
        """
        scheduler, head, retainer = await self._retained_sibling_card(
            predicted_peak_mb=8258.0, monkeypatch=monkeypatch
        )

        scheduler._classify_dispatch_stall(head, {})

        assert retainer.retained_resident_model == "model-b"
        retainer.pipe_connection.send.assert_not_called()  # type: ignore[attr-defined]

    async def test_a_fitting_card_leaves_the_retained_sibling_unnamed(self, monkeypatch) -> None:  # noqa: ANN001
        """Control: where the head fits beside the retained weights, this bucket does not claim the slot."""
        scheduler, head, _retainer = await self._retained_sibling_card(
            predicted_peak_mb=3000.0, monkeypatch=monkeypatch
        )

        bucket, _text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.UNEXPLAINED

    async def test_resident_idle_head_without_hold_classifies_unexplained(self) -> None:
        """The same resident-idle head with no reconciliation hold still falls through to UNEXPLAINED.

        Control for the reconcile bucket: an idle-resident head that no gate (and no reconcile hold) claims
        is the genuinely inexplicable scheduler stall, and must keep the ``no matching gate`` attribution.
        """
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        holder = make_mock_process_info(1, model_name="model-a", state=HordeProcessState.PRELOADED_MODEL)
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name="model-a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        scheduler = self._scheduler(ProcessMap({1: holder}), horde_model_map, job_tracker)

        bucket, text = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.UNEXPLAINED
        assert "no matching gate" in text


class TestStallReasonStability:
    """The stall reason names the block and nothing that merely advances with the clock.

    Two consumers compare the reason across cycles: the parked-head line throttles on it being unchanged, and
    recovery judges whether a rung moved the head's blocker by whether it changed. A reason carrying a
    per-cycle counter would defeat both, logging every cycle of a sustained stall and making every rung look
    effective. The counters still reach the operator, on the formatted line.
    """

    @staticmethod
    async def _scheduler_with_bypassed_head() -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
        """A parked cold head that resident-model jobs have been skipping past for some time."""
        job_tracker = JobTracker()
        head = make_job_pop_response(model="model-a")
        await job_tracker.record_popped_job(head)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({1: make_mock_process_info(1, model_name=None)}),
            horde_model_map=HordeModelMap(root={}),
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        scheduler._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head.id_),
            first_skip_time=time.time() - 300.0,
            skip_count=1,
        )
        scheduler._head_starvation_job_id = str(head.id_)
        scheduler._head_starvation_since = time.time() - 300.0
        return scheduler, head

    async def test_advancing_bypass_counters_leave_the_reason_unchanged(self) -> None:
        """Skips accumulating against an unchanged gate do not rewrite what the gate is."""
        scheduler, head = await self._scheduler_with_bypassed_head()

        first = scheduler._diagnose_dispatch_stall(head, {})
        state = scheduler._affinity_skip_state
        scheduler._affinity_skip_state = dataclasses.replace(
            state,
            first_skip_time=state.first_skip_time - 120.0,
            skip_count=state.skip_count + 1,
        )

        assert scheduler._diagnose_dispatch_stall(head, {}) == first

    async def test_a_sustained_block_logs_once_per_interval_and_still_shows_the_counters(self) -> None:
        """The line is throttled across cycles, and the quantities it drops from the reason appear in it."""
        scheduler, head = await self._scheduler_with_bypassed_head()

        with patch.object(inference_scheduler_module.logger, "opt") as opt:
            scheduler._log_dispatch_stall_if_needed({})
            state = scheduler._affinity_skip_state
            scheduler._affinity_skip_state = dataclasses.replace(
                state,
                first_skip_time=state.first_skip_time - 120.0,
            )
            scheduler._log_dispatch_stall_if_needed({})

        assert opt.return_value.warning.call_count == 1, (
            "a block that has not changed is stated once per interval, however its counters move"
        )
        emitted = str(opt.return_value.warning.call_args)
        assert "affinity line-skips" in emitted, "the line still tells the operator the head is being fed past"

    async def test_a_changed_block_speaks_immediately(self) -> None:
        """A different gate is new information, so it is not held back by the previous gate's interval."""
        scheduler, head = await self._scheduler_with_bypassed_head()

        with patch.object(inference_scheduler_module.logger, "opt") as opt:
            scheduler._log_dispatch_stall_if_needed({})
            scheduler._record_preload_admission(
                AdmissionDecision.DEFER_BUDGET,
                job=head,
                reason="the card cannot hold these weights beside the live contexts",
            )
            scheduler._log_dispatch_stall_if_needed({})

        assert opt.return_value.warning.call_count == 2
        assert "cannot hold these weights" in str(opt.return_value.warning.call_args)


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
