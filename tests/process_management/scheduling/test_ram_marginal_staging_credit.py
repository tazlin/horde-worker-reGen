"""Marginal RAM accounting for checkpoint staging at the scheduler seam.

The RAM-verdict path (:meth:`InferenceScheduler._apply_ram_verdict`) credits a reusable staging target's
retained pages so an in-place swap is priced at its marginal growth, prefers to spare that target from the
reclaim cycle, still contains an unbounded creep leak, and reconciles the credit against measured truth.

Contracts asserted here:
- a preload onto an idle retaining target is admitted where the cold-load charge would defer, and the credited
  admission is recorded for reconciliation;
- a busy or fresh target earns no credit;
- the stale-unload reclaim cycle spares the head's protected staging target, but the creep-containment override
  cycles a genuinely bloated idle slot regardless of protection or resident model;
- when even the credited verdict cannot fit, the reclaim still escalates to cycling a *different* stale slot;
- the measured-truth reconciliation flags a credit whose target grew past its charge and stays quiet otherwise.

Interaction with the head-priority RAM-defer barrier (tested in
``tests/process_management/regressions/test_head_of_queue_ram_defer_starvation_repro.py``): that suite pins
``_ram_budget.check_job`` to a ``Mock`` returning a hard non-fit and mocks ``_replace_stale_ram_unload_process``,
so the new credit and protect-id arguments are swallowed by the mocks and never reach real logic. The pinned
non-fit still defers, the barrier still latches, and its premises hold unchanged.

These tests are authored but NOT executed here: a live GPU worker occupies the machine and the standing
constraint forbids running pytest beside it. Run with ``AI_HORDE_TESTING=True pytest`` once the box is free.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _CREEP_CONTAINMENT_RSS_BYTES,
    _FRESH_INFERENCE_CHILD_BASELINE_MB,
    _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS,
    InferenceScheduler,
    _ReuseCreditRecord,
)
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MB = 1024 * 1024


def _replace_mock(scheduler: InferenceScheduler) -> Mock:
    """The mocked ``_replace_inference_process`` on the factory's mock lifecycle, typed for assertion access."""
    return scheduler._process_lifecycle._replace_inference_process  # type: ignore[return-value]


def _retaining_target(process_id: int = 0, *, rss_mb: float) -> HordeProcessInfo:
    """An idle, model-less inference slot that kept ``rss_mb`` of pages after unloading its prior model."""
    target = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    target.ram_usage_bytes = int(rss_mb * _MB)
    target.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
    return target


class TestStagingReuseCredit:
    """The retained-RSS credit measurement excludes busy and fresh targets."""

    def test_retaining_idle_target_yields_excess_over_baseline(self) -> None:
        """The credit is the target's resident RSS above a fresh child's baseline."""
        scheduler = _make_inference_scheduler()
        target = _retaining_target(rss_mb=8000.0)
        assert scheduler._staging_reuse_credit_mb(target) == pytest.approx(8000.0 - _FRESH_INFERENCE_CHILD_BASELINE_MB)

    def test_busy_target_earns_no_credit(self) -> None:
        """A busy target's pages are in live use, so it contributes no reusable credit."""
        scheduler = _make_inference_scheduler()
        busy = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_PRIMED)
        busy.ram_usage_bytes = int(9000.0 * _MB)
        assert scheduler._staging_reuse_credit_mb(busy) == 0.0

    def test_fresh_target_earns_no_credit(self) -> None:
        """A just-spawned child at baseline RSS yields zero credit (collapses to the full charge)."""
        scheduler = _make_inference_scheduler()
        fresh = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        fresh.ram_usage_bytes = int((_FRESH_INFERENCE_CHILD_BASELINE_MB - 200.0) * _MB)
        assert scheduler._staging_reuse_credit_mb(fresh) == 0.0


class TestCreditedAdmission:
    """A retaining target admits a swap the cold-load charge would defer, and the admission is recorded."""

    def test_credited_admit_records_pending_reconciliation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The credit admits the live-window SDXL swap and records it for the measured-truth check."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 16000.0)
        target = _retaining_target(rss_mb=8000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        scheduler._measured_available_ram_mb = lambda: 17946.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 4800.0  # type: ignore[method-assign]

        job = make_job_pop_response("head_model")
        admitted = scheduler._apply_ram_verdict(
            job,
            "x",
            target,
            is_head_blocker=False,
            no_live_resource_consumer=True,
        )
        assert admitted is True
        assert 0 in scheduler._pending_reuse_credits
        assert scheduler._pending_reuse_credits[0].model == "head_model"

    def test_credited_defer_escalates_to_cycle_of_a_different_stale_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When even the credited charge cannot fit, reclaim cycles a stale slot other than the target."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 16000.0)
        target = _retaining_target(0, rss_mb=4000.0)
        stale_other = _retaining_target(1, rss_mb=2000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target, 1: stale_other}))
        scheduler._measured_available_ram_mb = lambda: 5000.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 1024.0  # type: ignore[method-assign]
        scheduler.unload_models = Mock(return_value=False)  # type: ignore[method-assign]

        job = make_job_pop_response("head_model")
        admitted = scheduler._apply_ram_verdict(
            job,
            "x",
            target,
            is_head_blocker=False,
            no_live_resource_consumer=False,
        )
        assert admitted is False
        # The target was spared and the other stale slot was cycled instead.
        replace = _replace_mock(scheduler)
        assert replace.call_count == 1
        assert replace.call_args.args[0] is stale_other


class TestReclaimRetargeting:
    """Cycling spares the protected reuse target but the creep override still contains an unbounded leak."""

    def test_cycle_spares_protected_reuse_target(self) -> None:
        """The head's staging target is not cycled by the stale-unload reclaim when protected."""
        target = _retaining_target(rss_mb=5000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        assert scheduler._replace_stale_ram_unload_process(protect_process_id=0) is False
        assert _replace_mock(scheduler).called is False

    def test_unprotected_stale_slot_is_cycled(self) -> None:
        """Without protection the same retaining stale slot is cycled (the original last-resort behavior)."""
        target = _retaining_target(rss_mb=5000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        assert scheduler._replace_stale_ram_unload_process() is True
        assert _replace_mock(scheduler).call_args.args[0] is target

    def test_creep_override_cycles_bloated_slot_even_with_model_and_protection(self) -> None:
        """A slot above the creep ceiling is cycled regardless of a resident model or protection."""
        bloated = make_mock_process_info(0, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
        bloated.ram_usage_bytes = _CREEP_CONTAINMENT_RSS_BYTES + _MB
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: bloated}))
        scheduler._pending_reuse_credits[0] = _ReuseCreditRecord("resident", 100.0, 100.0, time.time())

        assert scheduler._replace_stale_ram_unload_process(protect_process_id=0) is True
        replace = _replace_mock(scheduler)
        assert replace.call_args.args[0] is bloated
        assert replace.call_args.kwargs["intentional_reclaim"] is True
        # A cycled slot's pending credit is void (its successor cold-loads).
        assert 0 not in scheduler._pending_reuse_credits

    def test_creep_victim_preferred_over_stale_victim(self) -> None:
        """When both a stale slot and a crept slot exist, creep containment cycles the crept one first."""
        stale = _retaining_target(0, rss_mb=2000.0)
        bloated = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
        bloated.ram_usage_bytes = _CREEP_CONTAINMENT_RSS_BYTES + _MB
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: stale, 1: bloated}))

        assert scheduler._replace_stale_ram_unload_process() is True
        assert _replace_mock(scheduler).call_args.args[0] is bloated


class TestCreditReconciliation:
    """The measured-truth check flags an over-generous credit and stays quiet within slack."""

    def _settled_target(
        self,
        scheduler: InferenceScheduler,
        *,
        admit_rss_mb: float,
        now_rss_mb: float,
        charge_mb: float,
    ) -> None:
        """Seat a settled credited target on ``scheduler`` whose RSS has grown since admit time."""
        proc = make_mock_process_info(0, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
        proc.ram_usage_bytes = int(now_rss_mb * _MB)
        scheduler._process_map = ProcessMap({0: proc})
        scheduler._pending_reuse_credits[0] = _ReuseCreditRecord(
            model="m",
            rss_at_admit_mb=admit_rss_mb,
            effective_charge_mb=charge_mb,
            admitted_at=time.time() - _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS - 1.0,
        )

    def test_over_generous_credit_is_flagged_once_and_cleared(self) -> None:
        """Growth exceeding the charge by more than the slack logs the discrepancy and drops the record."""
        scheduler = _make_inference_scheduler()
        # growth = 3049 MB against charge 1000 MB; 3049 > 1000 + 2048 slack -> flagged.
        self._settled_target(scheduler, admit_rss_mb=2000.0, now_rss_mb=5049.0, charge_mb=1000.0)
        messages: list[object] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            scheduler._reconcile_reuse_credit()
        finally:
            logger.remove(sink_id)
        assert any("too generous" in str(record) for record in messages)
        assert 0 not in scheduler._pending_reuse_credits

    def test_credit_within_slack_is_silent(self) -> None:
        """Growth within the charge plus slack clears the record without a discrepancy warning."""
        scheduler = _make_inference_scheduler()
        # growth = 3000 MB against charge 1000 MB; 3000 < 1000 + 2048 slack -> silent.
        self._settled_target(scheduler, admit_rss_mb=2000.0, now_rss_mb=4000.0, charge_mb=1000.0)
        messages: list[object] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            scheduler._reconcile_reuse_credit()
        finally:
            logger.remove(sink_id)
        assert not any("too generous" in str(record) for record in messages)
        assert 0 not in scheduler._pending_reuse_credits
