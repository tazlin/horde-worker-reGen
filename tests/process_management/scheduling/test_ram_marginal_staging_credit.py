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
from horde_sdk.generation_parameters import KNOWN_FACEFIXERS, KNOWN_UPSCALERS
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo, LoadCompletionSample
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import BudgetVerdict
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.ram_reclaim import (
    CREEP_CONTAINMENT_RSS_BYTES,
    FRESH_INFERENCE_CHILD_BASELINE_MB,
    ReuseCreditKind,
    ReuseCreditRecord,
    staging_reuse_credit_mb,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    mark_ram_unload_settled,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MB = 1024 * 1024


def _replace_mock(scheduler: InferenceScheduler) -> Mock:
    """The mocked ``_replace_inference_process`` on the factory's mock lifecycle, typed for assertion access."""
    return scheduler._process_lifecycle._replace_inference_process  # type: ignore[return-value]


_PINNED_HOST_RAM_MB = 32000.0
"""Host total RAM these rows price against.

The stale-unload retention threshold is derived from the host, so leaving it to the runner's own RAM would
make every ``rss_mb`` in this module mean something different per machine.
"""


def _pin_host_ram(scheduler: InferenceScheduler) -> None:
    """Pin the host total RAM so the reclaim's retention threshold is the same on every runner."""
    scheduler._measured_total_ram_mb = lambda: _PINNED_HOST_RAM_MB  # type: ignore[method-assign]


def _retaining_target(process_id: int = 0, *, rss_mb: float) -> HordeProcessInfo:
    """An idle, model-less inference slot that kept ``rss_mb`` of pages after unloading its prior model."""
    target = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    target.ram_usage_bytes = int(rss_mb * _MB)
    target.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
    mark_ram_unload_settled(target)
    return target


class TestStagingReuseCredit:
    """The retained-RSS credit measurement excludes busy and fresh targets."""

    def test_retaining_idle_target_yields_excess_over_baseline(self) -> None:
        """The credit is the target's resident RSS above a fresh child's baseline."""
        target = _retaining_target(rss_mb=8000.0)
        assert staging_reuse_credit_mb(target) == pytest.approx(8000.0 - FRESH_INFERENCE_CHILD_BASELINE_MB)

    def test_busy_target_earns_no_credit(self) -> None:
        """A busy target's pages are in live use, so it contributes no reusable credit."""
        busy = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_PRIMED)
        busy.ram_usage_bytes = int(9000.0 * _MB)
        assert staging_reuse_credit_mb(busy) == 0.0

    def test_fresh_target_earns_no_credit(self) -> None:
        """A just-spawned child at baseline RSS yields zero credit (collapses to the full charge)."""
        fresh = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        fresh.ram_usage_bytes = int((FRESH_INFERENCE_CHILD_BASELINE_MB - 200.0) * _MB)
        assert staging_reuse_credit_mb(fresh) == 0.0

    def test_a_reading_older_than_the_unload_earns_no_credit(self) -> None:
        """A reading sampled before the RAM unload still holds the released model, so it is no evidence of reuse."""
        unsettled = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        unsettled.ram_usage_bytes = int(17000.0 * _MB)
        unsettled.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        unsettled.last_ram_unload_requested_at = time.time()
        unsettled.report_sampled_at = unsettled.last_ram_unload_requested_at - 1.0
        assert staging_reuse_credit_mb(unsettled) == 0.0

        mark_ram_unload_settled(unsettled)
        assert staging_reuse_credit_mb(unsettled) == pytest.approx(17000.0 - FRESH_INFERENCE_CHILD_BASELINE_MB)


_MAPPING_MB = 12533.0
"""A checkpoint mapping larger than the committable memory the commit-bound rows leave."""


def _credit_lines(messages: list[str]) -> list[str]:
    return [message for message in messages if "RAM credit admitting" in message]


class TestCreditAgainstCommit:
    """The retained-page credit prices physical RAM only, and its log line says which test it lowered."""

    @staticmethod
    async def _admit(
        *,
        available_commit_mb: float | None,
    ) -> tuple[InferenceScheduler, bool, list[str], float]:
        target = _retaining_target(rss_mb=20000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        scheduler._measured_available_ram_mb = lambda: 60000.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 4800.0  # type: ignore[method-assign]
        scheduler._checkpoint_staging_charge_mb = lambda _job: _MAPPING_MB  # type: ignore[method-assign]
        scheduler.set_available_commit_mb_provider(lambda: available_commit_mb)
        job = make_job_pop_response("head_model")
        await track_popped_job_async(scheduler._job_tracker, job)
        features_mb = scheduler.snapshot().queue.jobs[str(job.id_)].feature_ram_mb
        messages: list[str] = []
        sink_id = logger.add(lambda record: messages.append(str(record)), level="INFO")
        try:
            admitted = scheduler._apply_ram_verdict(job, target, is_head_blocker=False, no_live_resource_consumer=True)
        finally:
            logger.remove(sink_id)
        return scheduler, admitted, messages, features_mb

    async def test_a_commit_bound_host_defers_the_mapping_the_credit_would_admit(self) -> None:
        """Committable memory short of the whole mapping defers the load and records no credited admission."""
        scheduler, admitted, messages, _ = await self._admit(available_commit_mb=9000.0)

        assert admitted is False
        assert 0 not in scheduler.ram_reclaim.pending_reuse_credits
        assert _credit_lines(messages) == []
        assert any("commit-bound: 9000 MB committable" in message for message in messages)

    async def test_the_credit_line_names_physical_ram_and_the_whole_commit_charge(self) -> None:
        """An admitted credit reads as a physical-RAM credit beside the uncredited commit charge."""
        _, admitted, messages, features_mb = await self._admit(available_commit_mb=1_000_000.0)

        assert admitted is True
        (line,) = _credit_lines(messages)
        assert "applied to physical RAM" in line
        assert f"commit charged the whole mapping ~{_MAPPING_MB + features_mb:.0f} MB" in line

    async def test_the_credit_line_without_a_commit_figure_says_commit_is_not_priced(self) -> None:
        """A host with no commit figure admits as before, and the line says commit was not priced."""
        scheduler, admitted, messages, _ = await self._admit(available_commit_mb=None)

        assert admitted is True
        assert 0 in scheduler.ram_reclaim.pending_reuse_credits
        (line,) = _credit_lines(messages)
        assert "applied to physical RAM" in line
        assert "commit not priced" in line


class TestCreditedAdmission:
    """A retaining target admits a swap the cold-load charge would defer, and the admission is recorded."""

    async def test_credited_admit_records_pending_reconciliation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The credit admits the live-window SDXL swap and records it for the measured-truth check."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 16000.0)
        target = _retaining_target(rss_mb=8000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        scheduler._measured_available_ram_mb = lambda: 17946.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 4800.0  # type: ignore[method-assign]

        job = make_job_pop_response("head_model")
        await track_popped_job_async(scheduler._job_tracker, job)
        admitted = scheduler._apply_ram_verdict(
            job,
            target,
            is_head_blocker=False,
            no_live_resource_consumer=True,
        )
        assert admitted is True
        assert 0 in scheduler.ram_reclaim.pending_reuse_credits
        assert scheduler.ram_reclaim.pending_reuse_credits[0].model == "head_model"

    async def test_credited_defer_escalates_to_cycle_of_a_different_stale_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When even the credited charge cannot fit, reclaim cycles a stale slot other than the target."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 16000.0)
        target = _retaining_target(0, rss_mb=4000.0)
        stale_other = _retaining_target(1, rss_mb=2000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target, 1: stale_other}))
        _pin_host_ram(scheduler)
        scheduler._measured_available_ram_mb = lambda: 5000.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 1024.0  # type: ignore[method-assign]
        scheduler.unload_models = Mock(return_value=False)  # type: ignore[method-assign]

        job = make_job_pop_response("head_model")
        await track_popped_job_async(scheduler._job_tracker, job)
        admitted = scheduler._apply_ram_verdict(
            job,
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
        _pin_host_ram(scheduler)
        assert scheduler._replace_stale_ram_unload_process(protect_process_id=0) is False
        assert _replace_mock(scheduler).called is False

    def test_unprotected_stale_slot_is_cycled(self) -> None:
        """Without protection the same retaining stale slot is cycled (the original last-resort behavior)."""
        target = _retaining_target(rss_mb=5000.0)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        _pin_host_ram(scheduler)
        assert scheduler._replace_stale_ram_unload_process() is True
        assert _replace_mock(scheduler).call_args.args[0] is target

    def test_creep_override_cycles_bloated_slot_even_with_model_and_protection(self) -> None:
        """A slot above the creep ceiling is cycled regardless of a resident model or protection."""
        bloated = make_mock_process_info(0, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
        bloated.ram_usage_bytes = CREEP_CONTAINMENT_RSS_BYTES + _MB
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: bloated}))
        scheduler.ram_reclaim.pending_reuse_credits[0] = ReuseCreditRecord("resident", 100.0, 100.0, time.time())

        assert scheduler._replace_stale_ram_unload_process(protect_process_id=0) is True
        replace = _replace_mock(scheduler)
        assert replace.call_args.args[0] is bloated
        assert replace.call_args.kwargs["intentional_reclaim"] is True
        # A cycled slot's pending credit is void (its successor cold-loads).
        assert 0 not in scheduler.ram_reclaim.pending_reuse_credits

    def test_creep_victim_preferred_over_stale_victim(self) -> None:
        """When both a stale slot and a crept slot exist, creep containment cycles the crept one first."""
        stale = _retaining_target(0, rss_mb=2000.0)
        bloated = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
        bloated.ram_usage_bytes = CREEP_CONTAINMENT_RSS_BYTES + _MB
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: stale, 1: bloated}))
        _pin_host_ram(scheduler)

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
        proc.load_completion_sample = LoadCompletionSample(
            process_launch_identifier=proc.process_launch_identifier,
            model=proc.loaded_horde_model_name,
            sampled_at=time.time(),
            private_bytes=proc.ram_usage_bytes,
            peak_private_bytes=proc.ram_usage_bytes,
        )
        scheduler._process_map = ProcessMap({0: proc})
        scheduler.ram_reclaim.pending_reuse_credits[0] = ReuseCreditRecord(
            model="m",
            private_at_admit_mb=admit_rss_mb,
            effective_charge_mb=charge_mb,
            admitted_at=time.time() - 1.0,
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
        assert 0 not in scheduler.ram_reclaim.pending_reuse_credits

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
        assert 0 not in scheduler.ram_reclaim.pending_reuse_credits


def _sdxl_pricing_scheduler() -> InferenceScheduler:
    """A scheduler pricing an SDXL checkpoint of 6500 MB with a dedicated post-processing lane."""
    scheduler = _make_inference_scheduler()
    scheduler._model_metadata.get_baseline = Mock(return_value="stable_diffusion_xl")
    checkpoint = Mock()
    checkpoint.stat.return_value.st_size = 6500 * _MB
    scheduler._resolve_checkpoint_path = Mock(return_value=checkpoint)
    scheduler._process_lifecycle.post_process_lane_enabled = Mock(return_value=True)
    return scheduler


_HEAVY_POST_PROCESSING = [KNOWN_UPSCALERS.RealESRGAN_x4plus.value, KNOWN_FACEFIXERS.CodeFormers.value]


def test_learned_checkpoint_price_keeps_each_jobs_feature_ram() -> None:
    """A price learned from plain loads still charges a feature-heavy job its own features, and the reverse."""
    scheduler = _sdxl_pricing_scheduler()
    for _ in range(5):
        scheduler.ram_reclaim.learned_ram.observe("sdxl", ReuseCreditKind.WHOLE, 6500)
    plain = make_job_pop_response("sdxl", width=1024, height=1024)
    heavy = make_job_pop_response("sdxl", width=1024, height=1024, post_processing=_HEAVY_POST_PROCESSING)
    plain_features = scheduler._job_feature_ram(plain)
    heavy_features = scheduler._job_feature_ram(heavy)
    assert heavy_features.post_processing_mb > 0
    assert scheduler._whole_job_ram_charge_mb(plain) == pytest.approx(7150 + plain_features.total_mb)
    assert scheduler._whole_job_ram_charge_mb(heavy) == pytest.approx(7150 + heavy_features.total_mb)
    # The admitted charge's checkpoint share is what load-completion growth is checked against and learned
    # beside, so feature-heavy admissions cannot raise a plain job's price.
    verdict = BudgetVerdict(fits=True, predicted_mb=6500 + heavy_features.total_mb, available_mb=0.0, reserve_mb=0.0)
    assert scheduler._checkpoint_part_of_charge_mb(heavy, verdict) == pytest.approx(6500)


def test_an_unseen_checkpoint_is_priced_from_its_baselines_loads() -> None:
    """Cold loads of three SDXL checkpoints price a fourth before it has loaded once."""
    scheduler = _sdxl_pricing_scheduler()
    for model in ("sdxl_a", "sdxl_b", "sdxl_c", "sdxl_a", "sdxl_b"):
        scheduler.ram_reclaim.learned_ram.observe(
            model, ReuseCreditKind.WHOLE, 8000, baseline="stable_diffusion_xl", size_mb=6500
        )
    job = make_job_pop_response("sdxl_d", width=1024, height=1024)
    assert scheduler._checkpoint_staging_charge_mb(job) == pytest.approx(8000 * 1.1)


def test_without_a_lane_post_processing_ram_lands_on_inference() -> None:
    """With no dedicated lane the inference process allocates the post-processors' RAM too."""
    scheduler = _sdxl_pricing_scheduler()
    scheduler._process_lifecycle.post_process_lane_enabled = Mock(return_value=False)
    heavy = make_job_pop_response("sdxl", width=1024, height=1024, post_processing=_HEAVY_POST_PROCESSING)
    features = scheduler._job_feature_ram(heavy)
    assert features.post_processing_mb == 0
    assert features.sampling_mb > 0


def test_checkpoint_charge_uses_file_metadata_and_trusted_growth_in_both_directions() -> None:
    """Cold-context seeds do not set swap prices; measured load peaks can lower and raise them."""
    scheduler = _sdxl_pricing_scheduler()
    job = make_job_pop_response("sdxl", width=1024, height=1024)
    static = scheduler._checkpoint_staging_charge_mb(job)
    assert static == pytest.approx(6500)
    for _ in range(5):
        scheduler.ram_reclaim.learned_ram.observe("sdxl", ReuseCreditKind.WHOLE, 5000)
    lower = scheduler._checkpoint_staging_charge_mb(job)
    assert lower == pytest.approx(5500)
    assert lower < static
    scheduler.ram_reclaim.learned_ram.observe("sdxl", ReuseCreditKind.WHOLE, 10000)
    assert scheduler._checkpoint_staging_charge_mb(job) == 11000
