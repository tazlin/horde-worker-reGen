"""UNet-only RAM component charging for disaggregation-class jobs at the scheduler seam.

The RAM-verdict path (:meth:`InferenceScheduler._apply_ram_verdict`) prices a disaggregation-class preload at
its UNet-only component charge (read torch-free from the checkpoint's component-identity sidecar) instead of the
whole checkpoint, credits the charge to zero when the checkpoint is already staged on the target, falls back to
the whole-checkpoint charge when no sidecar is available, and reconciles the component charge against measured
RSS growth. A job the whole-checkpoint charge would defer is dispatched at the UNet-only price.

Contracts asserted here:
- the component charge is the sidecar UNet residual for a disaggregation-class job, None (whole) for a
  monolithic or declined job, and None (whole) when no sidecar resolves;
- a checkpoint already staged on the target credits the charge to zero;
- a job deferred under whole-checkpoint pricing dispatches under UNet-only pricing (positive liveness);
- the measured-truth reconciliation flags a component charge whose target grew past it and stays quiet within
  slack, distinguishing the component wording from the page-reuse wording.

These tests are authored to run in fake mode with no GPU. Run with ``AI_HORDE_TESTING=True pytest``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot, HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.component_residency_map import ComponentResidencyMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import _COMPONENT_STAGING_CHARGE_FLOOR_MB
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _REUSE_CREDIT_KIND_COMPONENT,
    _REUSE_CREDIT_KIND_PAGE_REUSE,
    _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS,
    InferenceScheduler,
    _ReuseCreditRecord,
)
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MB = 1024 * 1024
_WHOLE_MB = 16000.0
# An SDXL-ish UNet residual, comfortably above the component floor and below the whole-checkpoint charge.
_RESIDUAL_MB = 6000.0


@dataclass(frozen=True)
class _FakeSidecar:
    """A minimal stand-in for ComponentIdentitySidecar exposing only the residual the charge is grounded on."""

    residual_tensor_bytes: int


def _idle_target(process_id: int = 0) -> HordeProcessInfo:
    """An idle, model-less inference slot eligible to receive a preload."""
    return make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)


def _retaining_target(process_id: int = 0, *, rss_mb: float) -> HordeProcessInfo:
    """An idle, model-less inference slot that kept ``rss_mb`` of reusable pages after unloading its model."""
    target = _idle_target(process_id)
    target.ram_usage_bytes = int(rss_mb * _MB)
    return target


def _mark_disaggregation_class(scheduler: InferenceScheduler) -> None:
    """Make every job read as disaggregation-class (the stable predicate the VRAM side also charges against)."""
    scheduler._is_disaggregation_class_eligible = lambda _job: True  # type: ignore[method-assign]


def _pin_sidecar(scheduler: InferenceScheduler, sidecar: _FakeSidecar | None) -> None:
    """Pin the torch-free sidecar read so the charge arithmetic is asserted without the filesystem."""
    scheduler._read_component_sidecar = lambda _model: sidecar  # type: ignore[assignment, method-assign, return-value]


class TestComponentChargeSelection:
    """The charge is UNet-only for a disaggregation-class job, whole for monolithic/declined/missing sidecar."""

    def test_disaggregation_class_charges_unet_residual(self) -> None:
        """A disaggregation-class job with a sidecar is charged the floored UNet residual."""
        scheduler = _make_inference_scheduler()
        _mark_disaggregation_class(scheduler)
        _pin_sidecar(scheduler, _FakeSidecar(int(_RESIDUAL_MB * _MB)))
        job = make_job_pop_response("disagg_model")
        assert scheduler._disaggregated_component_charge_mb(job, _idle_target()) == pytest.approx(_RESIDUAL_MB)

    def test_monolithic_job_charges_whole(self) -> None:
        """A non-disaggregation-class job returns None so the whole-checkpoint charge stands."""
        scheduler = _make_inference_scheduler()  # default predicate is False
        _pin_sidecar(scheduler, _FakeSidecar(int(_RESIDUAL_MB * _MB)))
        job = make_job_pop_response("mono_model")
        assert scheduler._disaggregated_component_charge_mb(job, _idle_target()) is None

    def test_missing_sidecar_charges_whole(self) -> None:
        """A disaggregation-class job with no resolvable sidecar returns None (whole-checkpoint fallback)."""
        scheduler = _make_inference_scheduler()
        _mark_disaggregation_class(scheduler)
        _pin_sidecar(scheduler, None)
        job = make_job_pop_response("disagg_model")
        assert scheduler._disaggregated_component_charge_mb(job, _idle_target()) is None

    def test_small_residual_is_floored(self) -> None:
        """A degenerate residual is charged the component floor, never near zero."""
        scheduler = _make_inference_scheduler()
        _mark_disaggregation_class(scheduler)
        _pin_sidecar(scheduler, _FakeSidecar(1))
        job = make_job_pop_response("disagg_model")
        assert scheduler._disaggregated_component_charge_mb(job, _idle_target()) == pytest.approx(
            _COMPONENT_STAGING_CHARGE_FLOOR_MB,
        )


class TestResidencyCredit:
    """A checkpoint already staged on the target credits the component charge to zero; absent it is full."""

    def _scheduler_with_residency(self, held_model: str | None) -> InferenceScheduler:
        """A disaggregation-class scheduler whose residency map holds ``held_model`` staged on process 0."""
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: _idle_target()}))
        _mark_disaggregation_class(scheduler)
        _pin_sidecar(scheduler, _FakeSidecar(int(_RESIDUAL_MB * _MB)))
        residency = ComponentResidencyMap()
        held = (
            [HeldComponentSnapshot(kind="checkpoint", identity=held_model, approx_ram_mb=_RESIDUAL_MB)]
            if held_model is not None
            else []
        )
        residency.update_from_report(process_id=0, launch_identifier=0, held=held)
        scheduler._component_residency_map = residency
        return scheduler

    def test_held_checkpoint_credits_charge_to_zero(self) -> None:
        """When the target already holds the checkpoint staged, the stage materialises nothing (charge 0)."""
        scheduler = self._scheduler_with_residency("disagg_model")
        job = make_job_pop_response("disagg_model")
        target = scheduler._process_map.get(0)
        assert target is not None
        assert scheduler._disaggregated_component_charge_mb(job, target) == pytest.approx(0.0)

    def test_absent_checkpoint_charges_full_residual(self) -> None:
        """When the target holds a different checkpoint, the full UNet residual is charged."""
        scheduler = self._scheduler_with_residency("other_model")
        job = make_job_pop_response("disagg_model")
        target = scheduler._process_map.get(0)
        assert target is not None
        assert scheduler._disaggregated_component_charge_mb(job, target) == pytest.approx(_RESIDUAL_MB)


class TestDeclinedJobRepricesWhole:
    """A job re-routed to monolithic (declined) is priced whole-checkpoint even with a sidecar present."""

    def test_declined_job_charges_whole(self) -> None:
        """A predicate that reads the declined state as not-disaggregation-class yields the whole charge (None).

        The manager's ``_disaggregation_class_eligible`` returns False once a job is disaggregation-declined
        (pinned in ``test_disaggregation_integration``), so a re-routed job re-enters ``_apply_ram_verdict`` and
        is priced whole-checkpoint: the component seam is bypassed at the class predicate, not by a separate
        code path.
        """
        scheduler = _make_inference_scheduler()
        declined = {"disagg_model"}
        scheduler._is_disaggregation_class_eligible = lambda job: job.model not in declined  # type: ignore[method-assign]
        _pin_sidecar(scheduler, _FakeSidecar(int(_RESIDUAL_MB * _MB)))
        job = make_job_pop_response("disagg_model")
        assert scheduler._disaggregated_component_charge_mb(job, _idle_target()) is None


class TestMissingSidecarParity:
    """A disaggregation-class job with no sidecar admits exactly as the pre-feature reuse-credit path."""

    def test_no_sidecar_matches_prefeature_reuse_credit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sidecar-less disaggregation-class job is priced identically to the monolithic reuse-credit path.

        A model that can never carry a sidecar (a ``.ckpt`` pickle) must not be admitted more strictly than
        before component charging existed, or it would re-introduce head-of-queue RAM-defer starvation. Both
        arms fall through to the ordinary path with the retained-page reuse credit; the effective charge and the
        pending record's kind must match.
        """
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: _WHOLE_MB)

        def _run(*, disaggregation_class: bool) -> tuple[bool, _ReuseCreditRecord | None]:
            target = _retaining_target(rss_mb=8000.0)
            scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
            scheduler._measured_available_ram_mb = lambda: 17946.0  # type: ignore[method-assign]
            scheduler._ram_danger_floor_mb = lambda: 4800.0  # type: ignore[method-assign]
            if disaggregation_class:
                _mark_disaggregation_class(scheduler)
            _pin_sidecar(scheduler, None)  # no sidecar in either arm
            job = make_job_pop_response("no_sidecar_model")
            admitted = scheduler._apply_ram_verdict(
                job,
                "x",
                target,
                is_head_blocker=False,
                no_live_resource_consumer=True,
            )
            return admitted, scheduler._pending_reuse_credits.get(0)

        disagg_admitted, disagg_record = _run(disaggregation_class=True)
        mono_admitted, mono_record = _run(disaggregation_class=False)

        # The reuse credit admits both (retained 6900 MB -> credit 4830 MB -> effective 11170 MB fits 17946).
        assert disagg_admitted is True
        assert mono_admitted is True
        assert disagg_record is not None
        assert mono_record is not None
        assert disagg_record.effective_charge_mb == pytest.approx(mono_record.effective_charge_mb)
        assert disagg_record.effective_charge_mb == pytest.approx(11170.0)
        assert disagg_record.kind == mono_record.kind == _REUSE_CREDIT_KIND_PAGE_REUSE


class TestComponentAdmissionLiveness:
    """A job the whole-checkpoint charge defers is dispatched under UNet-only pricing."""

    def test_unet_only_pricing_dispatches_what_whole_pricing_defers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The UNet-only charge admits a preload the whole-checkpoint charge would defer, and records it."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: _WHOLE_MB)
        target = _idle_target()
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: target}))
        scheduler._measured_available_ram_mb = lambda: 17946.0  # type: ignore[method-assign]
        scheduler._ram_danger_floor_mb = lambda: 1024.0  # type: ignore[method-assign]
        _mark_disaggregation_class(scheduler)
        _pin_sidecar(scheduler, _FakeSidecar(int(_RESIDUAL_MB * _MB)))

        job = make_job_pop_response("disagg_model")
        # Premise: the whole-checkpoint charge (16000 + 4096 reserve) does not fit 17946 and would defer.
        whole_verdict = scheduler._ram_budget.check_job(job, "x", 17946.0)
        assert whole_verdict.fits is False

        admitted = scheduler._apply_ram_verdict(
            job,
            "x",
            target,
            is_head_blocker=False,
            no_live_resource_consumer=True,
        )
        assert admitted is True
        assert 0 in scheduler._pending_reuse_credits
        record = scheduler._pending_reuse_credits[0]
        assert record.model == "disagg_model"
        assert record.kind == _REUSE_CREDIT_KIND_COMPONENT
        assert record.effective_charge_mb == pytest.approx(_RESIDUAL_MB)


class TestComponentChargeReconciliation:
    """The measured-truth check flags an under-priced component charge and stays quiet within slack."""

    def _settled_component_target(
        self,
        scheduler: InferenceScheduler,
        *,
        admit_rss_mb: float,
        now_rss_mb: float,
        charge_mb: float,
    ) -> None:
        """Seat a settled component-kind credited target on ``scheduler`` whose RSS grew since admit time."""
        proc = make_mock_process_info(0, model_name="disagg_model", state=HordeProcessState.WAITING_FOR_JOB)
        proc.ram_usage_bytes = int(now_rss_mb * _MB)
        scheduler._process_map = ProcessMap({0: proc})
        scheduler._pending_reuse_credits[0] = _ReuseCreditRecord(
            model="disagg_model",
            rss_at_admit_mb=admit_rss_mb,
            effective_charge_mb=charge_mb,
            admitted_at=time.time() - _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS - 1.0,
            kind=_REUSE_CREDIT_KIND_COMPONENT,
        )

    def test_under_priced_component_charge_flagged_once(self) -> None:
        """Growth exceeding the component charge by more than the slack logs the UNet-only wording and clears."""
        scheduler = _make_inference_scheduler()
        # growth = 9049 MB against charge 6000 MB; 9049 > 6000 + 2048 slack -> flagged.
        self._settled_component_target(scheduler, admit_rss_mb=2000.0, now_rss_mb=11049.0, charge_mb=6000.0)
        messages: list[object] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            scheduler._reconcile_reuse_credit()
        finally:
            logger.remove(sink_id)
        assert any("under-priced the stage" in str(record) for record in messages)
        assert not any("too generous" in str(record) for record in messages)
        assert 0 not in scheduler._pending_reuse_credits

    def test_component_charge_within_slack_is_silent(self) -> None:
        """Growth within the component charge plus slack (expected mmap sharing) clears the record silently."""
        scheduler = _make_inference_scheduler()
        # growth = 6000 MB against charge 6000 MB; 6000 < 6000 + 2048 slack -> silent.
        self._settled_component_target(scheduler, admit_rss_mb=2000.0, now_rss_mb=8000.0, charge_mb=6000.0)
        messages: list[object] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            scheduler._reconcile_reuse_credit()
        finally:
            logger.remove(sink_id)
        assert not any("under-priced the stage" in str(record) for record in messages)
        assert 0 not in scheduler._pending_reuse_credits

    def test_vanished_component_slot_is_dropped(self) -> None:
        """A component record whose slot has vanished is dropped without a warning."""
        scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
        scheduler._pending_reuse_credits[0] = _ReuseCreditRecord(
            model="disagg_model",
            rss_at_admit_mb=2000.0,
            effective_charge_mb=6000.0,
            admitted_at=time.time(),
            kind=_REUSE_CREDIT_KIND_COMPONENT,
        )
        scheduler._reconcile_reuse_credit()
        assert 0 not in scheduler._pending_reuse_credits

    def test_page_reuse_record_keeps_its_wording(self) -> None:
        """A page-reuse record (the default kind) still logs the too-generous wording, not the component one."""
        scheduler = _make_inference_scheduler()
        proc = make_mock_process_info(0, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
        proc.ram_usage_bytes = int(9000.0 * _MB)
        scheduler._process_map = ProcessMap({0: proc})
        scheduler._pending_reuse_credits[0] = _ReuseCreditRecord(
            model="m",
            rss_at_admit_mb=2000.0,
            effective_charge_mb=1000.0,
            admitted_at=time.time() - _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS - 1.0,
            kind=_REUSE_CREDIT_KIND_PAGE_REUSE,
        )
        messages: list[object] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            scheduler._reconcile_reuse_credit()
        finally:
            logger.remove(sink_id)
        assert any("too generous" in str(record) for record in messages)
        assert not any("under-priced the stage" in str(record) for record in messages)
