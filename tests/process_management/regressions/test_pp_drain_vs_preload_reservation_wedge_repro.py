"""Drain-vs-reservation deadlock: post-processing priced against a RAM-staged head's planned charge.

The circular wait this reproduces: a head model is preloaded into system RAM and records a preload-flow
planned charge; its VRAM materialisation is held by the dispatch reconciliation gate until the card has room;
the room only frees when the previous job's post-processing completes; but the PP admission was priced against
the head's planned charge, so PP defers, ages out at its patience window, and the finished job is faulted
without images while the recovery supervisor escalates through soft resets to the queue-deadlock give-up.

The contract under test: a PP_JOB request is priced against physical truth (device-free minus noise) plus the
dispatch-flow reservations only. Preload-flow planned charges are bookkeeping for loads that cannot claim the
card before this very drain completes (their dispatch gate re-prices them against fresh measured truth), so
they must not withhold the drain. Dispatch-flow reservations (in-flight sampling about to spike) still do.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.scheduling.workload_flow import (
    DISPATCH_ADMISSION_FLOW,
    PRELOAD_ADMISSION_FLOW,
)

_TOTAL_MB = 16376.0
_NOISE_MB = admission_noise_buffer_mb(_TOTAL_MB)


def _arbiter_for(
    *,
    device_free_mb: float,
    planned_mb: float,
    preload_planned_mb: float,
) -> VramArbiter:
    """An arbiter frozen on one card with the given overlay split (mirrors the production snapshot fields)."""
    state = DeviceVramState(
        total_vram_mb=_TOTAL_MB,
        baseline_mb=1400.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=planned_mb,
        committed_is_stale=False,
        preload_planned_unmaterialized_mb=preload_planned_mb,
        noise_buffer_mb=_NOISE_MB,
        device_free_mb=device_free_mb,
    )
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter


def _pp_request(candidate_mb: float) -> VramRequest:
    return VramRequest(
        kind=VramRequestKind.PP_JOB,
        job_label="post_process:job",
        baseline=None,
        device_index=0,
        target_process_id=1,
        candidate_delta_mb=candidate_mb,
        is_head_of_queue=True,
    )


class TestPostProcessDrainIsNotHeldByPreloadPlans:
    """The drain admits against physical room; only dispatch-flow reservations may withhold it."""

    def test_pp_admits_over_a_ram_staged_heads_planned_charge(self) -> None:
        """The production arithmetic: candidate 3590 vs device-free 7131 with a 4208 MB preload-flow charge.

        Physically the chain fits (7131 - noise leaves well over 3590); only the staged head's bookkeeping
        made it defer, age out at the patience window, and fault the finished job without images.
        """
        arbiter = _arbiter_for(device_free_mb=7131.0, planned_mb=4208.0, preload_planned_mb=4208.0)
        verdict = arbiter.evaluate(_pp_request(3590.0))
        assert verdict.disposition is VramDisposition.FITS

    def test_pp_is_still_held_by_dispatch_flow_reservations(self) -> None:
        """An in-flight sampling reservation is a real imminent spike: it still withholds the drain."""
        arbiter = _arbiter_for(device_free_mb=7131.0, planned_mb=4208.0, preload_planned_mb=0.0)
        verdict = arbiter.evaluate(_pp_request(3590.0))
        assert verdict.disposition is VramDisposition.DEFER

    def test_pp_is_still_held_when_the_card_genuinely_has_no_room(self) -> None:
        """Excluding preload plans never fabricates room: a physically full card still defers the drain."""
        arbiter = _arbiter_for(device_free_mb=2500.0, planned_mb=4208.0, preload_planned_mb=4208.0)
        verdict = arbiter.evaluate(_pp_request(3590.0))
        assert verdict.disposition is VramDisposition.DEFER

    def test_a_preload_is_still_charged_the_full_overlay(self) -> None:
        """The exemption is drain-only: another unit's preload charge still defers a competing preload."""
        arbiter = _arbiter_for(device_free_mb=7131.0, planned_mb=4208.0, preload_planned_mb=4208.0)
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.PRELOAD,
                job_label="competing_model",
                baseline=None,
                device_index=0,
                target_process_id=5,
                candidate_delta_mb=3590.0,
                is_head_of_queue=False,
            ),
        )
        assert verdict.disposition is VramDisposition.DEFER


class TestSnapshotCarriesThePreloadShare:
    """The scheduler's frozen device state carries the preload flow's share alongside the combined overlay."""

    def test_build_device_state_reports_the_preload_flow_share(self) -> None:
        """A live in-flight preload charge surfaces in both the combined overlay and its preload-flow share."""
        from horde_worker_regen.process_management.ipc.messages import (
            HordeProcessState,
            ModelInfo,
            ModelLoadState,
        )
        from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
        from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
        from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
        from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

        loading = make_mock_process_info(4, model_name="staged_head", state=HordeProcessState.PRELOADING_MODEL)
        loading.process_reserved_mb = 0
        model_map = HordeModelMap(
            root={
                "staged_head": ModelInfo(
                    horde_model_name="staged_head",
                    horde_model_load_state=ModelLoadState.LOADING,
                    process_id=4,
                ),
            },
        )
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({4: loading}),
            horde_model_map=model_map,
            bridge_data=make_mock_bridge_data(image_models_to_load=["staged_head"]),
        )
        scheduler._reserve_ledger.set_planned(
            PRELOAD_ADMISSION_FLOW,
            "4",
            vram_mb=4208.0,
            target_process_id=4,
            reserved_at_admit_mb=0.0,
        )

        state = scheduler.build_vram_arbiter_device_state(None)

        assert state.preload_planned_unmaterialized_mb == 4208.0
        assert state.planned_unmaterialized_mb == 4208.0


class TestLedgerPerFlowPlannedShare:
    """The per-flow accessor reports only its flow's outstanding charges and never ratchets watermarks."""

    def test_sums_only_the_requested_flow_and_decays_by_materialisation(self) -> None:
        """The preload-flow share excludes other flows' charges and decays with the target's reservation."""
        ledger = CommittedReserveLedger()
        ledger.set_planned(
            PRELOAD_ADMISSION_FLOW,
            "4",
            vram_mb=4208.0,
            target_process_id=4,
            reserved_at_admit_mb=100.0,
        )
        ledger.set_planned(
            DISPATCH_ADMISSION_FLOW,
            "job-a",
            vram_mb=6000.0,
            target_process_id=5,
            reserved_at_admit_mb=0.0,
        )
        # Nothing materialised: the preload flow reports its full charge, the dispatch flow's is excluded.
        assert ledger.effective_planned_vram_mb_for_flow(PRELOAD_ADMISSION_FLOW, {}) == 4208.0
        # The target's reservation grew 1000 MB past admit: the outstanding share decays one-for-one.
        assert ledger.effective_planned_vram_mb_for_flow(PRELOAD_ADMISSION_FLOW, {4: 1100.0}) == 3208.0

    def test_read_is_side_effect_free_on_the_materialisation_watermark(self) -> None:
        """The per-cycle total accessor owns the ratchet; the per-flow read must not advance it."""
        ledger = CommittedReserveLedger()
        ledger.set_planned(
            PRELOAD_ADMISSION_FLOW,
            "4",
            vram_mb=4208.0,
            target_process_id=4,
            reserved_at_admit_mb=0.0,
        )
        assert ledger.effective_planned_vram_mb_for_flow(PRELOAD_ADMISSION_FLOW, {4: 4208.0}) == 0.0
        # The transient growth above was not persisted: a later read against a lower reservation still
        # reports the full outstanding charge (only effective_planned_vram_mb may ratchet it down).
        assert ledger.effective_planned_vram_mb_for_flow(PRELOAD_ADMISSION_FLOW, {4: 0.0}) == 4208.0
