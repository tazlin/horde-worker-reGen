"""Head-vs-staged-sibling deadlock: the head of queue priced against a queued job's staged preload plan.

The circular wait this reproduces: a job behind the head is preloaded into system RAM and records a
preload-flow planned charge; the head's dispatch (or preload) is then priced against that charge and defers,
though the card physically has the room; the staged sibling's VRAM claim can only materialise when its own
dispatch is admitted, which cannot precede the head's; nothing clears the standoff but the recovery
supervisor's soft resets and, past its patience, the queue-deadlock give-up that faults the backlog.

The contract under test: a true head-of-queue request (PRELOAD or MONOLITHIC_DISPATCH) is priced against
physical truth (device-free minus noise) plus the dispatch-flow reservations only. Preload-flow planned
charges are bookkeeping for loads that are queued behind the head by definition, so they must not withhold
it. Dispatch-flow reservations (in-flight sampling about to spike) still do, and a non-head request (a
line-skip) stays fully charged so it cannot consume a staged head's room.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
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


def _dispatch_request(candidate_mb: float, *, is_head_of_queue: bool) -> VramRequest:
    return VramRequest(
        kind=VramRequestKind.MONOLITHIC_DISPATCH,
        job_label="head_model",
        baseline=None,
        device_index=0,
        target_process_id=3,
        candidate_delta_mb=candidate_mb,
        is_head_of_queue=is_head_of_queue,
    )


class TestHeadDispatchIsNotHeldByStagedSiblingPlans:
    """The head's dispatch admits against physical room; only dispatch-flow reservations may withhold it."""

    def test_head_dispatch_admits_over_a_queued_siblings_staged_charge(self) -> None:
        """The production arithmetic: candidate 11144 vs device-free 12226 with a 4000 MB preload-flow charge.

        Physically the head fits (12226 minus noise leaves over 11144); only a queued sibling's staged
        bookkeeping made it defer until the recovery supervisor faulted the backlog.
        """
        arbiter = _arbiter_for(device_free_mb=12226.0, planned_mb=4000.0, preload_planned_mb=4000.0)
        verdict = arbiter.evaluate(_dispatch_request(11144.0, is_head_of_queue=True))
        assert verdict.disposition is VramDisposition.FITS

    def test_mutual_standoff_between_two_staged_jobs_releases_the_head(self) -> None:
        """The two-staged-jobs standoff: the head admits over the sibling's plan, the sibling stays held.

        The production arithmetic: head candidate 6346 deferred on the sibling's 5356 charge while the
        sibling's 5356 deferred on the head's 6346, with device-free 11216 seating either. The head must
        win; the non-head stays charged so it cannot take the head's room.
        """
        head_over_siblings_plan = _arbiter_for(
            device_free_mb=11216.0,
            planned_mb=5356.0,
            preload_planned_mb=5356.0,
        )
        head_verdict = head_over_siblings_plan.evaluate(_dispatch_request(6346.0, is_head_of_queue=True))
        assert head_verdict.disposition is VramDisposition.FITS

        sibling_over_heads_plan = _arbiter_for(
            device_free_mb=11216.0,
            planned_mb=6346.0,
            preload_planned_mb=6346.0,
        )
        sibling_verdict = sibling_over_heads_plan.evaluate(_dispatch_request(5356.0, is_head_of_queue=False))
        assert sibling_verdict.disposition is VramDisposition.DEFER

    def test_head_dispatch_is_still_held_by_dispatch_flow_reservations(self) -> None:
        """An in-flight sampling reservation is a real imminent spike: it still withholds the head."""
        arbiter = _arbiter_for(device_free_mb=12226.0, planned_mb=4000.0, preload_planned_mb=0.0)
        verdict = arbiter.evaluate(_dispatch_request(11144.0, is_head_of_queue=True))
        assert verdict.disposition is VramDisposition.DEFER

    def test_head_dispatch_is_still_held_when_the_card_genuinely_has_no_room(self) -> None:
        """Excluding staged plans never fabricates room: a physically full card still defers the head."""
        arbiter = _arbiter_for(device_free_mb=6480.0, planned_mb=4000.0, preload_planned_mb=4000.0)
        verdict = arbiter.evaluate(_dispatch_request(11144.0, is_head_of_queue=True))
        assert verdict.disposition is VramDisposition.DEFER


class TestHeadPreloadIsNotHeldByStagedSiblingPlans:
    """The head's preload gets the same exemption: staged siblings materialise strictly after the head."""

    def test_head_preload_admits_over_a_queued_siblings_staged_charge(self) -> None:
        """A head preload is not withheld by a queued sibling's staged plan when the card physically fits it."""
        arbiter = _arbiter_for(device_free_mb=12226.0, planned_mb=5130.0, preload_planned_mb=5130.0)
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.PRELOAD,
                job_label="head_model",
                baseline=None,
                device_index=0,
                target_process_id=3,
                candidate_delta_mb=11144.0,
                is_head_of_queue=True,
            ),
        )
        assert verdict.disposition is VramDisposition.FITS

    def test_non_head_preload_is_still_charged_the_full_overlay(self) -> None:
        """A competing non-head preload stays charged so it cannot stage into a staged head's room."""
        arbiter = _arbiter_for(device_free_mb=12226.0, planned_mb=5130.0, preload_planned_mb=5130.0)
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.PRELOAD,
                job_label="competing_model",
                baseline=None,
                device_index=0,
                target_process_id=5,
                candidate_delta_mb=11144.0,
                is_head_of_queue=False,
            ),
        )
        assert verdict.disposition is VramDisposition.DEFER

    def test_netting_the_share_does_not_double_subtract_the_heads_own_charge(self) -> None:
        """The head's own staged charge lives inside the netted share; dispatch-flow charges stay intact.

        With the head's own 6000 MB plan netted via the share, a further per-target own subtraction would
        eat into the sibling's 5000 MB dispatch-flow reservation. The reservation must survive in full: the
        head still defers when the physical room minus that reservation cannot seat it.
        """
        arbiter = _arbiter_for(device_free_mb=12226.0, planned_mb=11000.0, preload_planned_mb=6000.0)
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.MONOLITHIC_DISPATCH,
                job_label="head_model",
                baseline=None,
                device_index=0,
                target_process_id=3,
                candidate_delta_mb=8000.0,
                own_planned_unmaterialized_mb=6000.0,
                is_head_of_queue=True,
            ),
        )
        assert verdict.disposition is VramDisposition.DEFER
        assert verdict.measured.outstanding_reservations_mb == 5000.0
