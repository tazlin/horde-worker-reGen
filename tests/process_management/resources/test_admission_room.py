"""The measured admission room decomposed by tenant and reclaim rung.

The admission identity refuses on one number; the room breakdown says who holds the card and which rung
could return enough of it. These tests pin the arithmetic against a card at its edge: a whole-card
candidate, two inference contexts, a post-processing lane, a utilities lane, and a foreign share.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.admission_identity import (
    AdmissionRoom,
    RoomRungKind,
    TenantLane,
    admission_room,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)

_TOTAL_MB = 24074.0
_CONTEXT_MB = 487.0
_NOISE_MB = 1203.7


def _edge_room(
    *,
    candidate_mb: float = 19424.0,
    post_process_permitted: bool = True,
    utilities_permitted: bool = False,
    safety_permitted: bool = True,
) -> AdmissionRoom:
    """A 24 GB card holding two inference contexts, a post-processing lane, a utilities lane and safety.

    The target slot (pid 2) and its idle sibling (pid 3) each report a small allocator reservation; the
    lanes report theirs; safety reports nothing and is priced at its footprint. The device-free reading is
    set so the foreign share comes out to a round figure.
    """
    reserved = {2: 100.0, 3: 100.0, 1: 808.0, 4: 554.0}
    lanes = {
        2: TenantLane.INFERENCE_IDLE,
        3: TenantLane.INFERENCE_IDLE,
        1: TenantLane.POST_PROCESS,
        4: TenantLane.UTILITIES,
        0: TenantLane.SAFETY,
    }
    attributed = (100 + _CONTEXT_MB) * 2 + (808 + _CONTEXT_MB) + (554 + _CONTEXT_MB) + 3044.0
    device_free = _TOTAL_MB - attributed - 500.0
    return admission_room(
        candidate_mb=candidate_mb,
        device_free_mb=device_free,
        total_vram_mb=_TOTAL_MB,
        outstanding_reservations_mb=0.0,
        noise_buffer_mb=_NOISE_MB,
        per_process_reserved_mb=reserved,
        lane_by_process_id=lanes,
        target_process_id=2,
        context_mb=_CONTEXT_MB,
        safety_footprint_mb=3044.0,
        post_process_reclaim_permitted=post_process_permitted,
        safety_reclaim_permitted=safety_permitted,
        utilities_reclaim_permitted=utilities_permitted,
    )


def test_tenancy_attributes_every_live_process_and_the_foreign_remainder() -> None:
    """Each live process is charged its reservation plus a context; safety at its footprint; the rest foreign."""
    room = _edge_room()

    assert room.tenancy_mb[TenantLane.INFERENCE_TARGET] == 100 + _CONTEXT_MB
    assert room.tenancy_mb[TenantLane.INFERENCE_IDLE] == 100 + _CONTEXT_MB
    assert room.tenancy_mb[TenantLane.POST_PROCESS] == 808 + _CONTEXT_MB
    assert room.tenancy_mb[TenantLane.UTILITIES] == 554 + _CONTEXT_MB
    assert room.tenancy_mb[TenantLane.SAFETY] == 3044.0
    assert room.tenancy_mb[TenantLane.FOREIGN] == 500.0


def test_deficit_is_the_identity_deficit() -> None:
    """The room's deficit is candidate minus (device_free - reservations - noise), the verdict's own figure."""
    room = _edge_room()

    assert room.available_mb == room.device_free_mb - _NOISE_MB
    assert room.deficit_mb == 19424.0 - room.available_mb


def test_rungs_are_cheapest_first_and_the_target_slot_is_never_a_rung() -> None:
    """One idle-sibling rung (never the target), then the post-processing lane, safety, utilities."""
    room = _edge_room()

    kinds = [rung.kind for rung in room.rungs]
    assert kinds == [
        RoomRungKind.IDLE_SIBLING_CONTEXT,
        RoomRungKind.POST_PROCESS_LANE,
        RoomRungKind.SAFETY_OFF_GPU,
        RoomRungKind.UTILITIES_LANE,
    ]
    assert room.rungs[0].promised_mb == 100 + _CONTEXT_MB
    assert room.rungs[3].permitted is False


def test_closable_counts_only_permitted_rungs_and_unpausable_the_rest() -> None:
    """A forbidden rung's tenancy is unpausable; the deficit is closable only by what policy permits."""
    room = _edge_room()
    permitted_total = (100 + _CONTEXT_MB) + (808 + _CONTEXT_MB) + 3044.0

    assert room.reclaimable_mb == permitted_total
    assert room.unpausable_mb == 500.0 + (554 + _CONTEXT_MB)
    assert room.closable is (room.deficit_mb <= permitted_total)

    forbidden = _edge_room(post_process_permitted=False, safety_permitted=False)
    assert forbidden.reclaimable_mb == 100 + _CONTEXT_MB
    assert forbidden.rungs_to_close() == (forbidden.rungs[0],)


def test_rungs_to_close_stops_at_the_first_cover() -> None:
    """The cheapest permitted rungs are taken until their sum covers the deficit."""
    room = _edge_room(candidate_mb=1.0)
    assert room.deficit_mb < 0
    assert room.rungs_to_close() == (room.rungs[0],)

    short_by_a_sibling = _edge_room()
    covering = short_by_a_sibling.rungs_to_close()
    covered = sum(rung.promised_mb for rung in covering)
    assert covered >= short_by_a_sibling.deficit_mb
    assert len(covering) >= 1


def test_describe_and_inputs_name_the_deficit_the_tenancy_and_the_rungs() -> None:
    """The log line and the record carry the same figures a reader needs to act on a hold."""
    room = _edge_room()

    line = room.describe()
    assert f"short {room.deficit_mb:.0f} MB" in line
    assert "inference_idle" in line and "utilities" in line and "(not permitted)" in line

    inputs = room.as_inputs()
    assert inputs["room_deficit_mb"] == round(room.deficit_mb, 1)
    assert inputs["rung_utilities_lane_permitted"] is False
    assert inputs["tenancy_foreign_mb"] == 500.0


def test_a_fitting_candidate_reports_no_deficit() -> None:
    """A candidate inside the room reads as fitting, and no rung is needed to close anything."""
    room = _edge_room(candidate_mb=1000.0)

    assert room.deficit_mb < 0
    assert room.closable is False
    assert room.describe().startswith("fits by")


def test_arbiter_attaches_the_room_to_a_non_fitting_verdict() -> None:
    """A DEFER on a card with a device-free reading carries the room; an admit carries none."""
    state = DeviceVramState(
        total_vram_mb=_TOTAL_MB,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=20157.0,
        noise_buffer_mb=_NOISE_MB,
        per_process_reserved_mb={2: 100.0, 3: 100.0},
        lane_by_process_id={2: TenantLane.INFERENCE_IDLE, 3: TenantLane.INFERENCE_IDLE},
        marginal_mb=_CONTEXT_MB,
    )
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))

    held = arbiter.evaluate(
        VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="Z-Image-Turbo",
            baseline="z_image_turbo",
            device_index=0,
            target_process_id=2,
            candidate_delta_mb=19424.0,
            is_head_of_queue=True,
        ),
    )
    assert held.disposition is VramDisposition.DEFER
    assert held.room is not None
    assert held.room.deficit_mb == 19424.0 - (20157.0 - _NOISE_MB)
    assert held.room.tenancy_mb[TenantLane.INFERENCE_TARGET] == 100 + _CONTEXT_MB
    assert held.room.rungs[0].kind is RoomRungKind.IDLE_SIBLING_CONTEXT

    admitted = arbiter.evaluate(
        VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="small",
            baseline="stable_diffusion_1",
            device_index=0,
            target_process_id=2,
            candidate_delta_mb=3000.0,
            is_head_of_queue=True,
        ),
    )
    assert admitted.admits
    assert admitted.room is None
