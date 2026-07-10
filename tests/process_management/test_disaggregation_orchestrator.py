"""Fakes-based unit tests for the disaggregation orchestrator's per-job DAG.

No GPU: fake role processes capture the control messages they are sent, and stage results are fed
in by hand. Exercises the txt2img and img2img DAGs, the fault path, the pinned-sampler booking and its
early release, and fault re-dispatch after a stage process is retired mid-stage (the parent re-dispatches
from the held intermediates).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeImageResult,
    HordeSampleResultMessage,
    HordeStageModelMixin,
    HordeTextEncodeResultMessage,
    HordeVaeDecodeResultMessage,
    HordeVaeEncodeResultMessage,
    SampleSliceResult,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.workers.disaggregation_orchestrator import (
    _RESOURCE_DEFER_SECONDS,
    _SAMPLING_LIVENESS_GRACE_SECONDS,
    _STAGE_PATIENCE_SECONDS,
    DisaggregatedFault,
    DisaggregationOrchestrator,
)

from .conftest import make_job_pop_response

_SAMPLER_PID = 2
_SAMPLER_PID_2 = 4


class _FakeProcess:
    """A stand-in role process that records the control messages it is sent.

    Carries a ``process_launch_identifier`` so the launch-aware dispatch/pin tracking can be exercised: an
    id-reusing replacement is modelled by swapping in a fake with the same ``process_id`` but a new launch.
    ``busy`` models the device-state the liveness escalation corroborates against: a running sampler reports
    busy (the default), an idle/finished one reports not-busy.
    """

    def __init__(
        self,
        process_id: int,
        *,
        process_launch_identifier: int = 0,
        busy: bool = True,
    ) -> None:
        self.process_id = process_id
        self.process_launch_identifier = process_launch_identifier
        self.busy = busy
        self.sent: list[object] = []
        # Read by the sample-completion observation seam (the pinned sampler's latest reported peak). None here
        # leaves nothing to observe, matching an off-GPU or not-yet-reported sampler.
        self.process_peak_reserved_mb: int | None = None

    def is_process_busy(self) -> bool:
        return self.busy

    def safe_send_message(self, message: object) -> bool:
        self.sent.append(message)
        return True


def _job(*, post_processing: list[str] | None = None) -> HordeJobInfo:
    """A minimal job_info carrying a real ImageGenerateJobPopResponse (messages validate it)."""
    # A duck-typed stand-in: only ``sdk_api_job_info`` is read by the orchestrator on this path.
    return SimpleNamespace(  # type: ignore[return-value]
        sdk_api_job_info=make_job_pop_response(model="SDXL 1.0", post_processing=post_processing),
    )


class _PeakStub:
    """A per-job sampling-peak estimator: returns a fixed MB per job id, else a default (or None)."""

    def __init__(self, *, default_mb: float | None = None, by_job_id: dict[object, float] | None = None) -> None:
        self.default_mb = default_mb
        self.by_job_id = by_job_id or {}

    def __call__(self, job_info: HordeJobInfo) -> float | None:
        return self.by_job_id.get(job_info.sdk_api_job_info.id_, self.default_mb)


def _identity(_job_info: object) -> HordeStageModelMixin:
    return HordeStageModelMixin(horde_model_name="SDXL 1.0", ckpt_name="sd_xl_base_1.0.safetensors")


def _headroom_cycle(headroom_mb: float) -> VramArbiter:
    """A VRAM arbiter frozen on a device state whose sampling headroom is exactly ``headroom_mb``.

    All the device-overhead terms are zeroed so ``DeviceVramState.sampling_headroom_mb`` reduces to the raw
    total, letting a test pin a clean headroom figure. The orchestrator supplies the live in-flight sampling
    total with each request, so the state's own active-peaks total is irrelevant here.
    """
    state = DeviceVramState(
        total_vram_mb=headroom_mb,
        baseline_mb=0.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
    )
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter


def _admission_cycle(*, total_mb: float, committed_mb: float) -> VramArbiter:
    """A VRAM arbiter frozen on a device state with a plain committed floor.

    Enough to exercise the stage-dispatch seams: a total and a committed floor let a test pin that encode and
    decode dispatches proceed regardless of the memory picture (stage dispatches are never withheld).
    """
    state = DeviceVramState(
        total_vram_mb=total_mb,
        baseline_mb=0.0,
        committed_vram_mb=committed_mb,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
    )
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter


def _make_harness(
    *,
    estimate_sampling_peak_mb: object = None,
    estimate_decode_spike_mb: object = None,
    sampling_headroom_mb: float | None = None,
) -> SimpleNamespace:
    """Build an orchestrator wired to fake role processes plus reservation/early-release recorders.

    ``estimate_sampling_peak_mb`` (a callable, or None for the no-op) stubs the sampling-peak estimator.
    ``estimate_decode_spike_mb`` (a callable, or None for the no-op) stubs the decode-spike estimator the
    decode gate charges.
    ``sampling_headroom_mb`` (a fixed MB figure) freezes a crafted arbiter cycle whose sampling headroom is
    that figure, so the concurrent-sampling gate's decision can be exercised without a scheduler; None leaves
    the arbiter unwired, where the gate admits on missing telemetry (matching the old None-headroom contract).
    """
    encode_service = _FakeProcess(1)
    sampler = _FakeProcess(_SAMPLER_PID)
    sampler2 = _FakeProcess(_SAMPLER_PID_2)
    image_lane = _FakeProcess(3)
    by_id: dict[int, _FakeProcess] = {
        1: encode_service,
        _SAMPLER_PID: sampler,
        _SAMPLER_PID_2: sampler2,
        3: image_lane,
    }
    reserved: set[int] = set()
    completed: list[tuple[object, list[HordeImageResult], GENERATION_STATE, DisaggregatedFault | None]] = []
    sampling_completed: list[object] = []
    rerouted: list[object] = []
    released: list[int] = []
    virtual_now = [0.0]

    peak_callable = estimate_sampling_peak_mb if estimate_sampling_peak_mb is not None else (lambda _job_info: None)
    decode_callable = estimate_decode_spike_mb if estimate_decode_spike_mb is not None else (lambda _job_info: None)

    orchestrator = DisaggregationOrchestrator(
        find_encode_service=lambda: encode_service,  # type: ignore[arg-type]
        find_sampler=lambda _model: by_id.get(_SAMPLER_PID),  # type: ignore[arg-type,return-value]
        find_image_lane=lambda: image_lane,  # type: ignore[arg-type]
        loader_identity=_identity,  # type: ignore[arg-type]
        on_images_ready=lambda ji, imgs, st, fault: completed.append((ji, imgs, st, fault)),
        find_process_by_id=lambda pid: by_id.get(pid),  # type: ignore[arg-type,return-value]
        reserve_sampler_process=reserved.add,
        release_sampler_process=reserved.discard,
        on_sampling_complete=sampling_completed.append,
        reroute_monolithic=rerouted.append,
        estimate_sampling_peak_mb=peak_callable,  # type: ignore[arg-type]
        estimate_decode_spike_mb=decode_callable,  # type: ignore[arg-type]
        clock=lambda: virtual_now[0],
    )
    if sampling_headroom_mb is not None:
        orchestrator.set_vram_arbiter(_headroom_cycle(sampling_headroom_mb))
    return SimpleNamespace(
        orchestrator=orchestrator,
        encode_service=encode_service,
        sampler=sampler,
        sampler2=sampler2,
        image_lane=image_lane,
        by_id=by_id,
        reserved=reserved,
        completed=completed,
        sampling_completed=sampling_completed,
        rerouted=rerouted,
        released=released,
        virtual_now=virtual_now,
    )


@pytest.mark.asyncio
async def test_txt2img_dag_runs_to_completion() -> None:
    """A txt2img job flows encode -> sample -> decode and hands off its images once."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_

    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    assert _SAMPLER_PID in h.reserved  # the sampler is booked at registration
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 1  # text-encode dispatched

    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # sample dispatched to the pinned sampler with injected conditioning
    assert len(h.sampler.sent[0].slices) == 1

    await h.orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=_SAMPLER_PID,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job_id, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )
    # Early release: the sampler slot is freed and the job moved on to decode the instant sampling finishes.
    assert _SAMPLER_PID not in h.reserved
    assert len(h.sampling_completed) == 1
    h.orchestrator.tick()
    assert len(h.image_lane.sent) == 1  # decode dispatched

    await h.orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=3,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )
    assert len(h.completed) == 1
    _ji, images, state, fault = h.completed[0]
    assert state == GENERATION_STATE.ok
    assert len(images) == 1
    assert fault is None  # a successful completion carries no fault context
    assert not h.orchestrator.has_job(job)  # removed from the pipeline
    assert h.reserved == set()  # no reservation leaked


@pytest.mark.asyncio
async def test_img2img_waits_for_both_source_latent_and_conditioning() -> None:
    """An img2img job only samples once both the source latent and the conditioning are in."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_

    h.orchestrator.register(job, needs_source_latent=True, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.image_lane.sent) == 1  # vae-encode dispatched first

    # Conditioning arrives before the source latent: must NOT sample yet.
    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 0  # still awaiting the source latent

    await h.orchestrator.handle_stage_result(
        HordeVaeEncodeResultMessage(
            process_id=3,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            latent_bytes=b"src_latent",
            state=GENERATION_STATE.ok,
        ),
    )
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # now both inputs present -> sample dispatched
    assert h.sampler.sent[0].slices[0].source_latent_bytes == b"src_latent"


@pytest.mark.asyncio
async def test_faulted_stage_finishes_job_faulted_and_releases_pin() -> None:
    """A faulted stage result hands the job off faulted with no images and leaks no reservation."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    assert _SAMPLER_PID in h.reserved

    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.sdk_api_job_info.id_,
            positive_conditioning_bytes=None,
            negative_conditioning_bytes=None,
            state=GENERATION_STATE.faulted,
        ),
    )
    assert len(h.completed) == 1
    assert h.completed[0][2] == GENERATION_STATE.faulted
    assert h.completed[0][1] == []
    # A faulted job must not leave its sampler booked: the pid is available again (no pin leak).
    assert h.reserved == set()


@pytest.mark.asyncio
async def test_child_stage_fault_surfaces_child_reason_and_process_id() -> None:
    """A stage fault delivers the child's exception text and the faulting child's process id to the callback."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()

    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,  # the encode service child that produced the fault
            process_launch_identifier=0,
            info="",
            job_id=job.sdk_api_job_info.id_,
            positive_conditioning_bytes=None,
            negative_conditioning_bytes=None,
            state=GENERATION_STATE.faulted,
            fault_reason="OutOfMemoryError: CUDA out of memory",
        ),
    )
    fault = h.completed[0][3]
    assert isinstance(fault, DisaggregatedFault)
    assert fault.reason == "OutOfMemoryError: CUDA out of memory"  # the child's text, not a blank/generic reason
    assert fault.faulted_process_id == 1  # attributed to the child that faulted, not the pinned sampler


def test_parent_side_ageout_surfaces_orchestrator_reason_and_pinned_sampler() -> None:
    """A patience age-out (no child result) delivers the orchestrator's reason and the pinned sampler's id."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator._find_encode_service = lambda: None  # no role process: the stage cannot dispatch

    h.orchestrator.tick()  # anchors first_stalled_at at t=0
    assert h.completed == []
    h.virtual_now[0] = _STAGE_PATIENCE_SECONDS + 1.0
    h.orchestrator.tick()  # past patience: the job is faulted parent-side

    assert len(h.completed) == 1
    fault = h.completed[0][3]
    assert isinstance(fault, DisaggregatedFault)
    assert "no role process" in fault.reason  # the orchestrator's own reason, there being no child text
    assert fault.faulted_process_id == _SAMPLER_PID  # falls back to the pinned sampler, never a wrong default


def test_retired_stage_process_triggers_redispatch() -> None:
    """A stage process retired mid-stage frees its job to re-dispatch from held state."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 1

    # The encode service dies before returning conditioning; a second tick must re-dispatch.
    h.orchestrator.on_stage_process_retired(1)
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 2  # re-dispatched from held state


def test_reconcile_redispatches_when_stage_process_crashes() -> None:
    """A crashed stage process (gone from the live set) is detected by reconcile and re-dispatched."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 1  # dispatched to the encode service (pid 1)

    # The encode service (pid 1) crashed and was reaped; only some other process remains alive. Reconcile
    # takes live (id, launch) pairs, so the crashed launch is absent from the set.
    h.orchestrator.reconcile_retired_processes(alive_launches={(_SAMPLER_PID, 0), (3, 0)})
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 2  # crash detected -> re-dispatched from held state


@pytest.mark.asyncio
async def test_id_reusing_replacement_is_detected_as_retirement() -> None:
    """A replacement reusing the sampler's slot id (new launch) is detected as a retirement, not the live proc.

    The observed wedge: a hung sampler was replaced under the same ``process_id`` (only the launch identifier
    differs), so a bare-id liveness check saw the slot as still alive, the mid-sample dispatch never retired,
    and its ledgered peak leaked forever. Reconcile now compares full ``(id, launch)`` pairs, so the stale
    launch is retired: the ledger peak is freed and the stage re-dispatches onto the replacement.
    """
    h = _make_harness()
    job = _job()
    await _bring_to_sampling(h, job, pinned_pid=_SAMPLER_PID)
    h.orchestrator.tick()  # dispatch the sample to the pinned sampler (pid _SAMPLER_PID, launch 0)
    assert len(h.sampler.sent) == 1
    assert h.orchestrator._active_sampling_peaks  # the sample's peak is ledgered

    # The sampler hangs and is watchdog-replaced: same slot id, new launch. A bare-id check would miss this.
    replacement = _FakeProcess(_SAMPLER_PID, process_launch_identifier=1)
    h.by_id[_SAMPLER_PID] = replacement

    h.orchestrator.reconcile_retired_processes(alive_launches={(1, 0), (_SAMPLER_PID, 1), (3, 0)})
    # Retirement fired: the leaked ledger peak is freed and the pin released (so it can re-resolve).
    assert h.orchestrator._active_sampling_peaks == {}

    # The next tick re-resolves the sampler to the live replacement and re-dispatches the held sample onto it.
    h.orchestrator.tick()
    assert len(replacement.sent) == 1
    assert _SAMPLER_PID in h.reserved


def test_release_job_clears_all_held_state_and_is_a_noop_for_unknown_jobs() -> None:
    """The external-release seam drops the pin, reservation, and ledger entry, and no-ops for unknown jobs."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    assert _SAMPLER_PID in h.reserved

    # An unknown job is a safe no-op: nothing held, nothing raised.
    h.orchestrator.release_job("not-a-real-job-id")
    assert h.orchestrator.has_job(job)
    assert _SAMPLER_PID in h.reserved

    # Releasing the held job (mid-stage) drops every trace of it: state, pin, reservation, ledger.
    h.orchestrator.release_job(job_id)
    assert not h.orchestrator.has_job(job)
    assert h.reserved == set()
    assert h.orchestrator._active_sampling_peaks == {}


@pytest.mark.asyncio
async def test_gate_deferral_clears_a_stale_ledger_entry_promptly_and_dispatches() -> None:
    """A leaked ledger peak (no live sampling behind it) is cleared at once, not at the sanity bound.

    A gate deferral is healthy backpressure only while a sampling is genuinely in flight. A peak left in the
    ledger for a sampling that never returns has no live sampler behind it, so the fast liveness escalation
    reclaims it within a tick (seconds, not the 180s sanity bound) and the job dispatches, because a candidate
    that fits alone on an idle card must always run.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    # A leaked ledger entry whose owning job the orchestrator no longer holds (no live sampling behind it).
    h.orchestrator._active_sampling_peaks["ghost-job"] = 8260.0

    job = _job()
    await _bring_to_sampling(h, job, pinned_pid=_SAMPLER_PID)
    h.orchestrator.tick()  # 8260 (ghost) + 8260 (job) > 15000: gate-deferred, but the ghost is not live sampling
    assert len(h.sampler.sent) == 0  # deferred this tick
    assert h.orchestrator._active_sampling_peaks == {}  # the ghost peak was cleared at once, no 180s wait

    h.orchestrator.tick()  # clean ledger now admits the job on the very next tick
    assert len(h.sampler.sent) == 1
    assert h.completed == []  # dispatched, not faulted


@pytest.mark.asyncio
async def test_gate_deferral_with_a_live_ledger_ages_the_job_into_patience_not_forever() -> None:
    """A deferral behind a genuinely live sampler that never returns ages into the patience fault, not forever.

    When the sanity bound elapses but the ledger entry belongs to a live sampler (its dispatch target is the
    current launch), the ledger is not the problem, so the job is not re-admitted. Instead it is aged through
    the normal patience machinery and faulted rather than deferring indefinitely.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # job A samples (its target stays live, its result never arrives)
    assert len(h.sampler2.sent) == 0  # job B gate-deferred behind A's ledgered peak

    # Past the sanity bound: A's ledger entry is live (target alive), so B ages rather than being re-admitted.
    h.virtual_now[0] = 181.0
    h.orchestrator.tick()
    assert h.completed == []  # not yet faulted, but now aging (first_stalled_at anchored)
    assert len(h.sampler2.sent) == 0

    # Past the patience window on top of the sanity bound: B faults rather than deferring forever.
    h.virtual_now[0] = 181.0 + _STAGE_PATIENCE_SECONDS + 1.0
    h.orchestrator.tick()
    assert len(h.completed) == 1
    assert h.completed[0][2] == GENERATION_STATE.faulted
    assert not h.orchestrator.has_job(job_b)
    assert h.orchestrator.has_job(job_a)  # A is untouched: it is genuinely sampling


@pytest.mark.asyncio
async def test_retired_sampler_releases_pin_and_reresolves() -> None:
    """A pinned sampler retired mid-flow releases its reservation, then re-resolution re-pins a live holder."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)

    # Encode completes so the job is ready to sample.
    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )

    # The pinned sampler crashes before the sample dispatch: its reservation must be released.
    h.orchestrator.on_stage_process_retired(_SAMPLER_PID)
    assert _SAMPLER_PID not in h.reserved

    # A live process still holds the model (find_sampler resolves it); the next tick re-pins and dispatches.
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1
    assert _SAMPLER_PID in h.reserved  # re-resolution re-booked the sampler


def _resource_faulted_text_encode(job_id: object) -> HordeTextEncodeResultMessage:
    """A text-encode result flagged as a resource-class (device out-of-memory) fault."""
    return HordeTextEncodeResultMessage(
        process_id=1,
        process_launch_identifier=0,
        info="",
        job_id=job_id,  # type: ignore[arg-type]
        positive_conditioning_bytes=None,
        negative_conditioning_bytes=None,
        state=GENERATION_STATE.faulted,
        fault_is_resource_class=True,
    )


@pytest.mark.asyncio
async def test_resource_class_fault_defers_then_redispatches_same_stage() -> None:
    """A resource-class stage fault defers the job (no forfeit) and the next tick re-dispatches the same stage."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 1  # text-encode dispatched
    assert _SAMPLER_PID in h.reserved

    await h.orchestrator.handle_stage_result(_resource_faulted_text_encode(job_id))
    # The resource fault neither forfeits the job nor re-routes it: it is deferred, still owned, pin held.
    assert h.completed == []
    assert h.rerouted == []
    assert h.orchestrator.has_job(job)
    assert _SAMPLER_PID in h.reserved

    # The next tick re-dispatches the same (text-encode) stage from held state, retrying as pressure clears.
    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 2


@pytest.mark.asyncio
async def test_resource_class_fault_recurrence_after_window_reroutes_monolithic() -> None:
    """A resource-class fault recurring past the defer window re-routes the job (pin freed, no images-faulted)."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()

    # First resource fault at t=0 anchors the defer window.
    await h.orchestrator.handle_stage_result(_resource_faulted_text_encode(job_id))
    assert h.rerouted == []

    # A second resource fault, now past the window, re-routes the job monolithically exactly once.
    h.virtual_now[0] = _RESOURCE_DEFER_SECONDS + 1.0
    await h.orchestrator.handle_stage_result(_resource_faulted_text_encode(job_id))
    assert h.rerouted == [job]
    assert not h.orchestrator.has_job(job)  # popped from the pipeline
    assert h.reserved == set()  # pin released, no reservation leaked
    assert h.completed == []  # no images-faulted hand-off: the job runs whole instead


@pytest.mark.asyncio
async def test_resource_defer_expiry_reroutes_when_stage_stays_undispatchable() -> None:
    """A deferred stage that stays undispatchable past the window is re-routed by the tick, not faulted."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()

    # Anchor the defer window via a resource-class fault, then remove the role so the stage cannot re-dispatch.
    await h.orchestrator.handle_stage_result(_resource_faulted_text_encode(job.sdk_api_job_info.id_))
    h.orchestrator._find_encode_service = lambda: None  # role gone: the stage cannot dispatch

    # Within the window the tick keeps the job (retrying); past it, the tick re-routes rather than faulting.
    h.virtual_now[0] = _RESOURCE_DEFER_SECONDS / 2
    h.orchestrator.tick()
    assert h.rerouted == []
    assert h.orchestrator.has_job(job)

    h.virtual_now[0] = _RESOURCE_DEFER_SECONDS + 1.0
    h.orchestrator.tick()
    assert h.rerouted == [job]
    assert h.completed == []  # re-routed, not faulted
    assert h.reserved == set()


@pytest.mark.asyncio
async def test_genuine_stage_fault_still_forfeits_and_is_not_rerouted() -> None:
    """A genuine (non-resource) stage fault keeps the existing forfeit path and is never re-routed."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()

    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.sdk_api_job_info.id_,
            positive_conditioning_bytes=None,
            negative_conditioning_bytes=None,
            state=GENERATION_STATE.faulted,
            fault_is_resource_class=False,
        ),
    )
    assert h.rerouted == []
    assert len(h.completed) == 1
    assert h.completed[0][2] == GENERATION_STATE.faulted
    assert h.reserved == set()


# -- concurrent-sampling admission gate ------------------------------------------------------------


async def _bring_to_sampling(h: SimpleNamespace, job: HordeJobInfo, *, pinned_pid: int) -> None:
    """Register a txt2img job and complete its text-encode so its next stage is SAMPLING."""
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=pinned_pid)
    h.orchestrator.tick()  # dispatch text-encode
    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )


def _sample_ok(job_id: object, process_id: int) -> HordeSampleResultMessage:
    """A successful sample result for a job (frees the ledgered peak and the pin)."""
    return HordeSampleResultMessage(
        process_id=process_id,
        process_launch_identifier=0,
        info="",
        results=[SampleSliceResult(job_id=job_id, latent_bytes=b"latent", state=GENERATION_STATE.ok)],  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_second_concurrent_sample_deferred_until_first_frees_headroom() -> None:
    """Two 1024-class SDXL samples cannot pair on a 16GB-class headroom: the second waits for the first."""
    # Headroom ~15000 net; each 1024 SDXL peak ~8260, so one fits but two (16520) do not.
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    # Job A (registered first) samples; job B is gate-deferred (empty ledger admits the first, the sum denies
    # the second), so only one HordeSampleControlMessage went out.
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 0

    # The deferred job must never fault from patience while it is only gate-blocked, even long past the window.
    h.virtual_now[0] = _STAGE_PATIENCE_SECONDS * 3
    h.orchestrator.tick()
    assert h.completed == []
    assert len(h.sampler2.sent) == 0
    assert h.orchestrator.has_job(job_b)

    # Job A's sample completes, freeing its ledgered peak; the next tick admits job B onto its own sampler.
    await h.orchestrator.handle_stage_result(_sample_ok(job_a.sdk_api_job_info.id_, _SAMPLER_PID))
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_two_small_samples_pair_within_headroom() -> None:
    """Two 512-class SDXL samples (~7312 each) DO pair under a 16GB-class headroom (~15000 net)."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=7312.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    # 7312 + 7312 = 14624 <= 15000: both sample concurrently.
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_large_pair_admits_on_24gb_class_headroom() -> None:
    """Two 1024-class SDXL samples (~8260 each) pair on a 24GB-class headroom (~22000 net)."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=22000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    # 8260 + 8260 = 16520 <= 22000: both sample concurrently.
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_single_over_peak_sample_always_admits_on_empty_ledger() -> None:
    """A lone sampling over the headroom is admitted (the monolithic status quo), never wedged by the gate."""
    # Headroom far below one job's peak: an 8GB-class card whose net headroom is under a single 1024 peak.
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=6000.0)
    job = _job()
    await _bring_to_sampling(h, job, pinned_pid=_SAMPLER_PID)

    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # empty ledger always admits, regardless of the estimate

    # And it never faults from patience: it is sampling, not stalled.
    h.virtual_now[0] = _STAGE_PATIENCE_SECONDS * 3
    h.orchestrator.tick()
    assert h.completed == []


@pytest.mark.asyncio
async def test_missing_peak_estimate_never_wedges_second_sample() -> None:
    """A None peak estimate admits the second sample rather than wedging on missing telemetry."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=None), sampling_headroom_mb=1000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    # Both peaks are unknown: the gate never blocks on a missing estimate, so both dispatch.
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_gate_frees_ledger_on_sample_fault_admitting_the_deferred_job() -> None:
    """A first sample faulting frees its ledgered peak, so the gated second job then admits."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 0

    # Job A's sample faults (genuine, non-resource): its peak must leave the ledger so B can proceed.
    await h.orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=_SAMPLER_PID,
            process_launch_identifier=0,
            info="",
            results=[
                SampleSliceResult(job_id=job_a.sdk_api_job_info.id_, latent_bytes=None, state=GENERATION_STATE.faulted)
            ],
        ),
    )
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 1  # A's headroom returned, B admitted


@pytest.mark.asyncio
async def test_gate_frees_ledger_on_sampler_retirement() -> None:
    """A sampler retired mid-sampling frees its ledgered peak, admitting the gated second job."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 0

    # Job A's sampler (pid 2) crashes mid-sampling: its peak must be freed, and job B admitted, before A even
    # re-dispatches. Remove A's sampler from the live set so A itself cannot re-sample and re-book the ledger.
    del h.by_id[_SAMPLER_PID]
    h.orchestrator.on_stage_process_retired(_SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 1  # retirement returned the headroom, B admitted


@pytest.mark.asyncio
async def test_gate_frees_ledger_on_reroute_monolithic() -> None:
    """A first sample re-routed monolithically (resource-fault past the window) frees its ledgered peak."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a = _job()
    job_b = _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)

    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 0

    def _resource_sample_fault() -> HordeSampleResultMessage:
        return HordeSampleResultMessage(
            process_id=_SAMPLER_PID,
            process_launch_identifier=0,
            info="",
            results=[
                SampleSliceResult(job_id=job_a.sdk_api_job_info.id_, latent_bytes=None, state=GENERATION_STATE.faulted)
            ],
            fault_is_resource_class=True,
        )

    # First resource fault anchors the defer window (A stays, deferred); a recurrence past it re-routes A.
    await h.orchestrator.handle_stage_result(_resource_sample_fault())
    h.virtual_now[0] = _RESOURCE_DEFER_SECONDS + 1.0
    await h.orchestrator.handle_stage_result(_resource_sample_fault())
    assert h.rerouted == [job_a]
    assert not h.orchestrator.has_job(job_a)

    # A left the pipeline with its peak freed: B now admits.
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 1


def _two_sampler_cycle(*, vae_lane_decode_spike_mb: float) -> VramArbiter:
    """A 16375MB card with one 6158 sampler resident and the lane's decode spike charged.

    The two-sampler acceptance figures at the device level: total 16375, a fixed per-process overhead of
    1288, one loaded inference context, and the lane decode spike varied by the caller.
    ``DeviceVramState.sampling_headroom_mb`` is then 16375 - 1288 - decode_spike.
    """
    state = DeviceVramState(
        total_vram_mb=16375.0,
        baseline_mb=0.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        num_loaded_inference_processes=1,
        per_process_overhead_mb=1288.0,
        marginal_mb=300.0,
        vram_reserve_mb=0.0,
        vae_lane_decode_spike_mb=vae_lane_decode_spike_mb,
    )
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter


def test_two_sampler_bounded_decode_spike_admits_through_the_flipped_gate() -> None:
    """End-to-end (orchestrator adapter -> arbiter): 2x6158 peaks + the bounded 2500 lane spike admit.

    headroom = 16375 - 1288 - 2500 = 12587; the live active 6158 plus the 6158 candidate = 12316 <= 12587.
    """
    h = _make_harness()
    h.orchestrator.set_vram_arbiter(_two_sampler_cycle(vae_lane_decode_spike_mb=2500.0))
    h.orchestrator._active_sampling_peaks["first"] = 6158.0

    assert h.orchestrator._admit_concurrent_sampling(6158.0) is True


def test_two_sampler_full_quota_charge_denies_through_the_flipped_gate() -> None:
    """End-to-end (orchestrator adapter -> arbiter): charging the full 8192 lane quota denies the second sampler.

    headroom = 16375 - 1288 - 8192 = 6895; the demand 6158 + 6158 = 12316 does not fit (the collapse tripwire).
    """
    h = _make_harness()
    h.orchestrator.set_vram_arbiter(_two_sampler_cycle(vae_lane_decode_spike_mb=8192.0))
    h.orchestrator._active_sampling_peaks["first"] = 6158.0

    assert h.orchestrator._admit_concurrent_sampling(6158.0) is False


async def _drive_to_decode_pending(h: SimpleNamespace, job: HordeJobInfo, sampler_pid: int) -> None:
    """Run a txt2img job through encode and sample, leaving it at AWAITING_LATENT_DECODE (pre-decode)."""
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=sampler_pid)
    h.orchestrator.tick()
    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    h.orchestrator.tick()
    await h.orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=sampler_pid,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job_id, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )


@pytest.mark.asyncio
async def test_stage_completions_never_release_allocator_caches_unprompted() -> None:
    """Stage completions leave every process's allocator pool in place; reclaim is on-demand only.

    An unconditional post-stage cache release forces a collection pause plus a full pool rebuild on the
    next slice for every job, costing far more than the reservation it returns. The admission arbiter's
    escalation ladder is the sole reclaim path: it targets a specific process's cache only when a
    competing demand actually needs the memory.
    """
    h = _make_harness()
    job = _job()
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)
    h.orchestrator.tick()  # decode dispatched
    await h.orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=3,
            process_launch_identifier=0,
            info="",
            job_id=job.sdk_api_job_info.id_,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )
    assert h.released == []


def test_encode_dispatches_even_when_committed_tops_the_physical_total() -> None:
    """An encode dispatches whatever the memory picture: stage dispatches are never withheld.

    The concurrent-sampling gate downstream is the pipeline's admission point (this job only samples if that
    gate admits it), so gating the encode adds no admission control and would only serialise the stage
    overlap. The resource-defer/reroute machinery remains reserved for genuine resource-class stage FAULTS
    reported by a child, not for parent-side verdicts.
    """
    h = _make_harness()
    h.orchestrator.set_vram_arbiter(_admission_cycle(total_mb=16000.0, committed_mb=17000.0))
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)

    h.orchestrator.tick()
    assert len(h.encode_service.sent) == 1  # dispatched despite committed 17000 > total 16000
    assert h.rerouted == []
    assert h.completed == []


@pytest.mark.asyncio
async def test_decode_dispatches_even_when_demand_tops_the_physical_total() -> None:
    """A pending decode dispatches even when committed-plus-spike tops the total: draining ends pressure.

    Withholding a decode freezes finished sampling work behind the stage-patience window and re-routes it
    monolithically, re-running the sampling on a card that is already pressured. The lane's tiled decode and
    its allocation self-heal bound the transient spike, so the decode always proceeds.
    """
    h = _make_harness(estimate_decode_spike_mb=lambda _job_info: 8000.0)
    h.orchestrator.set_vram_arbiter(_admission_cycle(total_mb=16000.0, committed_mb=10000.0))
    job = _job()
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)
    h.orchestrator.tick()
    assert len(h.image_lane.sent) == 1  # decode dispatched despite 10000+8000 > 16000
    assert h.rerouted == []  # nothing sampled is ever thrown away over memory pressure


# --------------------------------------------------------------------------------------------------------- #
#  Part 1 liveness: the concurrent-sampling gate may serialize samplers but must never deadlock.             #
#  When no sampling is verifiably in flight, a gate-deferred sample escalates to admission within a tick;    #
#  when one genuinely is, the deferral is bounded by that sampling's completion. The cells below sweep the   #
#  matrix (peaks co-fit or not; first sampling live / finished / crashed / leaked; one lane or two).         #
# --------------------------------------------------------------------------------------------------------- #


async def _first_sampling_second_deferred(h: SimpleNamespace, job_a: HordeJobInfo, job_b: HordeJobInfo) -> None:
    """Bring both jobs to SAMPLING and tick once: job A is booked-sampling, job B gate-deferred behind its peak."""
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # job A admitted (empty ledger admits the first)
    assert len(h.sampler2.sent) == 0  # job B gate-deferred behind A's ledgered peak


@pytest.mark.asyncio
async def test_liveness_peaks_cofit_admits_both_at_once() -> None:
    """Cell (peaks co-fit): both samplers admit immediately; no deferral and no escalation are involved."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=7000.0), sampling_headroom_mb=15000.0)
    job_a, job_b = _job(), _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_liveness_genuine_first_bounds_the_deferral_then_admits_on_completion() -> None:
    """Cell (peaks don't co-fit, first genuinely sampling): the second waits, then admits on the first's result.

    A busy first sampler is verifiably in flight, so the deferral is healthy backpressure bounded by that
    job's completion: it is never escalated early, and the second admits the moment the first frees headroom.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a, job_b = _job(), _job()
    await _first_sampling_second_deferred(h, job_a, job_b)

    h.virtual_now[0] = _SAMPLING_LIVENESS_GRACE_SECONDS + 10.0
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 0  # a live sampling genuinely holds the headroom: correctly still deferred

    await h.orchestrator.handle_stage_result(_sample_ok(job_a.sdk_api_job_info.id_, _SAMPLER_PID))
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 1  # bounded by the first's completion, the second admits


@pytest.mark.asyncio
async def test_liveness_leaked_ledger_no_owner_escalates_within_a_tick() -> None:
    """Cell (pin/peak leaked, one lane): a ledger peak whose job is gone escalates the head within a tick.

    Models a job that faulted out of the pipeline without releasing its peak: nothing is sampling behind it,
    so the fast escalation reclaims the headroom at once rather than waiting on the 180s sanity bound.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    h.orchestrator._active_sampling_peaks["leaked-job"] = 8260.0
    job = _job()
    await _bring_to_sampling(h, job, pinned_pid=_SAMPLER_PID)

    h.orchestrator.tick()  # 8260 leaked + 8260 candidate > 15000: deferred, but the leak is not live sampling
    assert h.orchestrator._active_sampling_peaks == {}  # cleared at once
    h.orchestrator.tick()
    assert len(h.sampler.sent) == 1  # admitted within a tick
    assert h.completed == []  # dispatched, not faulted


@pytest.mark.asyncio
async def test_liveness_crashed_first_sampler_launch_dead_escalates_the_second() -> None:
    """Cell (two lanes, first sampler crashed): the second escalates within a tick, not at the sanity bound."""
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a, job_b = _job(), _job()
    await _first_sampling_second_deferred(h, job_a, job_b)

    h.by_id.pop(_SAMPLER_PID)  # the in-flight first sampler vanished: its peak is backed by no live launch

    h.orchestrator.tick()  # the fast escalation clears A's dead-launch peak
    assert h.orchestrator._active_sampling_peaks == {}
    h.orchestrator.tick()  # the clean ledger admits job B
    assert len(h.sampler2.sent) == 1


@pytest.mark.asyncio
async def test_liveness_hostile_idle_sampler_that_never_returns_admits_not_faults() -> None:
    """Hostile self-infliction: a live-launch first sampler goes idle and never returns its result.

    Under a launch-only staleness check this entry is never cleared (its launch stays live) and the deferred
    second job would age into the patience fault: a wedge that faults a job for another's lost result. The
    device-state corroboration deems the idle sampler's lingering peak stale past the grace, so the second is
    admitted (not faulted) instead.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a, job_b = _job(), _job()
    await _first_sampling_second_deferred(h, job_a, job_b)

    h.sampler.busy = False  # A's sample finished but its result never reached the orchestrator (launch still live)

    # Within the grace the just-idle reading is not yet trusted (filters the post-dispatch transition window).
    h.virtual_now[0] = _SAMPLING_LIVENESS_GRACE_SECONDS - 1.0
    h.orchestrator.tick()
    assert len(h.sampler2.sent) == 0

    # Past the grace the lingering peak is provably stale: the second is admitted, and job B is never faulted.
    h.virtual_now[0] = _SAMPLING_LIVENESS_GRACE_SECONDS + 1.0
    h.orchestrator.tick()  # clears the stale peak
    h.orchestrator.tick()  # admits job B
    assert len(h.sampler2.sent) == 1
    assert h.completed == []


@pytest.mark.asyncio
async def test_liveness_just_dispatched_sampler_is_not_reclaimed_within_the_grace() -> None:
    """Protective: a sample dispatched this tick (child not yet busy) is never mistaken for a stale one.

    Without the grace the second sampler would be admitted alongside a genuinely-just-dispatched first,
    over-committing the card with two non-fitting peaks. The gate keeps the second deferred until the grace.
    """
    h = _make_harness(estimate_sampling_peak_mb=_PeakStub(default_mb=8260.0), sampling_headroom_mb=15000.0)
    job_a, job_b = _job(), _job()
    await _bring_to_sampling(h, job_a, pinned_pid=_SAMPLER_PID)
    await _bring_to_sampling(h, job_b, pinned_pid=_SAMPLER_PID_2)
    h.sampler.busy = False  # models the window before the child flips to a busy state after dispatch

    h.orchestrator.tick()  # job A dispatched this tick, job B deferred
    assert len(h.sampler.sent) == 1
    assert len(h.sampler2.sent) == 0  # NOT admitted: two non-fitting peaks must never co-run
    assert h.orchestrator._active_sampling_peaks  # A's peak retained (not judged stale within the grace)


def test_paused_encode_lane_reroutes_monolithic_instead_of_aging_to_a_fault() -> None:
    """A conditioning stage whose encode lane is policy-paused reroutes at once: no patience fault, no wait.

    The production wedge: whole-card residency stops the component lane for a heavy head, an in-flight
    disaggregated job then has no role process for awaiting_conditioning, ages out the 90s patience window,
    and is faulted for horde reissue even though it could run whole on its own sampler. A deliberate pause is
    a routing decision, not a crash, so the job must return to the monolithic path immediately.
    """
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator._find_encode_service = lambda: None  # the lane was stopped off-GPU
    h.orchestrator._encode_lane_paused = lambda: True  # by a policy holder, with a live restore path

    h.orchestrator.tick()

    assert h.rerouted == [job]  # rerouted on the first tick, not parked for the patience window
    assert h.completed == []  # never faulted: the job runs whole instead
    assert not h.orchestrator.has_job(job)
    assert h.reserved == set()  # pin released, no reservation leaked


def test_paused_image_lane_reroutes_a_source_latent_stage() -> None:
    """An img2img job whose VAE lane is policy-paused reroutes rather than stalling on the source latent."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=True, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator._find_image_lane = lambda: None
    h.orchestrator._image_lane_paused = lambda: True

    h.orchestrator.tick()

    assert h.rerouted == [job]
    assert h.completed == []
    assert h.reserved == set()


def test_crashed_encode_lane_without_a_pause_still_ages_to_the_patience_fault() -> None:
    """A genuinely missing lane (no policy pause) keeps the patience fault so the horde reissues the job."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator._find_encode_service = lambda: None  # crashed and not yet replaced

    h.orchestrator.tick()  # anchors the patience clock; no reroute
    assert h.rerouted == []
    h.virtual_now[0] = _STAGE_PATIENCE_SECONDS + 1.0
    h.orchestrator.tick()

    assert h.rerouted == []
    assert len(h.completed) == 1
    assert h.completed[0][2] == GENERATION_STATE.faulted


def test_a_paused_image_lane_does_not_reroute_a_conditioning_stage() -> None:
    """The pause predicate is stage-scoped: the other lane's pause never reroutes a conditioning stall."""
    h = _make_harness()
    job = _job()
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator._find_encode_service = lambda: None
    h.orchestrator._image_lane_paused = lambda: True  # unrelated lane paused

    h.orchestrator.tick()

    assert h.rerouted == []  # the conditioning stall ages through patience instead
    assert h.orchestrator.has_job(job)


@pytest.mark.asyncio
async def test_paused_image_lane_reroutes_a_decode_stage() -> None:
    """A job whose latent is ready but whose VAE lane got policy-paused reroutes instead of faulting."""
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    h.orchestrator.register(job, needs_source_latent=False, pinned_sampler_process_id=_SAMPLER_PID)
    h.orchestrator.tick()
    await h.orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    h.orchestrator.tick()
    await h.orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=_SAMPLER_PID,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job_id, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )

    # The whole-card claim stops the VAE lane before the decode dispatches.
    h.orchestrator._find_image_lane = lambda: None
    h.orchestrator._image_lane_paused = lambda: True
    h.orchestrator.tick()

    assert h.rerouted == [job]
    assert h.completed == []
    assert h.reserved == set()
