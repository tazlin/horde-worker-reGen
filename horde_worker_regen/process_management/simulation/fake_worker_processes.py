"""Protocol-faithful worker-lane fakes for orchestration testing.

These classes speak the exact same pipe/queue message protocol as the real
worker lanes, but never import hordelib,
torch, or any other ML dependency. They allow the full multiprocessing
orchestration layer (process manager, scheduler, safety orchestrator,
job tracker) to be exercised end-to-end on machines with no GPU and without
the heavy dependency stack loaded into the child processes.

The module-level process entry points mirror their production signatures so they
can be passed directly as ``multiprocessing.Process`` targets (they must remain
module-level functions to stay picklable under spawn). The utilities lane is a
parent-resident adapter in production, so its fake uses the matching factory seam
and keeps the same control/message behavior without a subprocess or socket.
"""

from __future__ import annotations

import os
import time

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore
from typing import override

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ANNOTATION_CONTROL_TYPES
from hordelib.metrics import JobPhaseMetrics, SamplingStats
from loguru import logger

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    AuxPrefetchOutcome,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeAuxPrefetchControlMessage,
    HordeAuxPrefetchResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeDownloadAvailabilityMessage,
    HordeDownloadControlMessage,
    HordeEvictComponentsControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordePostProcessControlMessage,
    HordePostProcessResultMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    HordeRestoreComponentsControlMessage,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
    HordeSampleControlMessage,
    HordeSampleResultMessage,
    ModelLoadState,
    PipelineStageTag,
    SampleSliceResult,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
)
from horde_worker_regen.process_management.lifecycle.child_crash_capture import (
    enable_child_faulthandler,
    write_startup_crash,
)
from horde_worker_regen.process_management.lifecycle.debug_attach import maybe_wait_for_process_debugger
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.process_management.lifecycle.utilities_adapter import UtilitiesProcessAdapter
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    ClearanceLeaseProxy,
)
from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes
from horde_worker_regen.process_management.simulation.fault_injection import (
    FAULT_INFO_PREFIX,
    FaultKind,
    FaultProfile,
)
from horde_worker_regen.process_management.simulation.sim_vram import (
    SimVramLedger,
    simulate_post_processing_allocation,
)

_DEFAULT_FAKE_SAMPLING_STEPS = 30
"""Step count a fake sampling window reports when its job carries none, so a beat always has a position."""


def _hang_forever(process_label: str, reason: str) -> None:
    """Block the calling fake process forever, emitting nothing (simulates a wedged, unresponsive child).

    A truly hung child stops servicing its control pipe too, so recovery must come from the parent's
    watchdog killing it. This models that worst case rather than a cooperative stall the child could
    itself escape.
    """
    logger.warning(f"{process_label} hanging (injected fault): {reason}")
    while True:
        time.sleep(3600)


class _FakeUtilitiesMemoryReport:
    """Zero-cost memory reading shaped like the capability client's response."""

    process_rss_bytes = 0
    torch_reserved_bytes = 0
    torch_allocated_bytes = 0


class _FakeUtilitiesClient:
    """In-memory implementation of the capability-client operations the adapter polls."""

    def health(self, *, timeout: float | None = None) -> bool:
        """Report the simulated service as healthy."""
        return True

    def get_memory_report(self) -> _FakeUtilitiesMemoryReport:
        """Return a zeroed process-memory reading."""
        return _FakeUtilitiesMemoryReport()

    def release_cache(self) -> _FakeUtilitiesMemoryReport:
        """Acknowledge allocator-cache release without changing simulated residency."""
        return self.get_memory_report()

    def shutdown(self) -> None:
        """Acknowledge a graceful shutdown request."""


class _FakeUtilitiesServer:
    """Capability-server lifecycle seam with no subprocess, socket, or external environment."""

    def __init__(self) -> None:
        self._running = False
        self._exit_code: int | None = None
        self._client = _FakeUtilitiesClient()

    @property
    def is_running(self) -> bool:
        """Return whether the simulated service has started and not stopped."""
        return self._running

    @property
    def exit_code(self) -> int | None:
        """Return zero after shutdown and None while the service is active."""
        return self._exit_code

    @property
    def base_url(self) -> str:
        """Return a non-network address; fake adapter operations never dereference it."""
        return "memory://image-utilities"

    @property
    def pid(self) -> int | None:
        """Return None because this simulation deliberately owns no OS process."""
        return None

    @property
    def client(self) -> _FakeUtilitiesClient:
        """Return the in-memory capability client."""
        return self._client

    def start(self) -> None:
        """Mark the simulated service ready."""
        self._running = True
        self._exit_code = None

    def stop(self) -> None:
        """Mark the simulated service stopped."""
        self._running = False
        self._exit_code = 0


class FakeUtilitiesProcessAdapter(UtilitiesProcessAdapter):
    """Protocol-faithful utilities lane that performs image operations in memory."""

    @override
    def annotate(self, control_type: str, image_bytes: bytes, resolution: int = 512) -> bytes:
        """Return a valid canned control-map PNG without HTTP or detector execution."""
        return make_dummy_png_bytes()

    @override
    def remove_background(self, image_bytes: bytes) -> bytes:
        """Return the supplied image as the simulated background-removal result."""
        return image_bytes

    @override
    def list_annotators(self) -> list[dict[str, object]]:
        """Advertise every named annotator as available to the fake worker."""
        return [
            {
                "name": control_type.value,
                "available": True,
                "weights_present": "present",
                "loaded": False,
            }
            for control_type in KNOWN_ANNOTATION_CONTROL_TYPES
        ]


def create_fake_utilities_adapter(
    process_id: int,
    process_message_queue: ProcessQueue,
    control_connection: Connection,
    process_launch_identifier: int,
    *,
    device_index: int,
    python_executable: str,
    child_env: dict[str, str],
    log_path: str | None = None,
) -> FakeUtilitiesProcessAdapter:
    """Build a utilities adapter with no subprocess, network, or external package environment."""
    _ = (python_executable, child_env, log_path)
    return FakeUtilitiesProcessAdapter(
        process_id=process_id,
        process_message_queue=process_message_queue,
        control_connection=control_connection,
        process_launch_identifier=process_launch_identifier,
        # The production protocol names the concrete HTTP client, while this in-memory client intentionally
        # implements only the operations the adapter calls and must not import that external package.
        server=_FakeUtilitiesServer(),  # pyrefly: ignore [bad-argument-type]
        device_index=device_index,
        heartbeat_interval_seconds=0.1,
        memory_interval_seconds=0.5,
    )


class FakeInferenceProcess(HordeProcess):
    """A lightweight stand-in for ``HordeInferenceProcess``.

    Reproduces the message sequences the main process expects (preload,
    inference start/complete, unloads) without performing any real work.
    """

    _active_model_name: str | None = None
    _periodic_report_includes_vram = True
    _inference_semaphore: Semaphore
    _job_delay_seconds: float
    _fail_every_n: int
    _jobs_started: int = 0
    _fault_profile: FaultProfile
    _sim_vram_ledger: SimVramLedger | None = None
    _sim_total_vram_mb: float = 0.0
    _sim_weights_mb: float = 0.0
    _sim_context_mb: float = 0.0
    _sim_weights_mb_by_model: dict[str, float]
    _sim_sampling_activation_mb_by_model: dict[str, float]
    _gpu_sampling_lease: ClearanceLeaseProxy | None = None

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        job_delay_seconds: float = 0.0,
        gpu_sampling_lease: ClearanceLeaseProxy | None = None,
        fail_every_n: int = 0,
        fault_profile: FaultProfile | None = None,
        sim_vram_ledger: SimVramLedger | None = None,
        sim_total_vram_mb: float = 0.0,
        sim_weights_mb: float = 0.0,
        sim_context_mb: float = 0.0,
        sim_weights_mb_by_model: dict[str, float] | None = None,
        sim_sampling_activation_mb_by_model: dict[str, float] | None = None,
    ) -> None:
        """Initialise the fake inference process.

        Args:
            process_id (int): The ID of the process. This is not the same as the PID.
            process_message_queue (ProcessQueue): The queue to send messages to the main process.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            inference_semaphore (Semaphore): The semaphore limiting concurrent inference; acquired and \
                released around each fake job so concurrency control is still exercised.
            disk_lock (Lock): The lock to use for disk access.
            process_launch_identifier (int): The unique identifier for this launch.
            device_index (int, optional): The stable index of the GPU this process is attributed to. Defaults to 0.
            job_delay_seconds (float, optional): How long each fake inference job takes. Defaults to 0.0.
            gpu_sampling_lease (ClearanceLeaseProxy | None, optional): The parent's per-child
                clearance proxy. When given, this fake honours the same handshake the real child does
                around its sampling window (prime, wait for the grant, release when the window closes),
                so a harness run exercises the parent's clearance controller end to end. Defaults to
                None (no lease, and the fake's message sequence is unchanged).
            fail_every_n (int, optional): If > 0, every nth job reports a faulted result instead of \
                images. Defaults to 0 (never fail).
            fault_profile (FaultProfile | None, optional): A misbehaviour script (hang, crash, drop \
                heartbeats, slow, OOM, corrupt message). Defaults to a no-op profile.
            sim_vram_ledger (SimVramLedger | None, optional): A shared simulated-VRAM ledger. When given, \
                the process registers its weights/context on it and reports ledger-derived device VRAM, so \
                the orchestrator's budget/forecast see a simulated device. Defaults to None (inert).
            sim_total_vram_mb (float, optional): Card capacity reported when no mutable VRAM ledger is wired. \
                This lets ordinary fake runs exercise measured-capacity admission without a manager process. \
                Defaults to 0.0 (no device telemetry).
            sim_weights_mb (float, optional): This process's resident model-weight footprint to register on \
                the ledger when the slot samples. Defaults to 0.0.
            sim_context_mb (float, optional): This process's fixed CUDA-context overhead to register on the \
                ledger at startup. Defaults to 0.0.
            sim_weights_mb_by_model (dict[str, float] | None, optional): Per-model resident weight
                footprints (MB), consulted before ``sim_weights_mb``, so one card carries the
                differently-sized residency a mixed-model scenario really commits. Defaults to None.
            sim_sampling_activation_mb_by_model (dict[str, float] | None, optional): Per-model sampling
                activation (MB) charged on top of the resident weights for the length of a sampling
                window and released when it closes. Defaults to None (no sampling transient).
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )
        self._inference_semaphore = inference_semaphore
        self._job_delay_seconds = job_delay_seconds
        self._gpu_sampling_lease = gpu_sampling_lease
        self._fail_every_n = fail_every_n
        self._fault_profile = fault_profile if fault_profile is not None else FaultProfile()
        self._sim_vram_ledger = sim_vram_ledger
        self._sim_total_vram_mb = sim_total_vram_mb
        self._sim_weights_mb = sim_weights_mb
        self._sim_context_mb = sim_context_mb
        self._sim_weights_mb_by_model = dict(sim_weights_mb_by_model or {})
        self._sim_sampling_activation_mb_by_model = dict(sim_sampling_activation_mb_by_model or {})

        if self._sim_vram_ledger is not None:
            # A slot's number is reused by whatever replaces it, so drop anything the previous tenant left
            # charged: a replaced process's VRAM goes back to the card when its context dies, and a
            # simulation that kept charging for it would drift the card down every recovery.
            self._sim_vram_ledger.free_own_models(self.device_index, self.process_id)
            # A process's CUDA context is committed for its whole life and only a teardown reclaims it, so
            # register it up front (its weights are added later, when the slot actually samples).
            self._sim_vram_ledger.set_context_overhead(self.device_index, self.process_id, self._sim_context_mb)

        if self._fault_profile.crash_on_start:
            # Die while still in PROCESS_STARTING (super().__init__ already announced it), simulating a
            # child that fails during import/CUDA init. os._exit skips cleanup like a real hard crash.
            os._exit(70)

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def get_vram_usage_mb(self) -> int:
        """Return device-wide used VRAM from the simulated ledger, or a fixed 0 when none is wired in.

        The real child reports ``torch_total - torch_free`` (device-wide used), which the orchestrator
        turns into free VRAM; reporting the ledger's device-wide used figure feeds the same seam.
        """
        if self._sim_vram_ledger is not None:
            return int(self._sim_vram_ledger.device_used_mb(self.device_index))
        return 0

    @override
    def _offthread_vram_sampling_ready(self) -> bool:
        """Allow periodic synthetic VRAM reports; no device initialization can occur in this fake."""
        return True

    @override
    def get_vram_total_mb(self) -> int:
        """Return the simulated card's total VRAM from the mutable ledger or fixed harness topology."""
        if self._sim_vram_ledger is not None:
            return int(self._sim_vram_ledger.total_mb(self.device_index))
        return int(self._sim_total_vram_mb)

    @override
    def get_process_vram_stats(self) -> tuple[int, int, int, int] | None:
        """Return plausible per-process allocator stats so parent-side attribution plumbing is exercised.

        The fake has no torch allocator nor direct-IO residency pool to read; it reports zeros so the memory
        report still carries the fields (allocated, reserved, peak, aimdo) and the ledger/drift accounting
        runs. A slot holding simulated weights reports none of them per-process, so the parent's
        committed-VRAM attribution sees a card whose device-wide figure moves while its per-process
        reservations stay flat: attribution drift is not modelled here.
        """
        return 0, 0, 0, 0

    def on_horde_model_state_change(
        self,
        horde_model_name: str,
        process_state: HordeProcessState,
        horde_model_state: ModelLoadState,
        time_elapsed: float | None = None,
    ) -> None:
        """Send a model state change message followed by a memory report, as the real process does."""
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_state=process_state,
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Model {horde_model_name} {horde_model_state.name}",
                horde_model_name=horde_model_name,
                horde_model_state=horde_model_state,
                time_elapsed=time_elapsed,
            ),
        )
        self.send_memory_report_message(include_vram=True)

    def preload_model(self, horde_model_name: str) -> None:
        """Pretend to preload a model, emitting the same state sequence as the real process."""
        if self._active_model_name == horde_model_name:
            return

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADING_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADING,
        )

        if self._fault_profile.stall_in_preload:
            # Announce PRELOADING_MODEL but never PRELOADED_MODEL: a stuck model load the parent's
            # preload-timeout watchdog must catch.
            _hang_forever(f"fake inference {self.process_id}", "stalled in preload")

        time_start = time.time()
        self._active_model_name = horde_model_name

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADED_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_RAM,
            time_elapsed=time.time() - time_start,
        )

    def _resident_weights_mb_for(self, horde_model_name: str) -> float:
        """Return the resident weight footprint (MB) this slot commits while ``horde_model_name`` is loaded.

        A per-model figure when the caller supplied one, so a card carrying a large checkpoint beside a small
        one prices them apart; otherwise the single flat figure, which is what a caller wiring the ledger by
        hand supplies.
        """
        return self._sim_weights_mb_by_model.get(horde_model_name, self._sim_weights_mb)

    def _open_sampling_window_vram(self) -> None:
        """Charge this slot's weights and sampling activation to the simulated device for the window ahead.

        Both land here rather than at preload because preload only stages weights in host RAM: the device
        pays nothing until the job actually samples. Charging them earlier makes the card look occupied by a
        model that is not on it, and the parent's admission gates then refuse to dispatch the very job whose
        weights they are reading, having no residency to credit the charge against.

        The weights persist after the window closes (a slot holds them until it is told to unload, which is
        what retention is); the activation is the per-step working set on top of them, and it goes back when
        the denoise ends. Without the activation a fake card looks equally occupied whether a slot is idle or
        sampling, and every admission, governor and reclaim judgement downstream reads a card that never
        moves.
        """
        if self._sim_vram_ledger is None or self._active_model_name is None:
            return
        self._sim_vram_ledger.set_resident_weights(
            self.device_index,
            self.process_id,
            self._resident_weights_mb_for(self._active_model_name),
        )
        activation_mb = self._sim_sampling_activation_mb_by_model.get(self._active_model_name, 0.0)
        if activation_mb > 0:
            self._sim_vram_ledger.set_transient(self.device_index, self.process_id, activation_mb)

    def _close_sampling_transient(self) -> None:
        """Release the sampling activation, leaving the slot's weights resident on the card."""
        if self._sim_vram_ledger is not None:
            self._sim_vram_ledger.clear_transient(self.device_index, self.process_id)

    def _emit_corrupt_result(self, job_info: ImageGenerateJobPopResponse) -> None:
        """Emit a stale-launch duplicate result just before the real one.

        Models a late message from a *prior* launch of this slot arriving after the slot was replaced.
        The main process should ignore it (its launch identifier will not match the slot's current one);
        if instead it is folded in, the lifecycle auditor sees a double-finalize, which is the bug this
        probes for.
        """
        self.process_message_queue.put(
            HordeInferenceResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier + 9999,
                info=f"{FAULT_INFO_PREFIX}{FaultKind.CORRUPT_MESSAGE}",
                state=GENERATION_STATE.ok,
                time_elapsed=0.0,
                job_image_results=[HordeImageResult(image_bytes=make_dummy_png_bytes())],
                sdk_api_job_info=job_info,
            ),
        )

    def _begin_lease_job(self) -> None:
        """Reset the clearance grant for the job or sample stage about to run, as the real child does.

        One grant covers one dispatched unit of work; without this reset the first acquire latches the
        proxy open and every later window passes straight through, which is the failure mode a harness
        run under the lease exists to catch.
        """
        if self._gpu_sampling_lease is not None:
            self._gpu_sampling_lease.begin_job()

    def _await_clearance(self) -> None:
        """Block where hordelib blocks the real child: at the sample call, until the parent grants clearance.

        The caller has already reported itself primed (its pipeline staged, nothing sampling yet), which is
        what makes it visible to the parent as a clearance waiter. A starved child is never wedged: the
        bounded acquire returns after the same timeout the real child registers and it samples unpriced,
        so a controller that never grants shows up as a slow, warned run rather than a deadlock.
        """
        if self._gpu_sampling_lease is None:
            return
        if not self._gpu_sampling_lease.acquire(True, CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS):
            logger.warning(
                f"Fake inference {self.process_id} sampled without a clearance grant "
                f"(waited {CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS}s)",
            )

    def _close_sampling_window(self) -> None:
        """Signal the parent that this sampling window closed, as hordelib's lease wrapper does on return."""
        if self._gpu_sampling_lease is not None:
            self._gpu_sampling_lease.release()

    def _emit_sampling_step_beat(self, *, progress_fraction: float, total_steps: int) -> None:
        """Report one sampling step, carrying its position when this fake is under the clearance lease.

        The parent reads the position for two things the lease depends on: a primed slot advances to
        ``INFERENCE_STARTING`` on its first positioned step (so the parent learns the denoise loop was
        entered and stops treating the slot as a waiter), and the tail-overlap handoff sizes its window
        from the observed step rate. An unleased run keeps the position-free beat this fake has always
        sent, so nothing shifts for it.
        """
        if self._gpu_sampling_lease is None:
            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.INFERENCE_STEP)
            return
        bounded = min(1.0, max(0.0, progress_fraction))
        current_step = min(total_steps, int(bounded * total_steps) + 1)
        self.send_heartbeat_message(
            heartbeat_type=HordeHeartbeatType.INFERENCE_STEP,
            percent_complete=int(bounded * 100),
            current_step=current_step,
            total_steps=total_steps,
        )

    def _sample_for(self, duration_seconds: float, *, total_steps: int, drop_heartbeats: bool) -> None:
        """Spend a sampling window, beating out step progress across it.

        A leased slot's zero-length window still emits one beat: the parent has to see the loop entered, or a
        slot that primed and finished between two control-loop ticks would still read as a staged waiter.
        An unleased slot stays silent through a zero-length window, as this fake has always been.
        """
        started_at = time.time()
        if duration_seconds <= 0:
            if self._gpu_sampling_lease is not None and not drop_heartbeats:
                self._emit_sampling_step_beat(progress_fraction=1.0, total_steps=total_steps)
            return
        deadline = started_at + duration_seconds
        while time.time() < deadline:
            if not drop_heartbeats:
                self._emit_sampling_step_beat(
                    progress_fraction=(time.time() - started_at) / duration_seconds,
                    total_steps=total_steps,
                )
            time.sleep(min(0.05, duration_seconds))

    def _run_fake_inference(self, job_info: ImageGenerateJobPopResponse) -> None:
        """Pretend to run inference and send the result messages for it, honoring any fault profile."""
        self._jobs_started += 1
        profile = self._fault_profile
        self._begin_lease_job()

        if profile.crash_on_job_n == self._jobs_started:
            # Acquire the semaphore first, then die holding it: exercises the parent's semaphore-orphan
            # handling on a mid-inference crash. os._exit mimics a segfault / OS OOM-kill (no cleanup).
            self._inference_semaphore.acquire()
            logger.error(f"Fake inference {self.process_id} crashing on job {self._jobs_started} (injected)")
            os._exit(71)

        if profile.hang_after_n_jobs is not None and self._jobs_started > profile.hang_after_n_jobs:
            self._inference_semaphore.acquire()
            _hang_forever(f"fake inference {self.process_id}", f"hung on job {self._jobs_started}")

        should_oom = profile.oom_on_job_n == self._jobs_started
        should_fail = should_oom or (self._fail_every_n > 0 and self._jobs_started % self._fail_every_n == 0)

        self._inference_semaphore.acquire()
        # The elapsed figure starts here, before the clearance wait, exactly as the real child's does: a job
        # held at the lease really did take that long from the slot's point of view.
        time_start = time.time()
        self._await_clearance()
        effective_delay = self._job_delay_seconds * profile.delay_factor_for_ordinal(self._jobs_started)
        self._open_sampling_window_vram()
        try:
            self._sample_for(
                effective_delay,
                total_steps=job_info.payload.ddim_steps or _DEFAULT_FAKE_SAMPLING_STEPS,
                drop_heartbeats=profile.drop_heartbeats,
            )
        finally:
            # The window closes when sampling ends, before the result is assembled, mirroring where
            # hordelib's lease wrapper returns around the sample call.
            self._close_sampling_transient()
            self._close_sampling_window()
            self._inference_semaphore.release()

        if profile.corrupt_on_job_n == self._jobs_started:
            self._emit_corrupt_result(job_info)

        n_iter = job_info.payload.n_iter if job_info.payload.n_iter else 1
        job_image_results = None
        if not should_fail:
            job_image_results = [HordeImageResult(image_bytes=make_dummy_png_bytes()) for _ in range(n_iter)]

        if should_oom:
            result_info = f"{FAULT_INFO_PREFIX}{FaultKind.OOM}"
        elif should_fail:
            result_info = f"{FAULT_INFO_PREFIX}fail_every_n"
        else:
            result_info = "fake inference"

        self.process_message_queue.put(
            HordeInferenceResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=result_info,
                state=GENERATION_STATE.ok if not should_fail else GENERATION_STATE.faulted,
                time_elapsed=time.time() - time_start,
                job_image_results=job_image_results,
                sdk_api_job_info=job_info,
            ),
        )

        # The real process snapshots hordelib's metrics collector after each job; emit a
        # synthetic equivalent so the pipe -> dispatcher -> run-metrics chain is exercised
        # without any GPU.
        steps = job_info.payload.ddim_steps if job_info.payload.ddim_steps else 30
        elapsed = max(time.time() - time_start, 0.001)
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {job_info.id_}",
                job_id=str(job_info.id_),
                phase_metrics=JobPhaseMetrics(
                    sampling=SamplingStats(
                        steps_completed=steps,
                        total_steps=steps,
                        duration_seconds=elapsed,
                        iterations_per_second=steps / elapsed,
                    ),
                    vram_used_high_water_mb=1234,
                    ram_used_high_water_mb=2345,
                ),
            ),
        )

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=(
                    HordeProcessState.INFERENCE_COMPLETE if not should_fail else HordeProcessState.INFERENCE_FAILED
                ),
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.LOADED_IN_VRAM,
            )

        self.send_process_state_change_message(
            HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _run_fake_sample(self, message: HordeSampleControlMessage) -> None:
        """Pretend to run the disaggregated sample stage, mirroring the real sampler's message sequence.

        The real sampler reports ``INFERENCE_STARTING`` as a plain process-state change (not a model-state
        change: the pinned sampler already holds its UNet and the parent keeps that residency), returns one
        LATENT per slice in a single ``HordeSampleResultMessage``, then returns to idle. The
        ``INFERENCE_STARTING`` this emits is the disaggregation state traffic the parent's dispatcher must
        tolerate on a sampler holding no whole-job model bookkeeping.

        Under the clearance lease the sequence is the real sampler's: one grant covers the whole control
        message, the stage reports itself primed and waits for that grant before any slice samples, and the
        parent advances the slot to ``INFERENCE_STARTING`` on the first positioned step. Each slice spends
        the configured job delay so a sampler has a denoise tail for the parent to hand off against; an
        unleased run keeps the original immediate, position-free sequence.
        """
        self._begin_lease_job()
        if self._gpu_sampling_lease is not None:
            self.send_process_state_change_message(HordeProcessState.INFERENCE_PRIMED, info="Priming")
            self._await_clearance()
        else:
            self.send_process_state_change_message(HordeProcessState.INFERENCE_STARTING, info="Sampling")
        time_start = time.time()
        results = []
        self._open_sampling_window_vram()
        for job_slice in message.slices:
            self._sample_for(
                self._job_delay_seconds,
                total_steps=job_slice.sdk_api_job_info.payload.ddim_steps or _DEFAULT_FAKE_SAMPLING_STEPS,
                drop_heartbeats=self._fault_profile.drop_heartbeats,
            )
            results.append(
                SampleSliceResult(
                    job_id=job_slice.job_id,
                    latent_bytes=self._make_dummy_latent_bytes(),
                    state=GENERATION_STATE.ok,
                ),
            )
            self.send_stage_job_metrics_message(str(job_slice.job_id), stage=PipelineStageTag.SAMPLE)
        self._close_sampling_transient()
        self._close_sampling_window()
        self.process_message_queue.put(
            HordeSampleResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="fake sample",
                time_elapsed=time.time() - time_start,
                results=results,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, info="Waiting for job")

    @staticmethod
    def _make_dummy_latent_bytes() -> bytes:
        """A tiny opaque stand-in for a serialized LATENT; the fake pipeline never deserializes it."""
        return b"fake-latent"

    def _run_fake_alchemy(self, form: AlchemyFormSpec) -> None:
        """Pretend to run an alchemy form, emitting the same message sequence as the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )
        time_start = time.time()
        if self._job_delay_seconds > 0:
            time.sleep(self._job_delay_seconds)

        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=GENERATION_STATE.ok,
                image_bytes=make_dummy_png_bytes(),
            ),
        )
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {form.form_id}",
                job_id=form.form_id,
                is_alchemy=True,
                phase_metrics=JobPhaseMetrics(vram_used_high_water_mb=600, ram_used_high_water_mb=1200),
            ),
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_COMPLETE,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle control messages with the same observable behavior as the real inference process."""
        logger.debug(f"Fake inference process received {type(message).__name__}: {message.control_flag}")

        if isinstance(message, HordeAlchemyControlMessage):
            self._run_fake_alchemy(message.form)
        elif isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(message.horde_model_name)
        elif isinstance(message, HordeSampleControlMessage):
            self._run_fake_sample(message)
        elif isinstance(message, HordeInferenceControlMessage) and (
            message.control_flag == HordeControlFlag.START_INFERENCE
        ):
            if message.horde_model_name != self._active_model_name:
                self.preload_model(message.horde_model_name)

            # Under the lease the slot is primed, not sampling: its weights land at clearance and the parent
            # advances it to INFERENCE_STARTING on its first step, exactly as the real child arranges. With
            # no lease the slot goes straight to sampling, as this fake has always reported.
            self.on_horde_model_state_change(
                horde_model_name=message.horde_model_name,
                process_state=(
                    HordeProcessState.INFERENCE_PRIMED
                    if self._gpu_sampling_lease is not None
                    else HordeProcessState.INFERENCE_STARTING
                ),
                horde_model_state=ModelLoadState.IN_USE,
            )

            self._run_fake_inference(message.sdk_api_job_info)
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            if self._sim_vram_ledger is not None:
                # The orchestrator told this slot to free VRAM: it frees only its own model (the
                # cross-process rule). This is the lever by which orchestrator reclaim makes room for a
                # *sibling's* imminent post-processing peak, the dynamic the harness exists to observe.
                self._sim_vram_ledger.free_own_models(self.device_index, self.process_id)
            if self._active_model_name is not None:
                self.on_horde_model_state_change(
                    process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
                    horde_model_name=self._active_model_name,
                    horde_model_state=ModelLoadState.LOADED_IN_RAM,
                )
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from VRAM",
            )
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            if self._sim_vram_ledger is not None:
                # Weights dropped from host RAM cannot still be on the device, so the card gets them back
                # here too; the VRAM unload the parent usually sends first has normally done it already.
                self._sim_vram_ledger.free_own_models(self.device_index, self.process_id)
            if self._active_model_name is not None:
                self.on_horde_model_state_change(
                    process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                    horde_model_name=self._active_model_name,
                    horde_model_state=ModelLoadState.ON_DISK,
                )
            self._active_model_name = None
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from RAM",
            )
        elif message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
            # Mirror the real child: releasing the allocator cache keeps every model resident and only
            # sends a fresh memory report, so the fake emits the same report traffic and no model-unload
            # state change (the ledger residency is untouched).
            self.send_memory_report_message(include_vram=True)
        elif isinstance(message, HordeEvictComponentsControlMessage):
            # The fake holds no real component cache, so the base handler no-ops; accepting the message
            # keeps the fake from faulting when the RAM-pressure rung targets it in a simulation.
            self.evict_held_components(message.identities)
        elif isinstance(message, HordeRestoreComponentsControlMessage):
            # Accepted on the same terms as eviction: no real cache to restore, and a fake targeted by the
            # reclaim ladder's cheaper rung must not fault the simulation.
            self.restore_held_components(message.identities)

    @override
    def cleanup_for_exit(self) -> None:
        """Give this slot's simulated VRAM back to the card and report the final state."""
        self._release_sim_vram()
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )

    def _release_sim_vram(self) -> None:
        """Return every charge this slot holds on the simulated card.

        A dying context takes its weights, its activation and the context itself with it, so a slot that
        exits and is not replaced must stop occupying the card. Without this a run that cycles processes
        walks the card's free reading down by one context per recovery and eventually strands itself on
        memory nothing holds. A slot killed outright cannot run this; its replacement clears the slot at
        startup instead.
        """
        if self._sim_vram_ledger is None:
            return
        self._sim_vram_ledger.free_own_models(self.device_index, self.process_id)
        self._sim_vram_ledger.set_context_overhead(self.device_index, self.process_id, 0.0)


class FakeSafetyProcess(HordeProcess):
    """A lightweight stand-in for ``HordeSafetyProcess`` that approves every image."""

    _fault_profile: FaultProfile
    _evals_started: int = 0

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        evaluation_delay_seconds: float = 0.0,
        fault_profile: FaultProfile | None = None,
    ) -> None:
        """Initialise the fake safety process.

        Args:
            process_id (int): The ID of the process. This is not the same as the PID.
            process_message_queue (ProcessQueue): The queue to send messages to the main process.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            disk_lock (Lock): The lock to use for disk access.
            process_launch_identifier (int): The unique identifier for this launch.
            evaluation_delay_seconds (float, optional): How long each fake evaluation takes. Defaults to 0.0.
            fault_profile (FaultProfile | None, optional): A misbehaviour script applied to the safety-eval
                path (crash on start, crash/hang on the nth evaluation, slow). Defaults to a no-op profile.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self._evaluation_delay_seconds = evaluation_delay_seconds
        self._fault_profile = fault_profile if fault_profile is not None else FaultProfile()

        if self._fault_profile.crash_on_start:
            os._exit(70)

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def get_vram_usage_mb(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_mb(self) -> int:
        """Return a fixed fake VRAM total value."""
        return 0

    @override
    def get_process_vram_stats(self) -> tuple[int, int, int, int] | None:
        """Return zeroed per-process allocator stats (allocated, reserved, peak, aimdo) so plumbing is exercised."""
        return 0, 0, 0, 0

    def _run_fake_alchemy(self, form: AlchemyFormSpec) -> None:
        """Pretend to run a CLIP-class alchemy form (caption/interrogation/nsfw)."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )
        time_start = time.time()
        if self._evaluation_delay_seconds > 0:
            time.sleep(self._evaluation_delay_seconds)

        result_payload: dict = {form.form: "a fake caption"} if form.form == "caption" else {form.form: False}
        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=GENERATION_STATE.ok,
                result_payload=result_payload,
            ),
        )
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {form.form_id}",
                job_id=form.form_id,
                is_alchemy=True,
                phase_metrics=JobPhaseMetrics(ram_used_high_water_mb=800),
            ),
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_COMPLETE,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Evaluate any safety request as safe and report back immediately."""
        if isinstance(message, HordeAlchemyControlMessage):
            self._run_fake_alchemy(message.form)
            return

        if message.control_flag in (
            HordeControlFlag.DEMOTE_SAFETY_WEIGHTS,
            HordeControlFlag.PROMOTE_SAFETY_WEIGHTS,
        ):
            # The fake holds no real weights; like the real child it answers with a fresh memory report.
            self.send_memory_report_message(include_vram=False)
            return

        if not isinstance(message, HordeSafetyControlMessage):
            logger.critical(f"Fake safety process received unexpected message type: {type(message).__name__}")
            return

        self._evals_started += 1
        profile = self._fault_profile

        if profile.crash_on_job_n == self._evals_started:
            logger.error(f"Fake safety {self.process_id} crashing on evaluation {self._evals_started} (injected)")
            os._exit(71)

        if profile.hang_after_n_jobs is not None and self._evals_started > profile.hang_after_n_jobs:
            _hang_forever(f"fake safety {self.process_id}", f"hung on evaluation {self._evals_started}")

        self.send_process_state_change_message(
            process_state=HordeProcessState.EVALUATING_SAFETY,
            info="Evaluating safety",
        )

        time_start = time.time()
        effective_delay = self._evaluation_delay_seconds * profile.delay_factor_for_ordinal(self._evals_started)
        if effective_delay > 0:
            time.sleep(effective_delay)

        self.process_message_queue.put(
            HordeSafetyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="fake safety evaluation",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                safety_evaluations=[
                    HordeSafetyEvaluation(
                        is_nsfw=False,
                        is_csam=False,
                        replacement_image_bytes=None,
                    )
                    for _ in message.images_bytes
                ],
            ),
        )

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def cleanup_for_exit(self) -> None:
        """No resources to release; report the final state like the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )


def start_fake_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
    vae_decode_semaphore: Semaphore,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    low_memory_mode: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    vram_heavy_models: bool = False,
    dry_run_skip_inference: bool = False,
    dry_run_inference_delay: float = 1.0,
    gpu_sampling_lease: ClearanceLeaseProxy | None = None,
    expect_image_models: bool = True,
    legacy_comfy_vram_unload: bool = False,
    fail_every_n: int = 0,
    fault_profile: FaultProfile | None = None,
    sim_vram_ledger: SimVramLedger | None = None,
    sim_total_vram_mb_by_device: dict[int, float] | None = None,
    sim_weights_mb: float = 0.0,
    sim_context_mb: float = 0.0,
    sim_weights_mb_by_model: dict[str, float] | None = None,
    sim_sampling_activation_mb_by_model: dict[str, float] | None = None,
) -> None:
    """Start a fake inference process.

    Signature-compatible with ``worker_entry_points.start_inference_process`` so it can
    be injected as a drop-in multiprocessing target. Memory/GPU related arguments (and
    ``expect_image_models``, which only gates the real worker's image-model presence check)
    are accepted and ignored; ``dry_run_inference_delay`` controls how long fake jobs take.
    ``gpu_sampling_lease`` is honoured rather than ignored: given a proxy, the fake runs the same clearance
    handshake the real child does around every sampling window, so a harness run exercises the parent's
    clearance controller. Passing None keeps the fake's original message sequence.
    ``fail_every_n`` makes every nth job report a faulted result (0 = never), and
    ``fault_profile`` scripts richer misbehaviour (hang, crash, drop heartbeats, slow, OOM,
    corrupt message), letting harnesses exercise the recovery paths. ``sim_vram_ledger`` (with
    ``sim_weights_mb`` / ``sim_context_mb``, or the per-model ``sim_weights_mb_by_model`` /
    ``sim_sampling_activation_mb_by_model`` tables) wires this fake to a shared simulated-VRAM ledger, so the
    card's occupancy tracks what the scenario's models and in-flight sampling actually commit and a
    ``fault_profile.post_processing_peak_mb`` drives deterministic post-processing VRAM pressure
    (stall-and-recover vs. complete) without a GPU. Without a mutable ledger,
    ``sim_total_vram_mb_by_device`` makes the fake report the harness's injected device capacity while
    keeping device usage at zero. Inject any of these with ``functools.partial`` (partials of module-level
    functions stay picklable under spawn).
    """
    enable_child_faulthandler(f"fake_inference_{process_id}")
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake inference")
    try:
        worker_process = FakeInferenceProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            inference_semaphore=inference_semaphore,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            job_delay_seconds=dry_run_inference_delay,
            gpu_sampling_lease=gpu_sampling_lease,
            fail_every_n=fail_every_n,
            fault_profile=fault_profile,
            sim_vram_ledger=sim_vram_ledger,
            sim_total_vram_mb=(sim_total_vram_mb_by_device or {}).get(device_index, 0.0),
            sim_weights_mb=sim_weights_mb,
            sim_context_mb=sim_context_mb,
            sim_weights_mb_by_model=sim_weights_mb_by_model,
            sim_sampling_activation_mb_by_model=sim_sampling_activation_mb_by_model,
        )
        worker_process.main_loop()
    except Exception as e:
        # Mirror the real entry points: a fake child that dies during startup runs with no loguru sink
        # (logger.remove() above), so without this its traceback goes nowhere and the warm worker just
        # wedges until the per-level timeout with nothing in logs/. See child_crash_capture.
        write_startup_crash(f"fake_inference_{process_id}", e)
        raise


def start_fake_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    cpu_only: bool = True,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_safety: bool = False,
    fault_profile: FaultProfile | None = None,
) -> None:
    """Start a fake safety process.

    Signature-compatible with ``worker_entry_points.start_safety_process`` so it can
    be injected as a drop-in multiprocessing target. GPU related arguments are
    accepted and ignored. ``fault_profile`` scripts misbehaviour on the safety-eval path;
    inject it with ``functools.partial`` (partials of module-level functions stay picklable).
    """
    enable_child_faulthandler(f"fake_safety_{process_id}")
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake safety")
    try:
        worker_process = FakeSafetyProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            fault_profile=fault_profile,
        )
        worker_process.main_loop()
    except Exception as e:
        # See start_fake_inference_process: leave a discoverable trace for a startup death that would
        # otherwise be silent (logger.remove() drops all sinks).
        write_startup_crash(f"fake_safety_{process_id}", e)
        raise


class FakePostProcessProcess(HordeProcess):
    """A lightweight stand-in for ``HordePostProcessProcess`` that echoes images back unchanged."""

    _fault_profile: FaultProfile
    _jobs_started: int = 0

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        post_processing_delay_seconds: float = 0.0,
        fault_profile: FaultProfile | None = None,
        sim_vram_ledger: SimVramLedger | None = None,
        sim_context_mb: float = 0.0,
    ) -> None:
        """Initialise the fake post-processing process.

        Args:
            process_id (int): The ID of the process. This is not the same as the PID.
            process_message_queue (ProcessQueue): The queue to send messages to the main process.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            disk_lock (Lock): The lock to use for disk access.
            process_launch_identifier (int): The unique identifier for this launch.
            post_processing_delay_seconds (float, optional): How long each fake job takes. Defaults to 0.0.
            fault_profile (FaultProfile | None, optional): A misbehaviour script applied to the
                post-processing path (crash on start, crash/hang on the nth job, slow). With
                ``post_processing_peak_mb`` set and a ledger wired in, each job allocates that peak against
                the simulated device and stalls (hangs silently) when it does not fit. Defaults to a
                no-op profile.
            sim_vram_ledger (SimVramLedger | None, optional): A shared simulated-VRAM ledger this process
                reports from and allocates its post-processing peaks against. Defaults to None.
            sim_context_mb (float, optional): This process's fixed CUDA-context overhead to register on the
                ledger. Defaults to 0.0.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self.process_type = HordeProcessType.POST_PROCESS
        self._post_processing_delay_seconds = post_processing_delay_seconds
        self._fault_profile = fault_profile if fault_profile is not None else FaultProfile()
        self._sim_vram_ledger = sim_vram_ledger
        if self._sim_vram_ledger is not None:
            self._sim_vram_ledger.set_context_overhead(self.device_index, self.process_id, sim_context_mb)

        if self._fault_profile.crash_on_start:
            os._exit(72)

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def get_vram_usage_mb(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_mb(self) -> int:
        """Return a fixed fake VRAM total value."""
        return 0

    @override
    def get_process_vram_stats(self) -> tuple[int, int, int, int] | None:
        """Return zeroed per-process allocator stats (allocated, reserved, peak, aimdo) so plumbing is exercised."""
        return 0, 0, 0, 0

    def _run_fake_post_processing(self, message: HordePostProcessControlMessage) -> None:
        """Pretend to post-process a job's images, echoing them back unchanged."""
        self._jobs_started += 1
        profile = self._fault_profile

        if profile.crash_on_job_n == self._jobs_started:
            logger.error(f"Fake post-process {self.process_id} crashing on job {self._jobs_started} (injected)")
            os._exit(73)

        if profile.hang_after_n_jobs is not None and self._jobs_started > profile.hang_after_n_jobs:
            _hang_forever(f"fake post-process {self.process_id}", f"hung on job {self._jobs_started}")

        self.send_process_state_change_message(
            process_state=HordeProcessState.POST_PROCESSING,
            info=f"Post-processing job {message.job_id}",
        )

        peak_mb = profile.post_processing_peak_mb
        if peak_mb is not None and self._sim_vram_ledger is not None:
            fits = simulate_post_processing_allocation(
                self._sim_vram_ledger,
                device_index=self.device_index,
                process_id=self.process_id,
                post_processing_peak_mb=float(peak_mb),
            )
            if not fits:
                _hang_forever(
                    f"fake post-process {self.process_id}",
                    f"post-processing peak {peak_mb}MB does not fit "
                    f"{self._sim_vram_ledger.device_free_mb(self.device_index):.0f}MB free",
                )
            self._sim_vram_ledger.clear_transient(self.device_index, self.process_id)

        time_start = time.time()
        effective_delay = self._post_processing_delay_seconds * profile.delay_factor_for_ordinal(self._jobs_started)
        if effective_delay > 0:
            time.sleep(effective_delay)

        self.process_message_queue.put(
            HordePostProcessResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Post-processing for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                job_image_results=[HordeImageResult(image_bytes=image_bytes) for image_bytes in message.images_bytes],
                state=GENERATION_STATE.ok,
            ),
        )

        self.send_process_state_change_message(
            process_state=HordeProcessState.POST_PROCESSING_COMPLETE,
            info=f"Finished job {message.job_id}",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _run_fake_graph_alchemy(self, form: AlchemyFormSpec) -> None:
        """Pretend to run a graph-backed alchemy form, echoing the source image back."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )
        time_start = time.time()
        if self._post_processing_delay_seconds > 0:
            time.sleep(self._post_processing_delay_seconds)

        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=GENERATION_STATE.ok,
                image_bytes=form.source_image_bytes,
            ),
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_COMPLETE,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle post-processing jobs and graph alchemy forms with echo results."""
        if isinstance(message, HordePostProcessControlMessage):
            self._run_fake_post_processing(message)
            return

        if isinstance(message, HordeAlchemyControlMessage):
            self._run_fake_graph_alchemy(message.form)
            return

        logger.critical(f"Fake post-process received unexpected message type: {type(message).__name__}")

    @override
    def cleanup_for_exit(self) -> None:
        """No resources to release; report the final state like the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )


def start_fake_post_process_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_post_processing: bool = False,
    fault_profile: FaultProfile | None = None,
    sim_vram_ledger: SimVramLedger | None = None,
    sim_context_mb: float = 0.0,
) -> None:
    """Start a fake post-processing process.

    Signature-compatible with ``worker_entry_points.start_post_process_process`` so it can be injected
    as a drop-in multiprocessing target. GPU related arguments are accepted and ignored.
    ``fault_profile`` scripts misbehaviour on the post-processing path; inject it with
    ``functools.partial`` (partials of module-level functions stay picklable).
    """
    enable_child_faulthandler(f"fake_post_process_{process_id}")
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake post-process")
    try:
        worker_process = FakePostProcessProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            fault_profile=fault_profile,
            sim_vram_ledger=sim_vram_ledger,
            sim_context_mb=sim_context_mb,
        )
        worker_process.main_loop()
    except Exception as e:
        # See start_fake_inference_process: leave a discoverable trace for a startup death that would
        # otherwise be silent (logger.remove() drops all sinks).
        write_startup_crash(f"fake_post_process_{process_id}", e)
        raise


def start_fake_vae_lane_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_vae_lane: bool = False,
) -> None:
    """Start a fake VAE lane process.

    Signature-compatible with ``worker_entry_points.start_vae_lane_process`` so it can be injected as a
    drop-in multiprocessing target. The real ``HordeVaeLaneProcess`` already runs ML-free under ``dry_run``
    (it returns plausible stand-in latent/image bytes), so the fake reuses it directly; GPU arguments are
    accepted and ignored.
    """
    from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess

    _ = (accelerator_kind, amd_gpu, directml, dry_run_skip_vae_lane)
    enable_child_faulthandler(f"fake_vae_lane_{process_id}")
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake vae lane")
    try:
        worker_process = HordeVaeLaneProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            dry_run=True,
        )
        worker_process.main_loop()
    except Exception as e:
        write_startup_crash(f"fake_vae_lane_{process_id}", e)
        raise


def start_fake_component_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    horde_model_names: list[str] | None = None,
    dry_run_skip_component_lane: bool = False,
) -> None:
    """Start a fake component lane process.

    Signature-compatible with ``worker_entry_points.start_component_process`` so it can be injected as a
    drop-in multiprocessing target. The real ``HordeComponentLaneProcess`` already runs ML-free under
    ``dry_run`` (it holds nothing and just reports ready), so the fake reuses it directly; GPU/endpoint
    arguments are accepted and ignored.
    """
    from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess

    _ = (accelerator_kind, amd_gpu, directml, horde_model_names)
    enable_child_faulthandler(f"fake_component_{process_id}")
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake component")
    try:
        worker_process = HordeComponentLaneProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            dry_run=True,
        )
        worker_process.main_loop()
    except Exception as e:
        write_startup_crash(f"fake_component_{process_id}", e)
        raise


class FakeDownloadProcess(HordeProcess):
    """A lightweight stand-in for ``HordeDownloadProcess`` that imports no ML dependencies.

    Starts from a scripted on-disk set and "downloads" any requested model (after an optional
    per-model delay) by adding it to that set, unless it is in ``fail_models``. Emits the same
    ``HordeDownloadAvailabilityMessage`` snapshots the real process does.
    """

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        scripted_present: list[str] | None = None,
        download_delay_seconds: float = 0.0,
        fail_models: list[str] | None = None,
        rate_limit_kbps: int | None = None,
        paused: bool = False,
        fault_profile: FaultProfile | None = None,
    ) -> None:
        """Initialise with a scripted present-set and download behaviour."""
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self.process_type = HordeProcessType.DOWNLOAD
        self._fault_profile = fault_profile if fault_profile is not None else FaultProfile()
        if self._fault_profile.crash_on_start:
            os._exit(70)
        self._present: set[str] = set(scripted_present or [])
        self._download_delay_seconds = download_delay_seconds
        self._fail_models = set(fail_models or [])
        self._pending: list[str] = []
        self._failed: list[str] = []
        self._currently_downloading: str | None = None
        self._paused = paused
        self._rate_limit_kbps = rate_limit_kbps if (rate_limit_kbps or 0) > 0 else None
        self._send_availability()

    def _handle_aux_prefetch_request(self, message: HordeAuxPrefetchControlMessage) -> None:
        """Report every requested auxiliary file as placed on disk.

        A simulated job's LoRAs and textual inversions are references only: the fake inference lane reads
        no files, so there is nothing to fetch and every entry succeeds. Answering is what matters, since
        a popped job carrying auxiliary references stays pending until its prefetch reports back, and the
        scripted download delay is charged once for the batch so a prefetch still costs time.
        """
        if not message.entries:
            return
        effective_delay = self._download_delay_seconds * self._fault_profile.slow_factor
        if effective_delay > 0:
            time.sleep(effective_delay)
        self.process_message_queue.put(
            HordeAuxPrefetchResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="aux prefetch complete",
                outcomes=[
                    AuxPrefetchOutcome(
                        kind=entry.kind,
                        name=entry.name,
                        is_version=entry.is_version,
                        ok=True,
                        requesting_job_ids=[entry.requesting_job_id],
                    )
                    for entry in message.entries
                ],
            ),
        )

    def _status_snapshot(self) -> DownloadStatusSnapshot:
        """Project the fake's state into the same rich snapshot the real process emits."""
        if self._currently_downloading is not None:
            phase = DownloadPhase.PAUSED if self._paused else DownloadPhase.DOWNLOADING
            current = CurrentDownloadStatus(
                model_name=self._currently_downloading,
                feature="image model",
                target_dir="models/compvis",
            )
        else:
            phase = DownloadPhase.PAUSED if self._paused and self._pending else DownloadPhase.IDLE
            current = None
        return DownloadStatusSnapshot(
            phase=phase,
            current=current,
            pending=[DownloadItem(model_name=name, feature="image model") for name in self._pending],
            failures=[
                DownloadFailure(model_name=name, feature="image model", reason="failed") for name in self._failed
            ],
            present_model_names=sorted(self._present),
            paused=self._paused,
            rate_limit_kbps=self._rate_limit_kbps,
        )

    def _send_availability(self, info: str = "download availability") -> None:
        self.process_message_queue.put(
            HordeDownloadAvailabilityMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=info,
                available_model_names=sorted(self._present),
                currently_downloading=self._currently_downloading,
                pending_downloads=list(self._pending),
                failed_downloads=list(self._failed),
                # The dry-run harness has no real safety models; report them present so the parent starts
                # the (dry-run) safety process immediately, matching the worker's pre-deferral behaviour.
                safety_models_present=True,
                safety_models_attempted=True,
                status=self._status_snapshot(),
            ),
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if message.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE:
            # The fake holds no real model managers; a reference reload is a no-op here.
            return
        if isinstance(message, HordeAuxPrefetchControlMessage):
            self._handle_aux_prefetch_request(message)
            return
        if not isinstance(message, HordeDownloadControlMessage):
            logger.warning(f"Fake download process received unexpected message: {type(message).__name__}")
            return
        if message.set_paused is not None:
            self._paused = message.set_paused
        if message.set_rate_limit_kbps is not None:
            self._rate_limit_kbps = message.set_rate_limit_kbps if message.set_rate_limit_kbps > 0 else None
        # Mirror the real process: an authoritative configured set prunes queued downloads and drops an
        # in-flight one the config no longer wants, so a model removed from config stops downloading.
        if message.desired_image_models is not None:
            desired = set(message.desired_image_models)
            self._pending = [name for name in self._pending if name in desired]
            if self._currently_downloading is not None and self._currently_downloading not in desired:
                self._currently_downloading = None
        for model_name in message.model_names:
            if model_name in self._present or model_name in self._pending:
                continue
            self._pending.append(model_name)
        self._send_availability("download request received")

    @override
    def worker_cycle(self) -> None:
        if self._paused or not self._pending:
            return
        model_name = self._pending.pop(0)
        self._currently_downloading = model_name
        self._send_availability(f"downloading {model_name}")
        effective_delay = self._download_delay_seconds * self._fault_profile.slow_factor
        if effective_delay > 0:
            time.sleep(effective_delay)
        self._currently_downloading = None
        if model_name in self._fail_models:
            self._failed.append(model_name)
        else:
            self._present.add(model_name)
        self._send_availability(f"finished {model_name}")

    @override
    def cleanup_for_exit(self) -> None:
        return


def start_fake_download_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    download_bandwidth_semaphore: Semaphore,
    process_launch_identifier: int,
    *,
    nsfw: bool = True,
    allow_lora: bool = False,
    allow_controlnet: bool = False,
    allow_sdxl_controlnet: bool = False,
    allow_post_processing: bool = True,
    purge_loras: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    rate_limit_kbps: int | None = None,
    paused: bool = False,
    max_parallel_downloads: int = 4,
    per_host_concurrency: int = 1,
    connections_per_file: int = 4,
    scripted_present: list[str] | None = None,
    download_delay_seconds: float = 0.0,
    fail_models: list[str] | None = None,
    fault_profile: FaultProfile | None = None,
) -> None:
    """Start a fake download process.

    Signature-compatible with ``worker_entry_points.start_download_process``; the worker-config
    arguments are accepted and ignored, except ``rate_limit_kbps``/``paused`` which the fake honors so
    the pause/throttle controls can be exercised. ``max_parallel_downloads``/``per_host_concurrency``/
    ``connections_per_file`` are accepted for signature parity and ignored (the fake downloads serially,
    in a single stream). ``fault_profile`` scripts
    crash-on-start and slow downloads. Inject the scripting arguments with ``functools.partial`` (partials
    of module-level functions stay picklable under spawn).
    """
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake download")
    worker_process = FakeDownloadProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
        scripted_present=scripted_present,
        download_delay_seconds=download_delay_seconds,
        fail_models=fail_models,
        rate_limit_kbps=rate_limit_kbps,
        paused=paused,
        fault_profile=fault_profile,
    )
    worker_process.main_loop()
