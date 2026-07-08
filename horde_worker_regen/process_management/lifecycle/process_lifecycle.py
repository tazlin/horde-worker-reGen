"""Manages process start, stop, replace, and hung-process detection."""

from __future__ import annotations

import contextlib
import enum
import math
import os
import sys
import time
from collections.abc import Callable
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore
from typing import TYPE_CHECKING

import psutil
from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData

from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.compute_mode import is_cpu_only_install
from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeDownloadControlMessage,
    HordeHeartbeatType,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.child_crash_capture import read_last_startup_crash
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.model_sizing import any_offered_model_wants_whole_card
from horde_worker_regen.process_management.resources.resource_budget import ram_pressure_floor_mb
from horde_worker_regen.process_management.scheduling.performance_model import (
    BatchBucket,
    ResolutionBucket,
    signature_from_job,
)
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID


class PauseOwner(enum.StrEnum):
    """Which subsystem holds a lane's off-GPU pause, so its restore path alone can clear it.

    A lane's off-GPU pause has two independent initiators that must not clear each other's hold: the whole-card
    residency (restored when the residency drains, by the completion loop) and the verified reclaim ladder
    (restored when the card's saturation episode ends, by the ladder's LIFO unwind). Recording the initiator at
    pause time lets each restore path act only on the pause it owns, so a ladder-initiated pause is never
    stranded by a residency that has no grant to complete, and a residency pause is never lifted early by the
    ladder unwinding its own rungs. Whichever initiator transitions the lane into the paused state owns it; a
    second initiator finding the lane already paused is a no-op and does not take ownership.
    """

    WHOLE_CARD = "whole_card"
    """The whole-card residency stopped the lane; its completion loop owns the restore."""
    RECLAIM_LADDER = "reclaim_ladder"
    """The verified reclaim ladder stopped the lane; the ladder's episode-end unwind owns the restore."""


SAFETY_PROCESS_ID: int = 0
"""The reserved process-map slot for the safety process, by convention always PID 0.

Inference and safety processes share one integer slot space in the process map. The (currently single)
safety process owns slot 0; inference processes are allocated from 1 upward (:meth:`_allocate_inference_pid`)
so they can never collide with it, regardless of which on-disk gate (image models vs safety models) opens
first and starts its pool. The download process lives outside the map at its own reserved id.
"""

CRASH_LOOP_WINDOW_SECONDS: float = 300.0
"""Sliding window over which an inference slot's replacements are counted for crash-loop detection."""

CRASH_LOOP_MAX_REPLACEMENTS: int = 3
"""Replacements of a single slot within ``CRASH_LOOP_WINDOW_SECONDS`` before it is quarantined.

A slot that dies (or hangs) faster than it can do useful work is not worth respawning indefinitely:
each restart costs a model (re)load and starves the worker. Past this count the slot is quarantined
(left out of the pool) and the lost capacity is surfaced as a severity signal for the higher-level
recovery supervisor rather than papered over by an unbounded respawn loop.
"""

CRASH_LOOP_MAX_START_FAILURES: int = 3
"""Consecutive replacements while still in ``PROCESS_STARTING`` before a slot is quarantined.

This is the rate-independent companion to the sliding-window breaker above. A slot that *never*
advances past ``PROCESS_STARTING`` before dying has not proven it can initialise at all (a broken
dependency, a missing model, an import error). Such a failure is deterministic, so each restart costs
the full, slow cold-start before failing again, and if that cold-start is slower than
``CRASH_LOOP_WINDOW_SECONDS / CRASH_LOOP_MAX_REPLACEMENTS`` the window breaker can never accumulate
enough replacements *within the window* to trip (the early ones age out), so the slot would respawn
forever. Counting consecutive start-failures regardless of spacing catches exactly that case. The
streak resets the moment the slot reaches any later state (it did initialise, then failed differently).
"""

SAFETY_CRASH_LOOP_MAX: int = 3
"""Safety-pool replacements within ``CRASH_LOOP_WINDOW_SECONDS`` before the pool is reported as failing.

The crash-loop circuit breaker quarantines individual *inference* slots, but the safety pool has no such
per-slot breaker. This count is the equivalent signal for safety: a pool that has been rebuilt more than
this many times in the window is failing (e.g. a safety process that crashes on every start), which the
recovery supervisor escalates rather than rebuilding the pool forever."""

MODEL_LOAD_FAILURE_WINDOW_SECONDS: float = 600.0
"""Sliding window over which a single model's load failures are counted for quarantine."""

MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD: int = 3
"""Load failures of one model within ``MODEL_LOAD_FAILURE_WINDOW_SECONDS`` before it is quarantined.

A model that faults the backend every time it is loaded (an unsupported/corrupt checkpoint) is poison: the
slot crash-loop breaker keys on the *slot*, not the model, so re-dispatching the model round-robin across
fresh slots burns the whole pool down without any single slot tripping its breaker. Past this count the
model itself is taken out of rotation (its queued jobs faulted for reissue, further preloads skipped) so
one bad model can no longer cascade into a pool-wide recovery storm. The window is wider than the slot
breaker's because a poison model surfaces once per job dispatch, not once per fast respawn."""

SLOWDOWN_NOTICE_RATIO: float = 2.0
"""Sampling time past this multiple of the job's expected time logs a soft notice (rung 1 of the ladder)."""

SLOWDOWN_WARN_RATIO: float = 4.0
"""Sampling time past this multiple warns, audits, and counts toward the recovery-supervisor severity.

The hard kill remains the ``inference_step_timeout`` in :meth:`replace_hung_processes`; these softer,
evidence-based rungs sit below it so a measurable slowdown is logged before the slot is replaced."""

SLOWDOWN_WARN_LEVEL: int = 2
"""The ``current_job_slowdown_level`` value the WARN rung sets (``SLOWDOWN_WARN_RATIO`` reached).

The paged-slowdown watchdog only acts at this rung, never at the softer NOTICE rung (level 1)."""

WDDM_PAGING_VICTIM_MAX_AGE_SECONDS: float = 5.0
"""How recently the parent must have attributed WDDM paging to a slot for the paged-slowdown watchdog to
act on it.

The paging verdict and the hung-process watchdog run in the same control-loop tick (a handful of ~0.2s
loop-interval sleeps apart), and the verdict itself only fires after two consecutive elevated telemetry
samples, so a few seconds comfortably spans the freshest verdict plus normal sampling jitter. Kept short
so that a paging episode which has genuinely cleared (or telemetry that has stopped arriving) ages out
before it can drive a kill: the corroboration must be current, not a stale memory of past pressure."""

SATURATION_KILL_MIN_SECONDS: float = 10.0
"""How long a card must have been continuously SATURATED, with the reclaim ladder exhausted, before a
crawling sampler on it is replaced as the last reclaim rung.

The kill is the terminal rung of the reclaim ladder, not a first response: it fires only once the governor
has called the card SATURATED for this long without interruption AND the verified ladder has run every
softer rung (idle-model unload, cache release, lane pause, safety off-GPU) without relieving the card. Ten
seconds spans several governor samples plus the ladder's own per-rung verification windows, so a card that
any softer rung could still rescue is never killed; only a card genuinely wedged over the cliff reaches it."""

STEP_TIMEOUT_WORK_FACTOR: float = 2.0
"""Per-step hang grace scales with this multiple of a job's expected sampling time when no other widening
applies. A heavier job (more steps, larger pixels, hires) has longer legitimate heartbeat-silent stretches
than a light one, so the budget tracks its expected work, floored at ``inference_step_timeout`` and capped
at ``contended_step_timeout`` so a light job stays tight and a genuine wedge is still reaped."""

FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS: float = 60.0
"""Shortened stuck-aux-download grace applied while a LoRA-download backoff is active.

The first stall is reaped at the configured ``download_timeout`` and registers a backoff strike; once
the download path is known to be failing there is no value in letting a *requeued* job ride the full
window again, so further stalls are reaped at this much shorter grace (floored against an
unusually-low configured ``download_timeout``). This is the fast-fault half of the backoff: it bounds
how long a job that keeps timing out can hold a slot, instead of burning the full window per attempt."""

AUX_DOWNLOAD_DEADLINE_MARGIN_SECONDS: float = 15.0
"""How far before the stuck-aux watchdog the child's own aux-download deadline is set.

The parent hands each dispatched job a deadline of (its effective aux-download watchdog timeout minus
this margin); the child cancels a stalled download and faults the job *itself* a beat before the
watchdog would tear the whole process down, turning a teardown+respawn into a slot-local fault. The
margin covers the round trip: child cancel -> faulted result -> parent frees the slot, all before the
next watchdog pass."""

MIN_AUX_DOWNLOAD_DEADLINE_SECONDS: float = 10.0
"""Floor on the child-side aux-download deadline, so an unusually low configured ``download_timeout``
cannot drive it to zero (which would fault every LoRA job before it could fetch anything)."""


def _job_is_feature_heavy(process_info: HordeProcessInfo) -> bool:
    """Whether the slot's current job carries features that lengthen its heartbeat-silent work.

    ControlNet (an aux model plus a heavier graph), hires-fix (a whole second sampling pass), batching,
    and a large output resolution each raise a job's per-step wall time and add non-sampling phases (the
    hires pass and VAE decode emit no ``INFERENCE_STEP`` beat). The perf-model signature already buckets
    every one of these, so we read them from it rather than re-deriving from the payload. Returns False
    when the job cannot be characterised (no baseline / malformed payload).
    """
    job = process_info.last_job_referenced
    if job is None:
        return False
    signature = signature_from_job(job, process_info.loaded_horde_model_baseline)
    if signature is None:
        return False
    return (
        signature.has_controlnet
        or signature.has_hires_fix
        or signature.batch_bucket != BatchBucket.SINGLE
        or signature.resolution_bucket in (ResolutionBucket.LARGE, ResolutionBucket.HUGE)
    )


class ProcessLifecycleManager:
    """Owns process start/stop/replace logic and related state."""

    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _job_tracker: JobTracker
    _process_message_queue: ProcessQueue
    _card_runtimes: dict[int, CardRuntime]
    _disk_lock: Lock_MultiProcessing
    _aux_model_lock: Lock_MultiProcessing
    _download_bandwidth_semaphore: Semaphore
    _gpu_sampling_lease_enabled: bool
    _runtime_config: RuntimeConfig
    _max_inference_processes: int
    _max_safety_processes: int
    _amd_gpu: bool
    _directml: int | None
    _abort_callback: Callable[[], None]
    _state: WorkerState
    _entry_points: ProcessEntryPoints
    _download_process_info: HordeProcessInfo | None
    _owned_registry: OwnedProcessRegistry | None
    _action_ledger: ActionLedger

    num_processes_launched: int
    _num_process_recoveries: int
    _safety_processes_should_be_replaced: bool
    _safety_processes_ending: bool
    _post_process_processes_should_be_replaced: bool
    _post_process_processes_ending: bool
    _recently_recovered: bool
    _hung_processes_detected: bool
    _hung_processes_detected_time: float
    _slot_recovery_history: dict[int, list[float]]
    _slot_consecutive_start_failures: dict[int, int]
    _quarantined_inference_slots: set[int]
    _num_slots_quarantined: int
    _safety_recovery_history: list[float]
    _model_load_failure_history: dict[str, list[float]]
    _quarantined_models: set[str]
    _recent_load_failure_by_process: dict[int, tuple[str, float]]

    def __init__(
        self,
        *,
        ctx: BaseContext,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        job_tracker: JobTracker,
        process_message_queue: ProcessQueue,
        card_runtimes: dict[int, CardRuntime],
        disk_lock: Lock_MultiProcessing,
        aux_model_lock: Lock_MultiProcessing,
        download_bandwidth_semaphore: Semaphore,
        gpu_sampling_lease_enabled: bool = False,
        runtime_config: RuntimeConfig,
        max_safety_processes: int,
        amd_gpu: bool,
        directml: int | None,
        abort_callback: Callable[[], None],
        state: WorkerState,
        entry_points: ProcessEntryPoints | None = None,
        owned_registry: OwnedProcessRegistry | None = None,
        action_ledger: ActionLedger | None = None,
        wddm_paging_victims_provider: Callable[[float], dict[int, float]] | None = None,
        device_saturation_duration_provider: Callable[[int], float] | None = None,
        saturation_unresolved_provider: Callable[[int], bool] | None = None,
    ) -> None:
        """Initialize with shared references and callbacks from the parent manager."""
        # All child processes are created from this context (ctx.Process/ctx.Pipe), not the bare
        # multiprocessing.* helpers. Those resolve against the process-global default start method,
        # which on POSIX is 'fork'; a forked child of a parent that has initialized CUDA dies with
        # "Cannot re-initialize CUDA in forked subprocess". Pinning the spawn context here makes the
        # worker correct regardless of how it was launched, instead of depending on an earlier
        # set_start_method('spawn') side effect having run in this interpreter.
        self._ctx = ctx
        # typeshed declares Process only on the concrete context subclasses, not on BaseContext;
        # the spawn context we are handed has it at runtime. Bind it once as a typed factory so the
        # call sites stay statically checked (returning BaseProcess) without scattered ignores.
        self._new_process: Callable[..., BaseProcess] = ctx.Process  # pyrefly: ignore[missing-attribute]
        self._validate_spawn_start_method()
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_message_queue = process_message_queue
        self._card_runtimes = card_runtimes
        self._disk_lock = disk_lock
        self._aux_model_lock = aux_model_lock
        self._download_bandwidth_semaphore = download_bandwidth_semaphore
        self._gpu_sampling_lease_enabled = gpu_sampling_lease_enabled
        self._runtime_config = runtime_config
        # Total inference processes across all driven cards. Single-GPU = that one card's count, identical
        # to the old global value.
        self._max_inference_processes = sum(card.target_process_count for card in card_runtimes.values())
        self._max_safety_processes = max_safety_processes
        self._amd_gpu = amd_gpu
        self._directml = directml
        self._abort_callback = abort_callback
        self._state = state
        self._entry_points = entry_points if entry_points is not None else ProcessEntryPoints()
        self._owned_registry = owned_registry
        # The ledger is always present (an in-memory ring by default) so diagnostics work under test;
        # the parent manager injects a file-backed one in a real run.
        self._action_ledger = action_ledger if action_ledger is not None else ActionLedger()
        # Returns the fresh WDDM paging-victim map (os_pid -> shared MB) from the scheduler, or an empty
        # map when the parent has no current paging attribution. Injected lazily (the scheduler is built
        # after this manager), so a bare default here keeps the manager usable standalone in tests.
        self._wddm_paging_victims_provider: Callable[[float], dict[int, float]] = (
            wddm_paging_victims_provider if wddm_paging_victims_provider is not None else (lambda _max_age: {})
        )
        # How long a card has been continuously SATURATED (device-free below the hard floor), in seconds, and
        # whether the verified reclaim ladder has exhausted itself on that card without relieving it. These are
        # the two device-level gates the last-rung kill reads: they replace the old per-PID paging-victim gate,
        # which the measured LRU physics made structurally unsatisfiable (the demoted pid and the crawling pid
        # differ). Injected lazily by the parent (governor and ladder are built after this manager); bare
        # defaults keep the manager usable standalone in tests.
        self._device_saturation_duration_provider: Callable[[int], float] = (
            device_saturation_duration_provider
            if device_saturation_duration_provider is not None
            else (lambda _device_index: 0.0)
        )
        self._saturation_unresolved_provider: Callable[[int], bool] = (
            saturation_unresolved_provider
            if saturation_unresolved_provider is not None
            else (lambda _device_index: False)
        )

        self.num_processes_launched = 0
        self._num_process_recoveries = 0
        self._num_slowdown_events = 0
        # Count of inference slots replaced as the reclaim ladder's last rung (a crawling sampler on a card
        # that has been SATURATED past the kill horizon with every softer reclaim rung exhausted). Surfaced on
        # the run-metrics snapshot as ``paging_victim_replacements`` (the counter's name is retained; it now
        # counts last-rung replacements rather than per-PID paging-victim matches).
        self._paging_victim_replacements = 0
        self._safety_processes_should_be_replaced = False
        self._safety_processes_ending = False
        self._post_process_processes_should_be_replaced = False
        self._post_process_processes_ending = False
        self._post_process_results_known_lost: set[GenerationID] = set()
        self._component_processes_should_be_replaced = False
        self._component_processes_ending = False
        self._vae_lane_processes_should_be_replaced = False
        self._vae_lane_processes_ending = False
        # Runtime override forcing the safety process off-GPU (cpu_only) even when safety_on_gpu is configured.
        # Set while a whole-card (single-residency) job claims the device, so the safety process's CUDA
        # context (only reclaimable by the process exiting) is freed for the heavy model. Restored after.
        self._safety_gpu_paused = False
        self._safety_gpu_pause_count = 0
        self._safety_gpu_restore_count = 0
        # The driven card the safety process should be pinned to when it (re)spawns on-GPU, chosen by the
        # scheduler's headroom-aware placement identity and pushed here each control cycle. None (the default
        # and the only value on a single-GPU host) means "the lowest-index driven card", byte-identical to the
        # historical fixed pin. The re-promotion path respawns through the same bring-up, so honouring this at
        # spawn is all that is needed for both first bring-up and every restore to land on the chosen card.
        self._desired_safety_card: int | None = None
        # The device the currently-running on-GPU safety process is pinned to, recorded at its last spawn.
        # None whenever safety came up cpu_only. Read as the truthful "which card is safety on" signal so a
        # whole-card residency on a different card never needlessly evicts safety from a card it does not share.
        self._safety_pinned_card: int | None = None
        # Marks the *next* safety-pool rebuild as an intentional whole-card pause/restore cycle, so its
        # completion is not counted as a crash recovery and does not feed the safety crash-loop breaker.
        # Without this, repeated whole-card jobs cycling safety off/on read as a safety crash loop and trip
        # save-our-ship. Mirrors the intentional_reclaim path for inference RAM reclaim.
        self._safety_replacement_intentional = False
        # Runtime override forcing the dedicated post-processing lane off the GPU (stopped, not merely
        # model-unloaded) while a whole-card (single-residency) model claims the device. Unlike the safety
        # process, which cycles cpu_only, the post-processing lane has no useful CPU fallback (upscalers on
        # CPU are impractically slow), so the whole lane is stopped: its CUDA context AND any warm upscaler
        # models are freed for the heavy model. On a tight card (Flux ~11.5GB weights + ~3GB activation on a
        # 16GB card) the lane's ~1.4GB bare context alone tips the head into host-RAM weight streaming, so it
        # must vacate the card exactly as safety does. The lane is restarted after the residency restores;
        # in-flight post-processing work is recorded known-lost so the recovery coordinator requeues it
        # (bounded), then reports a no-image fault if the lane cannot return a result.
        self._post_process_gpu_paused = False
        # Which initiator (whole-card residency or the reclaim ladder) holds the current pause, so only its
        # restore path clears it; None while the lane is not paused. See :class:`PauseOwner`.
        self._post_process_pause_owner: PauseOwner | None = None
        self._post_process_gpu_pause_count = 0
        self._post_process_gpu_restore_count = 0
        # Marks the *next* post-processing-lane rebuild as an intentional whole-card pause/restore cycle, so
        # its completion is not counted as a crash recovery (mirrors ``_safety_replacement_intentional``).
        self._post_process_replacement_intentional = False
        # The dedicated VAE lane (the disaggregated pipeline's VAE-encode/decode stage) mirrors the
        # post-processing lane's whole-card off-GPU handling: its permanent CUDA context is real device-wide
        # VRAM only reclaimed by the process exiting, so a whole-card model stops the lane outright rather
        # than merely unloading its models. In-flight VAE stages are re-dispatched from held state by the
        # disaggregation orchestrator (it owns per-stage recovery), so no parent-side known-lost set is kept.
        self._vae_lane_gpu_paused = False
        self._vae_lane_pause_owner: PauseOwner | None = None
        self._vae_lane_gpu_pause_count = 0
        self._vae_lane_gpu_restore_count = 0
        # Marks the *next* VAE-lane rebuild as an intentional whole-card pause/restore cycle (mirrors
        # ``_post_process_replacement_intentional``), so its completion is not counted as a crash recovery.
        self._vae_lane_replacement_intentional = False
        # The dedicated component lane (the disaggregated pipeline's text-encode service) mirrors the VAE
        # lane's whole-card off-GPU handling: its permanent CUDA context and resident text encoders are
        # real device-wide VRAM only reclaimed by the process exiting, so a whole-card model stops the lane
        # outright rather than leaving its context resident. The lane serves no jobs, so no in-flight work is
        # accounted; disaggregation consumers that lose the producer fall back to loading their own copies,
        # and dispatch demotes to monolithic while the lane is down (its liveness gates disagg eligibility).
        self._component_gpu_paused = False
        self._component_pause_owner: PauseOwner | None = None
        self._component_gpu_pause_count = 0
        self._component_gpu_restore_count = 0
        # Marks the *next* component-lane rebuild as an intentional whole-card pause/restore cycle (mirrors
        # ``_vae_lane_replacement_intentional``), so its completion is not counted as a crash recovery.
        self._component_lane_replacement_intentional = False
        self._recently_recovered = False
        self._hung_processes_detected = False
        self._hung_processes_detected_time = 0.0
        self._any_replaced = False
        self._on_process_recovery: Callable[[HordeProcessInfo, str], None] | None = None
        self._download_process_info = None
        self._slot_recovery_history = {}
        self._slot_consecutive_start_failures = {}
        self._quarantined_inference_slots = set()
        self._num_slots_quarantined = 0
        self._safety_recovery_history = []
        self._model_load_failure_history = {}
        self._quarantined_models = set()
        self._recent_load_failure_by_process = {}

        self._print_config()

    def _print_config(self) -> None:
        """Emit a structured snapshot of the current configuration, so a future hang explains itself."""
        logger.info(
            f"ProcessLifecycleManager config: max_inference_processes={self._max_inference_processes}, "
            f"max_safety_processes={self._max_safety_processes}, "
            f"gpu_sampling_lease_enabled={self._gpu_sampling_lease_enabled}, "
            f"amd_gpu={self._amd_gpu}, directml={self._directml}, "
            f"runtime_config={self._runtime_config}"
        )

    def _validate_spawn_start_method(self) -> None:
        """Fail loudly if children would be created with a CUDA-unsafe start method.

        Inference/safety children import torch and ComfyUI, which initialize CUDA at import time. On
        POSIX the default start method is ``fork``; a forked child of a CUDA-initialized parent dies
        with "Cannot re-initialize CUDA in forked subprocess". The context is pinned to spawn at
        construction, so a non-spawn method here means a misconfigured launcher handed us the wrong
        context: surface it instead of crash-looping every child.
        """
        if sys.platform == "win32":
            return

        method = None

        try:
            method = self._ctx.get_start_method()
        except Exception:
            # A test double (Mock context) has no real start method; nothing to validate.
            return

        if not isinstance(method, str) or method == "spawn":
            return

        message = (
            f"Child processes would be created with the '{method}' start method, but CUDA requires "
            "'spawn'. Forked children of a CUDA-initialized parent raise 'Cannot re-initialize CUDA "
            "in forked subprocess'. Launch via the standard entry points so the spawn context is set."
        )
        logger.critical(message)
        if not os.environ.get("AI_HORDE_TESTING"):
            raise RuntimeError(message)

    def set_process_recovery_observer(self, observer: Callable[[HordeProcessInfo, str], None]) -> None:
        """Register a callback invoked with the process info and a reason on each recovery.

        Used by the run-metrics aggregator to record crash/hang events.
        """
        self._on_process_recovery = observer

    def _notify_process_recovery(self, process_info: HordeProcessInfo, reason: str) -> None:
        if self._on_process_recovery is None:
            return
        try:
            self._on_process_recovery(process_info, reason)
        except Exception as e:
            logger.warning(f"Process recovery observer failed: {type(e).__name__} {e}")

    @property
    def recently_recovered(self) -> bool:
        """Whether a process was recently recovered (read-only for manager)."""
        return self._recently_recovered

    @property
    def num_slots_quarantined(self) -> int:
        """How many inference slots the crash-loop circuit breaker has taken out of the pool."""
        return self._num_slots_quarantined

    @property
    def quarantined_inference_slots(self) -> frozenset[int]:
        """The process ids of inference slots currently quarantined (read-only)."""
        return frozenset(self._quarantined_inference_slots)

    @property
    def download_process_info(self) -> HordeProcessInfo | None:
        """The background download process, or None if one is not running."""
        return self._download_process_info

    @property
    def action_ledger(self) -> ActionLedger:
        """The self-audited record of lifecycle actions taken on child processes (read-only)."""
        return self._action_ledger

    _VRAM_MATERIALIZING_PROCESS_TYPES = (
        HordeProcessType.INFERENCE,
        HordeProcessType.SAFETY,
        HordeProcessType.POST_PROCESS,
        HordeProcessType.VAE_LANE,
        HordeProcessType.COMPONENT,
    )
    """GPU-context process types whose spawn is a VRAM-materializing event for LIFO reclaim ranking."""

    def _register_owned(self, process_info: HordeProcessInfo) -> None:
        """Record a just-started child in the action ledger and the owned-PID registry.

        A GPU-context process's spawn is also a VRAM-materializing event (its CUDA context forms on the card),
        so its monotonic materialization stamp is set here for the reclaim ladder's LIFO ranking; a model that
        later reports LOADED_IN_VRAM restamps it. The download process is not a GPU context and is skipped.
        """
        if process_info.process_type in self._VRAM_MATERIALIZING_PROCESS_TYPES:
            process_info.vram_materialized_monotonic = time.monotonic()
        self._action_ledger.record(
            LedgerEventType.PROCESS_SPAWNED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            detail={"process_type": process_info.process_type.name},
        )
        if self._owned_registry is not None:
            self._owned_registry.record(
                os_pid=process_info.os_pid,
                launch_identifier=process_info.process_launch_identifier,
                process_type=process_info.process_type.name,
            )

    def _log_recovery_diagnostics(self, process_info: HordeProcessInfo, reason: str) -> None:
        """Emit a structured snapshot of why a process is being recovered, so a future hang explains itself.

        Pulls together the OS identity, the last state and heartbeat the parent saw, the child's exit
        code (if it died), and the slot's recent ledger history into one log line. Called at the moment
        of replacement, while ``process_info`` still reflects the faulted state.
        """
        now = time.time()
        exitcode: int | None = None
        with contextlib.suppress(Exception):
            exitcode = process_info.mp_process.exitcode

        recent = self._action_ledger.recent(process_id=process_info.process_id, limit=10)
        recent_summary = "; ".join(
            f"{event.event_type.name}@-{now - event.timestamp:.1f}s" + (f"({event.reason})" if event.reason else "")
            for event in recent
        )
        last_job = process_info.last_job_referenced
        logger.error(
            f"Recovery diagnostics for process {process_info.process_id} (os_pid={process_info.os_pid}, "
            f"launch={process_info.process_launch_identifier}): reason='{reason}'; "
            f"last_state={process_info.last_process_state.name}; exitcode={exitcode}; "
            f"last_heartbeat_type={process_info.last_heartbeat_type.name}; "
            f"since_last_heartbeat={now - process_info.last_heartbeat_timestamp:.1f}s; "
            f"since_last_message={now - process_info.last_received_timestamp:.1f}s; "
            f"last_job={last_job.id_ if last_job is not None else None}; recent_actions=[{recent_summary}]",
        )

    def _forget_owned(self, process_info: HordeProcessInfo) -> None:
        """Drop a cleanly-ended child from the owned-PID registry (no-op if reaping is disabled)."""
        if self._owned_registry is not None:
            self._owned_registry.forget(process_info.os_pid)

    def kill_owned_children(self) -> list[int]:
        """Best-effort kill of every still-owned child by OS pid; for atexit / signal cleanup.

        Identity is re-verified per pid inside the registry, so a reused pid is never killed. Returns
        the pids actually killed. No-op (empty list) when orphan reaping is disabled.
        """
        if self._owned_registry is None:
            return []
        return self._owned_registry.kill_all_owned()

    def start_safety_processes(self) -> None:
        """Start all the safety processes configured to be used."""
        bridge_data = self._runtime_config.bridge_data
        num_processes_to_start = self._max_safety_processes - self._process_map.num_safety_processes()

        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_safety_processes()} safety processes running, but "
                f"max_safety_processes is set to {self._max_safety_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        for _ in range(num_processes_to_start):
            # By convention the safety process owns the reserved slot ``SAFETY_PROCESS_ID`` (0); inference
            # processes are allocated from 1 upward, so the two never collide no matter which pool starts
            # first. A hypothetical second safety process (not used today) falls back to the lowest free
            # inference-range slot rather than re-taking 0.
            pid = (
                SAFETY_PROCESS_ID if self._process_map.num_safety_processes() == 0 else self._allocate_inference_pid()
            )
            pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)

            # A CPU-only torch build has no CUDA device to pin to, so the safety process must come up
            # off-GPU regardless of config (otherwise loading the safety models on "cuda" raises). The
            # runtime pause likewise overrides the configured placement: while a whole-card job holds the
            # device, the safety process must come up off-GPU so it does not re-take a CUDA context.
            cpu_only = is_cpu_only_install() or (not bridge_data.safety_on_gpu) or self._safety_gpu_paused

            # When the safety model runs on-GPU it lives on the scheduler-chosen card (the driven card with the
            # most verified headroom net of its expected sampling peak); its mask_kind is None on a default
            # single-GPU host (so no pin, byte-identical) and set on a masked multi-GPU host. Absent a chosen
            # card (single-GPU, or before the scheduler has placed one) this is the lowest-index driven card,
            # the historical fixed pin.
            desired_card = self._desired_safety_card
            if desired_card is not None and desired_card in self._card_runtimes:
                safety_card = self._card_runtimes[desired_card]
            else:
                safety_card = self._card_runtimes[min(self._card_runtimes)]
            self._safety_pinned_card = None if cpu_only else safety_card.device_index

            process = self._new_process(
                target=self._entry_points.safety_entry_point,
                args=(
                    pid,
                    self._process_message_queue,
                    child_pipe_connection,
                    self._disk_lock,
                    self.num_processes_launched,
                    cpu_only,
                ),
                kwargs={
                    "device_index": safety_card.device_index,
                    "accelerator_kind": safety_card.mask_kind,
                    "amd_gpu": self._amd_gpu,
                    "directml": self._directml,
                    "dry_run_skip_safety": bridge_data.dry_run_skip_safety,
                    "comfy_smart_memory": bridge_data.comfy_smart_memory,
                },
            )

            process.start()

            self._process_map[pid] = HordeProcessInfo(
                mp_process=process,
                pipe_connection=pipe_connection,
                process_id=pid,
                process_type=HordeProcessType.SAFETY,
                last_process_state=HordeProcessState.PROCESS_STARTING,
                process_launch_identifier=self.num_processes_launched,
            )
            self._register_owned(self._process_map[pid])

            logger.info(f"Started safety process (id: {pid})")
            self.num_processes_launched += 1

    def post_process_lane_enabled(self) -> bool:
        """Whether the dedicated post-processing lane should be running.

        Pipeline disaggregation runs its VAE stages on the dedicated VAE lane (see
        :meth:`start_vae_lane_processes`), not this lane, so the post-processing lane follows its own
        configuration flag and is not forced on by disaggregation.
        """
        return self._runtime_config.bridge_data.post_processing_lane_enabled

    def _post_process_card(self) -> CardRuntime:
        """Return the card the dedicated post-processing lane is pinned to.

        The lane avoids sharing a card with an on-GPU safety context when another card exists; otherwise
        it takes the first configured card.
        """
        ordered_cards = [self._card_runtimes[index] for index in sorted(self._card_runtimes)]
        bridge_data = self._runtime_config.bridge_data
        safety_holds_first_card = (
            bridge_data.safety_on_gpu and not is_cpu_only_install() and not self._safety_gpu_paused
        )
        if safety_holds_first_card and len(ordered_cards) > 1:
            return ordered_cards[1]
        return ordered_cards[0]

    def post_process_lane_card_index(self) -> int:
        """Return the device index the dedicated post-processing lane is (or would be) pinned to."""
        return self._post_process_card().device_index

    def start_post_process_processes(self) -> None:
        """Start the dedicated post-processing process, if enabled and not already running."""
        if not self.post_process_lane_enabled():
            return

        # While a whole-card model holds the card the lane is deliberately kept off-GPU: this per-tick start
        # hook must not resurrect it until the residency restores and clears the pause.
        if self._post_process_gpu_paused:
            return

        if self._process_map.num_post_process_processes() > 0:
            return

        bridge_data = self._runtime_config.bridge_data
        pid = self._allocate_inference_pid()
        pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)

        lane_card = self._post_process_card()

        process = self._new_process(
            target=self._entry_points.post_process_entry_point,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                self._disk_lock,
                self.num_processes_launched,
            ),
            kwargs={
                "device_index": lane_card.device_index,
                "accelerator_kind": lane_card.mask_kind,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "dry_run_skip_post_processing": bridge_data.dry_run_skip_post_processing,
                "comfy_smart_memory": bridge_data.comfy_smart_memory,
            },
        )

        process.start()

        self._process_map[pid] = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.POST_PROCESS,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            device_index=lane_card.device_index,
        )
        self._register_owned(self._process_map[pid])

        logger.info(f"Started post-process process (id: {pid}, device_index: {lane_card.device_index})")
        self.num_processes_launched += 1

    def end_post_process_processes(self) -> None:
        """End any dedicated post-processing processes."""
        for process_info in self._process_map.get_stoppable_post_process_processes():
            process_info.end_intended = True
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
            self._process_map.on_process_ending(process_id=process_info.process_id)
            self._forget_owned(process_info)

            logger.info(f"Ended post-process process {process_info.process_id}")

    def _initiate_post_process_replacement(self) -> None:
        """Flag the post-processing lane for replacement so the control loop's state machine restarts it."""
        self._post_process_processes_should_be_replaced = True

    def _initiate_component_replacement(self) -> None:
        """Flag the component lane for replacement so the control loop's state machine restarts it.

        Set when the lane is found dead (``_reap_if_crashed``). Consumers that lose the producer fall back to
        loading their own copies, so a lane crash degrades sharing rather than faulting jobs; respawning
        restores the dedup.
        """
        self._component_processes_should_be_replaced = True

    def take_post_process_results_known_lost(self) -> set[GenerationID]:
        """Return and clear the jobs whose post-processing result was lost to a lane replacement.

        A single lane serves every post-processing job, so tearing it down positively loses the result of
        any job mid-flight on it; the recovery coordinator drains this set to requeue those jobs at once
        rather than waiting out the orphan watchdog's grace.
        """
        lost = self._post_process_results_known_lost
        self._post_process_results_known_lost = set()
        return lost

    def _replace_all_post_process_process(self) -> None:
        """Replace the dedicated post-processing process across control-loop ticks.

        Mirrors the safety replacement state machine: enter the ending phase, wait for the old process to
        drain out of the map, then start a fresh one. Entering the ending phase unconditionally covers a
        process that died while still PROCESS_STARTING (never "loaded"), the same startup-crash wedge the
        safety flow guards against.
        """
        if not self._post_process_processes_should_be_replaced:
            return

        if not self._post_process_processes_ending:
            self._post_process_processes_ending = True
            # The lane being torn down positively loses any in-flight result; record those jobs so the
            # recovery coordinator requeues them immediately instead of after the orphan grace.
            self._post_process_results_known_lost.update(
                job_info.sdk_api_job_info.id_
                for job_info in self._job_tracker.jobs_being_post_processed
                if job_info.sdk_api_job_info.id_ is not None
            )
            if self._process_map.num_loaded_post_process_processes() > 0:
                self.end_post_process_processes()
            return

        if (
            self._process_map.num_loaded_post_process_processes() == 0
            and self._process_map.num_post_process_processes() > 0
        ):
            self._process_map.delete_post_process_processes()

        if (
            self._post_process_processes_ending
            and self._process_map.num_loaded_post_process_processes() == 0
            and self._process_map.num_post_process_processes() == 0
        ):
            self.start_post_process_processes()
            self._post_process_processes_ending = False
            self._post_process_processes_should_be_replaced = False
            if self._post_process_replacement_intentional:
                # A deliberate whole-card pause (or its restore), not a lane crash: keep it out of the
                # recovery count so a burst of whole-card jobs cycling the lane is not read as a crash loop.
                self._post_process_replacement_intentional = False
            else:
                self._num_process_recoveries += 1

    @property
    def post_process_processes_should_be_replaced(self) -> bool:
        """Whether the dedicated post-processing lane is flagged for replacement."""
        return self._post_process_processes_should_be_replaced

    def _component_lane_enabled(self) -> bool:
        """Whether the dedicated text-encode service should run.

        Gated on pipeline disaggregation being enabled: the service is the encode stage of the
        disaggregated pipeline (it produces CONDITIONING for the UNet-only samplers), so it is spawned
        exactly when disaggregation is on.
        """
        return self._runtime_config.bridge_data.enable_pipeline_disaggregation

    def start_component_processes(self) -> None:
        """Start the dedicated text-encode service, if disaggregation is enabled and it is not running."""
        if not self._component_lane_enabled():
            return

        # While a whole-card model holds the card the lane is deliberately kept off-GPU: this per-tick start
        # hook must not resurrect it until the residency restores and clears the pause.
        if self._component_gpu_paused:
            return

        if self._process_map.num_component_processes() > 0:
            return

        bridge_data = self._runtime_config.bridge_data
        pid = self._allocate_inference_pid()
        pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)
        lane_card = self._post_process_card()

        process = self._new_process(
            target=self._entry_points.component_entry_point,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                self._disk_lock,
                self.num_processes_launched,
            ),
            kwargs={
                "device_index": lane_card.device_index,
                "accelerator_kind": lane_card.mask_kind,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "horde_model_names": list(bridge_data.image_models_to_load),
                "dry_run_skip_component_lane": bridge_data.dry_run_skip_inference,
                "comfy_smart_memory": bridge_data.comfy_smart_memory,
            },
        )
        process.start()

        self._process_map[pid] = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.COMPONENT,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            device_index=lane_card.device_index,
        )
        self._register_owned(self._process_map[pid])

        logger.info(f"Started component lane (id: {pid}, device_index: {lane_card.device_index})")
        self.num_processes_launched += 1

    def end_component_processes(self) -> None:
        """End the dedicated component lane process (its publications are withdrawn as it exits)."""
        for process_info in self._process_map.get_stoppable_component_processes():
            process_info.end_intended = True
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
            self._process_map.on_process_ending(process_id=process_info.process_id)
            self._forget_owned(process_info)
            logger.info(f"Ended component lane {process_info.process_id}")

    def _replace_all_component_process(self) -> None:
        """Replace the component lane across control-loop ticks (mirrors the post-processing state machine).

        Enter the ending phase (unconditionally, covering a lane that died while still PROCESS_STARTING), wait
        for the old process to drain out of the map, then start a fresh one. Unlike the post-processing lane
        there are no in-flight job results to account for: the lane serves no jobs, and consumers that lose a
        producer fall back to loading their own copies.
        """
        if not self._component_processes_should_be_replaced:
            return

        if not self._component_processes_ending:
            self._component_processes_ending = True
            if self._process_map.num_loaded_component_processes() > 0:
                self.end_component_processes()
            return

        if self._process_map.num_loaded_component_processes() == 0 and self._process_map.num_component_processes() > 0:
            self._process_map.delete_component_processes()

        if (
            self._component_processes_ending
            and self._process_map.num_loaded_component_processes() == 0
            and self._process_map.num_component_processes() == 0
        ):
            self.start_component_processes()
            self._component_processes_ending = False
            self._component_processes_should_be_replaced = False
            if self._component_lane_replacement_intentional:
                # A deliberate whole-card pause (or its restore), not a lane crash: keep it out of the
                # recovery count so a burst of whole-card jobs cycling the lane is not read as a crash loop.
                self._component_lane_replacement_intentional = False
            else:
                self._num_process_recoveries += 1

    def _initiate_component_replacement(self) -> None:
        """Flag the component lane for replacement so the control loop's state machine restarts it."""
        self._component_processes_should_be_replaced = True

    def component_lane_enabled(self) -> bool:
        """Whether the dedicated component (text-encode) lane should run (public view of the config gate)."""
        return self._component_lane_enabled()

    def component_lane_card_index(self) -> int:
        """Return the device index the dedicated component lane is (or would be) pinned to."""
        return self._post_process_card().device_index

    @property
    def is_component_gpu_paused(self) -> bool:
        """Whether the dedicated component lane is being held off-GPU for a whole-card job."""
        return self._component_gpu_paused

    @property
    def component_pause_owner(self) -> PauseOwner | None:
        """Which initiator holds the component lane's off-GPU pause (None when it is not paused)."""
        return self._component_pause_owner

    @property
    def component_gpu_pause_count(self) -> int:
        """How many whole-card residency component-lane off-GPU pauses this manager initiated."""
        return self._component_gpu_pause_count

    @property
    def component_gpu_restore_count(self) -> int:
        """How many whole-card residency component-lane restores this manager initiated."""
        return self._component_gpu_restore_count

    def pause_component_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Stop the dedicated component lane so its CUDA context and text encoders free for a whole-card model.

        A no-op (returns False) when the lane is not enabled or is already paused. Otherwise records ``owner``
        as the pause holder (so only its restore path clears it), sets the override (which suppresses the
        per-tick restart in :meth:`start_component_processes`) and triggers the lane replacement state machine
        to end the running process; the intentional flag keeps that teardown out of the crash-recovery count.
        It stays stopped until :meth:`restore_component_off_gpu` is called by the same owner. The lane serves no
        jobs, so nothing in flight is stranded: while it is down, disaggregation dispatch demotes to monolithic
        (its liveness gates eligibility) rather than faulting.

        Args:
            owner: The subsystem initiating the pause; only this owner's restore path clears it.

        Returns:
            True if a pause was initiated, False if it was already paused or the lane is not enabled.
        """
        if not self._component_lane_enabled() or self._component_gpu_paused:
            return False
        self._component_gpu_paused = True
        self._component_pause_owner = owner
        self._component_gpu_pause_count += 1
        if self._process_map.num_component_processes() > 0:
            self._component_lane_replacement_intentional = True
            self._initiate_component_replacement()
        logger.info(
            "Whole-card residency: stopping the component lane to free its VRAM context for the heavy model.",
        )
        return True

    def restore_component_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Restart the dedicated component lane after its pausing owner has released the device.

        A no-op (returns False) when the lane is not currently paused, or when ``owner`` is not the initiator
        that holds the pause (a foreign owner must not lift another's hold). Otherwise clears the override and
        starts the lane directly, for the same reason :meth:`restore_vae_lane_off_gpu` does: the replacement
        state machine consumed its flag during the pause (its final start call was suppressed by the pause
        gate), and the bring-up callers are one-shot latches, so without this the lane would stay down for the
        rest of the session and disaggregation would remain demoted to monolithic.

        Args:
            owner: The subsystem requesting the restore; it must match the owner that initiated the pause.

        Returns:
            True if a restore was initiated, False if it was not paused or is held by a different owner.
        """
        if not self._component_gpu_paused or self._component_pause_owner is not owner:
            return False
        self._component_gpu_paused = False
        self._component_pause_owner = None
        self._component_gpu_restore_count += 1
        logger.info("Whole-card residency complete: restarting the component lane.")
        self.start_component_processes()
        return True

    @post_process_processes_should_be_replaced.setter
    def post_process_processes_should_be_replaced(self, value: bool) -> None:
        self._post_process_processes_should_be_replaced = value

    def vae_lane_enabled(self) -> bool:
        """Whether the dedicated VAE lane should run.

        Gated on pipeline disaggregation being enabled: the lane is the VAE-encode/decode stage of the
        disaggregated pipeline (it produces the source LATENT for img2img and decodes the sampler's LATENT
        to final images), so it is spawned exactly when disaggregation is on.
        """
        return self._runtime_config.bridge_data.enable_pipeline_disaggregation

    def vae_lane_card_index(self) -> int:
        """Return the device index the dedicated VAE lane is (or would be) pinned to."""
        return self._post_process_card().device_index

    def start_vae_lane_processes(self) -> None:
        """Start the dedicated VAE lane, if disaggregation is enabled and it is not already running."""
        if not self.vae_lane_enabled():
            return

        # While a whole-card model holds the card the lane is deliberately kept off-GPU: this per-tick start
        # hook must not resurrect it until the residency restores and clears the pause.
        if self._vae_lane_gpu_paused:
            return

        if self._process_map.num_vae_lane_processes() > 0:
            return

        bridge_data = self._runtime_config.bridge_data
        pid = self._allocate_inference_pid()
        pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)
        lane_card = self._post_process_card()

        process = self._new_process(
            target=self._entry_points.vae_lane_entry_point,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                self._disk_lock,
                self.num_processes_launched,
            ),
            kwargs={
                "device_index": lane_card.device_index,
                "accelerator_kind": lane_card.mask_kind,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "dry_run_skip_vae_lane": bridge_data.dry_run_skip_inference,
                "comfy_smart_memory": bridge_data.comfy_smart_memory,
            },
        )
        process.start()

        self._process_map[pid] = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.VAE_LANE,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            device_index=lane_card.device_index,
        )
        self._register_owned(self._process_map[pid])

        logger.info(f"Started VAE lane (id: {pid}, device_index: {lane_card.device_index})")
        self.num_processes_launched += 1

    def end_vae_lane_processes(self) -> None:
        """End any dedicated VAE lane processes."""
        for process_info in self._process_map.get_stoppable_vae_lane_processes():
            process_info.end_intended = True
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
            self._process_map.on_process_ending(process_id=process_info.process_id)
            self._forget_owned(process_info)
            logger.info(f"Ended VAE lane {process_info.process_id}")

    def _initiate_vae_lane_replacement(self) -> None:
        """Flag the VAE lane for replacement so the control loop's state machine restarts it."""
        self._vae_lane_processes_should_be_replaced = True

    def _replace_all_vae_lane_process(self) -> None:
        """Replace the dedicated VAE lane across control-loop ticks (mirrors the post-processing state machine).

        Enter the ending phase (unconditionally, covering a lane that died while still PROCESS_STARTING), wait
        for the old process to drain out of the map, then start a fresh one. Unlike the post-processing lane
        no parent-side known-lost set is drained: in-flight VAE stages are held by the disaggregation
        orchestrator, which re-dispatches an orphaned stage from held state once the replacement lane appears.
        """
        if not self._vae_lane_processes_should_be_replaced:
            return

        if not self._vae_lane_processes_ending:
            self._vae_lane_processes_ending = True
            if self._process_map.num_loaded_vae_lane_processes() > 0:
                self.end_vae_lane_processes()
            return

        if self._process_map.num_loaded_vae_lane_processes() == 0 and self._process_map.num_vae_lane_processes() > 0:
            self._process_map.delete_vae_lane_processes()

        if (
            self._vae_lane_processes_ending
            and self._process_map.num_loaded_vae_lane_processes() == 0
            and self._process_map.num_vae_lane_processes() == 0
        ):
            self.start_vae_lane_processes()
            self._vae_lane_processes_ending = False
            self._vae_lane_processes_should_be_replaced = False
            if self._vae_lane_replacement_intentional:
                # A deliberate whole-card pause (or its restore), not a lane crash: keep it out of the
                # recovery count so a burst of whole-card jobs cycling the lane is not read as a crash loop.
                self._vae_lane_replacement_intentional = False
            else:
                self._num_process_recoveries += 1

    @property
    def vae_lane_processes_should_be_replaced(self) -> bool:
        """Whether the dedicated VAE lane is flagged for replacement."""
        return self._vae_lane_processes_should_be_replaced

    @vae_lane_processes_should_be_replaced.setter
    def vae_lane_processes_should_be_replaced(self, value: bool) -> None:
        self._vae_lane_processes_should_be_replaced = value

    @property
    def is_vae_lane_gpu_paused(self) -> bool:
        """Whether the dedicated VAE lane is being held off-GPU for a whole-card job."""
        return self._vae_lane_gpu_paused

    @property
    def vae_lane_pause_owner(self) -> PauseOwner | None:
        """Which initiator holds the VAE lane's off-GPU pause (None when it is not paused)."""
        return self._vae_lane_pause_owner

    @property
    def vae_lane_gpu_pause_count(self) -> int:
        """How many whole-card residency VAE-lane off-GPU pauses this manager initiated."""
        return self._vae_lane_gpu_pause_count

    @property
    def vae_lane_gpu_restore_count(self) -> int:
        """How many whole-card residency VAE-lane restores this manager initiated."""
        return self._vae_lane_gpu_restore_count

    def pause_vae_lane_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Stop the dedicated VAE lane so its CUDA context and models free for a whole-card model.

        A no-op (returns False) when the lane is not enabled or is already paused. Otherwise records ``owner``
        as the pause holder (so only its restore path clears it), sets the override (which suppresses the
        per-tick restart in :meth:`start_vae_lane_processes`) and triggers the lane replacement state machine to
        end the running process; the intentional flag keeps that teardown out of the crash-recovery count. Like
        the post-processing lane it stays stopped until :meth:`restore_vae_lane_off_gpu` is called by the same
        owner. Any VAE stage in flight on the lane is re-dispatched from held state by the disaggregation
        orchestrator once a replacement lane appears.

        Args:
            owner: The subsystem initiating the pause; only this owner's restore path clears it.

        Returns:
            True if a pause was initiated, False if it was already paused or the lane is not enabled.
        """
        if not self.vae_lane_enabled() or self._vae_lane_gpu_paused:
            return False
        self._vae_lane_gpu_paused = True
        self._vae_lane_pause_owner = owner
        self._vae_lane_gpu_pause_count += 1
        if self._process_map.num_vae_lane_processes() > 0:
            self._vae_lane_replacement_intentional = True
            self._initiate_vae_lane_replacement()
        logger.info(
            "Whole-card residency: stopping the VAE lane to free its VRAM context for the heavy model.",
        )
        return True

    def restore_vae_lane_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Restart the dedicated VAE lane after its pausing owner has released the device.

        A no-op (returns False) when the lane is not currently paused, or when ``owner`` is not the initiator
        that holds the pause (a foreign owner must not lift another's hold). Otherwise clears the override and
        starts the lane directly, for the same reason :meth:`restore_post_process_off_gpu` does: the replacement
        state machine consumed its flag during the pause (its final start call was suppressed by the pause
        gate), and the bring-up callers are one-shot latches, so without this the lane would stay down for the
        rest of the session and every VAE stage would queue against a lane that never returns.

        Args:
            owner: The subsystem requesting the restore; it must match the owner that initiated the pause.

        Returns:
            True if a restore was initiated, False if it was not paused or is held by a different owner.
        """
        if not self._vae_lane_gpu_paused or self._vae_lane_pause_owner is not owner:
            return False
        self._vae_lane_gpu_paused = False
        self._vae_lane_pause_owner = None
        self._vae_lane_gpu_restore_count += 1
        logger.info("Whole-card residency complete: restarting the VAE lane.")
        self.start_vae_lane_processes()
        return True

    def start_download_process(self) -> None:
        """Start the singleton background download process, if not already running.

        The download process lives outside the process map (it serves no jobs and must not be
        swept up by the hung-process logic); its messages are routed by its reserved process id.
        """
        if self._download_process_info is not None:
            return

        bridge_data = self._runtime_config.bridge_data
        pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)

        process = self._new_process(
            target=self._entry_points.download_entry_point,
            args=(
                DOWNLOAD_PROCESS_ID,
                self._process_message_queue,
                child_pipe_connection,
                self._disk_lock,
                self._download_bandwidth_semaphore,
                self.num_processes_launched,
            ),
            kwargs={
                "nsfw": bridge_data.nsfw,
                "allow_lora": bridge_data.allow_lora,
                "allow_controlnet": bridge_data.allow_controlnet,
                "allow_sdxl_controlnet": bridge_data.allow_sdxl_controlnet,
                "allow_post_processing": bridge_data.allow_post_processing,
                "purge_loras": bridge_data.purge_loras_on_download,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "rate_limit_kbps": bridge_data.download_rate_limit_kbps,
                "paused": bridge_data.downloads_paused,
                "max_parallel_downloads": bridge_data.download_max_parallel_downloads,
                "per_host_concurrency": bridge_data.download_per_host_concurrency,
                "connections_per_file": bridge_data.download_connections_per_file,
            },
        )
        process.start()

        self._download_process_info = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=DOWNLOAD_PROCESS_ID,
            process_type=HordeProcessType.DOWNLOAD,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            capabilities=WorkerCapability(0),
        )
        self._register_owned(self._download_process_info)
        self.num_processes_launched += 1
        logger.info("Started background download process")

    def request_downloads(
        self,
        model_names: list[str],
        *,
        download_aux: bool = False,
        desired_image_models: list[str] | None = None,
    ) -> None:
        """Ask the download process to ensure the given image models are present on disk.

        ``desired_image_models``, when given, is the authoritative configured image-model set: the
        download process prunes any queued/in-flight download not in it (so a config edit that removes a
        model stops it downloading). A reconcile-only request (a removal with nothing new to fetch) is
        still sent, hence this does not short-circuit when only ``desired_image_models`` is set.
        """
        if self._download_process_info is None:
            logger.warning("Cannot request downloads: no download process is running")
            return
        if not model_names and not download_aux and desired_image_models is None:
            return
        self._download_process_info.safe_send_message(
            HordeDownloadControlMessage(
                model_names=list(model_names),
                download_aux=download_aux,
                desired_image_models=desired_image_models,
            ),
        )

    def set_download_controls(
        self,
        *,
        paused: bool | None = None,
        rate_limit_kbps: int | None = None,
        max_parallel_downloads: int | None = None,
        per_host_concurrency: int | None = None,
        connections_per_file: int | None = None,
    ) -> None:
        """Forward live download controls (pause/bandwidth/parallelism) to the download process.

        Used by both the config-reload path and the supervisor pause/resume/rate commands. A ``None``
        argument leaves that control unchanged; ``rate_limit_kbps`` of 0 (or negative) clears the cap.
        No-op if no download process is running, or if every argument is ``None``.
        """
        if self._download_process_info is None:
            return
        controls = (paused, rate_limit_kbps, max_parallel_downloads, per_host_concurrency, connections_per_file)
        if all(arg is None for arg in controls):
            return
        self._download_process_info.safe_send_message(
            HordeDownloadControlMessage(
                model_names=[],
                download_aux=False,
                set_paused=paused,
                set_rate_limit_kbps=rate_limit_kbps,
                set_max_parallel_downloads=max_parallel_downloads,
                set_per_host_concurrency=per_host_concurrency,
                set_connections_per_file=connections_per_file,
            ),
        )

    def set_download_gating(
        self,
        *,
        nsfw: bool | None = None,
        allow_lora: bool | None = None,
        allow_controlnet: bool | None = None,
        allow_sdxl_controlnet: bool | None = None,
        allow_post_processing: bool | None = None,
        purge_loras: bool | None = None,
    ) -> None:
        """Forward changed download-gating flags to the download process, applied live (no restart).

        These gate which auxiliary categories the download process fetches (and the nsfw/purge behaviour of
        the default-LoRa pass). They were once construction-time only, so a change to them restarted the
        process; the download process now applies them live and re-arms its one-shot aux pass when a category
        is newly enabled. A ``None`` argument leaves that flag unchanged; a no-op if no download process is
        running or every argument is ``None``.
        """
        if self._download_process_info is None:
            return
        gating = (nsfw, allow_lora, allow_controlnet, allow_sdxl_controlnet, allow_post_processing, purge_loras)
        if all(arg is None for arg in gating):
            return
        self._download_process_info.safe_send_message(
            HordeDownloadControlMessage(
                model_names=[],
                download_aux=False,
                set_nsfw=nsfw,
                set_allow_lora=allow_lora,
                set_allow_controlnet=allow_controlnet,
                set_allow_sdxl_controlnet=allow_sdxl_controlnet,
                set_allow_post_processing=allow_post_processing,
                set_purge_loras=purge_loras,
            ),
        )

    def broadcast_reload_model_database(self) -> None:
        """Tell every subprocess to reload its model managers' references from disk (no download).

        Sent after the parent refreshes the on-disk reference, or after the download process reports
        new LoRa/TI availability, so inference and download subprocesses pick up the changes live
        without a restart. Subprocesses never download references; they only re-read the parent's files.
        """
        message = HordeControlMessage(control_flag=HordeControlFlag.RELOAD_MODEL_DATABASE)
        for process_info in self._process_map.get_inference_processes():
            process_info.safe_send_message(message)
        if self._download_process_info is not None:
            self._download_process_info.safe_send_message(message)

    def end_download_process(self) -> None:
        """Stop the background download process, if running."""
        if self._download_process_info is None:
            return
        with contextlib.suppress(BrokenPipeError):
            self._download_process_info.safe_send_message(
                HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS),
            )
        try:
            self._download_process_info.mp_process.join(timeout=1)
            self._download_process_info.mp_process.kill()
        except Exception as e:
            logger.debug(f"Failed to stop download process: {e}")
        self._forget_owned(self._download_process_info)
        self._download_process_info = None

    def restart_download_process(self) -> None:
        """Stop and restart the download process (a hard reset that re-reads the current bridge data).

        The download-gating flags (nsfw / allow_lora / allow_controlnet / allow_post_processing / purge) are
        now applied live via :meth:`set_download_gating`, so a config change to them no longer needs this.
        It remains for the cases that genuinely need a fresh process (e.g. a structural restart).
        """
        self.end_download_process()
        self.start_download_process()

    def _device_for_new_process(self) -> int:
        """Pick the card a newly-spawned inference process should run on.

        Fills each driven card up to its ``target_process_count`` (lowest index first), so the initial
        spawn distributes processes across cards in proportion to their per-card config. When every card is
        already at target (e.g. a runtime scale-up beyond the configured count) it falls back to the lowest
        index. On a single-GPU host this is always that one card.
        """
        if not self._card_runtimes:
            return 0
        counts: dict[int, int] = {}
        for process_info in self._process_map.values():
            if process_info.process_type is HordeProcessType.INFERENCE:
                counts[process_info.device_index] = counts.get(process_info.device_index, 0) + 1
        for index in sorted(self._card_runtimes):
            if counts.get(index, 0) < self._card_runtimes[index].target_process_count:
                return index
        return min(self._card_runtimes)

    def start_inference_processes(self) -> None:
        """Start all the inference processes configured to be used, across every driven card."""
        # The dedicated post-processing lane rides the same readiness gates as the inference pool (its
        # models live in the same on-disk cache); starting it here covers every bring-up path and is a
        # no-op when the lane is disabled or already running.
        self.start_post_process_processes()
        # The component lane is the disaggregated pipeline's text-encode service; it rides the same bring-up
        # and is a no-op unless pipeline disaggregation is enabled.
        self.start_component_processes()
        # The VAE lane is the disaggregated pipeline's VAE-encode/decode stage; likewise a no-op unless
        # pipeline disaggregation is enabled.
        self.start_vae_lane_processes()

        num_processes_to_start = self._max_inference_processes - self._process_map.num_inference_processes()

        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_inference_processes()} inference processes running, but "
                f"max_inference_processes is set to {self._max_inference_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        for i in range(num_processes_to_start):
            # Allocate through the shared helper (lowest free slot from 1 upward) so the reserved safety slot
            # 0 is never taken and ids stay stable across scale cycles. A ``len(map)``-based id would grab 0
            # when the inference pool starts before the safety pool, colliding with the safety process.
            pid = self._allocate_inference_pid()
            self._start_inference_process(pid, device_index=self._device_for_new_process())

            logger.info(f"Started inference process (id: {pid})")

            if i == 0:
                time.sleep(4)

    def _start_inference_process(self, pid: int, *, device_index: int = 0) -> HordeProcessInfo:
        """Starts an inference process.

        :param pid: process ID to assign to the process
        :param device_index: stable index of the GPU this process is assigned to (0 on a single-GPU host)
        :return: The new HordeProcessInfo
        """
        bridge_data = self._runtime_config.bridge_data
        # A card whose index is not in the plan (e.g. an unexpected device_index) falls back to the lowest
        # configured card so a spawn never fails on a missing key; single-GPU always resolves to card 0.
        card = self._card_runtimes.get(device_index) or self._card_runtimes[min(self._card_runtimes)]
        logger.info(f"Starting inference process on PID {pid} (device {card.device_index})")
        vram_heavy_models = any_offered_model_wants_whole_card(bridge_data.image_models_to_load)

        # DirectML has no env-var device mask (unlike CUDA/ROCm/XPU); a process is pinned to a DirectML
        # adapter only by its ``--directml=N`` comfy arg. When this worker drives several DirectML cards
        # (mask_kind 'directml'), each process must target its own adapter index. The legacy explicit
        # ``--directml=N`` flag stays authoritative as a single-device selection: when it is set
        # (self._directml is not None) it is passed through unchanged for every process.
        directml_index = self._directml
        if card.mask_kind == "directml" and self._directml is None:
            directml_index = card.device_index

        pipe_connection, child_pipe_connection = self._ctx.Pipe(duplex=True)
        process = self._new_process(
            target=self._entry_points.inference_entry_point,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                card.inference_semaphore,
                self._disk_lock,
                self._aux_model_lock,
                card.vae_decode_semaphore,
                self.num_processes_launched,
            ),
            kwargs={
                "device_index": card.device_index,
                "accelerator_kind": card.mask_kind,
                "amd_gpu": self._amd_gpu,
                "directml": directml_index,
                "vram_heavy_models": vram_heavy_models,
                "dry_run_skip_inference": bridge_data.dry_run_skip_inference,
                "dry_run_inference_delay": bridge_data.dry_run_inference_delay,
                "gpu_sampling_lease": card.gpu_sampling_lease if self._gpu_sampling_lease_enabled else None,
                # An alchemist-only worker (no image models configured, e.g. a CPU install) must not
                # treat an empty image-model database as a fatal error in the child.
                "expect_image_models": bool(card.config.image_models_to_load),
                "comfy_smart_memory": bridge_data.comfy_smart_memory,
            },
        )
        process.start()
        process_info = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.INFERENCE,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            device_index=card.device_index,
        )
        self._process_map[pid] = process_info
        self._register_owned(process_info)
        self.num_processes_launched += 1
        return process_info

    def _allocate_inference_pid(self) -> int:
        """Return the lowest inference process id not currently in use.

        Allocation starts at 1: slot ``SAFETY_PROCESS_ID`` (0) is reserved for the safety process by
        convention, so an inference process never occupies it even when the inference pool starts before the
        safety pool. Slot ids are reused once freed, so this stays stable across scale-down/scale-up cycles
        (a ``len(map)``-based scheme would collide after removing a non-last slot). The download process lives
        outside the map at its own reserved id, so it never participates here.
        """
        used = set(self._process_map.keys())
        pid = SAFETY_PROCESS_ID + 1
        while pid in used:
            pid += 1
        return pid

    def refresh_max_inference_processes(self) -> None:
        """Recompute the cached worker-wide inference-process ceiling from the current per-card targets.

        The ceiling is summed once at construction; call this after the shared ``card_runtimes`` targets are
        changed at runtime (an alchemist-only collapse lowers every card to one) so the worker-wide scale
        bound and any reader of it agree with the new plan.
        """
        self._max_inference_processes = sum(card.target_process_count for card in self._card_runtimes.values())

    def scale_inference_processes(
        self,
        target_count: int,
        *,
        device_index: int | None = None,
        whole_card_model: str | None = None,
        pressure_shortfall_mb: float | None = None,
    ) -> int:
        """Grow or shrink the running inference processes toward ``target_count``.

        Growth spawns fresh processes (bounded by the launched-process ceiling). Shrink ends idle
        processes, preferring ones not holding a model needed by queued work, and removes them from
        the process map; busy processes are never killed, so the effective count may not reach the
        target in one call. Used by the benchmark to stage processes on demand and as a memory/VRAM
        pressure lever.

        Args:
            target_count: The desired inference-process count for the scoped pool.
            device_index: When given, grow/shrink only that card's pool toward ``target_count``; new
                processes spawn on that card and only idle processes on that card are stopped (the per-card
                lever a whole-card residency uses to reduce one card's live contexts on a multi-GPU host).
                When None, the worker-wide pool, bounded by the launched-process ceiling (the single-GPU /
                benchmark behaviour, unchanged).
            whole_card_model: When set, this shrink is a whole-card residency collapsing to sole residency
                for that model, so the usual "spare any process whose model is queued" protection is dropped:
                whole-card residency means the heavy head owns the card and the queued siblings deliberately
                wait (their models reload once the head drains, see
                :meth:`InferenceScheduler._restore_siblings_after_whole_card`). Only the residency holder (the
                process the head is staged/resident on) is spared; otherwise a sibling holding a model queued
                *behind* the head pins the count above the target and the residency can never converge, wedging
                the queue. Busy processes are still never killed (the victim selection skips them), so live work
                is unaffected. Leave None for the ordinary benchmark / pressure shrink.
            pressure_shortfall_mb: When set, choose the smallest idle victim that can plausibly clear this
                RAM shortfall before falling back to the usual first eligible victim. This keeps a small RAM
                dip from tearing down a much larger model-holding process when a smaller idle context suffices.

        Returns:
            The number of inference processes after scaling (scoped to ``device_index`` when given).
        """
        if device_index is None:
            ceiling = self._max_inference_processes
            current = self._process_map.num_loaded_inference_processes()
        else:
            card = self._card_runtimes.get(device_index)
            ceiling = card.target_process_count if card is not None else self._max_inference_processes
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
        target = max(0, min(target_count, ceiling))

        if target > current:
            for _ in range(target - current):
                pid = self._allocate_inference_pid()
                new_process_device = device_index if device_index is not None else self._device_for_new_process()
                self._start_inference_process(pid, device_index=new_process_device)
                logger.info(f"Scaled up: started inference process {pid}")
        elif target < current:
            if whole_card_model is not None:
                disallowed = self._whole_card_protected_processes(whole_card_model, device_index)
            else:
                disallowed = self.get_processes_with_model_for_queued_job()
                if device_index is not None:
                    # Confine the shrink to this card: every inference process on another card is off-limits, so
                    # only an idle process on the target card can be the victim.
                    disallowed = disallowed + self._other_card_inference_processes(device_index)
            for _ in range(current - target):
                victim = self._select_inference_process_to_scale_down(
                    disallowed_processes=disallowed,
                    pressure_shortfall_mb=pressure_shortfall_mb,
                )
                if victim is None:
                    logger.debug("Scale down: no idle inference process available to stop right now")
                    break
                self._end_inference_process(victim)
                self._process_map.retire_process(victim, "inference scale-down")
                logger.info(f"Scaled down: stopped inference process {victim.process_id}")

        return self._process_map.num_loaded_inference_processes(device_index=device_index)

    def _select_inference_process_to_scale_down(
        self,
        *,
        disallowed_processes: list[int],
        pressure_shortfall_mb: float | None,
    ) -> HordeProcessInfo | None:
        """Choose an idle inference process to stop for scale-down.

        Under RAM pressure, prefer the smallest eligible resident context whose reported RSS can clear the
        shortfall. If no eligible process has a sufficient report, fall back to the existing first-eligible
        behavior so ordinary scale-down semantics stay unchanged.
        """
        if pressure_shortfall_mb is None or pressure_shortfall_mb <= 0:
            return self._process_map._get_first_inference_process_to_kill(disallowed_processes=disallowed_processes)

        shortfall_bytes = pressure_shortfall_mb * 1024 * 1024
        candidates: list[HordeProcessInfo] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.process_id in disallowed_processes:
                continue
            if process_info.is_process_busy():
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            if process_info.ram_usage_bytes >= shortfall_bytes:
                candidates.append(process_info)

        if candidates:
            return min(candidates, key=lambda process_info: process_info.ram_usage_bytes)
        return self._process_map._get_first_inference_process_to_kill(disallowed_processes=disallowed_processes)

    def _other_card_inference_processes(self, device_index: int) -> list[int]:
        """Return inference processes pinned to a card other than ``device_index`` (off-limits for a scoped shrink)."""
        return [
            p.process_id
            for p in self._process_map.values()
            if p.process_type is HordeProcessType.INFERENCE and p.device_index != device_index
        ]

    def _whole_card_protected_processes(self, whole_card_model: str, device_index: int | None) -> list[int]:
        """Return the processes a whole-card convergence shrink must spare: the residency holder, plus other cards.

        The residency holder is whichever inference process the heavy head is staged or resident on (its
        ``loaded_horde_model_name`` is ``whole_card_model``; a ``PRELOADED_MODEL`` head sets that name). Unlike
        :meth:`get_processes_with_model_for_queued_job`, an idle sibling holding some *other* queued model is
        deliberately left stoppable: collapsing to sole residency is the whole point, and that sibling's
        queued job waits and reloads after the head drains. When ``device_index`` is set the shrink is scoped
        to one card, so every inference process on another card is also off-limits.
        """
        protected = [
            p.process_id
            for p in self._process_map.values()
            if p.process_type is HordeProcessType.INFERENCE and p.loaded_horde_model_name == whole_card_model
        ]
        if device_index is not None:
            protected += self._other_card_inference_processes(device_index)
        return protected

    def end_inference_processes(
        self,
        force: bool = False,
    ) -> None:
        """End any inference processes above the configured limit, or all of them if shutting down."""
        if force:
            if not self._state.shutting_down:
                logger.error("Forcing inference processes to end without shutting down")

            for process in self._process_map.get_inference_processes():
                self._end_inference_process(process)

        if len(self._job_tracker.jobs_pending_inference) > 0 and len(
            self._job_tracker.jobs_pending_inference,
        ) != len(self._job_tracker.jobs_in_progress):
            return

        processes_with_model_for_queued_job: list[int] = self.get_processes_with_model_for_queued_job()

        if (
            self._state.shutting_down
            and len(self._job_tracker.jobs_pending_inference) == 0
            and len(self._job_tracker.jobs_in_progress) == 0
        ):
            processes_with_model_for_queued_job = []

        process_info = self._process_map._get_first_inference_process_to_kill(
            disallowed_processes=processes_with_model_for_queued_job,
        )

        if process_info is not None:
            self._end_inference_process(process_info)

    def _end_inference_process(self, process_info: HordeProcessInfo) -> None:
        """Ends an inference process."""
        # Mark the end as supervisor-intended *before* doing anything else, so the crash reaper does not
        # mistake this slot's imminent exit for an unexpected death and try to "recover" (and re-count) it.
        process_info.end_intended = True
        self._process_map.on_process_ending(process_id=process_info.process_id)
        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)

        try:
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
        except BrokenPipeError:
            if not self._state.shutting_down:
                logger.debug(f"Process {process_info.process_id} control channel vanished")
        try:
            process_info.mp_process.join(timeout=1)
            if process_info.mp_process.is_alive():
                process_info.mp_process.kill()
                process_info.mp_process.join(timeout=1)
        except Exception as e:
            logger.error(f"Failed to kill process {process_info.process_id}: {e}")

        self._forget_owned(process_info)
        self._action_ledger.record(
            LedgerEventType.PROCESS_ENDED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
        )

        if not self._state.shutting_down:
            logger.info(f"Ended inference process {process_info.process_id}")

    def end_safety_processes(self) -> None:
        """End any safety processes above the configured limit, or all of them if shutting down."""
        for process_info in self._process_map.get_stoppable_safety_processes():
            # Mark the end as supervisor-intended before sending the command so the crash reaper does not
            # treat the child's expected exit as a safety-pool crash.
            process_info.end_intended = True
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
            self._process_map.on_process_ending(process_id=process_info.process_id)
            self._forget_owned(process_info)

            logger.info(f"Ended safety process {process_info.process_id}")

    @property
    def is_safety_gpu_paused(self) -> bool:
        """Whether the safety process is being forced off-GPU for a whole-card job."""
        return self._safety_gpu_paused

    @property
    def safety_gpu_pause_count(self) -> int:
        """How many whole-card residency safety-off-GPU pauses this lifecycle manager initiated."""
        return self._safety_gpu_pause_count

    @property
    def safety_gpu_restore_count(self) -> int:
        """How many whole-card residency safety-on-GPU restores this lifecycle manager initiated."""
        return self._safety_gpu_restore_count

    def set_desired_safety_card(self, device_index: int | None) -> None:
        """Record the driven card the safety process should pin to when it next (re)spawns on-GPU.

        Pushed by the scheduler's headroom-aware placement identity each control cycle. Takes effect only at
        the next safety bring-up (spawn or the re-promotion respawn), never migrating a live process; a value
        that is not a driven card index (or None) falls back to the lowest-index card at spawn time.

        Args:
            device_index: The chosen card's stable index, or None for the lowest-index default.
        """
        self._desired_safety_card = device_index

    def safety_gpu_card_index(self) -> int | None:
        """Return the device the on-GPU safety process occupies, or None when safety is off-GPU.

        Truthful "which card is safety on" signal: None while safety is paused off-GPU (residency or the
        runtime placement policy) or came up cpu_only, else the card it was pinned to at its last spawn.
        """
        if self._safety_gpu_paused:
            return None
        return self._safety_pinned_card

    def pause_safety_on_gpu(self) -> bool:
        """Move the safety process off-GPU (cpu_only) so its CUDA context frees for a whole-card model.

        A no-op (returns False) when safety is not configured on-GPU or is already paused. Otherwise sets
        the override and triggers the existing safety-replacement state machine, which ends the on-GPU
        safety process and brings a cpu_only one up over the next few control-loop ticks. Reusing that
        machinery (rather than ad-hoc end/spawn) keeps the churn on the tested recovery path.

        Returns:
            True if a pause was initiated, False if it was already paused or not applicable.
        """
        if not self._runtime_config.bridge_data.safety_on_gpu or self._safety_gpu_paused:
            return False
        self._safety_gpu_paused = True
        self._safety_gpu_pause_count += 1
        self._safety_replacement_intentional = True
        self._initiate_safety_replacement()
        logger.info("Whole-card residency: moving the safety process off-GPU to free its VRAM context.")
        return True

    def restore_safety_on_gpu(self) -> bool:
        """Bring the safety process back on-GPU after a whole-card job has released the device.

        A no-op (returns False) when not currently paused. Clears the override and triggers a replacement so
        the safety process comes back up on its configured (GPU) placement.

        Returns:
            True if a restore was initiated, False if it was not paused.
        """
        if not self._safety_gpu_paused:
            return False
        self._safety_gpu_paused = False
        self._safety_gpu_restore_count += 1
        self._safety_replacement_intentional = True
        self._initiate_safety_replacement()
        logger.info("Whole-card residency complete: restoring the safety process to the GPU.")
        return True

    @property
    def is_post_process_gpu_paused(self) -> bool:
        """Whether the dedicated post-processing lane is being held off-GPU for a whole-card job."""
        return self._post_process_gpu_paused

    @property
    def post_process_pause_owner(self) -> PauseOwner | None:
        """Which initiator holds the post-processing lane's off-GPU pause (None when it is not paused).

        The liveness floor for pending post-processing work reads this: a residency-owned pause has a live
        restore path (the residency completion loop), so waiting for it is safe, but a reclaim-ladder-owned
        pause must not suppress the patience clock, since its restore is not guaranteed within the window (a
        card stuck saturated may never recover) and a stranded job must age out to the raw-image fallback.
        """
        return self._post_process_pause_owner

    @property
    def post_process_gpu_pause_count(self) -> int:
        """How many whole-card residency post-processing-lane off-GPU pauses this manager initiated."""
        return self._post_process_gpu_pause_count

    @property
    def post_process_gpu_restore_count(self) -> int:
        """How many whole-card residency post-processing-lane restores this manager initiated."""
        return self._post_process_gpu_restore_count

    def pause_post_process_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Stop the dedicated post-processing lane so its CUDA context and models free for a whole-card model.

        A no-op (returns False) when the lane is not enabled or is already paused. Otherwise records ``owner``
        as the pause holder (so only its restore path clears it), sets the override (which suppresses the
        per-tick restart in :meth:`start_post_process_processes`) and triggers the lane replacement state
        machine to end the running process; the intentional flag keeps that teardown out of the crash-recovery
        count. Unlike safety, the lane does not come back cpu_only (post-processing on CPU is impractically
        slow): it stays stopped until :meth:`restore_post_process_off_gpu` is called by the same owner. Any
        post-processing job in flight on the lane is recorded known-lost so the recovery coordinator requeues it
        (bounded), then reports a no-image fault if the lane cannot return a result.

        Args:
            owner: The subsystem initiating the pause; only this owner's restore path clears it.

        Returns:
            True if a pause was initiated, False if it was already paused or the lane is not enabled.
        """
        if not self.post_process_lane_enabled() or self._post_process_gpu_paused:
            return False
        self._post_process_gpu_paused = True
        self._post_process_pause_owner = owner
        self._post_process_gpu_pause_count += 1
        if self._process_map.num_post_process_processes() > 0:
            self._post_process_replacement_intentional = True
            self._initiate_post_process_replacement()
        logger.info(
            "Whole-card residency: stopping the post-processing lane to free its VRAM context for the heavy model.",
        )
        return True

    def restore_post_process_off_gpu(self, *, owner: PauseOwner) -> bool:
        """Restart the dedicated post-processing lane after its pausing owner has released the device.

        A no-op (returns False) when the lane is not currently paused, or when ``owner`` is not the initiator
        that holds the pause (a foreign owner must not lift another's hold). Otherwise clears the override and
        starts the lane directly: no recurring hook exists to bring it back otherwise. The bring-up callers
        (:meth:`start_inference_processes` via the download coordinator) are one-shot latches, and the
        replacement state machine consumed its flag during the pause, when its final start call was suppressed
        by the pause gate, so without this call the lane would stay down for the rest of the session and every
        post-processing job would queue against a lane that never returns.

        Args:
            owner: The subsystem requesting the restore; it must match the owner that initiated the pause.

        Returns:
            True if a restore was initiated, False if it was not paused or is held by a different owner.
        """
        if not self._post_process_gpu_paused or self._post_process_pause_owner is not owner:
            return False
        self._post_process_gpu_paused = False
        self._post_process_pause_owner = None
        self._post_process_gpu_restore_count += 1
        logger.info("Whole-card residency complete: restarting the post-processing lane.")
        self.start_post_process_processes()
        return True

    def _initiate_safety_replacement(self) -> None:
        """Flag the safety pool for replacement so the control loop's state machine restarts it.

        Setting this flag is the trigger; `_replace_all_safety_process` (run each control-loop tick)
        then ends, deletes, and restarts the safety process across the next few ticks. The only other
        place this flag is set requires a *running* safety process and an in-flight job, so without
        this a safety process that wedges or dies during startup would never be recovered: it would
        sit pinned at PROCESS_STARTING while the stuck-detection logged a misleading "replacing it"
        forever without doing anything.
        """
        self._safety_processes_should_be_replaced = True

    def _reap_if_crashed(self, process_info: HordeProcessInfo) -> bool:
        """Recover a child that has already exited (crash, sys.exit, segfault) instead of waiting on a timer.

        A dead child sends no further messages, so the state-timeout checks in `replace_hung_processes`
        would otherwise leave it pinned at its last reported state forever. Detecting the exit directly
        lets us restart it promptly and log the exit code so the cause is visible.

        Recovery is gated on *intent*, not on the last reported state. A child reaches ``PROCESS_ENDED``
        both when the parent asked it to (shutdown/scale-down/replacement) and when it caught a fatal
        error and exited via its own graceful shutdown path, for example a ``PRELOAD_MODEL`` handler
        that raised, which sets the child's end flag and emits ``PROCESS_ENDED`` indistinguishably from
        an intended end (an observed soak wedge: process died mid-preload, reported ``PROCESS_ENDED``,
        and was never replaced, stranding the popped job forever). Keying off intent rather than state
        lets us recover that case while still leaving a genuinely intended end alone.

        Returns:
            True if the process was found dead and a replacement was initiated.
        """
        if process_info.end_intended:
            return False
        if process_info.mp_process.is_alive():
            return False

        exit_code = process_info.mp_process.exitcode
        ended_itself = exit_code == 0 and process_info.last_process_state in (
            HordeProcessState.PROCESS_ENDING,
            HordeProcessState.PROCESS_ENDED,
        )
        if ended_itself:
            # A clean exit the parent never asked for: the child hit a terminal condition and chose its own
            # graceful shutdown. Naming it as such (rather than "exited unexpectedly") points the operator at
            # the child's log, where the terminating error was recorded, instead of implying a hard crash.
            logger.error(
                f"{process_info} ended itself without a parent request (exitcode=0) while "
                f"{process_info.last_process_state.name}; the terminating error is in the child's own log; "
                f"recovering",
            )
        else:
            logger.error(
                f"{process_info} exited unexpectedly (exitcode={exit_code}) while "
                f"{process_info.last_process_state.name}; recovering",
            )
        if process_info.process_type == HordeProcessType.SAFETY:
            self._initiate_safety_replacement()
            self._replace_all_safety_process()
        elif process_info.process_type == HordeProcessType.POST_PROCESS:
            self._initiate_post_process_replacement()
            self._replace_all_post_process_process()
        elif process_info.process_type == HordeProcessType.COMPONENT:
            self._initiate_component_replacement()
            self._replace_all_component_process()
        elif process_info.process_type == HordeProcessType.VAE_LANE:
            self._initiate_vae_lane_replacement()
            self._replace_all_vae_lane_process()
        elif process_info.process_type == HordeProcessType.INFERENCE:
            self._replace_inference_process(process_info)
        return True

    def _replace_all_safety_process(self) -> None:
        """Replace all of the safety processes."""
        if not self._safety_processes_should_be_replaced:
            return

        if not self._safety_processes_ending:
            # Enter the ending phase on the first call regardless of whether a process is currently
            # loaded. A safety process that died while still PROCESS_STARTING is never "loaded", so the
            # old ``num_loaded > 0`` guard left ``_safety_processes_ending`` unset; the restart branch
            # below (gated on that flag) then never fired, leaving the worker without safety forever.
            # Setting the flag here covers both the normal end->delete->start flow and a startup crash.
            self._safety_processes_ending = True
            if self._process_map.num_loaded_safety_processes() > 0:
                self.end_safety_processes()
            return

        if self._process_map.num_loaded_safety_processes() == 0 and self._process_map.num_safety_processes() > 0:
            self._process_map.delete_safety_processes()

        if (
            self._safety_processes_ending
            and self._process_map.num_loaded_safety_processes() == 0
            and self._process_map.num_safety_processes() == 0
        ):
            self.start_safety_processes()
            self._safety_processes_ending = False
            self._safety_processes_should_be_replaced = False
            if self._safety_replacement_intentional:
                # A deliberate whole-card pause/restore cycle, not a crash: keep it out of the recovery
                # count and the crash-loop breaker so a burst of whole-card jobs is not mistaken for a
                # safety crash loop (which would trip save-our-ship).
                self._safety_replacement_intentional = False
            else:
                self._num_process_recoveries += 1
                self._record_safety_recovery()

    def _record_safety_recovery(self) -> None:
        """Record that the safety pool was just rebuilt, pruning the history to the crash-loop window."""
        now = time.time()
        self._safety_recovery_history = [
            t for t in self._safety_recovery_history if now - t <= CRASH_LOOP_WINDOW_SECONDS
        ]
        self._safety_recovery_history.append(now)

    @property
    def safety_pool_failing(self) -> bool:
        """Whether the safety pool has been rebuilt too many times recently (its crash-loop signal).

        The equivalent of inference-slot quarantine for the safety pool: True when a safety process has
        had to be rebuilt more than ``SAFETY_CRASH_LOOP_MAX`` times within the crash-loop window (e.g. it
        crashes on every start), which the recovery supervisor escalates instead of rebuilding forever.
        """
        now = time.time()
        recent = [t for t in self._safety_recovery_history if now - t <= CRASH_LOOP_WINDOW_SECONDS]
        return len(recent) > SAFETY_CRASH_LOOP_MAX

    def _release_held_primitives(self, process_info: HordeProcessInfo) -> None:
        """Release every shared primitive a replaced inference child might still be holding.

        A child acquires the inference/VAE/sampling semaphores and the disk/aux locks inside its own
        process, so a child that dies or hangs leaves them held; the parent must release on its behalf
        or that concurrency is lost for the lifetime of the worker (one orphaned inference permit at
        ``max_threads=1`` wedges everything). We deliberately do not infer which primitives are held
        from the last state the parent recorded: a child can crash after acquiring but before the
        parent processes the matching state-change message, and that exact race is what wedged the
        worker before. Releasing unconditionally is safe because every one of these is bounded (the
        semaphores are BoundedSemaphores, a Lock is bound to one), so releasing one the child did not
        hold raises ValueError, which we swallow as a harmless no-op rather than inflating a limit.
        """
        # Release the dead child's own card's GPU semaphores (single-GPU = the one card), plus the shared
        # disk/aux locks. A child with an unknown device falls back to the lowest configured card.
        card = self._card_runtimes.get(process_info.device_index) or self._card_runtimes[min(self._card_runtimes)]
        candidates: list[tuple[str, Semaphore | Lock_MultiProcessing]] = [
            ("inference_semaphore", card.inference_semaphore),
            ("disk_lock", self._disk_lock),
            ("aux_model_lock", self._aux_model_lock),
            ("vae_decode_semaphore", card.vae_decode_semaphore),
        ]
        if self._gpu_sampling_lease_enabled:
            candidates.append(("gpu_sampling_lease", card.gpu_sampling_lease))

        released: list[str] = []
        for name, primitive in candidates:
            try:
                primitive.release()
                released.append(name)
            except ValueError:
                # Not held by the dead child; the bounded primitive rejected the spurious release.
                pass

        if released:
            self._action_ledger.record(
                LedgerEventType.SEMAPHORE_RELEASED,
                process_id=process_info.process_id,
                os_pid=process_info.os_pid,
                detail={"released": ", ".join(released)},
            )
            logger.debug(
                f"Released primitives possibly held by replaced process {process_info.process_id}: "
                f"{', '.join(released)}",
            )

    def record_model_load_failure(self, process_id: int, model_name: str) -> bool:
        """Record that ``model_name`` failed to load on ``process_id``; return whether it is now quarantined.

        Keyed on the *model*, not the slot: a deterministically-unloadable checkpoint is re-dispatched
        round-robin across fresh slots, so without a per-model counter no single slot's crash-loop breaker
        ever trips and the bad model burns the whole pool down. Once a model crosses
        ``MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD`` failures within the window it is taken out of rotation
        (see :meth:`is_model_load_quarantined`). The process->model mapping is remembered so the imminent
        slot replacement can label the recovery as a model-load failure rather than a process crash.
        """
        now = time.time()
        self._recent_load_failure_by_process[process_id] = (model_name, now)
        prior = self._model_load_failure_history.get(model_name, [])
        recent = [t for t in prior if now - t <= MODEL_LOAD_FAILURE_WINDOW_SECONDS]
        recent.append(now)
        self._model_load_failure_history[model_name] = recent
        if len(recent) >= MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD and model_name not in self._quarantined_models:
            self._quarantined_models.add(model_name)
            logger.error(
                f"Model {model_name} failed to load {len(recent)} times within "
                f"{MODEL_LOAD_FAILURE_WINDOW_SECONDS:.0f}s; quarantining it (its jobs will be reissued and it "
                f"will not be preloaded) to stop it churning the inference pool.",
            )
            return True
        return model_name in self._quarantined_models

    def is_model_load_quarantined(self, model_name: str | None) -> bool:
        """Whether ``model_name`` has been quarantined for repeatedly failing to load."""
        return model_name is not None and model_name in self._quarantined_models

    def quarantined_models(self) -> frozenset[str]:
        """The set of models currently quarantined for repeated load failures."""
        return frozenset(self._quarantined_models)

    def _take_recent_load_failure_for_process(self, process_id: int) -> str | None:
        """Pop and return the model this process most recently failed to load, if that failure is fresh.

        Used by the slot-replacement path to tell a clean exit after a reported load failure apart from a
        genuine process crash/hang, so the recovery is labelled and counted correctly. Bounded to the
        crash-loop window so a stale prior failure cannot mislabel an unrelated later crash.
        """
        entry = self._recent_load_failure_by_process.pop(process_id, None)
        if entry is None:
            return None
        model_name, when = entry
        if time.time() - when > CRASH_LOOP_WINDOW_SECONDS:
            return None
        return model_name

    def _record_slot_recovery(self, process_id: int) -> int:
        """Record a replacement of the given slot and return how many happened within the window."""
        now = time.time()
        recent = [t for t in self._slot_recovery_history.get(process_id, []) if now - t <= CRASH_LOOP_WINDOW_SECONDS]
        recent.append(now)
        self._slot_recovery_history[process_id] = recent
        return len(recent)

    def reset_recovery_counter(self) -> None:
        """Zero the cumulative recovery/slowdown counters at a benchmark level boundary.

        The warm benchmark worker reuses one process pool across levels. These counters are otherwise
        only ever incremented, so without this every level after the first genuine recovery would
        inherit that level's count and read as having recovered itself. This mirrors
        ``WorkerRunMetrics.reset`` (which clears the per-level crash-event list); the slot-recovery
        *history* backing the crash-loop breaker is deliberately left intact so a genuine crash loop
        spanning levels is still caught.
        """
        self._num_process_recoveries = 0
        self._num_slowdown_events = 0
        self._paging_victim_replacements = 0

    def _record_start_failure(self, process_info: HordeProcessInfo) -> int:
        """Track consecutive replacements that never advanced past PROCESS_STARTING; return the streak.

        A replacement while still in ``PROCESS_STARTING`` means the slot died (or was killed) before it
        ever became job-capable, so it never proved it can initialise. These are counted consecutively
        and the streak resets the moment a slot is replaced from any later state (it did initialise, then
        failed differently). Unlike :meth:`_record_slot_recovery`, this is independent of how long each
        failed start took, so a slow but deterministic crash-on-start is still caught.
        """
        process_id = process_info.process_id
        if process_info.last_process_state == HordeProcessState.PROCESS_STARTING:
            streak = self._slot_consecutive_start_failures.get(process_id, 0) + 1
            self._slot_consecutive_start_failures[process_id] = streak
            return streak
        self._slot_consecutive_start_failures.pop(process_id, None)
        return 0

    def _quarantine_inference_slot(self, process_info: HordeProcessInfo, reason: str) -> None:
        """Take a crash-looping (or crash-on-start) inference slot out of the pool instead of respawning it.

        The slot has tripped one of the circuit breakers (too many replacements in the window, or too
        many consecutive failures before reaching readiness), so respawning it would just repeat the
        loop and keep starving the worker. Its OS process was already ended by the caller, and the
        caller already recorded the recovery event; here we only drop it from the process map and
        remember it so it is not silently refilled. The lost capacity is surfaced via
        ``num_slots_quarantined`` so the higher-level recovery supervisor (Phase 5) can escalate when
        too much capacity is lost.
        """
        self._quarantined_inference_slots.add(process_info.process_id)
        self._num_slots_quarantined += 1
        self._process_map.retire_process(process_info, f"inference slot quarantined: {reason}")
        self._action_ledger.record(
            LedgerEventType.PROCESS_QUARANTINED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            reason=reason,
        )
        logger.critical(
            f"Inference slot {process_info.process_id} quarantined ({reason}); not respawning it.",
        )

    def rebuild_inference_pool(self, *, reason: str) -> None:
        """Rebuild the inference pool in place: replace live slots and revive quarantined ones.

        The recovery supervisor's soft reset uses this to give a wedged worker (e.g. every slot
        quarantined by the crash-loop breaker) a clean start without restarting the parent process or
        detaching the TUI. The crash-loop history is cleared first: this is a deliberate, supervised
        rebuild, not the unbounded respawn loop the breaker guards against, so prior replacements must
        not immediately re-quarantine the fresh slots.
        """
        logger.error(f"Soft reset: rebuilding inference pool ({reason}).")
        self._slot_recovery_history.clear()
        self._slot_consecutive_start_failures.clear()

        revived = sorted(self._quarantined_inference_slots)
        self._quarantined_inference_slots.clear()

        live = [p for p in self._process_map.values() if p.process_type == HordeProcessType.INFERENCE]
        for process_info in live:
            self._replace_inference_process(process_info, intentional_reason=f"soft reset: {reason}")

        for slot_id in revived:
            if slot_id not in self._process_map:
                self._start_inference_process(slot_id, device_index=self._device_for_new_process())

        self._action_ledger.record(
            LedgerEventType.PROCESS_REPLACED,
            reason=f"soft reset: {reason}",
            detail={"rebuilt_live": len(live), "revived_quarantined": len(revived)},
        )

    def rebuild_safety_pool(self, *, reason: str) -> None:
        """Force the safety pool to be rebuilt (arm + replace), used by the recovery supervisor's soft reset.

        This is a deliberate, supervised rebuild (the soft reset cycles *both* pools to give a wedged
        worker a clean start), not a safety crash. So, mirroring ``rebuild_inference_pool``, it clears the
        safety crash-loop history and marks the replacement intentional: the rebuilt safety pool is not
        counted as a process recovery and does not feed the safety crash-loop breaker. Otherwise a single
        soft reset whose wedge was the *inference* pool double-counts as two recoveries (the healthy safety
        pool being collateral), and the soft reset could be re-triggered immediately by its own stale
        safety history.
        """
        logger.error(f"Soft reset: rebuilding safety pool ({reason}).")
        self._safety_recovery_history.clear()
        self._safety_replacement_intentional = True
        self._initiate_safety_replacement()
        self._replace_all_safety_process()

    def _looks_like_oom_kill(self, process_info: HordeProcessInfo) -> bool:
        """Whether a dead slot was SIGKILLed by the OS OOM-killer rather than crashing on its own.

        The fingerprint is a ``SIGKILL`` exit (``exitcode == -9``) while system RAM is below its danger
        floor: the kernel reaps the largest process to relieve memory pressure, so the slot vanishes with
        no exception and no fault ``info`` for the ordinary classifier to read. The low-RAM check is what
        distinguishes this from the worker's own hang-kill (also ``-9``) on a healthy host: when RAM is fine
        a ``-9`` stays an ordinary crash/hang. Reads the configured floor defensively (a partially-mocked or
        older config falls back to the module defaults) and never raises; any error reads False, leaving
        the slot on its ordinary crash path.
        """
        raw_exitcode = getattr(process_info.mp_process, "exitcode", None)
        if raw_exitcode != -9:
            return False
        try:
            bridge_data = self._runtime_config.bridge_data
            pause = getattr(bridge_data, "ram_pressure_pause_percent", 85.0)
            min_free = getattr(bridge_data, "ram_pressure_min_free_mb", 1024.0)
            pause_pct = float(pause) if isinstance(pause, (int, float)) and not isinstance(pause, bool) else 85.0
            min_free_mb = (
                float(min_free) if isinstance(min_free, (int, float)) and not isinstance(min_free, bool) else 1024.0
            )
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            floor_mb = ram_pressure_floor_mb(
                vm.total / (1024 * 1024),
                pause_percent=pause_pct,
                min_free_mb=min_free_mb,
            )
            return available_mb < floor_mb
        except Exception as e:
            logger.debug(f"OOM-kill check failed: {type(e).__name__} {e}")
            return False

    def _replace_inference_process(
        self,
        process_info: HordeProcessInfo,
        *,
        intentional_reclaim: bool = False,
        intentional_reason: str | None = None,
        resource_fault_reason: str | None = None,
    ) -> None:
        """Replace an inference process (because it crashed, hung, timed out, or by deliberate request).

        Frees any shared GPU/disk primitives the dead child may still hold (state-independently; see
        ``_release_held_primitives``), faults its in-flight job, then either respawns the slot or, if
        the slot has been replaced too many times in a short window, quarantines it (crash-loop
        circuit breaker) so a permanently-broken slot cannot spin in an unbounded respawn loop.

        Args:
            process_info: The slot to replace.
            intentional_reclaim: When True, this is a *deliberate* cycle of a healthy idle slot to return
                allocator-retained RAM to the OS, not a crash or hang. The crash bookkeeping is then
                skipped: it is not logged as a recovery, not counted in ``_num_process_recoveries``, and
                not fed to the crash-loop / start-failure breakers (which would otherwise quarantine a
                perfectly healthy slot under sustained RAM pressure, since reclaim-cycles of one slot can
                exceed ``CRASH_LOOP_MAX_REPLACEMENTS`` within the window).
            intentional_reason: When set, this is some *other* deliberate replacement of a healthy slot
                (e.g. the maintenance-mode pool reload), labelled with this reason. It takes the same
                no-crash-bookkeeping path as ``intentional_reclaim``: labelling a deliberate reload
                "crashed or hung" both pollutes the recovery diagnostics and feeds a
                phantom crash into the recovery count and crash-loop breaker. Ignored if
                ``intentional_reclaim`` is also set.
            resource_fault_reason: When set, this is a hang recovery whose cause is a VRAM-resource
                condition (the paged-slowdown watchdog), so the in-flight job is faulted as a resource
                failure. That routes it to the bounded degraded/isolated retry (which clears the device for
                it) rather than a plain re-dispatch onto another over-committed slot. Unlike the intentional
                paths this still takes the full crash/recovery bookkeeping (it *is* a recovery).
        """
        bridge_data = self._runtime_config.bridge_data
        logger.debug(f"Replacing {process_info}")
        # Fault the job *this* process was working on. It must be taken from process_info, never by
        # scanning the whole map for "the first process holding any in-flight job": a scan could fault
        # an unrelated (often healthy) process's job and can leave this dying process's actual job stuck in
        # INFERENCE_IN_PROGRESS with no owner, wedging the head of the queue indefinitely.
        # The orphaned-job watchdog in the manager is the backstop for any in-progress job that still
        # ends up without an owner; this is the primary, correct path.
        job_to_remove = process_info.last_job_referenced
        if job_to_remove is not None and job_to_remove not in self._job_tracker.jobs_lookup:
            job_to_remove = None

        self._release_held_primitives(process_info)

        aux_download_stall = (
            process_info.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL and not intentional_reclaim
        )
        # Decide retryability against the incident state *before* this strike is recorded: a lone transient
        # stall (no incident yet) still earns its one ordinary retry, but a stall while a download outage is
        # already active is faulted terminally below instead of requeued straight back into the same failing
        # download, which the logs show just stalls and tears the slot down a second time (every doomed job
        # was costing two process recoveries, not one).
        aux_stall_retryable = not self._state.lora_download_backoff.is_escalation_active(time.time())

        if aux_download_stall:
            if job_to_remove is not None:
                logger.error(
                    f"Job {job_to_remove.id_ or job_to_remove.ids} was in aux model preload on process "
                    f"{process_info.process_id} but it failed. Removing.",
                )
            # A slot torn down mid aux-download is the reliable signal that the ad-hoc download path is
            # failing; register a strike so the popper stops feeding new LoRA jobs (escalating window)
            # and the aux-download watchdog reaps further stalls sooner.
            window = self._state.lora_download_backoff.register_timeout(time.time())
            logger.warning(
                f"Auxiliary (LoRA) download stalled and was torn down (strike "
                f"{self._state.lora_download_backoff.strikes}); withholding LoRA job pops for "
                f"{window:.0f}s while downloads recover.",
            )

        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)
        # Also clear any entry still pointing at this slot by id. When a child reports PROCESS_ENDING
        # before we replace it, on_process_ending already nulled loaded_horde_model_name, so the line
        # above is a no-op and a stale LOADING entry would otherwise pin the model as "resident" on the
        # dead slot, starving the pending job of any re-preload (the soak re-dispatch wedge).
        expired_by_pid = self._horde_model_map.expire_entries_for_process(process_info.process_id)
        if expired_by_pid:
            logger.debug(
                f"Expired model-map entries {expired_by_pid} stranded on dead process {process_info.process_id}",
            )

        if job_to_remove is not None:
            # A slot crash/hang mid-job is normally retryable: the job is requeued to a fresh slot (bounded
            # by max_inference_attempts) rather than faulted outright. The crash gives no resource signal,
            # so it takes the ordinary retry, not the degraded path. The exception is an aux-download stall
            # during an active download outage: an immediate retry only re-enters the same failing download
            # and tears down a second slot, so it is faulted terminally (the horde reassigns the job); the
            # job outcome is the same as the eventual out-of-attempts fault, at half the process churn.
            self._job_tracker.handle_job_fault_now(
                faulted_job=job_to_remove,
                process_info=process_info,
                process_timeout=bridge_data.process_timeout,
                retryable=not (aux_download_stall and not aux_stall_retryable),
                is_resource_failure=resource_fault_reason is not None,
                fault_reason=resource_fault_reason,
            )

        if intentional_reclaim or intentional_reason is not None:
            # A healthy slot is being replaced *deliberately*, not because it crashed or hung: either to
            # return allocator-retained RAM to the OS (the model unload freed the model, but only a respawn
            # reclaims the pages), or for an operational reload such as the maintenance-mode pool refresh.
            # Either way this skips the recovery diagnostics, the process_recoveries count, and the
            # crash-loop / start-failure breakers entirely, so a deliberate replacement is never mistaken
            # for (or accumulated toward) a crash.
            if intentional_reclaim:
                reason = "idle process cycled to reclaim RAM"
                logger.info(
                    f"Cycling idle process {process_info.process_id} to reclaim "
                    f"{process_info.ram_usage_bytes} bytes of unreleased RAM.",
                )
            else:
                assert intentional_reason is not None
                reason = intentional_reason
                logger.info(
                    f"Replacing process {process_info.process_id} ({reason}); deliberate, not a crash or hang.",
                )
            self._action_ledger.record(
                LedgerEventType.PROCESS_REPLACED,
                process_id=process_info.process_id,
                os_pid=process_info.os_pid,
                launch_identifier=process_info.process_launch_identifier,
                reason=reason,
                detail={
                    "last_state": process_info.last_process_state.name,
                    "ram_usage_bytes": process_info.ram_usage_bytes,
                },
            )
            self._end_inference_process(process_info)
            self._start_inference_process(process_info.process_id, device_index=process_info.device_index)
            return

        failed_model = self._take_recent_load_failure_for_process(process_info.process_id)
        if failed_model is not None:
            # The slot reported it could not load this model, then exited cleanly (its backend may be in an
            # indeterminate state). The fault belongs to the *model*, not the slot, so this is labelled a
            # model-load failure (not the misleading "crashed or hung") and is deliberately kept out of
            # the slot crash-loop/start-failure breakers: those count *slot* sickness, and feeding a poison
            # model's failures into them would quarantine a perfectly healthy slot. The model itself is
            # quarantined by record_model_load_failure once it crosses the threshold.
            will_quarantine = False
            quarantine_reason = ""
            recovery_reason = f"inference process replaced (failed to load model {failed_model})"
        elif self._looks_like_oom_kill(process_info):
            # A SIGKILL (exitcode -9) while system RAM is critically low is the kernel OOM-killer, not the
            # slot crashing: labelling it "crashed or hung" both misleads the post-mortem and feeds the
            # per-slot crash-loop breaker, which would quarantine a healthy slot for a host-wide memory
            # problem no slot teardown can fix. So it is labelled an OS OOM kill and kept out of the slot
            # breakers, the same way a poison-model load failure is; the RAM-pressure governor and pop
            # throttle address the cause. The job still faults retryable (a resource failure earns a retry).
            will_quarantine = False
            quarantine_reason = ""
            recovery_reason = "inference process replaced (likely OS OOM-killed; system RAM critically low)"
        else:
            replacements_in_window = self._record_slot_recovery(process_info.process_id)
            consecutive_start_failures = self._record_start_failure(process_info)
            crash_looped = replacements_in_window > CRASH_LOOP_MAX_REPLACEMENTS
            crash_on_start = consecutive_start_failures >= CRASH_LOOP_MAX_START_FAILURES
            will_quarantine = crash_looped or crash_on_start
            if not will_quarantine:
                quarantine_reason = ""
                recovery_reason = "inference process replaced (crashed or hung)"
            elif crash_on_start:
                quarantine_reason = (
                    f"crash on start: {consecutive_start_failures} consecutive failures before reaching readiness"
                )
                recovery_reason = f"inference slot quarantined ({quarantine_reason})"
            else:
                quarantine_reason = (
                    f"crash loop: {replacements_in_window} replacements within {CRASH_LOOP_WINDOW_SECONDS:.0f}s"
                )
                recovery_reason = f"inference slot quarantined ({quarantine_reason})"
        # Record the recovery while the process info still reflects the faulted state: ending the
        # process first overwrites last_process_state with PROCESS_ENDING and loses that diagnostic.
        self._log_recovery_diagnostics(process_info, recovery_reason)
        raw_exitcode = getattr(process_info.mp_process, "exitcode", None)
        exitcode = raw_exitcode if isinstance(raw_exitcode, int) else None
        detail: dict[str, str | int | float | bool | None] = {
            "last_state": process_info.last_process_state.name,
            "exitcode": exitcode,
        }
        # A nonzero exit means the child actually crashed (vs. a hang, exitcode None): lift the why from
        # its startup-crash file into the ledger so the structured record explains itself even when the
        # per-subprocess human logs are not kept. Only on a real crash, to avoid attributing a stale crash.
        if exitcode not in (0, None):
            crash_signature = read_last_startup_crash(f"inference_{process_info.process_id}")
            if crash_signature is not None:
                detail["crash_signature"] = crash_signature
        self._action_ledger.record(
            LedgerEventType.PROCESS_REPLACED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            job_id=str(job_to_remove.id_) if job_to_remove is not None else None,
            reason=recovery_reason,
            detail=detail,
        )
        self._notify_process_recovery(process_info, recovery_reason)

        self._end_inference_process(process_info)
        self._num_process_recoveries += 1

        if will_quarantine:
            self._quarantine_inference_slot(process_info, quarantine_reason)
            return

        self._start_inference_process(process_info.process_id, device_index=process_info.device_index)

    def get_processes_with_model_for_queued_job(self) -> list[int]:
        """Get the processes that have the model for any queued job."""
        processes_with_model_for_queued_job: list[int] = []

        queued_models = {job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None}
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress if job.model is not None}

        for p in self._process_map.values():
            if (
                p.loaded_horde_model_name in queued_models
                or p.loaded_horde_model_name in in_progress_models
                or p.last_process_state == HordeProcessState.PRELOADED_MODEL
            ):
                processes_with_model_for_queued_job.append(p.process_id)

        return processes_with_model_for_queued_job

    def _hard_kill_processes(
        self,
        inference: bool = True,
        safety: bool = True,
        all_: bool = True,
    ) -> None:
        """Kill all processes immediately."""
        for process_info in self._process_map.values():
            if (
                (inference and process_info.process_type == HordeProcessType.INFERENCE)
                or (safety and process_info.process_type == HordeProcessType.SAFETY)
                or (all_)
            ):
                try:
                    process_info.mp_process.kill()
                    process_info.mp_process.kill()
                    process_info.mp_process.join(1)
                except Exception as e:
                    logger.error(f"Failed to kill process {process_info}: {e}")

        self._process_map.clear()
        self._horde_model_map.root.clear()
        if self._owned_registry is not None:
            self._owned_registry.clear()

    def _check_and_replace_process(
        self,
        process_info: HordeProcessInfo,
        timeout: float,
        state: HordeProcessState,
        error_message: str,
        *,
        use_state_duration: bool = False,
    ) -> bool:
        """Check if a process has been stuck in a state for too long and replace it if it has.

        Returns:
            True if the process was replaced, False otherwise
        """
        now = time.time()
        if use_state_duration:
            time_elapsed = now - process_info.last_process_state_started_at
        else:
            time_elapsed = now - process_info.last_received_timestamp
            time_elapsed = min(time_elapsed, now - process_info.last_heartbeat_timestamp)

        if time_elapsed > timeout and process_info.last_process_state == state:
            logger.error(f"{process_info} {error_message}, replacing it")
            self._action_ledger.record(
                LedgerEventType.TIMEOUT_DETECTED,
                process_id=process_info.process_id,
                os_pid=process_info.os_pid,
                launch_identifier=process_info.process_launch_identifier,
                reason=error_message,
                detail={
                    "state": state.name,
                    "elapsed_s": round(time_elapsed, 1),
                    "timeout_s": timeout,
                    "elapsed_source": "state_duration" if use_state_duration else "message_or_heartbeat_silence",
                },
            )
            if process_info.process_type == HordeProcessType.SAFETY:
                self._log_recovery_diagnostics(process_info, error_message)
                # Arm the replacement before driving it: `_replace_all_safety_process` no-ops unless the
                # flag is set, so omitting this (the historical bug) made the safety branch a silent no-op.
                self._initiate_safety_replacement()
                self._replace_all_safety_process()
            if process_info.process_type == HordeProcessType.POST_PROCESS:
                # A lane reaped silent mid-post-processing means the upscaler/face-fixer peak could not be
                # hosted (or the pass wedged). Feed the feature-level circuit breaker so a run of these
                # disables post-processing before the worker keeps faulting into forced-maintenance.
                if state == HordeProcessState.POST_PROCESSING:
                    self._job_tracker.note_post_processing_overcommit_fault()
                self._initiate_post_process_replacement()
                self._replace_all_post_process_process()
            if process_info.process_type == HordeProcessType.INFERENCE:
                self._replace_inference_process(process_info)
            return True
        return False

    def _grade_running_inference(self) -> None:
        """Grade in-flight inference against its expected sampling time, escalating notices for slow jobs.

        The hard kill remains the ``inference_step_timeout`` in :meth:`replace_hung_processes`; this adds
        the softer, evidence-based rungs below it (a job measurably slower than its signature's expected
        time) so a slowdown is logged, audited, and counted toward the recovery-supervisor severity
        before the watchdog resorts to replacing the slot. A job with no expected time (cold start) is
        skipped, so this never fires on an uncalibrated worker.

        Sampling time is measured from the first sampling step (``current_first_step_at``), not from
        dispatch: the pre-sampling work a job's features make legitimate (cold VRAM load, aux/ControlNet
        download, prompt encode, hires/post-processing framing) emits no step, so grading it against the
        sampling-only expectation would mislabel a heavy or cold job as a hang candidate. A slot that has
        not yet emitted a step is therefore left to the first-step grace, not graded here.
        """
        now = time.time()
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.last_process_state != HordeProcessState.INFERENCE_STARTING:
                continue
            first_step_at = process_info.current_first_step_at
            expected = process_info.current_job_expected_sampling_seconds
            if first_step_at is None or expected is None or expected <= 0:
                continue

            elapsed = now - first_step_at
            ratio = elapsed / expected
            level = 2 if ratio >= SLOWDOWN_WARN_RATIO else 1 if ratio >= SLOWDOWN_NOTICE_RATIO else 0
            if level <= process_info.current_job_slowdown_level:
                continue
            process_info.current_job_slowdown_level = level

            job_id = (
                str(process_info.last_job_referenced.id_) if process_info.last_job_referenced is not None else None
            )
            if level >= 2:
                self._num_slowdown_events += 1
                # Include the residency snapshot: a slowdown this severe on a small-VRAM card is usually
                # the over-commit thrash (weights spilling to system RAM), so capturing which models were
                # resident and the device-wide free VRAM at this moment is the key live diagnostic.
                logger.warning(
                    f"Inference on process {process_info.process_id} is {ratio:.1f}x its expected sampling time "
                    f"({elapsed:.0f}s vs ~{expected:.0f}s); watching for a hang. "
                    f"{self._process_map.residency_snapshot()}",
                )
                self._action_ledger.record(
                    LedgerEventType.SLOWDOWN_DETECTED,
                    process_id=process_info.process_id,
                    os_pid=process_info.os_pid,
                    launch_identifier=process_info.process_launch_identifier,
                    job_id=job_id,
                    reason=f"{ratio:.1f}x expected sampling time",
                    detail={"elapsed_s": round(elapsed, 1), "expected_s": round(expected, 1)},
                )
            else:
                logger.info(
                    f"Inference on process {process_info.process_id} is running slower than expected "
                    f"({ratio:.1f}x ~{expected:.0f}s); not yet a concern.",
                )

    def _replace_if_paged_and_slow(
        self,
        process_info: HordeProcessInfo,
        paging_victims: dict[int, float],
        now: float,
    ) -> bool:
        """Replace a crawling sampler as the last reclaim rung once a card is wedged over the paging cliff.

        The silence-based hang watchdog cannot see this failure mode: when WDDM demotes a card's VRAM to
        system memory, the sampling job keeps emitting steps at seconds-per-iteration pace, so every step
        refreshes the heartbeat and the silence timeout never trips. The card is effectively lost for minutes
        while the job limps to completion. This kill is the terminal rung of the reclaim ladder, reached only
        after every softer rung has failed, and it gates on device-level truth rather than per-PID attribution:

        - The device has been continuously SATURATED (device-free below the hard floor) for at least
          :data:`SATURATION_KILL_MIN_SECONDS`.
        - The verified reclaim ladder has exhausted itself on that card without relieving it
          (``saturation_unresolved``): idle-model unload, cache release, lane pause, and safety off-GPU all ran
          and the card is still over the cliff.
        - This slot is crawling: its per-step floor tripped (steps each running several times their expected
          pace) or its whole-job elapsed grade reached WARN.

        The per-PID PDH paging-victim map is deliberately NOT a gate. The measured LRU physics make it
        structurally unsatisfiable here: WDDM demotes the least-recently-touched allocator, so the process that
        goes slow (the active sampler) and the process whose shared memory grows (the idle squatter) are
        usually different pids, and requiring the slow slot's own pid to appear in the victim set almost never
        held. The map is retained only as a logging hint. Because the ladder is exhausted and the card is still
        SATURATED, the crawling sampler is the last thing left to give the card back. The killed job faults as a
        resource fault, so it earns the single degraded, isolated retry (``RETRY_DEGRADED``) that clears the
        card for it rather than a plain re-dispatch onto another over-committed slot. The crawl signals reset at
        the next job boundary, so one replacement per (slot, job) is guaranteed; the recovery debounce in
        :meth:`replace_hung_processes` further prevents re-entry while the fresh slot spins up. Returns True when
        the slot was replaced.
        """
        if process_info.process_type != HordeProcessType.INFERENCE:
            return False
        # Confine to a slot that is actually sampling. The crawl signals are only cleared on the next job's
        # dispatch, so a slot that finished a slow job and is now idle can still carry them; without this guard,
        # saturation lingering into that gap would reap a healthy idle slot.
        if process_info.last_process_state != HordeProcessState.INFERENCE_STARTING:
            return False

        device_index = process_info.device_index
        saturated_seconds = self._device_saturation_duration_provider(device_index)
        if saturated_seconds < SATURATION_KILL_MIN_SECONDS:
            return False
        if not self._saturation_unresolved_provider(device_index):
            return False

        crawling = (
            process_info.current_job_per_step_floor_tripped
            or process_info.current_job_slowdown_level >= SLOWDOWN_WARN_LEVEL
        )
        if not crawling:
            return False

        os_pid = process_info.os_pid
        expected = process_info.current_job_expected_sampling_seconds
        first_step_at = process_info.current_first_step_at
        ratio = (now - first_step_at) / expected if (first_step_at is not None and expected) else float("nan")
        model = process_info.loaded_horde_model_name
        job_id = str(process_info.last_job_referenced.id_) if process_info.last_job_referenced is not None else None
        # PDH per-PID attribution is hint-only now: surface the slot's shared figure if it happens to appear,
        # but never gate on it.
        shared_hint_mb = paging_victims.get(os_pid) if os_pid is not None else None

        reason = (
            f"device {device_index} SATURATED {saturated_seconds:.0f}s with the reclaim ladder exhausted and "
            f"this slot crawling ({ratio:.1f}x expected sampling time); replacing to reclaim the card"
        )
        logger.warning(
            f"Inference slot {process_info.process_id} (pid {os_pid}, model {model}) is the last reclaim rung: "
            f"device {device_index} has been SATURATED for {saturated_seconds:.0f}s, every softer reclaim rung "
            f"is exhausted, and the slot is crawling at {ratio:.1f}x its expected sampling time. Replacing it to "
            "reclaim the card."
            + (f" (shared-VRAM hint {shared_hint_mb:.0f}MB)" if shared_hint_mb is not None else ""),
        )
        self._action_ledger.record(
            LedgerEventType.TIMEOUT_DETECTED,
            process_id=process_info.process_id,
            os_pid=os_pid,
            launch_identifier=process_info.process_launch_identifier,
            job_id=job_id,
            reason=reason,
            detail={
                "slowdown_ratio": round(ratio, 2),
                "saturated_seconds": round(saturated_seconds, 1),
                "per_step_floor_tripped": process_info.current_job_per_step_floor_tripped,
            },
        )
        self._paging_victim_replacements += 1
        self._replace_inference_process(process_info, resource_fault_reason=reason)
        return True

    def _effective_inference_step_timeout(self, bridge_data: reGenBridgeData, process_info: HordeProcessInfo) -> int:
        """Per-step hang timeout for a slot, widened when it is doing legitimate heartbeat-silent heavy work.

        The flat ``inference_step_timeout`` was calibrated for a light job on an uncontended device. On a
        multi-process worker it false-kills healthy jobs in two ways the live logs show: a single sampling
        step stretched past it by co-residence contention, and a feature phase (hires second pass, VAE
        decode, post-processing setup) that runs inside ``INFERENCE_STARTING`` and emits no step beat for
        far longer than a step. Neither is a hang. This widens the per-step grace, up to
        ``contended_step_timeout`` (floored at ``inference_step_timeout``), only when there is positive
        evidence of such work, so a light single job keeps the tight timeout and a genuinely wedged slot is
        still reaped once it has been continuously silent past the (bounded) grace.

        Precedence, all floored at the base per-step timeout:

        - An over-budget / exclusive admit keeps its dedicated ``overbudget_step_timeout`` (a heavy model
          streaming weights through VRAM every step), unchanged from before and taking priority.
        - The full ``contended_step_timeout`` is granted when the last heartbeat was a
          ``PIPELINE_STATE_CHANGE`` (a heavy non-step phase is running), the slot has been graded
          contention-slowed (``current_job_slowdown_level``), or the job's signature is feature-heavy.
        - Otherwise the grace scales with the job's expected sampling work (heavier job, longer legitimate
          silences), bounded by ``contended_step_timeout``.
        """
        base = bridge_data.inference_step_timeout
        job = process_info.last_job_referenced
        if job is None:
            return base

        if self._job_tracker.is_admitted_over_budget(job) or self._job_tracker.is_admitted_exclusive(job):
            overbudget = bridge_data.overbudget_step_timeout
            if not isinstance(overbudget, int) or isinstance(overbudget, bool):
                return base
            return max(base, overbudget)

        contended = bridge_data.contended_step_timeout
        if not isinstance(contended, int) or isinstance(contended, bool):
            return base
        ceiling = max(base, contended)

        if process_info.last_heartbeat_type == HordeHeartbeatType.PIPELINE_STATE_CHANGE:
            return ceiling
        if process_info.current_job_slowdown_level >= 1:
            return ceiling
        if _job_is_feature_heavy(process_info):
            return ceiling

        expected = process_info.current_job_expected_sampling_seconds
        if isinstance(expected, (int, float)) and not isinstance(expected, bool) and expected > 0:
            scaled = math.ceil(expected * STEP_TIMEOUT_WORK_FACTOR)
            return max(base, min(ceiling, scaled))
        return base

    def _effective_aux_download_timeout(self, bridge_data: reGenBridgeData) -> float:
        """Stuck-aux-download grace, shortened once the LoRA-download backoff is active.

        A healthy worker reaps a stalled aux download at the configured ``download_timeout``. While a
        backoff incident is active the download path is known to be failing, so a requeued job that
        keeps timing out is reaped at the much shorter ``FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS`` (never
        above the configured timeout) rather than holding a slot for the full window again. The
        shortening self-expires with the incident, so a recovered worker reverts to the full grace.
        """
        configured = bridge_data.download_timeout
        if not self._state.lora_download_backoff.is_escalation_active(time.time()):
            return configured
        return min(configured, FAST_AUX_DOWNLOAD_TIMEOUT_SECONDS)

    def aux_download_deadline_for_dispatch(self, bridge_data: reGenBridgeData) -> float:
        """The child-side aux-download budget to hand a job being dispatched now.

        Set to the current (backoff-aware) stuck-aux watchdog timeout minus a margin, floored, so the
        child gives up on a stalled download and faults the job a beat before the watchdog would tear the
        process down. Computed at dispatch, so it reflects the backoff state at that moment.
        """
        watchdog = self._effective_aux_download_timeout(bridge_data)
        return max(MIN_AUX_DOWNLOAD_DEADLINE_SECONDS, watchdog - AUX_DOWNLOAD_DEADLINE_MARGIN_SECONDS)

    def replace_hung_processes(self) -> bool:
        """Replaces processes that haven't checked in since `process_timeout` seconds in bridgeData."""
        import threading

        bridge_data = self._runtime_config.bridge_data

        def timed_unset_recently_recovered() -> None:
            time.sleep(bridge_data.inference_step_timeout)
            self._recently_recovered = False

        now = time.time()

        # A live inference slot that has advanced past PROCESS_STARTING has proven it can initialise,
        # so clear any consecutive crash-on-start streak it accrued. Only slots that never get past
        # startup keep accumulating toward the crash-on-start breaker in `_replace_inference_process`.
        for live_process in self._process_map.values():
            if (
                live_process.process_type == HordeProcessType.INFERENCE
                and live_process.last_process_state != HordeProcessState.PROCESS_STARTING
            ):
                self._slot_consecutive_start_failures.pop(live_process.process_id, None)

        # Soft, evidence-based slowdown grading runs every tick (cheap, no side effects beyond logging
        # and audit) regardless of the recovery debounce below, which only gates hard replacement.
        self._grade_running_inference()

        any_replaced = False

        # Definitive crashes (the OS process has exited) are reaped immediately, even right after
        # another recovery. A dead child sends no further messages, so deferring its reap behind the
        # recent-recovery debounce would wedge the worker for the debounce window on every burst of
        # crashes (e.g. a replacement that also dies). Unbounded respawns are prevented by the
        # crash-loop circuit breaker in `_replace_inference_process`, not by this debounce.
        # Snapshot the values: recovering a process can mutate the map (safety end/delete/restart).
        for process_info in list(self._process_map.values()):
            if self._reap_if_crashed(process_info):
                any_replaced = True

        if any_replaced:
            self._recently_recovered = True
            threading.Thread(target=timed_unset_recently_recovered).start()

        # The hang/timeout heuristics below are debounced: a just-replaced process is still spinning
        # up and would otherwise trip the startup / all-processes-timed-out checks before it reports in.
        if self._recently_recovered:
            return any_replaced

        # The first sampling step (which trails the cold model-load/encode work) gets a longer grace than
        # a steady step; is_stuck_on_inference floors this at the per-step timeout, so a config value
        # below it is harmless.
        first_step_timeout = bridge_data.inference_first_step_timeout

        # A slot stuck repeating one sampling step is not silent (it keeps heart-beating), so the
        # silence-based timeout above cannot catch it; the child forwards a non-advancing-repeat count
        # that this limit reaps on instead. Guard the type so a mocked config in tests cannot misfire.
        stuck_step_limit = bridge_data.inference_stuck_step_repeat_limit
        if not isinstance(stuck_step_limit, int) or isinstance(stuck_step_limit, bool):
            stuck_step_limit = None

        # The parent's freshest WDDM paging attribution (os_pid -> shared MB), empty when there is no
        # current verdict. Read once so every slot is judged against the same snapshot this tick.
        paging_victims = self._wddm_paging_victims_provider(WDDM_PAGING_VICTIM_MAX_AGE_SECONDS)

        for process_info in list(self._process_map.values()):
            if self._replace_if_paged_and_slow(process_info, paging_victims, now):
                any_replaced = True
                self._recently_recovered = True
                threading.Thread(target=timed_unset_recently_recovered).start()
                continue
            if self._process_map.is_stuck_on_inference(
                process_info.process_id,
                self._effective_inference_step_timeout(bridge_data, process_info),
                first_step_timeout,
            ):
                logger.error(f"{process_info} seems to be stuck mid inference, replacing it")
                self._action_ledger.record(
                    LedgerEventType.TIMEOUT_DETECTED,
                    process_id=process_info.process_id,
                    os_pid=process_info.os_pid,
                    launch_identifier=process_info.process_launch_identifier,
                    reason="stuck mid inference (no step progress within inference_step_timeout)",
                )
                self._replace_inference_process(process_info)
                any_replaced = True
                self._recently_recovered = True
                threading.Thread(target=timed_unset_recently_recovered).start()
            elif stuck_step_limit is not None and self._process_map.is_stuck_on_nonadvancing_step(
                process_info.process_id,
                stuck_step_limit,
            ):
                logger.error(
                    f"Inference slot {process_info.process_id} is stuck on a non-advancing sampling step "
                    f"(reported step {process_info.last_current_step}/{process_info.last_total_steps} without "
                    f"advancing {process_info.nonadvancing_step_repeats} times); the ComfyUI generation will "
                    f"not return a result, replacing it (stuck-step watchdog).",
                )
                self._action_ledger.record(
                    LedgerEventType.TIMEOUT_DETECTED,
                    process_id=process_info.process_id,
                    os_pid=process_info.os_pid,
                    launch_identifier=process_info.process_launch_identifier,
                    reason="stuck on a non-advancing sampling step (stuck-step watchdog)",
                )
                self._replace_inference_process(process_info)
                any_replaced = True
                self._recently_recovered = True
                threading.Thread(target=timed_unset_recently_recovered).start()
            else:
                conditions: list[tuple[float, HordeProcessState, str, bool]] = [
                    (
                        bridge_data.preload_timeout,
                        HordeProcessState.PRELOADING_MODEL,
                        "seems to be stuck preloading a model",
                        False,
                    ),
                    (
                        self._effective_aux_download_timeout(bridge_data),
                        HordeProcessState.DOWNLOADING_AUX_MODEL,
                        "seems to be stuck downloading an auxiliary model (LoRa, etc)",
                        True,
                    ),
                    (
                        bridge_data.preload_timeout,
                        HordeProcessState.PROCESS_STARTING,
                        "seems to be stuck starting",
                        False,
                    ),
                    (
                        bridge_data.post_process_timeout + (3 * bridge_data.max_batch),
                        HordeProcessState.POST_PROCESSING,
                        "seems to be stuck post processing",
                        False,
                    ),
                    (
                        bridge_data.process_timeout,
                        HordeProcessState.WAITING_FOR_JOB,
                        "seems to be stuck idle (silent) while there is work to do",
                        False,
                    ),
                ]
                if self._state.last_pop_no_jobs_available:
                    continue

                for timeout, state, error_message, use_state_duration in conditions:
                    if self._check_and_replace_process(
                        process_info,
                        timeout,
                        state,
                        error_message,
                        use_state_duration=use_state_duration,
                    ):
                        any_replaced = True
                        self._recently_recovered = True

        if self._state.last_pop_no_jobs_available:
            return any_replaced

        # ``all(...)`` over an empty map is vacuously True, which would falsely declare "all processes
        # unresponsive" whenever no inference/safety process is running yet: during the startup
        # download-and-scan window and, deliberately, throughout download-only mode. Require at least one
        # process to exist before the all-timed-out verdict can hold, and never declare it while the worker
        # is explicitly held for downloads (it runs no inference by design; the hold is the authority that
        # this is intentional, not a wedge; it is cleared on go-live / start).
        all_processes_timed_out = (
            bool(self._process_map)
            and not self._state.downloads_only_hold
            and all(
                ((now - process_info.last_received_timestamp) > bridge_data.process_timeout)
                for process_info in self._process_map.values()
            )
        )

        shutdown_timed_out = self._state.shutting_down and (now - self._state.shutting_down_time) > (60 * 5)

        if (all_processes_timed_out and not (self._state.last_pop_no_jobs_available or self._recently_recovered)) or (
            shutdown_timed_out
        ):
            if not self._hung_processes_detected:
                self._hung_processes_detected = True
                self._hung_processes_detected_time = now

            last_detected_delta = now - self._hung_processes_detected_time

            if last_detected_delta < 20:
                return False

            self._job_tracker._purge_jobs()

            if bridge_data.exit_on_unhandled_faults or self._state.shutting_down:
                logger.error("All processes have been unresponsive for too long, exiting.")

                self._abort_callback()
                if bridge_data.exit_on_unhandled_faults:
                    logger.error("Exiting due to exit_on_unhandled_faults being enabled")

                return True

            logger.error("All processes have been unresponsive for too long, attempting to recover.")
            self._recently_recovered = True

            for process_info in self._process_map.values():
                if process_info.process_type == HordeProcessType.INFERENCE:
                    self._replace_inference_process(process_info)
                    self._any_replaced = True

            threading.Thread(target=timed_unset_recently_recovered).start()
        else:
            self._hung_processes_detected = False

        if any_replaced:
            threading.Thread(target=timed_unset_recently_recovered).start()

        return any_replaced

    @property
    def safety_processes_should_be_replaced(self) -> bool:
        """Whether the safety processes should be replaced."""
        return self._safety_processes_should_be_replaced

    @safety_processes_should_be_replaced.setter
    def safety_processes_should_be_replaced(self, value: bool) -> None:
        self._safety_processes_should_be_replaced = value
