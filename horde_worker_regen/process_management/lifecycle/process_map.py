"""A mapping of process IDs to HordeProcessInfo objects."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import override

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from hordelib.metrics import DownloadEvent, JobPhaseMetrics
from loguru import logger
from pydantic import ConfigDict

from horde_worker_regen.consts import KNOWN_CONTROLNET_WORKFLOWS, VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.ipc.messages import (
    HordeHeartbeatType,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo

_EXPECTED_PROCESS_STATE_SOURCES: dict[HordeProcessState, frozenset[HordeProcessState]] = {
    HordeProcessState.DOWNLOAD_COMPLETE: frozenset({HordeProcessState.DOWNLOADING_MODEL}),
    HordeProcessState.DOWNLOAD_AUX_COMPLETE: frozenset({HordeProcessState.DOWNLOADING_AUX_MODEL}),
    HordeProcessState.PRELOADED_MODEL: frozenset({HordeProcessState.PRELOADING_MODEL}),
    HordeProcessState.PRELOADING_MODEL: frozenset(
        {
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.JOB_RECEIVED,
            HordeProcessState.DOWNLOAD_COMPLETE,
            HordeProcessState.DOWNLOADING_AUX_MODEL,
            HordeProcessState.DOWNLOAD_AUX_COMPLETE,
            HordeProcessState.INFERENCE_COMPLETE,
            HordeProcessState.INFERENCE_FAILED,
            HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
            HordeProcessState.PRELOADED_MODEL,
        },
    ),
    HordeProcessState.INFERENCE_STARTING: frozenset(
        {
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.JOB_RECEIVED,
            HordeProcessState.PRELOADING_MODEL,
            HordeProcessState.PRELOADED_MODEL,
            HordeProcessState.INFERENCE_COMPLETE,
            HordeProcessState.DOWNLOADING_AUX_MODEL,
            HordeProcessState.DOWNLOAD_AUX_COMPLETE,
            HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
        },
    ),
    HordeProcessState.INFERENCE_POST_PROCESSING: frozenset({HordeProcessState.INFERENCE_STARTING}),
    HordeProcessState.INFERENCE_COMPLETE: frozenset(
        {HordeProcessState.INFERENCE_STARTING, HordeProcessState.INFERENCE_POST_PROCESSING},
    ),
    HordeProcessState.INFERENCE_FAILED: frozenset(
        {
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.JOB_RECEIVED,
        },
    ),
    HordeProcessState.EVALUATING_SAFETY: frozenset({HordeProcessState.WAITING_FOR_JOB}),
    HordeProcessState.SAFETY_FAILED: frozenset({HordeProcessState.EVALUATING_SAFETY}),
}
"""Expected previous states for selected process states.

States absent from this table (idle/teardown states and download/unload starts) may be
entered from anywhere; child processes are authoritative about their own state, so an
unexpected transition is logged but never refused. Divergence here usually means the
parent's optimistic bookkeeping and the child's reality have drifted apart.
"""

_RETIRED_LAUNCH_TTL_SECONDS = 300.0
"""How long intentionally retired process launches are kept for late IPC-message matching."""

_MAX_RETIRED_LAUNCHES = 256
"""Hard cap for the retired-launch tombstone registry."""


@dataclass(frozen=True)
class RetiredProcessLaunch:
    """A bounded tombstone for an intentionally retired child-process launch."""

    process_id: int
    process_launch_identifier: int
    process_type: HordeProcessType
    reason: str
    retired_at: float


class ProcessMap(dict[int, HordeProcessInfo]):
    """A mapping of process IDs to HordeProcessInfo objects.

    There are a number of helper methods on this class for querying the state of processes, such as how many are
    busy, how many are doing inference, etc. In addition, there are a number of methods for updating the state of
    processes based on messages received from them, such as heartbeats, memory reports, and process state changes.

    See `on_heartbeat`, `on_memory_report`, `on_process_state_change`, `on_last_job_reference_change`, and
    `on_model_load_state_change` for more details on how the process map is updated based on messages from processes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        initial: Mapping[int, HordeProcessInfo] | Iterable[tuple[int, HordeProcessInfo]] | None = None,
    ) -> None:
        """Initialize the process map and its retired-launch tombstone registry."""
        super().__init__()
        if initial is not None:
            self.update(initial)
        self._retired_launches: dict[tuple[int, int], RetiredProcessLaunch] = {}

    def retire_process(self, process_info: HordeProcessInfo, reason: str) -> HordeProcessInfo | None:
        """Remove a process from the active map and remember its launch for late IPC messages.

        Args:
            process_info: The process launch being intentionally removed from active scheduling state.
            reason: Short reason recorded with the tombstone for diagnostics.

        Returns:
            The removed process info, or None if the active map no longer contains that process id.
        """
        self._prune_retired_launches()
        key = (process_info.process_id, process_info.process_launch_identifier)
        self._retired_launches[key] = RetiredProcessLaunch(
            process_id=process_info.process_id,
            process_launch_identifier=process_info.process_launch_identifier,
            process_type=process_info.process_type,
            reason=reason,
            retired_at=time.time(),
        )
        self._prune_retired_launches()
        return self.pop(process_info.process_id, None)

    def get_retired_launch(
        self,
        process_id: int,
        process_launch_identifier: int,
    ) -> RetiredProcessLaunch | None:
        """Return the retired launch matching this exact process id and launch id, if still retained."""
        self._prune_retired_launches()
        return self._retired_launches.get((process_id, process_launch_identifier))

    def is_retired_launch(self, process_id: int, process_launch_identifier: int) -> bool:
        """Return whether the given process id/launch id pair was intentionally retired."""
        return self.get_retired_launch(process_id, process_launch_identifier) is not None

    def is_launch_active(self, process_id: int, process_launch_identifier: int) -> bool:
        """Return whether this exact process launch is the one currently occupying its slot.

        A launch stops being active once it is replaced (crash recovery installs a new, higher launch
        identifier under a fresh pid) or removed (scale-down, quarantine). Callers holding a reference to
        work that was dispatched to a now-dead launch use this to detect that its result will never
        arrive, since results only ever come from the exact launch the work was sent to.
        """
        info = self.get(process_id)
        return info is not None and info.process_launch_identifier == process_launch_identifier

    def _prune_retired_launches(self) -> None:
        """Prune expired tombstones and enforce the registry size cap."""
        now = time.time()
        expired_keys = [
            key
            for key, retired in self._retired_launches.items()
            if now - retired.retired_at > _RETIRED_LAUNCH_TTL_SECONDS
        ]
        for key in expired_keys:
            self._retired_launches.pop(key, None)

        overflow = len(self._retired_launches) - _MAX_RETIRED_LAUNCHES
        if overflow <= 0:
            return

        oldest_keys = sorted(self._retired_launches, key=lambda key: self._retired_launches[key].retired_at)
        for key in oldest_keys[:overflow]:
            self._retired_launches.pop(key, None)

    def on_heartbeat(
        self,
        process_id: int,
        heartbeat_type: HordeHeartbeatType,
        *,
        percent_complete: int | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        iterations_per_second: float | None = None,
    ) -> None:
        """Update the heartbeat for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            heartbeat_type (HordeHeartbeatType): The type of the heartbeat.
            percent_complete (int | None, optional): The percentage of the job that has been completed, \
                if applicable. Defaults to None.
            current_step (int | None, optional): The current sampling step, if applicable. Defaults to None.
            total_steps (int | None, optional): The total sampling steps, if applicable. Defaults to None.
            iterations_per_second (float | None, optional): The instantaneous sampling rate, \
                if applicable. Defaults to None.
        """
        self[process_id].last_heartbeat_delta = time.time() - self[process_id].last_heartbeat_timestamp
        self[process_id].last_received_timestamp = time.time()
        self[process_id].last_heartbeat_timestamp = time.time()
        self[process_id].last_heartbeat_type = heartbeat_type
        if heartbeat_type == HordeHeartbeatType.INFERENCE_STEP:
            self[process_id].heartbeats_inference_steps += 1
        else:
            self[process_id].heartbeats_inference_steps = 0

        self[process_id].last_heartbeat_percent_complete = percent_complete

        if heartbeat_type == HordeHeartbeatType.INFERENCE_STEP:
            if self[process_id].current_first_step_at is None:
                # First sampling step of this job: the slot has finished its one-time pre-sampling work,
                # so start the clock the graded-slowdown monitor measures sampling time against.
                self[process_id].current_first_step_at = time.time()
            self[process_id].last_current_step = current_step
            self[process_id].last_total_steps = total_steps
            self[process_id].last_iterations_per_second = iterations_per_second

    def on_process_ending(self, process_id: int) -> None:
        """Update the process map when a process has ended.

        Args:
            process_id (int): The ID of the process that has ended.
        """
        self[process_id].last_process_state = HordeProcessState.PROCESS_ENDING
        self[process_id].last_process_state_started_at = time.time()
        self[process_id].loaded_horde_model_name = None
        self[process_id].loaded_horde_model_baseline = None
        self[process_id].last_job_referenced = None
        self[process_id].batch_amount = 1

        # Drop this slot's last VRAM/RAM sample. A dead process reports nothing further, so its final
        # pre-death figure would otherwise persist in get_free_vram_mb()/RAM accounting and either keep
        # counting freed VRAM as "used" (under-counting headroom) or, once the device reclaims it,
        # over-state headroom. Zeroing total_vram_mb also drops the slot from get_free_vram_mb()'s
        # reporting set until its replacement re-reports real numbers.
        self[process_id].ram_usage_bytes = 0
        self[process_id].vram_usage_mb = 0
        self[process_id].total_vram_mb = 0

        self.reset_heartbeat_state(process_id)

        self[process_id].last_received_timestamp = time.time()

    def on_memory_report(
        self,
        process_id: int,
        ram_usage_bytes: int,
        vram_usage_mb: int | None = 0,
        total_vram_mb: int | None = 0,
    ) -> None:
        """Update the memory usage for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            ram_usage_bytes (int): The amount of RAM used by this process.
            vram_usage_mb (int): The amount of VRAM used by this process.
            total_vram_mb (int): The total amount of VRAM available to this process.
        """
        self[process_id].ram_usage_bytes = ram_usage_bytes
        self[process_id].vram_usage_mb = vram_usage_mb or 0
        self[process_id].total_vram_mb = total_vram_mb or 0

        self[process_id].last_received_timestamp = time.time()

        logger.debug(
            f"Process {process_id} memory report: "
            f"ram: {ram_usage_bytes} vram: {vram_usage_mb} total vram: {total_vram_mb}",
        )

    def on_job_metrics(self, process_id: int, phase_metrics: JobPhaseMetrics) -> None:
        """Record a finished job's metrics snapshot for the given process ID."""
        process_info = self[process_id]
        process_info.last_job_metrics = phase_metrics
        process_info.last_received_timestamp = time.time()

        if phase_metrics.vram_used_high_water_mb is not None:
            process_info.vram_used_high_water_mb = max(
                process_info.vram_used_high_water_mb,
                phase_metrics.vram_used_high_water_mb,
            )
        if phase_metrics.ram_used_high_water_mb is not None:
            process_info.ram_used_high_water_mb = max(
                process_info.ram_used_high_water_mb,
                phase_metrics.ram_used_high_water_mb,
            )

    def on_download_metrics(self, process_id: int, events: list[DownloadEvent]) -> None:
        """Record ad-hoc download events reported by the given process ID."""
        self[process_id].cumulative_download_events.extend(events)
        self[process_id].last_received_timestamp = time.time()

    def on_process_state_change(self, process_id: int, new_state: HordeProcessState) -> None:
        """Update the process state for the given process ID.

        Unexpected transitions (per ``_EXPECTED_PROCESS_STATE_SOURCES``) are logged but
        still applied, since the reporting process is the source of truth for its own state.

        Args:
            process_id (int): The ID of the process to update.
            new_state (HordeProcessState): The new state of the process.
        """
        old_state = self[process_id].last_process_state
        expected_sources = _EXPECTED_PROCESS_STATE_SOURCES.get(new_state)
        if expected_sources is not None and old_state != new_state and old_state not in expected_sources:
            logger.warning(
                f"Process {process_id} made an unexpected state transition: {old_state.name} -> {new_state.name}",
            )

        now = time.time()
        if old_state != new_state:
            self[process_id].last_process_state_started_at = now
        self[process_id].last_process_state = new_state
        self[process_id].last_received_timestamp = now

        if (
            new_state == HordeProcessState.INFERENCE_COMPLETE
            or new_state == HordeProcessState.INFERENCE_FAILED
            or new_state == HordeProcessState.PRELOADED_MODEL
            or new_state == HordeProcessState.WAITING_FOR_JOB
        ):
            self.reset_heartbeat_state(process_id)

    def on_last_job_reference_change(
        self,
        process_id: int,
        last_job_referenced: ImageGenerateJobPopResponse | None,
    ) -> None:
        """Update the job reference for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            last_job_referenced (ImageGenerateJobPopResponse | None): The last job referenced by this process.
        """
        if last_job_referenced is not None and (last_job_referenced != self[process_id].last_job_referenced):
            logger.debug(f"Resetting heartbeat for process {process_id}")
            self[process_id].last_heartbeat_delta = 0
            self[process_id].last_heartbeat_timestamp = time.time()
            self[process_id].heartbeats_inference_steps = 0

        self[process_id].last_job_referenced = last_job_referenced
        self[process_id].last_received_timestamp = time.time()

    def on_model_load_state_change(
        self,
        process_id: int,
        horde_model_name: str | None,
        horde_model_baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None = None,
        last_job_referenced: ImageGenerateJobPopResponse | None = None,
    ) -> None:
        """Update the model load state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            horde_model_name (str): The name of the horde model to update.
            horde_model_baseline (KNOWN_IMAGE_GENERATION_BASELINE): The baseline of the horde model to update.
            last_job_referenced (ImageGenerateJobPopResponse | None, optional): The last job referenced by this \
                 process. Defaults to None.
        """
        if horde_model_name is not None:
            self[process_id].recently_unloaded_from_ram = False

        self[process_id].loaded_horde_model_name = horde_model_name
        self[process_id].loaded_horde_model_baseline = horde_model_baseline

        self[process_id].last_received_timestamp = time.time()
        if last_job_referenced is not None:
            if (
                self[process_id].last_job_referenced is not None
                and last_job_referenced != self[process_id].last_job_referenced
            ):
                logger.debug(f"Resetting heartbeat for process {process_id}")
                self.reset_heartbeat_state(process_id)
            self[process_id].last_job_referenced = last_job_referenced

    def on_model_ram_clear(
        self,
        process_id: int,
    ) -> None:
        """Update the model load state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
        """
        self[process_id].loaded_horde_model_name = None
        self[process_id].loaded_horde_model_baseline = None
        self[process_id].last_job_referenced = None
        self[process_id].recently_unloaded_from_ram = True
        self[process_id].last_received_timestamp = time.time()

    def reset_heartbeat_state(self, process_id: int) -> None:
        """Reset the heartbeat state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
        """
        logger.debug(f"Resetting heartbeat for process {process_id}")
        self[process_id].last_heartbeat_delta = 0
        self[process_id].last_heartbeat_timestamp = time.time()
        self[process_id].heartbeats_inference_steps = 0
        self[process_id].last_heartbeat_percent_complete = None
        # Sampling progress is job-scoped: clear it on reset so a finished job's final step/it-s do not
        # linger and get rendered over an idle process (these are only ever set by INFERENCE_STEP beats).
        self[process_id].last_current_step = None
        self[process_id].last_total_steps = None
        self[process_id].last_iterations_per_second = None
        self[process_id].current_first_step_at = None

    def delete_safety_processes(self) -> None:
        """Clear all safety processes."""
        processes_to_delete = []
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                processes_to_delete.append(p)

        for process_info in processes_to_delete:
            logger.debug(f"Deleting safety process {process_info.process_id} from process map")
            self.retire_process(process_info, "safety process replacement")

    def is_stuck_on_inference(
        self,
        process_id: int,
        inference_step_timeout: int,
        first_step_timeout: int | None = None,
    ) -> bool:
        """Return true if a process is in inference but has shown no progress within the timeout.

        ``last_heartbeat_timestamp`` advances on every heartbeat the slot emits (including the
        PRELOAD_MODEL heartbeats it sends while staged in ``PRELOADED_MODEL``) and on every
        INFERENCE_STEP, so once the slot enters ``INFERENCE_STARTING`` the wall-clock gap since it is the
        time the slot has gone silent with no sampling progress. Measuring that gap live
        (rather than the previously-used ``last_heartbeat_delta``, which freezes at its last computed
        value the instant heartbeats stop arriving) catches a true hang where the child simply goes
        silent. It also catches a slot wedged *before* its first step (hung at 0%, having never
        emitted an INFERENCE_STEP): the old heartbeat-type and ``percent_complete < 1`` gates gave
        that case a free pass, which is exactly the 0%-hang gap this overhaul closes.

        The first sampling step is special: before it arrives (``last_current_step is None``, cleared at
        every job boundary) the slot is still doing one-time pre-sampling work (streaming a large model's
        components through VRAM, the initial prompt encode) that is legitimately far slower than a
        steady step. When a ``first_step_timeout`` is supplied it governs that pre-first-step window so a
        slow cold start is not mistaken for a hang; the tighter per-step timeout applies once at least one
        step has been observed.
        """
        process_info = self[process_id]
        if process_info.last_process_state != HordeProcessState.INFERENCE_STARTING:
            return False

        timeout = inference_step_timeout
        if first_step_timeout is not None and process_info.last_current_step is None:
            timeout = max(inference_step_timeout, first_step_timeout)

        return (time.time() - process_info.last_heartbeat_timestamp) > timeout

    def get_capable_processes(self, capability: WorkerCapability) -> list[HordeProcessInfo]:
        """Return all processes declaring the given capability.

        Job routing keys on capabilities, not process types; new job kinds (alchemy now,
        audio/video later) dispatch through this rather than growing per-type query methods.
        """
        return [p for p in self.values() if capability in p.capabilities]

    def get_first_available(
        self,
        capability: WorkerCapability,
        disallowed_processes: list[int] | None = None,
        *,
        device_index: int | None = None,
    ) -> HordeProcessInfo | None:
        """Return the first process with the capability that can accept a job, or None.

        Processes without a loaded model are preferred (cheapest to (re)target).

        Args:
            capability: The worker capability the process must have.
            disallowed_processes: Process ids to skip.
            device_index: When given, only consider processes pinned to that card (so a multi-GPU preload
                can target a chosen card); when None, any card.
        """
        disallowed = disallowed_processes or []

        for p in self.get_capable_processes(capability):
            if device_index is not None and p.device_index != device_index:
                continue
            if (
                p.last_process_state in (HordeProcessState.WAITING_FOR_JOB, HordeProcessState.PRELOADED_MODEL)
                and p.loaded_horde_model_name is None
                and p.process_id not in disallowed
            ):
                return p

        for p in self.get_capable_processes(capability):
            if device_index is not None and p.device_index != device_index:
                continue
            if p.can_accept_job() and p.process_id not in disallowed:
                return p

        return None

    def get_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the most conservative free VRAM (MB) across inference processes, or None.

        Child processes report ``vram_usage_mb``/``total_vram_mb`` computed as
        ``torch_total - torch_free`` and ``torch_total``, so ``total - usage`` is the
        device-wide free VRAM at sample time. A device-pinned (masked) child sees only its own card as
        ``cuda:0``, so its report is that card's free VRAM; the minimum across reporting processes is used
        as a conservative estimate. Returns None when no inference process has reported VRAM yet
        (cold start, or a CPU-only deployment).

        Args:
            device_index: When given, restrict to inference processes pinned to that card so the figure is
                that card's free VRAM (the per-card budget on a multi-GPU host); when None, the most
                conservative figure across every card (the single-GPU / worker-wide reading).
        """
        free_values = [
            p.total_vram_mb - p.vram_usage_mb
            for p in self.values()
            if p.process_type == HordeProcessType.INFERENCE
            and p.total_vram_mb > 0
            and (device_index is None or p.device_index == device_index)
        ]
        if not free_values:
            return None
        return float(min(free_values))

    def get_reported_total_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the device's total VRAM (MB) as reported by inference processes, or None.

        Children report ``total_vram_mb`` (``torch_total``); a masked child reports its own card's total, so
        the max across reporting processes is the (per-card, when filtered) device total. None until a
        process has reported (cold start, or a CPU-only deployment). Used by the streaming forecast to derive
        ComfyUI's inference reserve and the free VRAM achievable under sole residency.

        Args:
            device_index: When given, restrict to processes pinned to that card (the per-card total on a
                multi-GPU host); when None, the max across every card.
        """
        totals = [
            p.total_vram_mb
            for p in self.values()
            if p.process_type == HordeProcessType.INFERENCE
            and p.total_vram_mb > 0
            and (device_index is None or p.device_index == device_index)
        ]
        if not totals:
            return None
        return float(max(totals))

    def residency_snapshot(self) -> str:
        """One-line 'which model is resident on which inference slot' summary, for over-commit diagnostics.

        ``vram_usage_mb`` from the memory reports is *device-wide* used (children compute
        ``torch_total - torch_free``), not a per-slot figure, so this reports the per-slot resident model
        and state plus the single device-wide free VRAM. Logged when an over-commit is admitted or a
        slowdown is graded, so a live log shows the residency at the moment free VRAM ran out.
        """
        parts: list[str] = []
        for process_id, process_info in sorted(self.items()):
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            model = process_info.loaded_horde_model_name or "-"
            parts.append(f"#{process_id}:{model}[{process_info.last_process_state.name}]")
        free = self.get_free_vram_mb()
        free_str = f"{free:.0f}" if free is not None else "?"
        return f"slots=[{', '.join(parts) if parts else 'none'}] device_free_vram={free_str}MB"

    def num_inference_processes(self) -> int:
        """Return the number of inference processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE:
                count += 1
        return count

    def num_loaded_inference_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of inference processes that haven't been ended.

        Args:
            device_index: When given, count only processes pinned to that card (the per-card live-context
                count the residency forecast reasons about on a multi-GPU host); when None, count across
                every card.
        """
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.INFERENCE
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
                and (device_index is None or p.device_index == device_index)
            ):
                count += 1
        return count

    def num_available_inference_processes(self) -> int:
        """Return the number of inference processes that can actually accept a job.

        Keyed on ``can_accept_job()`` (the same predicate the scheduler dispatches against), not on the
        looser ``not is_process_busy()``. The two disagree at the edges: ``not is_process_busy()`` counts
        a dead, ending, failed, or just-unloaded slot (none of which can take a job) as available, and
        omits a ``PRELOADED_MODEL`` slot (which can). Reporting those as capacity is the phantom-capacity
        trap, so this mirrors ``can_accept_job()`` exactly.
        """
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and p.can_accept_job():
                count += 1
        return count

    def num_starting_processes(self) -> int:
        """Return the number of processes that are currently starting."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.PROCESS_STARTING:
                count += 1
        return count

    def keep_single_inference(
        self,
        *,
        stable_diffusion_model_reference: dict[str, ImageGenerationModelRecord],
        post_process_job_overlap: bool,
    ) -> tuple[bool, str]:
        """Return true if we should keep only a single inference process running.

        This is used to prevent overloading the system with inference processes, such as with batched jobs.
        """
        for p in self.values():
            if p.batch_amount > 1 and p.last_process_state == HordeProcessState.INFERENCE_STARTING:
                return True, "Batched job"

            if (
                (
                    p.last_process_state == HordeProcessState.INFERENCE_STARTING
                    or (
                        p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                        and not post_process_job_overlap
                    )
                )
                and p.last_job_referenced is not None
                and p.last_job_referenced.model in VRAM_HEAVY_MODELS
            ):
                return True, "VRAM heavy model"

            if (
                p.last_job_referenced is not None
                and p.last_job_referenced.payload.workflow in KNOWN_CONTROLNET_WORKFLOWS
            ):
                model = p.last_job_referenced.model
                if model is None:
                    logger.error(
                        f"Model is None for process {p.process_id} but workflow is "
                        f"{p.last_job_referenced.payload.workflow}",
                    )
                    continue

                model_info = stable_diffusion_model_reference.get(model)
                if model_info is None:
                    logger.debug(f"Model {model} not found in stable diffusion model reference. Is it a custom model?")
                    continue

                if model_info.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl and (
                    p.can_accept_job() or p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                ):
                    return True, "ControlNet XL"

            if p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING and not post_process_job_overlap:
                return True, "Post processing overlap"

        return False, "None"

    def get_inference_processes(self) -> list[HordeProcessInfo]:
        """Return a list of all inference processes."""
        return [p for p in self.values() if p.process_type == HordeProcessType.INFERENCE]

    def get_first_available_inference_process(
        self,
        disallowed_processes: list[int] | None = None,
        *,
        device_index: int | None = None,
    ) -> HordeProcessInfo | None:
        """Return the first available inference process, or None if there are none available.

        Args:
            disallowed_processes: Process ids to skip.
            device_index: When given, only consider processes pinned to that card (a multi-GPU preload
                targeting a chosen card); when None, any card.
        """
        return self.get_first_available(WorkerCapability.IMAGE_GEN, disallowed_processes, device_index=device_index)

    def _get_first_inference_process_to_kill(
        self,
        disallowed_processes: list[int] | None = None,
    ) -> HordeProcessInfo | None:
        """Return the first inference process eligible to be killed, or None if there are none.

        Used during shutdown.
        """
        if disallowed_processes is None:
            disallowed_processes = []

        for p in self.values():
            if p.process_type != HordeProcessType.INFERENCE:
                continue

            if p.process_id in disallowed_processes:
                continue

            if p.is_process_busy():
                continue

            # Already ending or ended — the pipe is already closing; do not re-target.
            if p.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue

            return p

        return None

    def get_safety_process(self) -> HordeProcessInfo | None:
        """Return the safety process."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                return p
        return None

    def num_safety_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of safety processes.

        Args:
            device_index: When given, count only safety processes pinned to that card, so a per-card
                residency forecast charges the safety CUDA context only against the card it actually sits
                on (the worker runs a single safety process pinned to the first configured card); when None,
                count across every card.
        """
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY and (device_index is None or p.device_index == device_index):
                count += 1
        return count

    def num_loaded_safety_processes(self) -> int:
        """Return the number of safety processes that are loaded."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.SAFETY
                and p.last_process_state != HordeProcessState.PROCESS_STARTING
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1

        return count

    def get_first_available_safety_process(self) -> HordeProcessInfo | None:
        """Return the first available safety process, or None if there are none available."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY and p.last_process_state == HordeProcessState.WAITING_FOR_JOB:
                return p
        return None

    def get_stoppable_safety_processes(self) -> list[HordeProcessInfo]:
        """Return safety processes that can be sent an end command.

        This is deliberately broader than ``get_first_available_safety_process``. Dispatch and job-popping
        need a safety process that can accept work; lifecycle teardown needs any live safety process that
        has not already entered its terminal shutdown states.
        """
        return [
            p
            for p in self.values()
            if p.process_type == HordeProcessType.SAFETY
            and p.last_process_state not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
        ]

    def get_process_by_horde_model_name(self, horde_model_name: str) -> HordeProcessInfo | None:
        """Return the process that has the given horde model loaded, or None if there is none."""
        for p in self.values():
            if p.loaded_horde_model_name == horde_model_name:
                return p
        return None

    def get_processes_by_horde_model_name(
        self,
        horde_model_name: str,
        *,
        allowed_cards: set[int] | None = None,
    ) -> list[HordeProcessInfo]:
        """Return every process that has the given horde model loaded (a model may be resident on >1 card).

        On a multi-GPU host the same model can be loaded on processes pinned to different cards, so this
        returns all such processes rather than the first. ``allowed_cards``, when given, restricts the result
        to processes whose pinned ``device_index`` is in that set (the dispatch router passes the job's
        eligible cards). Single-GPU callers get a one- or zero-element list mirroring
        :meth:`get_process_by_horde_model_name`.
        """
        return [
            p
            for p in self.values()
            if p.loaded_horde_model_name == horde_model_name
            and (allowed_cards is None or p.device_index in allowed_cards)
        ]

    def num_busy_processes(self) -> int:
        """Return the number of processes that are actively engaged in a task.

        This does not include processes which are starting up or shutting down, or in a faulted state.
        """
        count = 0
        for p in self.values():
            if p.is_process_busy():
                count += 1
        return count

    def num_busy_with_inference(self) -> int:
        """Return the number of processes that are actively engaged in an inference task."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.INFERENCE_STARTING:
                count += 1
        return count

    def has_inference_in_progress(self) -> bool:
        """Whether a live inference slot is actively running a job (worker-wide).

        True only while a slot is mid-inference (INFERENCE_STARTING or INFERENCE_POST_PROCESSING) on a
        process still alive. This is the "real inference is advancing" fact the deadlock clear and the wedge
        assessment both key on, kept in one place so they cannot drift apart. Deliberately narrower than
        ``is_process_busy`` (which also counts PROCESS_STARTING / preloading / downloading): a slot merely
        starting or loading a model is not running a job and must keep the anti-flap guard. Worker-wide on
        purpose: the queue-deadlock premise is itself all-cards-idle, so any one card mid-inference is enough
        to disprove it.
        """
        for process_info in self.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            if process_info.last_process_state in (
                HordeProcessState.INFERENCE_STARTING,
                HordeProcessState.INFERENCE_POST_PROCESSING,
            ):
                return True
        return False

    def num_busy_with_post_processing(self, *, device_index: int | None = None) -> int:
        """Return the number of processes actively engaged in a post-processing task.

        Args:
            device_index: When given, count only processes pinned to that card (the per-card
                concurrency gate scopes the post-processing overlap bump to one card); when None,
                count across every card.
        """
        count = 0
        for p in self.values():
            if p.last_process_state != HordeProcessState.INFERENCE_POST_PROCESSING:
                continue
            if device_index is not None and p.device_index != device_index:
                continue
            count += 1
        return count

    def num_preloading_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of processes that are preloading models.

        Args:
            device_index: When given, count only processes pinned to that card. The preload-serialization
                gate is per-card on a multi-GPU host: one card loading a checkpoint must not block a load
                onto a different, idle card (separate VRAM, independent of each other's disk-read spike).
        """
        count = 0
        for p in self.values():
            if device_index is not None and p.device_index != device_index:
                continue
            if p.last_process_state == HordeProcessState.PRELOADING_MODEL:
                count += 1
        return count

    def num_preloaded_processes(self) -> int:
        """Return the number of processes that have preloaded models."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.PRELOADED_MODEL:
                count += 1
        return count

    @override
    def __repr__(self) -> str:
        """Return a string representation of the process map."""
        base_string = "Processes: "
        for string in self.get_process_info_strings():
            base_string += string

        return base_string

    def get_process_info_strings(self) -> list[str]:
        """Return a list of strings containing information about each process."""
        info_strings = []
        current_time = time.time()
        for process_id, process_info in self.items():
            if process_info.process_type == HordeProcessType.INFERENCE:
                time_passed_seconds = round((current_time - process_info.last_received_timestamp), 2)
                safe_last_control_flag = (
                    process_info.last_control_flag.name if process_info.last_control_flag is not None else None
                )

                process_state_detail = process_info.last_process_state.name

                if (
                    process_info.last_heartbeat_percent_complete is not None
                    and process_info.last_job_referenced is not None
                ):
                    process_state_detail = (
                        f"{process_info.last_heartbeat_percent_complete}% of "
                        f"{process_info.last_job_referenced.payload.ddim_steps} steps "
                        f"using {process_info.last_job_referenced.payload.sampler_name}"
                    )
                    if process_info.last_job_referenced.payload.n_iter > 1:
                        process_state_detail += f" ({process_info.last_job_referenced.payload.n_iter}x batch)"

                horde_model_name_and_baseline = (
                    f"<u>{process_info.loaded_horde_model_name}</u> {process_info.loaded_horde_model_baseline})"
                    if process_info.loaded_horde_model_name is not None
                    else "No model loaded"
                )
                last_heartbeat_delta_now = round((current_time - process_info.last_heartbeat_timestamp), 2)
                info_strings.append(
                    (
                        f"Process {process_id} ({process_state_detail}) "
                        f"({horde_model_name_and_baseline}) "
                        f"<fg #7b7d7d>[last message: {time_passed_seconds} secs ago: {safe_last_control_flag} "
                        f"heartbeat delta: {last_heartbeat_delta_now}]</>"
                    ),
                    # f"ram: {process_info.ram_usage_bytes} vram: {process_info.vram_usage_mb} ",
                )

            else:
                info_strings.append(
                    f"Process {process_id}: ({process_info.process_type.name}) "
                    f"{process_info.last_process_state.name} ",
                )

        return info_strings

    def all_waiting_for_job(self) -> bool:
        """Return true if all processes are waiting for a job."""
        return all(
            p.last_process_state in [HordeProcessState.WAITING_FOR_JOB, HordeProcessState.PRELOADED_MODEL]
            for p in self.values()
        )
