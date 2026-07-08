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

from horde_worker_regen.consts import KNOWN_CONTROLNET_WORKFLOWS
from horde_worker_regen.process_management.fd_limits import (
    FD_HEADROOM_WARN_FRACTION,
    descriptor_headroom_fraction,
)
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
    HordeProcessState.INFERENCE_COMPLETE: frozenset({HordeProcessState.INFERENCE_STARTING}),
    HordeProcessState.INFERENCE_FAILED: frozenset(
        {
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.JOB_RECEIVED,
        },
    ),
    HordeProcessState.POST_PROCESSING: frozenset(
        {
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.POST_PROCESSING_COMPLETE,
            HordeProcessState.POST_PROCESSING_FAILED,
        },
    ),
    HordeProcessState.POST_PROCESSING_COMPLETE: frozenset({HordeProcessState.POST_PROCESSING}),
    HordeProcessState.POST_PROCESSING_FAILED: frozenset({HordeProcessState.POST_PROCESSING}),
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
        # Inference process ids reserved as the pinned sampler of an in-flight disaggregated job. A pinned
        # sampler is booked for that job from the moment the scheduler routes it (in place of START_INFERENCE)
        # until its sampling finishes; while reserved it is skipped by the availability finders so the
        # scheduler cannot dispatch a second (monolithic or disaggregated) job onto it. Unlike a monolithic
        # dispatch, no child message marks the pin, so this parent-side set is the sole booking record.
        self._disaggregation_reserved_process_ids: set[int] = set()

    def reserve_for_disaggregation(self, process_id: int) -> None:
        """Reserve an inference process as a pinned disaggregation sampler (skipped by availability finders)."""
        self._disaggregation_reserved_process_ids.add(process_id)

    def release_disaggregation_reservation(self, process_id: int) -> None:
        """Release a disaggregation sampler reservation, returning the process to the available pool."""
        self._disaggregation_reserved_process_ids.discard(process_id)

    def is_reserved_for_disaggregation(self, process_id: int) -> bool:
        """Whether the process is currently pinned as an in-flight disaggregated job's sampler."""
        return process_id in self._disaggregation_reserved_process_ids

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
        nonadvancing_step_repeats: int = 0,
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
            nonadvancing_step_repeats (int, optional): The child's running count of consecutive \
                progress reports at the same sampling step without advancing. Defaults to 0.
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
        # The child sends its authoritative running count on every heartbeat (0 while advancing), so
        # store it verbatim rather than re-deriving it here from the (step-less) post-completion beats.
        self[process_id].nonadvancing_step_repeats = nonadvancing_step_repeats

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
        # A dead process's context and allocator reservation are reclaimed by the OS, so drop its last
        # per-process attribution: leaving it would keep charging freed VRAM against the committed ledger.
        self[process_id].process_reserved_mb = None
        self[process_id].process_allocated_mb = None
        self[process_id].process_peak_reserved_mb = None
        self[process_id].process_aimdo_mb = None
        self[process_id].report_sampled_at = None

        self.reset_heartbeat_state(process_id)

        self[process_id].last_received_timestamp = time.time()

    def on_memory_report(
        self,
        process_id: int,
        ram_usage_bytes: int,
        vram_usage_mb: int | None = 0,
        total_vram_mb: int | None = 0,
        open_fds: int | None = None,
        fd_soft_limit: int | None = None,
        process_reserved_mb: int | None = None,
        process_allocated_mb: int | None = None,
        process_peak_reserved_mb: int | None = None,
        process_aimdo_mb: int | None = None,
        report_sampled_at: float | None = None,
    ) -> None:
        """Update the memory usage for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            ram_usage_bytes (int): The amount of RAM used by this process.
            vram_usage_mb (int): The amount of VRAM used by this process.
            total_vram_mb (int): The total amount of VRAM available to this process.
            open_fds (int | None): Open descriptors/handles the process reported, or None if unavailable.
            fd_soft_limit (int | None): The process's soft ``RLIMIT_NOFILE`` ceiling, or None if unbounded.
            process_reserved_mb (int | None): This process's own committed device memory (MB) from the torch \
                allocator (excludes its CUDA context), or None off-GPU. The honest per-process VRAM charge.
            process_allocated_mb (int | None): This process's own live (in-use) device memory (MB), or None.
            process_peak_reserved_mb (int | None): This process's peak reserved device memory (MB) over the \
                last report interval, or None.
            process_aimdo_mb (int | None): This process's device memory (MB) in the engine's direct-IO \
                weight pool, captured only if that (currently inert) subsystem is initialised, else 0; or None \
                off-GPU. Disjoint from ``process_reserved_mb``.
            report_sampled_at (float | None): Wall-clock epoch when the child sampled these figures, or None \
                (older children). Used to age the process's contribution for staleness-aware VRAM reconciliation.
        """
        process_info = self[process_id]
        process_info.ram_usage_bytes = ram_usage_bytes
        process_info.vram_usage_mb = vram_usage_mb or 0
        process_info.total_vram_mb = total_vram_mb or 0
        process_info.process_reserved_mb = process_reserved_mb
        process_info.process_allocated_mb = process_allocated_mb
        process_info.process_peak_reserved_mb = process_peak_reserved_mb
        process_info.process_aimdo_mb = process_aimdo_mb
        process_info.report_sampled_at = report_sampled_at
        process_info.open_fds = open_fds
        process_info.fd_soft_limit = fd_soft_limit

        process_info.last_received_timestamp = time.time()

        self._warn_on_low_descriptor_headroom(process_info)

        logger.debug(
            f"Process {process_id} memory report: "
            f"ram: {ram_usage_bytes} vram: {vram_usage_mb} total vram: {total_vram_mb} "
            f"fds: {open_fds}/{fd_soft_limit}",
        )

    def _warn_on_low_descriptor_headroom(self, process_info: HordeProcessInfo) -> None:
        """Warn once when a process crosses the descriptor-headroom threshold, re-arming when it recovers.

        A file-descriptor leak (the classic case being PyTorch's ``file_descriptor`` tensor-sharing
        strategy) climbs toward the process's ``RLIMIT_NOFILE`` ceiling and ends in ``EMFILE``, from which
        the slot faults every job while still heart-beating. Surfacing the climb at ``WARNING`` while the
        slot is still serving turns that silent poisoning into an early, named signal. The warning latches
        (rising edge) so it does not spam every report, and re-arms only after usage falls well clear of the
        threshold, so a value hovering at the line does not flap.
        """
        fraction = descriptor_headroom_fraction(process_info.open_fds, process_info.fd_soft_limit)
        if fraction is None:
            return
        if fraction >= FD_HEADROOM_WARN_FRACTION:
            if not process_info.fd_headroom_warned:
                process_info.fd_headroom_warned = True
                logger.warning(
                    f"Process {process_info.process_id} "
                    f"({process_info.loaded_horde_model_name or 'no model loaded'}) is nearing its "
                    f"file-descriptor ceiling: {process_info.open_fds}/{process_info.fd_soft_limit} "
                    f"({fraction * 100:.0f}% in use). A descriptor leak ends in EMFILE ('Too many open "
                    f"files'), which faults every job on the slot until it is recycled.",
                )
        elif process_info.fd_headroom_warned and fraction < FD_HEADROOM_WARN_FRACTION * 0.9:
            process_info.fd_headroom_warned = False
            logger.info(
                f"Process {process_info.process_id} file-descriptor headroom recovered: "
                f"{process_info.open_fds}/{process_info.fd_soft_limit} ({fraction * 100:.0f}% in use).",
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

    def reconcile_reported_os_pid(self, process_id: int, reported_os_pid: int | None) -> None:
        """Adopt a child's self-reported ``os.getpid()`` as the authoritative OS pid for per-PID telemetry.

        The parent captures ``os_pid`` from the ``mp_process.pid`` spawn handle at launch, which is wrong when
        the interpreter runs behind a launcher-stub ``python.exe``: the handle is the stub's pid while the real
        interpreter (and its CUDA context) live in a grandchild, so PDH paging attribution and the NVML per-PID
        lookups keyed on the handle pid miss entirely. The child self-reports its real pid on every message;
        this overwrites the handle-derived value the first time they differ so per-PID telemetry addresses the
        process that actually holds the GPU context. A None report (an older child) leaves the handle value.
        """
        if reported_os_pid is None or process_id not in self:
            return
        process_info = self[process_id]
        if process_info.os_pid != reported_os_pid:
            logger.debug(
                f"Process {process_id} self-reported os_pid {reported_os_pid} "
                f"(handle-derived was {process_info.os_pid}); adopting the child-reported pid for per-PID "
                "telemetry.",
            )
            process_info.os_pid = reported_os_pid

    def note_vram_materialized(self, process_id: int) -> None:
        """Stamp the monotonic time the parent observed this process materialize VRAM.

        The LIFO ranking key for the reclaim ladder. Called on a VRAM-materializing event (a model reported
        LOADED_IN_VRAM, a GPU process spawned), so the reclaim engine can reclaim the most-recently-
        materialized tenant first.
        """
        self[process_id].vram_materialized_monotonic = time.monotonic()

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
        # The model left VRAM, so it no longer has a materialization time; the next materialization restamps.
        self[process_id].vram_materialized_monotonic = None

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
        self[process_id].nonadvancing_step_repeats = 0
        # The per-step floor crawl detector is job-scoped like the slowdown grade: clear its streak and the
        # tripped flag at every job boundary so a fresh job starts from a clean slate.
        self[process_id].consecutive_slow_per_steps = 0
        self[process_id].current_job_per_step_floor_tripped = False

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

    def is_stuck_on_nonadvancing_step(self, process_id: int, repeat_limit: int) -> bool:
        """Return true if a sampling slot keeps reporting the same step without ever advancing.

        This is the wedge :meth:`is_stuck_on_inference` is blind to. There, a hung child goes *silent*
        and the heartbeat-silence gap catches it. Here the child is not silent: the underlying ComfyUI
        generation loops on a single step (in practice the final one, which a healthy job reports exactly
        once), so the child keeps receiving identical progress callbacks and keeps emitting heartbeats.
        Every heartbeat refreshes ``last_heartbeat_timestamp``, so the silence gap never grows and the
        slot sits in ``INFERENCE_STARTING`` forever, holding VRAM and a queue slot while never returning
        a result. The child counts those non-advancing reports and forwards the running count; once it
        crosses ``repeat_limit`` the generation is wedged and the slot must be reaped despite its liveness.

        The limit sits far above the healthy ceiling of one same-step report, so a job that briefly
        re-reports its final step before returning is never mistaken for a wedge.
        """
        process_info = self[process_id]
        if process_info.last_process_state != HordeProcessState.INFERENCE_STARTING:
            return False
        return process_info.nonadvancing_step_repeats >= repeat_limit

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
            if self.is_reserved_for_disaggregation(p.process_id):
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
            if self.is_reserved_for_disaggregation(p.process_id):
                continue
            if p.can_accept_job() and p.process_id not in disallowed:
                return p

        return None

    def get_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the most conservative free VRAM (MB) across GPU-reporting child processes, or None.

        Child processes report ``vram_usage_mb``/``total_vram_mb`` computed as
        ``torch_total - torch_free`` and ``torch_total``, so ``total - usage`` is the
        device-wide free VRAM at sample time. A device-pinned (masked) child sees only its own card as
        ``cuda:0``, so its report is that card's free VRAM; the minimum across reporting processes is used
        as a conservative estimate. Returns None when no GPU-bearing child has reported VRAM yet
        (cold start, CPU-only deployment, or a disabled GPU lane).

        Args:
            device_index: When given, restrict to reporting processes pinned to that card so the figure is
                that card's free VRAM (the per-card budget on a multi-GPU host); when None, the most
                conservative figure across every card (the single-GPU / worker-wide reading).
        """
        free_values = [
            p.total_vram_mb - p.vram_usage_mb
            for p in self.values()
            if p.total_vram_mb > 0 and (device_index is None or p.device_index == device_index)
        ]
        if not free_values:
            return None
        return float(min(free_values))

    def get_reported_total_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the device's total VRAM (MB) as reported by GPU-bearing child processes, or None.

        Children report ``total_vram_mb`` (``torch_total``); a masked child reports its own card's total, so
        the max across reporting processes is the (per-card, when filtered) device total. None until a
        GPU-bearing process has reported (cold start, CPU-only deployment, or a disabled GPU lane). Used by
        the streaming forecast to derive ComfyUI's inference reserve and the free VRAM achievable under sole
        residency.

        Args:
            device_index: When given, restrict to processes pinned to that card (the per-card total on a
                multi-GPU host); when None, the max across every card.
        """
        totals = [
            p.total_vram_mb
            for p in self.values()
            if p.total_vram_mb > 0 and (device_index is None or p.device_index == device_index)
        ]
        if not totals:
            return None
        return float(max(totals))

    def committed_ledger_processes(self, device_index: int | None = None) -> list[HordeProcessInfo]:
        """Return the GPU processes whose footprint the committed-VRAM ledger charges (a shared predicate).

        A process is counted when it has reported a ``process_reserved_mb`` (a GPU-bearing process that has
        sent at least one VRAM-inclusive memory report) and has not entered its terminal shutdown states.
        :meth:`committed_vram_mb`, :meth:`oldest_committed_report_age_seconds`, and the scheduler's
        idle-context residency capture all key on this exact set so the ledger sum, its staleness assessment,
        and the per-context marginal derivation can never disagree about which tenants make it up.
        """
        processes: list[HordeProcessInfo] = []
        for process_info in self.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is None:
                continue
            if process_info.last_process_state in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                continue
            processes.append(process_info)
        return processes

    def committed_vram_mb(self, *, context_constant_mb: float, device_index: int | None = None) -> float:
        """Return the device VRAM (MB) attributable to this worker: the sum of every live GPU process's footprint.

        Each live GPU process's device footprint is ``context_constant_mb + process_reserved_mb +
        process_aimdo_mb`` (the fixed CUDA-context overhead, plus the process's own byte-exact allocator
        reservation from ``torch.cuda.memory_reserved``, plus any weights held in the engine's direct-IO pool
        the torch allocator cannot see). The aimdo term is inert (0) in the current embedding because nothing
        initialises that subsystem, so weights are counted by ``process_reserved_mb``; it is kept as a
        disjoint, future-proof complement, so summing the two never double-counts. Summed over the live
        GPU-bearing processes, this is the *exact committed device memory attributable to the worker*, the
        ledger arithmetic that is the ONLY way to see an over-commit coming on Windows/WDDM (where the driver
        silently spills to host RAM without ever failing an allocation or showing pressure in
        ``mem_get_info``). It deliberately excludes the shared device baseline (OS/desktop/other apps), which
        is attributable to no worker process; the drift reconciliation adds the baseline back at device level.

        The context constant is resolved by the caller via
        :func:`~horde_worker_regen.process_management.resources.resource_budget.platform_context_constant_mb`
        (measured marginal when available, else the platform seed).

        Args:
            context_constant_mb: The per-process CUDA-context VRAM charge (MB) to add for each live process.
            device_index: When given, sum only processes pinned to that card; when None, sum every card.
        """
        total = 0.0
        for process_info in self.committed_ledger_processes(device_index):
            total += (
                context_constant_mb + (process_info.process_reserved_mb or 0) + (process_info.process_aimdo_mb or 0)
            )
        return total

    def oldest_committed_report_age_seconds(self, *, now: float, device_index: int | None = None) -> float | None:
        """Return the oldest memory-report age (seconds) among the committed ledger's contributors, or None.

        Ages each contributing process's last VRAM-inclusive report against ``now`` and returns the maximum
        (the least-fresh tenant). Returns None when no process currently contributes to the ledger (nothing to
        age). A contributor that has never carried a ``report_sampled_at`` (an older child) is treated as
        maximally stale (``inf``): its contribution cannot be dated, so the ledger it is part of cannot be
        trusted as current. The reconciler uses this to treat a stale ledger as an UNKNOWN, incomparable
        tenant rather than reconciling a device anchor against figures a process may have moved far past.

        Args:
            now: The wall-clock epoch (``time.time()``) to age reports against.
            device_index: When given, consider only processes pinned to that card; when None, every card.
        """
        contributors = self.committed_ledger_processes(device_index)
        if not contributors:
            return None
        ages = [(now - p.report_sampled_at) if p.report_sampled_at is not None else float("inf") for p in contributors]
        return max(ages)

    def residency_snapshot(self) -> str:
        """One-line 'which model is resident on which inference slot' summary, for over-commit diagnostics.

        ``vram_usage_mb`` from the memory reports is *device-wide* used (children compute
        ``torch_total - torch_free``), not a per-slot figure, so this reports the per-slot resident model
        and state plus the single device-wide free VRAM. Logged when an over-commit is admitted or a
        slowdown is graded, so a live log shows the residency at the moment free VRAM ran out.
        """
        parts: list[str] = []
        for process_id, process_info in sorted(self.items()):
            if process_info.process_type not in {
                HordeProcessType.INFERENCE,
                HordeProcessType.POST_PROCESS,
                HordeProcessType.VAE_LANE,
            }:
                continue
            if process_info.process_type == HordeProcessType.POST_PROCESS:
                parts.append(f"#{process_id}:post-process[{process_info.last_process_state.name}]")
            elif process_info.process_type == HordeProcessType.VAE_LANE:
                parts.append(f"#{process_id}:vae-lane[{process_info.last_process_state.name}]")
            else:
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
    ) -> tuple[bool, str]:
        """Return true if we should keep only a single inference process running.

        This is a worker-wide, device-blind hold, reserved for a workflow that cannot coexist with any
        concurrent inference at all. It is checked before the scheduler's dispatch path, so anything held
        here never reaches the size-tier overlap gate.

        Batched and otherwise card-demanding jobs are deliberately not a rule here: their serialization
        belongs to the scheduler's size-tier overlap gate, which prices a batch's multiplied activation
        peak against the card's measured headroom, scopes to the card the in-flight job runs on, and admits
        an overlap once the running job has made size-appropriate headway. A worker-wide hold on batches
        would sit on top of that gate and force full serialization whenever any batch samples, blind to
        whether the card has room for a second lane.
        """
        for p in self.values():
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

                if model_info.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl and p.can_accept_job():
                    return True, "ControlNet XL"

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

            # Already ending or ended; the pipe is already closing; do not re-target.
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

    def get_post_process_process(self) -> HordeProcessInfo | None:
        """Return the dedicated post-processing process."""
        for p in self.values():
            if p.process_type == HordeProcessType.POST_PROCESS:
                return p
        return None

    def num_post_process_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of dedicated post-processing processes.

        Args:
            device_index: When given, count only post-processing processes pinned to that card, so a
                per-card residency forecast charges the lane's CUDA context only against the card it
                actually sits on; when None, count across every card.
        """
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.POST_PROCESS and (
                device_index is None or p.device_index == device_index
            ):
                count += 1
        return count

    def num_loaded_post_process_processes(self) -> int:
        """Return the number of dedicated post-processing processes that are loaded."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.POST_PROCESS
                and p.last_process_state != HordeProcessState.PROCESS_STARTING
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1

        return count

    def get_first_available_post_process_process(self) -> HordeProcessInfo | None:
        """Return the first available dedicated post-processing process, or None if none are available."""
        for p in self.values():
            if (
                p.process_type == HordeProcessType.POST_PROCESS
                and p.last_process_state == HordeProcessState.WAITING_FOR_JOB
            ):
                return p
        return None

    def get_stoppable_post_process_processes(self) -> list[HordeProcessInfo]:
        """Return dedicated post-processing processes that can be sent an end command.

        Deliberately broader than ``get_first_available_post_process_process``: dispatch needs a process
        that can accept work; lifecycle teardown needs any live process that has not already entered its
        terminal shutdown states.
        """
        return [
            p
            for p in self.values()
            if p.process_type == HordeProcessType.POST_PROCESS
            and p.last_process_state not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
        ]

    def delete_post_process_processes(self) -> None:
        """Clear all dedicated post-processing processes."""
        processes_to_delete = [p for p in self.values() if p.process_type == HordeProcessType.POST_PROCESS]

        for process_info in processes_to_delete:
            logger.debug(f"Deleting post-process process {process_info.process_id} from process map")
            self.retire_process(process_info, "post-process process replacement")

    def get_component_process(self) -> HordeProcessInfo | None:
        """Return the dedicated component lane process, or None if it is not running."""
        for p in self.values():
            if p.process_type == HordeProcessType.COMPONENT:
                return p
        return None

    def num_component_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of dedicated component lane processes (0 or 1).

        Args:
            device_index: When given, count only the component lane if it is pinned to that card, so a
                per-card residency teardown gate waits only on the lane that actually sits on its card;
                when None, count across every card.
        """
        return sum(
            1
            for p in self.values()
            if p.process_type == HordeProcessType.COMPONENT
            and (device_index is None or p.device_index == device_index)
        )

    def num_loaded_component_processes(self) -> int:
        """Return the number of component lane processes past startup and not yet shutting down."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.COMPONENT and p.last_process_state not in (
                HordeProcessState.PROCESS_STARTING,
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                count += 1
        return count

    def get_stoppable_component_processes(self) -> list[HordeProcessInfo]:
        """Return component lane processes that can be sent an end command (any not already shutting down)."""
        return [
            p
            for p in self.values()
            if p.process_type == HordeProcessType.COMPONENT
            and p.last_process_state not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
        ]

    def delete_component_processes(self) -> None:
        """Clear all dedicated component lane processes."""
        processes_to_delete = [p for p in self.values() if p.process_type == HordeProcessType.COMPONENT]
        for process_info in processes_to_delete:
            logger.debug(f"Deleting component lane process {process_info.process_id} from process map")
            self.retire_process(process_info, "component lane replacement")

    def get_first_available_vae_lane_process(self) -> HordeProcessInfo | None:
        """Return the first available dedicated VAE lane process, or None if none are available."""
        for p in self.values():
            if (
                p.process_type == HordeProcessType.VAE_LANE
                and p.last_process_state == HordeProcessState.WAITING_FOR_JOB
            ):
                return p
        return None

    def num_vae_lane_processes(self, *, device_index: int | None = None) -> int:
        """Return the number of dedicated VAE lane processes.

        Args:
            device_index: When given, count only VAE lane processes pinned to that card, so a per-card
                residency forecast charges the lane's CUDA context only against the card it actually sits
                on; when None, count across every card.
        """
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.VAE_LANE and (
                device_index is None or p.device_index == device_index
            ):
                count += 1
        return count

    def num_loaded_vae_lane_processes(self) -> int:
        """Return the number of dedicated VAE lane processes that are loaded."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.VAE_LANE
                and p.last_process_state != HordeProcessState.PROCESS_STARTING
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1

        return count

    def get_stoppable_vae_lane_processes(self) -> list[HordeProcessInfo]:
        """Return dedicated VAE lane processes that can be sent an end command.

        Deliberately broader than ``get_first_available_vae_lane_process``: dispatch needs a process that
        can accept work; lifecycle teardown needs any live process that has not already entered its terminal
        shutdown states.
        """
        return [
            p
            for p in self.values()
            if p.process_type == HordeProcessType.VAE_LANE
            and p.last_process_state not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
        ]

    def delete_vae_lane_processes(self) -> None:
        """Clear all dedicated VAE lane processes."""
        processes_to_delete = [p for p in self.values() if p.process_type == HordeProcessType.VAE_LANE]

        for process_info in processes_to_delete:
            logger.debug(f"Deleting VAE lane process {process_info.process_id} from process map")
            self.retire_process(process_info, "VAE lane replacement")

    def get_process_by_horde_model_name(
        self,
        horde_model_name: str,
        *,
        include_reserved: bool = False,
    ) -> HordeProcessInfo | None:
        """Return a process that has the given horde model loaded, or None if there is none.

        A process pinned as an in-flight disaggregated job's sampler is skipped by default: dispatch selection
        and the orchestrator's crash re-resolution both use this, and neither may steal a sampler already booked
        for another job. ``include_reserved=True`` includes pinned lanes, for residency and pricing queries that
        must know a model's weights are resident even on a lane no job may be dispatched onto yet (the caller
        must not then dispatch onto the returned process without its own can-accept-job check).
        """
        for p in self.values():
            if p.loaded_horde_model_name == horde_model_name and (
                include_reserved or not self.is_reserved_for_disaggregation(p.process_id)
            ):
                return p
        return None

    def get_processes_by_horde_model_name(
        self,
        horde_model_name: str,
        *,
        allowed_cards: set[int] | None = None,
        include_reserved: bool = False,
    ) -> list[HordeProcessInfo]:
        """Return every process that has the given horde model loaded (a model may be resident on >1 card).

        On a multi-GPU host the same model can be loaded on processes pinned to different cards, so this
        returns all such processes rather than the first. ``allowed_cards``, when given, restricts the result
        to processes whose pinned ``device_index`` is in that set (the dispatch router passes the job's
        eligible cards). Pinned disaggregation-sampler lanes are excluded by default and included only for
        residency/pricing queries via ``include_reserved=True``. Single-GPU callers get a one- or zero-element
        list mirroring :meth:`get_process_by_horde_model_name`.
        """
        return [
            p
            for p in self.values()
            if p.loaded_horde_model_name == horde_model_name
            and (allowed_cards is None or p.device_index in allowed_cards)
            and (include_reserved or not self.is_reserved_for_disaggregation(p.process_id))
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

    def num_busy_with_inference(self, *, device_index: int | None = None) -> int:
        """Return the number of processes that are actively sampling.

        Args:
            device_index: When given, count only processes pinned to that card; when None, count across
                every card.
        """
        count = 0
        for p in self.values():
            if p.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and p.device_index != device_index:
                continue
            if p.last_process_state == HordeProcessState.INFERENCE_STARTING:
                count += 1
        return count

    def has_inference_in_progress(self) -> bool:
        """Whether a live inference slot is actively running a job (worker-wide).

        True only while a slot is mid-inference (INFERENCE_STARTING) on a process still alive. This is the
        "real inference is advancing" fact the deadlock clear and the wedge assessment both key on, kept in
        one place so they cannot drift apart. Deliberately narrower than ``is_process_busy`` (which also
        counts PROCESS_STARTING / preloading / downloading): a slot merely starting or loading a model is
        not running a job and must keep the anti-flap guard. Worker-wide on purpose: the queue-deadlock
        premise is itself all-cards-idle, so any one card mid-inference is enough to disprove it.
        """
        for process_info in self.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            if process_info.last_process_state == HordeProcessState.INFERENCE_STARTING:
                return True
        return False

    def num_busy_with_post_processing(self, *, device_index: int | None = None) -> int:
        """Return the number of dedicated post-processing processes actively working a job.

        Args:
            device_index: When given, count only processes pinned to that card; when None, count across
                every card.
        """
        count = 0
        for p in self.values():
            if p.last_process_state != HordeProcessState.POST_PROCESSING:
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

    def any_model_downloading_aux_more_than_threshold(
        self,
        threshold_seconds: float,
        device_index: int | None = None,
    ) -> bool:
        """Return True if any process is downloading an auxiliary model for longer than the threshold."""
        now = time.time()
        for process in self.values():
            if device_index is not None and process.device_index != device_index:
                continue
            if (
                process.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL
                and (now - process.last_received_timestamp) > threshold_seconds
            ):
                return True
        return False
