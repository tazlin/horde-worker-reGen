"""A mapping of process IDs to HordeProcessInfo objects."""

from __future__ import annotations

import time
from typing import override

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger
from pydantic import ConfigDict

from horde_worker_regen.consts import KNOWN_CONTROLNET_WORKFLOWS, VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import (
    HordeHeartbeatType,
    HordeProcessState,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo

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


class ProcessMap(dict[int, HordeProcessInfo]):
    """A mapping of process IDs to HordeProcessInfo objects.

    There are a number of helper methods on this class for querying the state of processes, such as how many are
    busy, how many are doing inference, etc. In addition, there are a number of methods for updating the state of
    processes based on messages received from them, such as heartbeats, memory reports, and process state changes.

    See `on_heartbeat`, `on_memory_report`, `on_process_state_change`, `on_last_job_reference_change`, and
    `on_model_load_state_change` for more details on how the process map is updated based on messages from processes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def on_heartbeat(
        self,
        process_id: int,
        heartbeat_type: HordeHeartbeatType,
        *,
        percent_complete: int | None = None,
    ) -> None:
        """Update the heartbeat for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            heartbeat_type (HordeHeartbeatType): The type of the heartbeat.
            percent_complete (int | None, optional): The percentage of the job that has been completed, \
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

    def on_process_ending(self, process_id: int) -> None:
        """Update the process map when a process has ended.

        Args:
            process_id (int): The ID of the process that has ended.
        """
        self[process_id].last_process_state = HordeProcessState.PROCESS_ENDING
        self[process_id].loaded_horde_model_name = None
        self[process_id].loaded_horde_model_baseline = None
        self[process_id].last_job_referenced = None
        self[process_id].batch_amount = 1

        self.reset_heartbeat_state(process_id)

        self[process_id].last_received_timestamp = time.time()

    def on_memory_report(
        self,
        process_id: int,
        ram_usage_bytes: int,
        vram_usage_bytes: int | None = 0,
        total_vram_bytes: int | None = 0,
    ) -> None:
        """Update the memory usage for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            ram_usage_bytes (int): The amount of RAM used by this process.
            vram_usage_bytes (int): The amount of VRAM used by this process.
            total_vram_bytes (int): The total amount of VRAM available to this process.
        """
        self[process_id].ram_usage_bytes = ram_usage_bytes
        self[process_id].vram_usage_bytes = vram_usage_bytes or 0
        self[process_id].total_vram_bytes = total_vram_bytes or 0

        self[process_id].last_received_timestamp = time.time()

        logger.debug(
            f"Process {process_id} memory report: "
            f"ram: {ram_usage_bytes} vram: {vram_usage_bytes} total vram: {total_vram_bytes}",
        )

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

        self[process_id].last_process_state = new_state
        self[process_id].last_received_timestamp = time.time()

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
            load_state (ModelLoadState): The load state of the model.
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

    def delete_safety_processes(self) -> None:
        """Clear all safety processes."""
        ids_to_delete = []
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                ids_to_delete.append(p.process_id)

        for process_id in ids_to_delete:
            logger.debug(f"Deleting safety process {process_id} from process map")
            self.pop(process_id)

    def is_stuck_on_inference(
        self,
        process_id: int,
        inference_step_timeout: int,
    ) -> bool:
        """Return true if the process is actively doing inference but we haven't received a heartbeat in a while."""
        if self[process_id].last_process_state != HordeProcessState.INFERENCE_STARTING:
            return False

        last_heartbeat_percent_complete = self[process_id].last_heartbeat_percent_complete
        if last_heartbeat_percent_complete is not None and last_heartbeat_percent_complete < 1:
            return False

        return bool(
            self[process_id].last_heartbeat_type == HordeHeartbeatType.INFERENCE_STEP
            and self[process_id].last_heartbeat_delta > inference_step_timeout,
        )

    def num_inference_processes(self) -> int:
        """Return the number of inference processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE:
                count += 1
        return count

    def num_loaded_inference_processes(self) -> int:
        """Return the number of inference processes that haven't been ended."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.INFERENCE
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1
        return count

    def num_available_inference_processes(self) -> int:
        """Return the number of inference processes that are available to accept jobs."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and not p.is_process_busy():
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
    ) -> HordeProcessInfo | None:
        """Return the first available inference process, or None if there are none available."""
        if disallowed_processes is None:
            disallowed_processes = []

        for p in self.values():
            if (
                p.process_type == HordeProcessType.INFERENCE
                and (
                    p.last_process_state == HordeProcessState.WAITING_FOR_JOB
                    or p.last_process_state == HordeProcessState.PRELOADED_MODEL
                )
                and p.loaded_horde_model_name is None
                and p.process_id not in disallowed_processes
            ):
                return p

        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and p.can_accept_job():
                if p.process_id in disallowed_processes:
                    continue
                return p

        return None

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

    def num_safety_processes(self) -> int:
        """Return the number of safety processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
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

    def get_process_by_horde_model_name(self, horde_model_name: str) -> HordeProcessInfo | None:
        """Return the process that has the given horde model loaded, or None if there is none."""
        for p in self.values():
            if p.loaded_horde_model_name == horde_model_name:
                return p
        return None

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

    def num_busy_with_post_processing(self) -> int:
        """Return the number of processes that are actively engaged in a post-processing task."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING:
                count += 1
        return count

    def num_preloading_processes(self) -> int:
        """Return the number of processes that are preloading models."""
        count = 0
        for p in self.values():
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
                    # f"ram: {process_info.ram_usage_bytes} vram: {process_info.vram_usage_bytes} ",
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
