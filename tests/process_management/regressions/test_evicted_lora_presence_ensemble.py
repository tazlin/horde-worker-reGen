"""A job for a LoRA the download process evicted waits for the download process's answer before it dispatches.

The download process is the one owner of whether a LoRA or textual inversion is on disk: it holds the files, runs
every evictor (the free-disk floor guard and the ad-hoc budget evictor) and answers a request for a file already
present with an immediate success, after applying the eviction pins that request carries. The parent keeps no
belief of its own. A LoRA an earlier job downloaded can be evicted before a later job for it is popped, and the
eviction is not reported to the parent, so the later job's request names the LoRA again and the job stays gated
until the answer arrives.

The row runs the manager's own wiring: the prefetch coordinator sends through the process lifecycle to the download
process's pipe, the result comes back through the message dispatcher, and dispatch runs the real scheduler.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import (
    AuxPrefetchOutcome,
    HordeAuxPrefetchControlMessage,
    HordeAuxPrefetchResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_MODEL = "stable_diffusion"
_LORA = "evictable-style"


class _FakeDownloadProcess:
    """The download process's presence rule over a scripted disk, read from the pipe the parent writes to.

    A requested entry already on disk is answered with an immediate success and no fetch. An absent entry is
    fetched (recorded in ``fetched``) and then answered with a success. ``evict`` removes a file as the floor
    guard does, sending nothing to the parent.
    """

    def __init__(self, process_info: HordeProcessInfo) -> None:
        self._process_info = process_info
        self._answered = 0
        self.on_disk: set[str] = set()
        self.fetched: list[str] = []

    def requests(self) -> list[HordeAuxPrefetchControlMessage]:
        """Return every entry-bearing prefetch request the parent has sent so far."""
        send = self._process_info.pipe_connection.send
        assert isinstance(send, Mock)
        return [
            call.args[0]
            for call in send.call_args_list
            if isinstance(call.args[0], HordeAuxPrefetchControlMessage) and call.args[0].entries
        ]

    def evict(self, name: str) -> None:
        """Remove a file from disk without telling the parent."""
        self.on_disk.discard(name)

    def answer_new_requests(self) -> HordeAuxPrefetchResultMessage | None:
        """Answer every request not yet answered, or return None when there is none."""
        pending = self.requests()[self._answered :]
        self._answered += len(pending)
        outcomes: list[AuxPrefetchOutcome] = []
        for request in pending:
            for entry in request.entries:
                if entry.name not in self.on_disk:
                    self.fetched.append(entry.name)
                    self.on_disk.add(entry.name)
                outcomes.append(
                    AuxPrefetchOutcome(
                        kind=entry.kind,
                        name=entry.name,
                        is_version=entry.is_version,
                        ok=True,
                        requesting_job_ids=[entry.requesting_job_id],
                    ),
                )
        if not outcomes:
            return None
        return HordeAuxPrefetchResultMessage(
            process_id=DOWNLOAD_PROCESS_ID,
            process_launch_identifier=self._process_info.process_launch_identifier,
            info="aux prefetch result",
            outcomes=outcomes,
        )


def _lora_job() -> ImageGenerateJobPopResponse:
    return make_job_pop_response(
        _MODEL,
        width=512,
        height=512,
        ddim_steps=20,
        loras=[LorasPayloadEntry(name=_LORA, model=1.0, clip=1.0, is_version=False)],
    )


async def _pop(manager: HordeWorkerProcessManager, job: ImageGenerateJobPopResponse) -> None:
    """Track a popped job and run the pop-time prefetch trigger, in the order the popper runs them."""
    await track_popped_job_async(manager._job_tracker, job)
    manager._aux_prefetch_coordinator.on_job_popped(job)


async def _deliver(manager: HordeWorkerProcessManager, download: _FakeDownloadProcess) -> None:
    """Route the download process's answer through the manager's message dispatcher."""
    result = download.answer_new_requests()
    assert result is not None, "the parent sent no request to answer"
    await manager._message_dispatcher._dispatch_buffered_message(result)


def _resident_lane(manager: HordeWorkerProcessManager) -> HordeProcessInfo:
    lane = make_mock_process_info(0, model_name=_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    lane.total_vram_mb = 16_375
    lane.process_reserved_mb = 1_372
    manager._process_map.clear()
    manager._process_map[0] = lane
    manager._horde_model_map.update_entry(_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
    return lane


def _sent_start_inference(lane: HordeProcessInfo) -> bool:
    send = lane.pipe_connection.send
    assert isinstance(send, Mock)
    return any(
        isinstance(call.args[0], HordeControlMessage) and call.args[0].control_flag is HordeControlFlag.START_INFERENCE
        for call in send.call_args_list
    )


async def test_job_for_an_evicted_lora_dispatches_only_after_the_download_process_answers() -> None:
    """The second job for an evicted LoRA is held until the download process re-fetches and answers it."""
    manager = make_testable_process_manager(
        stable_diffusion_reference={_MODEL: make_mock_model_reference_record(_MODEL)},
        max_threads=1,
    )
    download_info = make_mock_process_info(
        DOWNLOAD_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.DOWNLOAD,
    )
    manager._process_lifecycle._download_process_info = download_info
    download = _FakeDownloadProcess(download_info)
    tracker = manager._job_tracker

    lane = _resident_lane(manager)
    first = _lora_job()
    await _pop(manager, first)
    assert len(download.requests()) == 1
    await _deliver(manager, download)
    assert download.fetched == [_LORA]
    assert tracker.are_job_aux_models_prepared(first) is True
    assert await manager._inference_scheduler.start_inference() is True
    assert first in tracker.jobs_in_progress

    # The first job leaves the tracker, its pin lapses, and the floor guard evicts the file.
    tracker.handle_job_fault_now(first, retryable=False, fault_reason="finished")
    download.evict(_LORA)
    lane = _resident_lane(manager)

    second = _lora_job()
    await _pop(manager, second)

    assert len(download.requests()) == 2
    assert [entry.name for entry in download.requests()[-1].entries] == [_LORA]
    assert tracker.job_requires_aux_preparation(second) is True
    await manager._inference_scheduler.start_inference()
    assert second not in tracker.jobs_in_progress
    assert _sent_start_inference(lane) is False

    await _deliver(manager, download)
    assert download.fetched == [_LORA, _LORA]
    assert tracker.are_job_aux_models_prepared(second) is True
    assert await manager._inference_scheduler.start_inference() is True
    assert second in tracker.jobs_in_progress
    assert _sent_start_inference(lane) is True
