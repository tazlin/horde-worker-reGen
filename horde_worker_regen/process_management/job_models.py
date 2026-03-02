"""Job-related data models for the horde worker process management."""

from __future__ import annotations

import enum
from enum import auto

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.fields import JobID
from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from horde_worker_regen.process_management.messages import HordeImageResult
from horde_worker_regen.process_management.process_info import HordeProcessInfo


class HordeJobInfo(BaseModel):
    """Contains information about a job that has been generated.

    It is used to track the state of the job as it goes through the safety process and \
        then when it is returned to the requesting user.
    """

    sdk_api_job_info: ImageGenerateJobPopResponse
    """The API response which has all of the information about the job as sent by the API."""
    job_image_results: list[HordeImageResult] | None = None
    """A list of base64 encoded images and their generation faults that are the result of the job."""
    state: GENERATION_STATE | None
    """The state of the job to send to the API."""
    censored: bool | None = None
    """Whether or not the job was censored. This is set by the safety process."""

    time_popped: float
    time_submitted: float | None = None

    time_to_generate: float | None = None
    """The time it took to generate the job. This is set by the inference process."""

    time_to_download_aux_models: float | None = None

    @property
    def is_job_checked_for_safety(self) -> bool:
        """Return true if the job has been checked for safety."""
        return self.censored is not None

    @property
    def images_base64(self) -> list[str]:
        """Return a list containing all base64 images."""
        if self.job_image_results is None:
            return []
        return [r.image_base64 for r in self.job_image_results]

    def fault_job(self) -> None:
        """Mark the job as faulted."""
        self.state = GENERATION_STATE.faulted
        self.job_image_results = None


class JobSubmitState(enum.Enum):
    """The state of a job submit process."""

    PENDING = auto()
    """The job submit still needs to be done or retried."""
    SUCCESS = auto()
    """The job submit finished succesfully."""
    FAULTED = auto()
    """The job submit faulted for some reason."""


class PendingJob(BaseModel):
    """Base class for all PendingJobs async tasks."""

    state: JobSubmitState = JobSubmitState.PENDING
    _max_consecutive_failed_job_submits: int = 10
    _consecutive_failed_job_submits: int = 0

    @property
    def is_finished(self) -> bool:
        """Return true if the job submit has finished."""
        return self.state != JobSubmitState.PENDING

    @property
    def is_faulted(self) -> bool:
        """Return true if the job submit has faulted."""
        return self.state == JobSubmitState.FAULTED

    @property
    def retry_attempts_string(self) -> str:
        """Return a string containing the number of consecutive failed job submits and the maximum allowed."""
        return f"{self._consecutive_failed_job_submits}/{self._max_consecutive_failed_job_submits}"

    def retry(self) -> None:
        """Mark the job as needing to be retried. Fault the job if it has been retried too many times."""
        self._consecutive_failed_job_submits += 1
        if self._consecutive_failed_job_submits > self._max_consecutive_failed_job_submits:
            self.state = JobSubmitState.FAULTED

    def succeed(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Mark the job as successfully submitted."""
        self.state = JobSubmitState.SUCCESS

    def fault(self) -> None:
        """Mark the job as faulted."""
        self.state = JobSubmitState.FAULTED


class PendingSubmitJob(PendingJob):
    """Information about a job to submit to the horde."""

    completed_job_info: HordeJobInfo
    gen_iter: int
    kudos_reward: int = 0
    kudos_per_second: float = 0.0

    @property
    def image_result(self) -> HordeImageResult | None:
        """Return the image result for the job."""
        if self.completed_job_info.job_image_results is not None:
            return self.completed_job_info.job_image_results[self.gen_iter]
        return None

    @property
    def job_id(self) -> JobID:
        """Return the job ID for the job."""
        return self.completed_job_info.sdk_api_job_info.ids[self.gen_iter]

    @property
    def r2_upload(self) -> str:
        """Return the r2 upload for the job."""
        if self.completed_job_info.sdk_api_job_info.r2_uploads is None:
            return ""  # FIXME: Is this ever None? Or just a bad declaration on sdk?
        return self.completed_job_info.sdk_api_job_info.r2_uploads[self.gen_iter]

    @property
    def batch_count(self) -> int:
        """Return the number of jobs in the batch."""
        return len(self.completed_job_info.sdk_api_job_info.ids)

    @override
    def succeed(self, kudos_reward: int = 0, kudos_per_second: float = 0) -> None:
        """Mark the job as successfully submitted.

        Args:
            kudos_reward: The amount of kudos to reward the user.
            kudos_per_second: The amount of kudos per second to reward the user.
        """
        self.kudos_reward = kudos_reward
        self.kudos_per_second = kudos_per_second
        super().succeed()


class NextJobAndProcess(BaseModel):
    """Contains information about the next job to process and the process to process it with."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    next_job: ImageGenerateJobPopResponse
    process_with_model: HordeProcessInfo
    skipped_line: bool = False
    skipped_line_for: ImageGenerateJobPopResponse | None


class APIWorkerMessage(BaseModel):
    """A message sent to the worker from the API."""

    message_id: str
    """The ID of the message."""

    message_text: str | None
    """The text of the message."""

    message_origin: str | None
    """The origin (author) of the message."""

    message_expiry: str | None
    """The expiry time of the message."""
