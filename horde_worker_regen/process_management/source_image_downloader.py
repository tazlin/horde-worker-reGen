"""Download source images, masks, and extra source images for popped jobs."""

from __future__ import annotations

import asyncio
from asyncio import Task

from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

from horde_worker_regen.consts import MAX_SOURCE_IMAGE_RETRIES
from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.job_tracker import JobTracker


class SourceImageDownloader:
    """Download URL-based source images attached to a job pop response.

    Records download failures as job faults on the ``JobTracker`` so they
    can be reported back to the API when the job is submitted.
    """

    _api_sessions: ApiSessions
    _job_tracker: JobTracker

    def __init__(
        self,
        *,
        api_sessions: ApiSessions,
        job_tracker: JobTracker,
    ) -> None:
        """Initialize with an api-sessions holder and job tracker."""
        self._api_sessions = api_sessions
        self._job_tracker = job_tracker

    async def download_source_images(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
    ) -> ImageGenerateJobPopResponse:
        """Download any URL-based source images, masks, and extras.

        Returns:
            The same ``job_pop_response``, potentially with downloaded image
            data populated.  Download failures are recorded as faults on
            ``self._job_tracker``.
        """
        if job_pop_response.id_ is None:
            logger.error("Received ImageGenerateJobPopResponse with id_ is None. Please let the devs know!")
            return job_pop_response

        source_image_is_url = job_pop_response.source_image is not None and job_pop_response.source_image.startswith(
            "http",
        )
        if source_image_is_url:
            logger.debug(f"Source image for job {job_pop_response.id_} is a URL")

        source_mask_is_url = job_pop_response.source_mask is not None and job_pop_response.source_mask.startswith(
            "http",
        )
        if source_mask_is_url:
            logger.debug(f"Source mask for job {job_pop_response.id_} is a URL")

        any_extra_source_images_are_urls = False
        if job_pop_response.extra_source_images is not None:
            for extra_source_image in job_pop_response.extra_source_images:
                if extra_source_image.image.startswith("http"):
                    any_extra_source_images_are_urls = True
                    logger.debug(f"Extra source image for job {job_pop_response.id_} is a URL")

        attempts = 0
        while attempts < MAX_SOURCE_IMAGE_RETRIES:
            download_tasks: list[Task] = []

            if (
                source_image_is_url
                and job_pop_response.source_image is not None
                and job_pop_response.get_downloaded_source_image() is None
            ):
                download_tasks.append(
                    job_pop_response.async_download_source_image(self._api_sessions.require_aiohttp_session()),
                )
            if (
                source_mask_is_url
                and job_pop_response.source_mask is not None
                and job_pop_response.get_downloaded_source_mask() is None
            ):
                download_tasks.append(
                    job_pop_response.async_download_source_mask(self._api_sessions.require_aiohttp_session()),
                )

            download_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
            if (
                any_extra_source_images_are_urls
                and job_pop_response.extra_source_images is not None
                or (
                    download_extra_source_images is not None
                    and job_pop_response.extra_source_images is not None
                    and len(download_extra_source_images) != len(job_pop_response.extra_source_images)
                )
            ):
                download_tasks.append(
                    asyncio.create_task(
                        job_pop_response.async_download_extra_source_images(
                            self._api_sessions.require_aiohttp_session(),
                            max_retries=MAX_SOURCE_IMAGE_RETRIES,
                        ),
                    ),
                )

            gather_results = await asyncio.gather(*download_tasks, return_exceptions=True)

            for result in gather_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to download source image: {result}")
                    attempts += 1
                    break
            else:
                break

        if attempts >= MAX_SOURCE_IMAGE_RETRIES:
            await self._record_download_faults(
                job_pop_response,
                source_image_is_url=source_image_is_url,
                source_mask_is_url=source_mask_is_url,
                any_extra_source_images_are_urls=any_extra_source_images_are_urls,
            )

        return job_pop_response

    async def _record_download_faults(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
        *,
        source_image_is_url: bool,
        source_mask_is_url: bool,
        any_extra_source_images_are_urls: bool,
    ) -> None:
        """Record download failures as job faults for later reporting."""
        assert job_pop_response.id_ is not None  # caller guarantees this
        job_id = job_pop_response.id_

        if source_image_is_url and job_pop_response.get_downloaded_source_image() is None:
            logger.error(f"Failed to download source image for job {job_id}")
            await self._job_tracker.record_source_image_fault(
                job_id,
                GenMetadataEntry(
                    type=METADATA_TYPE.source_image,
                    value=METADATA_VALUE.download_failed,
                    ref="source_image",
                ),
            )

        if source_mask_is_url and job_pop_response.get_downloaded_source_mask() is None:
            logger.error(f"Failed to download source mask for job {job_id}")
            await self._job_tracker.record_source_image_fault(
                job_id,
                GenMetadataEntry(
                    type=METADATA_TYPE.source_mask,
                    value=METADATA_VALUE.download_failed,
                    ref="source_mask",
                ),
            )

        downloaded_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
        if (
            any_extra_source_images_are_urls
            and downloaded_extra_source_images is None
            or (
                downloaded_extra_source_images is not None
                and job_pop_response.extra_source_images is not None
                and len(downloaded_extra_source_images) != len(job_pop_response.extra_source_images)
            )
        ):
            logger.error(f"Failed to download extra source images for job {job_id}")

            ref = []
            if job_pop_response.extra_source_images is not None and downloaded_extra_source_images is not None:
                for predownload_extra_source_image in job_pop_response.extra_source_images:
                    if predownload_extra_source_image.image.startswith("http"):
                        if any(
                            predownload_extra_source_image.original_url == extra_source_image.image
                            for extra_source_image in downloaded_extra_source_images
                        ):
                            continue
                        ref.append(str(job_pop_response.extra_source_images.index(predownload_extra_source_image)))
            elif job_pop_response.extra_source_images is not None and downloaded_extra_source_images is None:
                ref = [str(i) for i in range(len(job_pop_response.extra_source_images))]

            for r in ref:
                await self._job_tracker.record_source_image_fault(
                    job_id,
                    GenMetadataEntry(
                        type=METADATA_TYPE.extra_source_images,
                        value=METADATA_VALUE.download_failed,
                        ref=r,
                    ),
                )
