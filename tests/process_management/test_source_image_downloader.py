"""Tests for SourceImageDownloader."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from horde_worker_regen.consts import MAX_SOURCE_IMAGE_RETRIES
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.source_image_downloader import SourceImageDownloader


def _make_downloader(
    *,
    job_tracker: JobTracker | None = None,
    aiohttp_session: object | None = None,
) -> SourceImageDownloader:
    if job_tracker is None:
        job_tracker = JobTracker()
    if aiohttp_session is None:
        aiohttp_session = Mock()

    return SourceImageDownloader(
        get_aiohttp_session=lambda: aiohttp_session,
        job_tracker=job_tracker,
    )


def _make_job_response(
    *,
    job_id: str | None = "test-job-123",
    source_image: str | None = None,
    source_mask: str | None = None,
    extra_source_images: list[object] | None = None,
) -> Mock:
    """Create a mock ImageGenerateJobPopResponse with configurable source images."""
    job = Mock()
    job.id_ = job_id
    job.source_image = source_image
    job.source_mask = source_mask
    job.extra_source_images = extra_source_images

    # Download methods return async tasks
    job.async_download_source_image = AsyncMock()
    job.async_download_source_mask = AsyncMock()
    job.async_download_extra_source_images = AsyncMock()

    # Downloaded results — None means not yet downloaded
    job.get_downloaded_source_image = Mock(return_value=None)
    job.get_downloaded_source_mask = Mock(return_value=None)
    job.get_downloaded_extra_source_images = Mock(return_value=None)

    return job


class TestDownloadSourceImagesHappyPath:
    """Successful download scenarios."""

    def test_no_source_images_returns_unchanged(self) -> None:
        """When the job has no source images at all, the response is returned as-is."""
        downloader = _make_downloader()
        job = _make_job_response(source_image=None, source_mask=None)

        result = asyncio.run(downloader.download_source_images(job))

        assert result is job
        job.async_download_source_image.assert_not_called()
        job.async_download_source_mask.assert_not_called()

    def test_non_url_source_image_not_downloaded(self) -> None:
        """Base64 source images (not starting with 'http') should not trigger downloads."""
        downloader = _make_downloader()
        job = _make_job_response(source_image="data:image/png;base64,abc123")

        result = asyncio.run(downloader.download_source_images(job))

        assert result is job
        job.async_download_source_image.assert_not_called()

    def test_url_source_image_triggers_download(self) -> None:
        """A source_image starting with 'http' should trigger a download."""
        session = Mock()
        downloader = _make_downloader(aiohttp_session=session)
        job = _make_job_response(source_image="https://example.com/img.png")

        # After first download attempt, report it as downloaded
        job.get_downloaded_source_image.return_value = None  # not yet
        job.async_download_source_image = AsyncMock(return_value=None)

        asyncio.run(downloader.download_source_images(job))

        job.async_download_source_image.assert_called()

    def test_url_source_mask_triggers_download(self) -> None:
        session = Mock()
        downloader = _make_downloader(aiohttp_session=session)
        job = _make_job_response(source_mask="https://example.com/mask.png")

        job.async_download_source_mask = AsyncMock(return_value=None)

        asyncio.run(downloader.download_source_images(job))

        job.async_download_source_mask.assert_called()

    def test_non_url_source_mask_not_downloaded(self) -> None:
        """A non-URL source mask should not be downloaded."""
        downloader = _make_downloader()
        job = _make_job_response(source_mask="base64data")

        asyncio.run(downloader.download_source_images(job))

        job.async_download_source_mask.assert_not_called()


class TestDownloadSourceImagesEdgeCases:
    """Edge cases and error paths."""

    def test_none_job_id_returns_early(self) -> None:
        """If job has no id_, we return early without attempting downloads."""
        downloader = _make_downloader()
        job = _make_job_response(job_id=None, source_image="https://example.com/img.png")

        result = asyncio.run(downloader.download_source_images(job))

        assert result is job
        job.async_download_source_image.assert_not_called()

    def test_already_downloaded_source_image_not_re_downloaded(self) -> None:
        """If the source image is already downloaded, don't download again."""
        downloader = _make_downloader()
        job = _make_job_response(source_image="https://example.com/img.png")
        job.get_downloaded_source_image.return_value = b"already downloaded"

        asyncio.run(downloader.download_source_images(job))

        # Should not have created a download task for the source image
        job.async_download_source_image.assert_not_called()


class TestDownloadRetryBehavior:
    """Retry logic when downloads fail."""

    def test_retries_on_exception(self) -> None:
        """When a download raises an exception, it should retry up to MAX_SOURCE_IMAGE_RETRIES."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)
        job = _make_job_response(source_image="https://example.com/img.png")

        call_count = 0

        async def failing_download(*args: object) -> None:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("download failed")

        job.async_download_source_image = failing_download  # type: ignore[assignment]
        jt.job_faults[job.id_] = []

        asyncio.run(downloader.download_source_images(job))

        assert call_count == MAX_SOURCE_IMAGE_RETRIES

    def test_records_fault_after_max_retries_exhausted(self) -> None:
        """After exhausting retries, a fault should be recorded on the job tracker."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)
        job = _make_job_response(source_image="https://example.com/img.png")

        async def failing_download(*args: object) -> None:
            raise ConnectionError("download failed")

        job.async_download_source_image = failing_download  # type: ignore[assignment]

        asyncio.run(downloader.download_source_images(job))

        assert job.id_ in jt.job_faults
        faults = jt.job_faults[job.id_]
        assert len(faults) >= 1
        assert any(f.ref == "source_image" for f in faults)

    def test_records_mask_fault_after_max_retries(self) -> None:
        """Mask download failures should also be recorded as faults."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)
        job = _make_job_response(source_mask="https://example.com/mask.png")

        async def failing_download(*args: object) -> None:
            raise ConnectionError("download failed")

        job.async_download_source_mask = failing_download  # type: ignore[assignment]

        asyncio.run(downloader.download_source_images(job))

        assert job.id_ in jt.job_faults
        faults = jt.job_faults[job.id_]
        assert any(f.ref == "source_mask" for f in faults)


class TestExtraSourceImages:
    """Extra source image download paths."""

    def test_extra_source_images_with_urls_downloaded(self) -> None:
        """Extra source images that are URLs should trigger download."""
        downloader = _make_downloader()

        extra1 = Mock()
        extra1.image = "https://example.com/extra1.png"
        extra2 = Mock()
        extra2.image = "data:image/png;base64,abc"

        job = _make_job_response(extra_source_images=[extra1, extra2])
        job.async_download_extra_source_images = AsyncMock(return_value=None)

        asyncio.run(downloader.download_source_images(job))

        job.async_download_extra_source_images.assert_called()

    def test_extra_source_images_no_urls_not_downloaded(self) -> None:
        """Extra source images that are all base64 should not trigger download."""
        downloader = _make_downloader()

        extra1 = Mock()
        extra1.image = "data:image/png;base64,abc"

        job = _make_job_response(extra_source_images=[extra1])
        # Not URL, so no download
        # get_downloaded_extra_source_images returns None initially
        job.get_downloaded_extra_source_images.return_value = None

        asyncio.run(downloader.download_source_images(job))

        job.async_download_extra_source_images.assert_not_called()


class TestRecordDownloadFaults:
    """Unit tests for _record_download_faults."""

    def test_creates_fault_list_if_missing(self) -> None:
        """If job_faults doesn't have an entry for this job, one should be created."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)

        job = _make_job_response(source_image="https://example.com/img.png")
        job.get_downloaded_source_image.return_value = None
        # Key NOT pre-created in job_faults

        downloader._record_download_faults(
            job,
            source_image_is_url=True,
            source_mask_is_url=False,
            any_extra_source_images_are_urls=False,
        )

        assert job.id_ in jt.job_faults
        assert len(jt.job_faults[job.id_]) == 1

    def test_appends_to_existing_fault_list(self) -> None:
        """If faults already exist for this job, new ones should be appended."""
        jt = JobTracker()
        existing_fault = Mock()
        jt.job_faults["test-job-123"] = [existing_fault]

        downloader = _make_downloader(job_tracker=jt)
        job = _make_job_response(source_image="https://example.com/img.png")
        job.get_downloaded_source_image.return_value = None

        downloader._record_download_faults(
            job,
            source_image_is_url=True,
            source_mask_is_url=False,
            any_extra_source_images_are_urls=False,
        )

        assert len(jt.job_faults["test-job-123"]) == 2
        assert jt.job_faults["test-job-123"][0] is existing_fault

    def test_no_fault_when_download_succeeded(self) -> None:
        """If the source image WAS downloaded, no fault should be recorded."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)

        job = _make_job_response(source_image="https://example.com/img.png")
        job.get_downloaded_source_image.return_value = b"downloaded"

        downloader._record_download_faults(
            job,
            source_image_is_url=True,
            source_mask_is_url=False,
            any_extra_source_images_are_urls=False,
        )

        faults = jt.job_faults.get(job.id_, [])
        assert len(faults) == 0

    def test_both_image_and_mask_failures(self) -> None:
        """Both source image and mask failures produce separate faults."""
        jt = JobTracker()
        downloader = _make_downloader(job_tracker=jt)

        job = _make_job_response(
            source_image="https://example.com/img.png",
            source_mask="https://example.com/mask.png",
        )
        job.get_downloaded_source_image.return_value = None
        job.get_downloaded_source_mask.return_value = None

        downloader._record_download_faults(
            job,
            source_image_is_url=True,
            source_mask_is_url=True,
            any_extra_source_images_are_urls=False,
        )

        faults = jt.job_faults[job.id_]
        refs = [f.ref for f in faults]
        assert "source_image" in refs
        assert "source_mask" in refs
