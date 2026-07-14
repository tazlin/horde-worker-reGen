"""The LoRA-pop guard tracks download *bandwidth*, not the mere existence of a download task.

The worker suppresses LoRA job pops only while the background download process is actively consuming
download bandwidth (``ModelAvailability.background_download_active`` ->
``lora_pops_blocked_by_downloads`` and ``job_popper._effective_allow_lora``): the point of the guard
is to avoid competing for the network with an in-flight model fetch.

A download task remains the reported ``current`` download for the whole of ``download_one_model``,
which transfers bytes (the chunk callback drives progress to 100%) and then runs ``validate_model`` to
hash the file. That sha256 pass has no progress callback and can take minutes on a large file, during
which ``current`` stays pinned at 100% (``downloaded_bytes == total_bytes``). The transfer is finished
and no bandwidth is in use, so the guard must release: ``background_download_active`` is True only while
some download still has bytes in flight (``percent < 100``, or an unknown total). A completed-bytes
download in its verification window does not block LoRA pops, including when it is the final model and
nothing follows to mask a latched guard.
"""

from __future__ import annotations

import queue
from unittest.mock import Mock

from horde_worker_regen.model_download_core import ChunkPacer
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    FEATURE_LORA_ADHOC,
    FEATURE_TI_ADHOC,
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
)
from horde_worker_regen.process_management.models.download_scheduler import DownloadKind
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.workers.download_process import (
    DOWNLOAD_PROCESS_ID,
    FEATURE_IMAGE_MODEL,
    HordeDownloadProcess,
    _TaskRuntime,
)


def _make_process() -> HordeDownloadProcess:
    """A bare download process; the verify-window state is pure (no managers/hordelib/GPU needed)."""
    return HordeDownloadProcess(
        process_id=DOWNLOAD_PROCESS_ID,
        process_message_queue=queue.Queue(),  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        process_launch_identifier=0,
    )


class TestCompletedBytesNotBandwidthActive:
    """A current download whose bytes are complete is verifying, not consuming bandwidth."""

    def test_model_availability_clears_when_bytes_complete(self) -> None:
        """A 100%-byte current under DOWNLOADING does not assert background_download_active.

        The parent stores a snapshot whose only download has finished transferring
        (``downloaded_bytes == total_bytes``) and is now hashing. No bandwidth is in use, so the guard
        releases and LoRA pops resume.
        """
        availability = ModelAvailability()
        availability.update(
            present={"a"},
            currently_downloading="big-model",
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                current=CurrentDownloadStatus(
                    model_name="big-model",
                    feature=FEATURE_IMAGE_MODEL,
                    target_dir="",
                    downloaded_bytes=4_000_000_000,
                    total_bytes=4_000_000_000,
                ),
            ),
        )

        assert availability.background_download_active is False

    def test_model_availability_still_active_while_transferring(self) -> None:
        """A genuinely in-flight transfer (bytes < total) still blocks LoRA pops."""
        availability = ModelAvailability()
        availability.update(
            present={"a"},
            currently_downloading="big-model",
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                current=CurrentDownloadStatus(
                    model_name="big-model",
                    feature=FEATURE_IMAGE_MODEL,
                    target_dir="",
                    downloaded_bytes=1_000_000_000,
                    total_bytes=4_000_000_000,
                ),
            ),
        )

        assert availability.background_download_active is True


class TestVerifyWindowReporting:
    """The download process must not advertise a verifying (post-transfer) task as bandwidth-active."""

    def test_build_status_during_validate_is_not_bandwidth_active(self) -> None:
        """A task still in ``_active`` after its bytes hit 100% reports as verifying, not transferring.

        ``download_one_model`` keeps the task in ``_active`` while ``validate_model`` runs its sha256, so
        ``_build_status`` emits a 100%-byte ``current``. Consumed by the parent, that status leaves the
        LoRA guard released, since the verification consumes no network bandwidth.
        """
        process = _make_process()
        runtime = _TaskRuntime(
            status=CurrentDownloadStatus(
                model_name="big-model",
                feature=FEATURE_IMAGE_MODEL,
                target_dir="",
                downloaded_bytes=4_000_000_000,
                total_bytes=4_000_000_000,
            ),
            pacer=ChunkPacer(),
        )
        process._active[(DownloadKind.IMAGE_MODEL, "", "big-model")] = runtime

        status = process._build_status(DownloadPhase.DOWNLOADING)
        assert status.current is not None
        assert status.current.percent == 100.0

        availability = ModelAvailability()
        availability.update(
            present={"big-model"},
            currently_downloading=status.current.model_name,
            pending=(),
            failed=(),
            status=status,
        )

        assert availability.background_download_active is False


class TestLastDownloadDoesNotHangGuard:
    """The final download releases the guard on its own: no later download follows to mask a latch."""

    def test_final_model_verify_does_not_keep_lora_blocked(self) -> None:
        """The guard stays released across repeated verify-window reports of the final model.

        While the final model's sha256 runs, the download process settles on a steady DOWNLOADING status
        with ``current`` pinned at 100% and no further bandwidth activity, and re-emits it each tick. Every
        reading of availability reports no active background download.
        """
        availability = ModelAvailability()
        completed_current = CurrentDownloadStatus(
            model_name="last-model",
            feature=FEATURE_IMAGE_MODEL,
            target_dir="",
            downloaded_bytes=6_000_000_000,
            total_bytes=6_000_000_000,
        )
        # The task stays in _active through verification, so the process re-emits the same verify-window
        # snapshot each tick; feeding it repeatedly confirms the guard is read as released every time.
        for _ in range(3):
            availability.update(
                present={"last-model"},
                currently_downloading="last-model",
                pending=(),
                failed=(),
                status=DownloadStatusSnapshot(phase=DownloadPhase.DOWNLOADING, current=completed_current),
            )
            assert availability.background_download_active is False


class TestPrefetchDownloadsDoNotGateLoraAdvertising:
    """A job-driven ad-hoc LoRA/TI prefetch is how a LoRA job becomes dispatchable, so it must not gate pops.

    The worker-wide LoRA-advertising guard (``background_download_active``) suppresses LoRA pops only for
    non-prefetch downloads (bulk/default seeding, image/aux fetches). A single job-driven prefetch flipping
    the guard would refuse cache-hit LoRA jobs and self-serialize LoRA intake under the prefetch design.
    """

    @staticmethod
    def _availability_for(*downloads: CurrentDownloadStatus) -> ModelAvailability:
        availability = ModelAvailability()
        availability.update(
            present=set(),
            currently_downloading=downloads[0].model_name if downloads else None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                current=downloads[0] if downloads else None,
                active=list(downloads),
            ),
        )
        return availability

    def test_adhoc_prefetch_only_leaves_advertising_on(self) -> None:
        """A snapshot whose only in-flight download is an ad-hoc prefetch does not suppress LoRA advertising."""
        for feature in (FEATURE_LORA_ADHOC, FEATURE_TI_ADHOC):
            availability = self._availability_for(
                CurrentDownloadStatus(
                    model_name="733630",
                    feature=feature,
                    target_dir="",
                    downloaded_bytes=1_000_000,
                    total_bytes=50_000_000,
                ),
            )
            assert availability.background_download_active is False

    def test_bulk_image_download_suppresses(self) -> None:
        """A bulk image-model download in flight suppresses LoRA advertising as before."""
        availability = self._availability_for(
            CurrentDownloadStatus(
                model_name="big-model",
                feature=FEATURE_IMAGE_MODEL,
                target_dir="",
                downloaded_bytes=1_000_000_000,
                total_bytes=4_000_000_000,
            ),
        )
        assert availability.background_download_active is True

    def test_mixed_prefetch_and_bulk_suppresses(self) -> None:
        """With both an ad-hoc prefetch and a bulk download in flight, the bulk download still suppresses."""
        availability = self._availability_for(
            CurrentDownloadStatus(
                model_name="733630",
                feature=FEATURE_LORA_ADHOC,
                target_dir="",
                downloaded_bytes=1_000_000,
                total_bytes=50_000_000,
            ),
            CurrentDownloadStatus(
                model_name="big-model",
                feature=FEATURE_IMAGE_MODEL,
                target_dir="",
                downloaded_bytes=1_000_000_000,
                total_bytes=4_000_000_000,
            ),
        )
        assert availability.background_download_active is True
