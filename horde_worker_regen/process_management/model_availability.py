"""Tracks which image models are present on disk, as reported by the download process.

Public members:
    ``ModelAvailability`` -- single-writer/many-reader holder for the on-disk model set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horde_worker_regen.process_management.supervisor_channel import DownloadStatusSnapshot

_DOWNLOADING_PHASE_VALUE = "downloading"


class ModelAvailability:
    """Holds the set of image models currently present on disk, plus the live download status.

    Single-writer (the message dispatcher, on download-process reports) / many-reader (job popper,
    process lifecycle, snapshot builder). The set is ``None`` until the download process makes its
    first report; while unknown, readers treat every configured model as present, preserving the
    legacy behaviour of workers that pre-download everything and run without a download process
    (tests, harness, dry-run). ``scan_complete`` distinguishes the authoritative post-scan reports
    from the early initializing/scanning ones, which are not yet authoritative about disk contents.
    """

    _present: set[str] | None
    _currently_downloading: str | None
    _pending: tuple[str, ...]
    _failed: tuple[str, ...]
    _status: DownloadStatusSnapshot | None
    _scan_complete: bool
    _safety_present: bool
    _safety_attempted: bool
    _controlnet_present: bool | None
    _sdxl_controlnet_present: bool | None
    _post_processing_present: bool | None

    def __init__(self) -> None:
        """Initialise with availability unknown (no report received yet)."""
        self._present = None
        self._currently_downloading = None
        self._pending = ()
        self._failed = ()
        self._status = None
        self._scan_complete = False
        self._safety_present = False
        self._safety_attempted = False
        self._controlnet_present = None
        self._sdxl_controlnet_present = None
        self._post_processing_present = None

    @property
    def is_known(self) -> bool:
        """Whether the download process has reported at all (even an early initializing report)."""
        return self._present is not None

    @property
    def scan_complete(self) -> bool:
        """Whether the latest report is an authoritative post-disk-scan one."""
        return self._scan_complete

    @property
    def safety_present(self) -> bool:
        """Whether the required safety models (DeepDanbooru + CLIP) are confirmed on disk."""
        return self._safety_present

    @property
    def safety_attempted(self) -> bool:
        """Whether the download process has finished its one-shot ensure of the safety models."""
        return self._safety_attempted

    @property
    def controlnet_present(self) -> bool | None:
        """On-disk readiness of the ControlNet feature (models + annotators); None until reported."""
        return self._controlnet_present

    @property
    def sdxl_controlnet_present(self) -> bool | None:
        """On-disk readiness of SDXL-ControlNet (models + annotators + miscellaneous); None until reported."""
        return self._sdxl_controlnet_present

    @property
    def post_processing_present(self) -> bool | None:
        """On-disk readiness of the post-processing feature (GFPGAN/ESRGAN/CodeFormer); None until reported."""
        return self._post_processing_present

    @property
    def status(self) -> DownloadStatusSnapshot | None:
        """The latest rich download-status snapshot, if any has been reported."""
        return self._status

    @property
    def background_download_active(self) -> bool:
        """Whether the background download process is actively consuming download bandwidth.

        A task stays a reported in-flight download for the whole of ``download_one_model``, which
        includes the post-transfer ``validate_model`` sha256 pass. That verification consumes no network
        bandwidth, so a download whose bytes have completed (``downloaded_bytes >= total_bytes``) does not
        count as active here: otherwise the LoRA-pop guard would stay latched through a minutes-long hash
        of the final model with no further download to mask it. A download with an unknown total
        (``percent is None``) is still treated as active, since completion cannot be ruled out. When
        several downloads run in parallel (``active``), the guard holds while *any* of them is still
        transferring.
        """
        if self._status is None or self._status.phase.value != _DOWNLOADING_PHASE_VALUE:
            return False
        in_flight = self._status.active or (
            [self._status.current] if self._status.current is not None else []
        )
        return any(download.percent is None or download.percent < 100.0 for download in in_flight)

    @property
    def present(self) -> set[str] | None:
        """The models present on disk, or ``None`` if not yet reported."""
        return set(self._present) if self._present is not None else None

    @property
    def currently_downloading(self) -> str | None:
        """The model being downloaded right now, if any."""
        return self._currently_downloading

    @property
    def pending(self) -> tuple[str, ...]:
        """Models still queued to download (excludes the one in progress)."""
        return self._pending

    @property
    def failed(self) -> tuple[str, ...]:
        """Models whose download was attempted and failed."""
        return self._failed

    def update(
        self,
        *,
        present: set[str],
        currently_downloading: str | None,
        pending: tuple[str, ...],
        failed: tuple[str, ...],
        status: DownloadStatusSnapshot | None = None,
        scan_complete: bool = True,
        safety_present: bool = False,
        safety_attempted: bool = False,
        controlnet_present: bool | None = None,
        sdxl_controlnet_present: bool | None = None,
        post_processing_present: bool | None = None,
    ) -> None:
        """Replace the availability snapshot with a fresh report from the download process."""
        self._present = set(present)
        self._currently_downloading = currently_downloading
        self._pending = pending
        self._failed = failed
        self._status = status
        self._scan_complete = scan_complete
        self._safety_present = safety_present
        self._safety_attempted = safety_attempted
        self._controlnet_present = controlnet_present
        self._sdxl_controlnet_present = sdxl_controlnet_present
        self._post_processing_present = post_processing_present

    def is_present(self, model_name: str) -> bool:
        """Return whether ``model_name`` is present on disk.

        Returns True while availability is unknown, so callers do not gate work before the first
        report (a worker that pre-downloaded everything keeps behaving as it always has).
        """
        if self._present is None:
            return True
        return model_name in self._present

    def filter_present(self, model_names: set[str]) -> set[str]:
        """Return the subset of ``model_names`` present on disk (all of them while unknown)."""
        if self._present is None:
            return set(model_names)
        return {name for name in model_names if name in self._present}
