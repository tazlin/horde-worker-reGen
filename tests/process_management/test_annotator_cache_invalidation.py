"""RED reproduction: the per-file validation cache survives a taint + re-download.

``HordeDownloadProcess._feature_model_present`` caches a validated feature file for the session
(``_validated_feature_files``) so the event-driven presence refresh never re-hashes a known-good file. But
``_redownload_annotators`` deliberately replaces the on-disk bytes (``taint_models`` clears them, then each
checkpoint is re-fetched). A cached "valid" verdict is about the *previous* file, not the one now on disk, so
after a re-download the cache must be evicted for those keys. It currently is not, which produces two faults:

* A corrupt re-download is reported *present* from a stale cache entry (the feature is offered, then faults at
  job time -- exactly the failure the validation gate exists to prevent).
* The annotator verify's recovery window cannot withhold ControlNet on the presence path as its own comments
  intend, because the refresh reads the stale "valid" verdict throughout.

These tests assert the post-fix behaviour (the cache is invalidated by a re-download), so they fail RED
against the current code.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import Mock

from horde_worker_regen.model_download_core import CompVisLike
from horde_worker_regen.process_management.download_process import (
    DOWNLOAD_PROCESS_ID,
    HordeDownloadProcess,
)


class _FakeAnnotatorManager:
    """A first-class ``controlnet_annotator`` manager with a flippable per-file checksum verdict.

    Mirrors the duck-typed surface the download core uses: per-record presence, taint, and download, plus a
    ``validate_model`` whose answer can be changed mid-test to model a checkpoint that is on disk but whose
    bytes have gone bad (a corrupt or interrupted re-download).
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        """Start with every named checkpoint on disk and validating."""
        self.model_reference: dict[str, object] = {name: object() for name in names}
        self.model_folder_path = "/cn/annotators"
        self._checksum_ok = True
        self.tainted: list[str] = []
        self.download_calls: list[str] = []

    def set_checksum_ok(self, *, ok: bool) -> None:
        """Flip what a fresh ``validate_model`` would report for every checkpoint."""
        self._checksum_ok = ok

    def is_model_available(self, _model_name: str) -> bool:
        """The checkpoints stay on disk for the whole scenario (a re-download lands a file, good or bad)."""
        return True

    def validate_model(self, _model_name: str, skip_checksum: bool = False) -> bool | None:
        """Return the current checksum verdict (True passes, False is a corrupt file)."""
        return self._checksum_ok

    def taint_models(self, models: list[str]) -> None:
        """Record the taint (the real manager would clear the on-disk files here)."""
        self.tainted.extend(models)

    def download_model(self, model_name: str, *, callback: object = None, connections: int = 1) -> bool:
        """Record a fetch and report success (the bytes landed; validity is ``validate_model``'s job)."""
        self.download_calls.append(model_name)
        return True


def _download_process() -> HordeDownloadProcess:
    """A download process with ControlNet allowed and the heavyweight pipe/lock dependencies mocked out."""
    return HordeDownloadProcess(
        process_id=DOWNLOAD_PROCESS_ID,
        process_message_queue=queue.Queue(),  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        process_launch_identifier=0,
        allow_lora=False,
        allow_post_processing=False,
        allow_sdxl_controlnet=False,
        allow_controlnet=True,
    )


_ANNOTATOR_NAMES = ("annotator_hed", "annotator_depth")


def test_redownload_evicts_the_validated_cache_for_refetched_checkpoints() -> None:
    """A taint + re-download must evict the re-fetched checkpoints from the session validation cache.

    The cache exists to avoid re-hashing a *known-good* file. Once a file is tainted and re-fetched it is a
    different file on disk, so its cached verdict is stale and must be dropped; otherwise the next presence
    refresh trusts a verdict about bytes that no longer exist.
    """
    process = _download_process()
    annotator_manager: CompVisLike = _FakeAnnotatorManager(_ANNOTATOR_NAMES)  # type: ignore[assignment]

    # Prime the cache: each checkpoint validates, so it is recorded as present-and-valid for the session.
    for name in _ANNOTATOR_NAMES:
        assert process._feature_model_present(annotator_manager, "controlnet_annotator", name) is True
        assert ("controlnet_annotator", name) in process._validated_feature_files

    process._redownload_annotators(annotator_manager)

    for name in _ANNOTATOR_NAMES:
        assert ("controlnet_annotator", name) not in process._validated_feature_files, (
            f"{name!r} is still cached as valid after a re-download; the next presence refresh will report a "
            "re-fetched (possibly corrupt) file as present without re-validating it."
        )


def test_corrupt_redownload_is_reported_absent_not_served_from_stale_cache() -> None:
    """After a re-download lands a corrupt file, the feature must read not-present, not stale-present.

    This is the user-visible fault the validation gate is meant to prevent: a checkpoint on disk that fails
    its checksum should withhold the feature. With the stale cache it is offered, and a ControlNet job that
    uses the bad annotator then faults.
    """
    process = _download_process()
    annotator_manager = _FakeAnnotatorManager(_ANNOTATOR_NAMES)
    manager = SimpleNamespace(controlnet_annotator=annotator_manager)

    # The checkpoints are present and valid, then primed into the cache by an initial presence read.
    assert process._manager_all_present(manager, "controlnet_annotator") is True

    # The re-download lands corrupt bytes for every checkpoint.
    annotator_manager.set_checksum_ok(ok=False)
    process._redownload_annotators(annotator_manager)  # type: ignore[arg-type]

    assert process._manager_all_present(manager, "controlnet_annotator") is False, (
        "a corrupt re-download is reported present from a stale 'valid' cache entry; the feature is offered "
        "and then faults at job time."
    )


def test_recovery_window_withholds_controlnet_on_the_presence_path() -> None:
    """During the verify recovery window the presence refresh must re-read the just-re-fetched files.

    ``_verify_annotators`` re-downloads on a failed verify and then re-runs ``_refresh_feature_presence`` to
    report interim presence (its own comment: "withhold ControlNet during the recovery window"). That can
    only hold if the re-download invalidates the cache; otherwise the refresh keeps reading the pre-taint
    "valid" verdict and ControlNet stays offered against unverified annotators.
    """
    process = _download_process()
    annotator_manager = _FakeAnnotatorManager(_ANNOTATOR_NAMES)
    manager = SimpleNamespace(
        controlnet=SimpleNamespace(
            model_reference={},
            model_folder_path="/cn",
            is_model_available=lambda _name: True,
        ),
        controlnet_annotator=annotator_manager,
    )

    # Cache the annotators as valid (the state when the verify is first enqueued).
    assert process._manager_all_present(manager, "controlnet_annotator") is True

    # The verify failed; the recovery re-downloads files that are still bad. The presence path must now show
    # the annotators as absent so ControlNet is withheld during the recovery window.
    annotator_manager.set_checksum_ok(ok=False)
    process._redownload_annotators(annotator_manager)  # type: ignore[arg-type]

    assert process._annotators_present_now(manager) is False, (
        "the recovery window still reports annotators present (from a stale cache), so ControlNet is offered "
        "against unverified annotators instead of being withheld until the re-verify settles."
    )
