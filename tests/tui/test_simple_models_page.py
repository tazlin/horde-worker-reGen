"""Tests for what the Simple experience's models-and-downloads page tells a contributor.

This page answers one question for a first-time contributor: is this working, or is it stuck? A bar and a
percentage cannot answer it (a stalled transfer and a slow one both show a bar that does not move much,
and a source that never declared a size shows no bar at all), and a download the worker is fully gated on
must not be described as something it contributes through. So the page is judged here on what it says:
the bytes, the rate and the estimate behind each transfer, the reason anything failed, and an honest
account of whether the worker can serve while the downloads run.
"""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    FEATURE_SAFETY,
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.simple import SimpleModelStatusView


def _render(renderable: object) -> str:
    console = Console(width=160)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _snapshot(
    *,
    downloads: DownloadStatusSnapshot | None = None,
    loaded_model: str | None = None,
) -> WorkerStateSnapshot:
    processes = []
    if loaded_model is not None:
        processes.append(
            ProcessSnapshot(
                process_id=1,
                process_type="INFERENCE",
                last_process_state="WAITING_FOR_JOB",
                is_alive=True,
                is_busy=False,
                loaded_horde_model_name=loaded_model,
            ),
        )
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="worker", worker_version="0.0.0"),
        active_models=["Deliberate"],
        processes=processes,
        downloads=downloads,
    )


def _safety_transfer(*, downloaded: int, total: int) -> CurrentDownloadStatus:
    return CurrentDownloadStatus(
        model_name="safety models",
        feature=FEATURE_SAFETY,
        target_dir="/models/clip_blip",
        downloaded_bytes=downloaded,
        total_bytes=total,
        speed_bps=212_000,
        eta_seconds=2344,
    )


class TestTransferProgress:
    """A transfer is shown with the figures that distinguish slow from stopped."""

    def test_bytes_rate_and_estimate_accompany_the_bar(self) -> None:
        """The bar alone cannot say whether a download is moving; the numbers under it can."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                active=[_safety_transfer(downloaded=178_257_920, total=675_282_944)],
            ),
        )

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "safety models" in rendered
        assert "26%" in rendered
        assert "170.0 MB of 644.0 MB" in rendered
        assert "207.0 KB/s" in rendered
        assert "39m 04s left" in rendered

    def test_a_transfer_of_unknown_size_still_reports_its_bytes(self) -> None:
        """No declared total means no bar, which is exactly when the byte counter has to carry the answer."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                active=[
                    CurrentDownloadStatus(
                        model_name="safety models",
                        feature=FEATURE_SAFETY,
                        target_dir="/models/clip_blip",
                        downloaded_bytes=52_428_800,
                        total_bytes=0,
                        speed_bps=212_000,
                    ),
                ],
            ),
        )

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "size unknown" in rendered
        assert "50.0 MB so far" in rendered

    def test_a_phase_with_nothing_in_flight_is_named(self) -> None:
        """Scanning and initializing are work; reading them as "nothing is downloading" reads as a stall."""
        snapshot = _snapshot(downloads=DownloadStatusSnapshot(phase=DownloadPhase.SCANNING))

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "already on this computer" in rendered
        assert "Nothing is downloading right now." not in rendered


class TestWhatTheDownloadsCost:
    """The page does not claim the worker is contributing while it is gated on the download."""

    def test_a_first_run_is_told_the_worker_waits_on_these(self) -> None:
        """With no model loaded, nothing can be served until a download lands, and the page says so."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                active=[_safety_transfer(downloaded=1, total=675_282_944)],
            ),
        )

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "starts contributing once the first of these finishes" in rendered

    def test_a_serving_worker_is_told_the_downloads_are_background(self) -> None:
        """Once a model is loaded the background claim is true, and it is made."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                active=[_safety_transfer(downloaded=1, total=675_282_944)],
            ),
            loaded_model="Deliberate",
        )

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "keeps contributing meanwhile" in rendered

    def test_queued_work_is_counted(self) -> None:
        """A queue behind the current transfer is part of how long the wait is."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                active=[_safety_transfer(downloaded=1, total=675_282_944)],
                pending=[DownloadItem(model_name="Deliberate", feature="image model")],
            ),
        )

        rendered = _render(SimpleModelStatusView._render_downloads(snapshot))

        assert "1 more queued after these." in rendered


class TestSomethingWentWrong:
    """A download that failed is stated at the top of the page with its reason."""

    def test_a_failure_and_its_reason_are_surfaced(self) -> None:
        """A count of failures in a grid tells a contributor nothing they can act on."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.IDLE,
                failures=[
                    DownloadFailure(
                        model_name="safety models",
                        feature=FEATURE_SAFETY,
                        reason="RuntimeError: checksum mismatch",
                    ),
                ],
            ),
        )

        banner = SimpleModelStatusView._render_problem(snapshot)
        assert banner is not None
        rendered = _render(banner)

        assert "safety models could not be downloaded" in rendered
        assert "RuntimeError: checksum mismatch" in rendered

    def test_a_download_subsystem_error_is_surfaced(self) -> None:
        """An error phase stops every download, so it is stated rather than left to an empty downloads card."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(
                phase=DownloadPhase.ERROR,
                error_message="the model reference could not be loaded",
            ),
        )

        banner = SimpleModelStatusView._render_problem(snapshot)
        assert banner is not None

        assert "the model reference could not be loaded" in _render(banner)

    def test_a_healthy_worker_shows_no_banner(self) -> None:
        """The banner is for something to act on; a working worker must not carry one."""
        snapshot = _snapshot(
            downloads=DownloadStatusSnapshot(phase=DownloadPhase.IDLE, present_model_names=["Deliberate"]),
            loaded_model="Deliberate",
        )

        assert SimpleModelStatusView._render_problem(snapshot) is None
