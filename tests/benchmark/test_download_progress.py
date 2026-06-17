"""Tests for the line-delimited download-progress channel the TUI parses from noisy subprocess stdout."""

from __future__ import annotations

from horde_worker_regen.benchmark.download_progress import (
    DownloadEvent,
    DownloadModelRow,
    decode_download_events,
    encode_download_event,
)


def test_encode_decode_round_trips_a_planned_event() -> None:
    """A planned event survives encode -> (noisy stdout) -> decode with its model rows intact."""
    event = DownloadEvent(
        kind="planned",
        models=[DownloadModelRow(name="m1", size_bytes=123, on_disk=True, target_path="/a")],
        present_bytes=123,
        to_download_bytes=0,
        free_disk_bytes=999,
        fits=True,
    )
    decoded = decode_download_events(encode_download_event(event))
    assert len(decoded) == 1
    assert decoded[0].kind == "planned"
    assert decoded[0].models[0].target_path == "/a"


def test_decode_isolates_events_from_interleaved_log_lines() -> None:
    """Decoding ignores surrounding log/banner noise and recovers every sentinel-wrapped event in order."""
    started = encode_download_event(DownloadEvent(kind="model_started", name="m1", index=1, total=2))
    finished = encode_download_event(DownloadEvent(kind="model_finished", name="m1", index=1, total=2, ok=True))
    noisy = f"INFO loading hordelib...\n{started}\nsome banner line\n{finished}\nDONE\n"

    decoded = decode_download_events(noisy)

    assert [event.kind for event in decoded] == ["model_started", "model_finished"]
    assert decoded[1].ok is True


def test_decode_tolerates_a_truncated_trailing_event() -> None:
    """A partially-flushed final line (no closing sentinel) is skipped rather than raising."""
    good = encode_download_event(DownloadEvent(kind="complete", downloaded=1, failed=0))
    decoded = decode_download_events(f"{good}\n<<<HORDE_BENCHMARK_DL>>>{{partial")
    assert [event.kind for event in decoded] == ["complete"]
