"""Structured, line-delimited progress for the ``horde-benchmark download`` subcommand.

The TUI runs ``download`` as a subprocess and streams its stdout to show live, per-model progress. That
stdout is *not* pure JSON: the benchmark imports the inference stack and loguru/hordelib write banners and
log lines onto the same stream. So each progress event is emitted on its own line, wrapped in unmistakable
sentinels, and the reader scans line-by-line for the payload (mirroring the ``plan`` JSON convention in
:mod:`horde_worker_regen.benchmark.progress_channel`).
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

_DL_BEGIN = "<<<HORDE_BENCHMARK_DL>>>"
_DL_END = "<<<END_HORDE_BENCHMARK_DL>>>"


class DownloadModelRow(BaseModel):
    """One model in the download plan: its size, whether it is already present, and where it lives."""

    name: str
    size_bytes: int | None = None
    on_disk: bool = False
    target_path: str = ""
    is_aux: bool = False
    """True for an auxiliary/feature file (controlnet checkpoint, post-processor, annotator) rather than an
    image checkpoint. These are fetched through the download subsystem's aux pass (each via its own model
    manager), NOT requested by name as image models -- doing so routes them to the image manager, which has no
    record of them and fails. A surface requesting a download must keep these out of the image-model set."""


class DownloadEvent(BaseModel):
    """A single line of download progress, discriminated by :attr:`kind`.

    One lean model (rather than a class per kind) keeps the line encoder/decoder trivial; unused fields
    simply stay at their defaults for a given kind.
    """

    kind: Literal["planned", "model_started", "model_progress", "model_finished", "complete"]

    # kind == "planned"
    models: list[DownloadModelRow] = Field(default_factory=list)
    present_bytes: int = 0
    to_download_bytes: int = 0
    free_disk_bytes: int | None = None
    fits: bool = True
    shortfall_bytes: int = 0

    # kind in {"model_started", "model_progress", "model_finished"}
    name: str = ""
    index: int = 0
    """1-based position of this model among those being downloaded."""
    total: int = 0
    """How many models are being downloaded in total."""
    ok: bool = True
    detail: str = ""

    # kind == "model_progress" (live per-chunk progress for the current model)
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float | None = None
    eta_seconds: float | None = None

    # kind == "complete"
    downloaded: int = 0
    failed: int = 0


class DownloadControl(BaseModel):
    """A control command the TUI sends to the running ``download`` subprocess over its stdin."""

    cmd: Literal["pause", "resume", "rate"]
    kbps: int = 0
    """For ``rate``: the bandwidth cap in kB/s (0 or negative clears it)."""


def encode_download_control(control: DownloadControl) -> str:
    """Serialise one control command as a single JSON line for the subprocess's stdin."""
    return control.model_dump_json()


def decode_download_control(line: str) -> DownloadControl | None:
    """Parse one stdin line into a control command, or None when it is blank/not a control line."""
    line = line.strip()
    if not line:
        return None
    try:
        return DownloadControl.model_validate_json(line)
    except ValueError:
        return None


def encode_download_event(event: DownloadEvent) -> str:
    """Serialise one event as a sentinel-wrapped line the reader can isolate from log noise."""
    return f"{_DL_BEGIN}{event.model_dump_json()}{_DL_END}"


def decode_download_events(raw_stdout: str) -> list[DownloadEvent]:
    """Extract every sentinel-wrapped event from a (possibly noisy) chunk of subprocess stdout."""
    events: list[DownloadEvent] = []
    cursor = 0
    while True:
        start = raw_stdout.find(_DL_BEGIN, cursor)
        if start == -1:
            return events
        end = raw_stdout.find(_DL_END, start + len(_DL_BEGIN))
        if end == -1:
            return events
        payload = raw_stdout[start + len(_DL_BEGIN) : end]
        cursor = end + len(_DL_END)
        try:
            events.append(DownloadEvent.model_validate(json.loads(payload)))
        except (json.JSONDecodeError, ValueError):
            continue


__all__ = [
    "DownloadControl",
    "DownloadEvent",
    "DownloadModelRow",
    "decode_download_control",
    "decode_download_events",
    "encode_download_control",
    "encode_download_event",
]
