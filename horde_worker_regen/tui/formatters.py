"""Small presentation helpers shared across the TUI widgets."""

from __future__ import annotations

from collections.abc import Sequence

_SPARK_TICKS = "▁▂▃▄▅▆▇█"
"""Eight-level block glyphs, low to high, for compact trend sparklines."""

STATE_LABELS: dict[str, str] = {
    "PROCESS_STARTING": "Starting",
    "WAITING_FOR_JOB": "Idle",
    "PRELOADING_MODEL": "Preloading",
    "PRELOADED_MODEL": "Preloaded",
    "DOWNLOADING_MODEL": "Downloading",
    "DOWNLOADING_AUX_MODEL": "Fetching aux",
    "INFERENCE_STARTING": "Sampling",
    "INFERENCE_POST_PROCESSING": "Post-proc",
    "INFERENCE_COMPLETE": "Inference done",
    "INFERENCE_FAILED": "Failed",
    "JOB_RECEIVED": "Job received",
    "ALCHEMY_STARTING": "Alchemy",
    "ALCHEMY_COMPLETE": "Alchemy done",
    "ALCHEMY_FAILED": "Alchemy failed",
    "SAFETY_STARTING": "Safety check",
    "EVALUATING_SAFETY": "Safety check",
    "SAFETY_FAILED": "Safety failed",
    "PROCESS_ENDING": "Ending",
    "PROCESS_ENDED": "Ended",
}
"""Human-readable labels for ``HordeProcessState`` names carried in a snapshot."""


def label_state(state: str) -> str:
    """Return a human-readable label for a process-state name, title-casing unknowns."""
    return STATE_LABELS.get(state, state.replace("_", " ").title())


def human_bytes(num_bytes: float | None) -> str:
    """Render a byte count as a human-readable string (e.g. ``12.3 GB``)."""
    if num_bytes is None:
        return "-"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def human_mb(num_mb: int | float | None) -> str:
    """Render a megabyte count as a human-readable string."""
    if num_mb is None:
        return "-"
    return human_bytes(float(num_mb) * 1024 * 1024)


def human_duration(seconds: float | None) -> str:
    """Render a duration in seconds as ``1h 02m 03s`` / ``2m 03s`` / ``45s``."""
    if seconds is None:
        return "-"
    total = int(max(seconds, 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_its(its: float | None) -> str:
    """Render an iterations-per-second value, treating the ``-1.0`` sentinel as unknown."""
    if its is None or its < 0:
        return "-"
    return f"{its:.2f} it/s"


def format_percent(value: float | None, *, digits: int = 0) -> str:
    """Render a percentage value, or a dash when unknown."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def shorten(text: str | None, length: int = 28) -> str:
    """Truncate a string to ``length`` characters with an ellipsis, or dash when empty."""
    if not text:
        return "-"
    return text if len(text) <= length else text[: length - 1] + "…"


def mini_bar(fraction: float, width: int) -> str:
    """Render a fixed-width filled/unfilled block bar (no percentage label).

    Shared by the live view's per-process progress and the overview's inline lanes so a single
    fill convention is used everywhere.
    """
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def sparkline(values: Sequence[float]) -> str:
    """Render a sequence of values as a unicode block sparkline, scaled to its own min/max.

    An empty sequence yields an empty string; a flat sequence renders as a low, even baseline so a
    steady (but non-zero) signal still reads as present rather than blank.
    """
    if not values:
        return ""
    lowest = min(values)
    highest = max(values)
    span = highest - lowest
    if span <= 0:
        # Flat line: show a mid-low baseline so a steady signal is visibly "there".
        return _SPARK_TICKS[2] * len(values)
    last_index = len(_SPARK_TICKS) - 1
    return "".join(_SPARK_TICKS[int((value - lowest) / span * last_index)] for value in values)
