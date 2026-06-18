"""Small presentation helpers shared across the TUI widgets."""

from __future__ import annotations

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
