"""Small presentation helpers shared across the TUI widgets."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text

from horde_worker_regen.process_management.lifecycle.process_temperature import ProcessTemperature

TEMPERATURE_COLOURS: dict[ProcessTemperature, str] = {
    ProcessTemperature.HOT: "green",
    ProcessTemperature.NEXT: "bright_yellow",
    ProcessTemperature.WARM: "cyan",
    ProcessTemperature.PRIMING: "deep_sky_blue1",
    ProcessTemperature.COLD: "grey62",
    ProcessTemperature.DOWN: "red",
}
"""Display colour per process temperature, shared by the overview table and the live-view panels."""


def temperature_colour(temperature: ProcessTemperature) -> str:
    """Return the display colour for a process temperature (active=green, cold=dim grey)."""
    return TEMPERATURE_COLOURS.get(temperature, "yellow")


_SPARK_TICKS = "▁▂▃▄▅▆▇█"
"""Eight-level block glyphs, low to high, for compact trend sparklines."""

_SPARK_TICKS_ASCII = " .:-=+*#"
"""Eight-level ASCII fallback for terminals that lack Unicode block elements."""

_low_fidelity: bool = False
"""Process-wide flag: True when the terminal cannot reliably render Unicode block elements."""


def configure_fidelity(low_fidelity: bool) -> None:
    """Set the process-wide rendering fidelity (call once at app startup, before any render).

    When low_fidelity is True, sparklines and progress bars use plain ASCII characters instead
    of Unicode block glyphs, so the display stays readable on PuTTY and other legacy terminals.
    """
    global _low_fidelity
    _low_fidelity = low_fidelity


def is_low_fidelity() -> bool:
    """Return whether ASCII-only rendering is active for this process."""
    return _low_fidelity


_JOB_ID_PALETTE: tuple[str, ...] = (
    "#5aa2ff",
    "#56d364",
    "#e3b341",
    "#b283f0",
    "#5ec8d8",
    "#f0883e",
    "#db61a2",
    "#6cb6ff",
    "#aada6c",
    "#f4a3a3",
    "#7ee787",
    "#d2a8ff",
)
"""Bright, terminal-readable hues a job id is deterministically mapped onto.

Chosen for legibility on the dark TUI background and for being visually distinct from each other, so two
jobs in flight at once almost always read as two different colours across every table that names them.
"""

_BASELINE_LABELS: dict[str, str] = {
    "stable_diffusion_1": "SD1.5",
    "stable_diffusion_2_512": "SD2",
    "stable_diffusion_2_768": "SD2",
    "stable_diffusion_xl": "SDXL",
    "stable_cascade": "Cascade",
    "flux_1": "Flux",
    "qwen_image": "Qwen",
    "z_image_turbo": "Z-Image",
}
"""Compact labels for the known image baselines, so a Baseline column stays narrow."""


def short_baseline(baseline: str | None) -> str:
    """Abbreviate a model baseline (e.g. ``stable_diffusion_xl`` -> ``SDXL``), or a dash when unknown.

    Unknown baselines fall back to a cleaned-up form of the raw name rather than being dropped, so a new
    baseline the table does not yet special-case still reads as something rather than blank.
    """
    if not baseline:
        return "-"
    if baseline in _BASELINE_LABELS:
        return _BASELINE_LABELS[baseline]
    return baseline.replace("stable_diffusion", "SD").replace("_", " ").strip()


def short_job_id(job_id: str | None, length: int = 8) -> str:
    """Return the first group of a job's UUID (the human-scannable prefix), or a dash when absent."""
    if not job_id:
        return "-"
    first_group = job_id.split("-", 1)[0]
    return first_group[:length] if first_group else "-"


def job_id_color(job_id: str | None) -> str:
    """Deterministically map a job id onto a palette colour, keyed on its first UUID group.

    The first group of a v4 UUID is eight hex digits, plenty of entropy to spread ids across the palette
    while staying stable: the same job is always the same colour everywhere it appears (process table,
    queue, recent jobs, live view), which is what lets the operator follow one job across the dashboard.
    Non-UUID ids fall back to a character-sum so they still colour rather than error.
    """
    if not job_id:
        return "grey50"
    first_group = job_id.split("-", 1)[0]
    try:
        index = int(first_group, 16)
    except ValueError:
        index = sum(ord(character) for character in first_group)
    return _JOB_ID_PALETTE[index % len(_JOB_ID_PALETTE)]


def job_id_text(job_id: str | None, length: int = 8) -> Text:
    """Render a job id's prefix in its deterministic colour, ready to drop into any table cell."""
    if not job_id:
        return Text("-", style="grey50")
    return Text(short_job_id(job_id, length), style=job_id_color(job_id))


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


def gpu_label(device_index: int, device_name: str | None, kind: str = "cuda") -> str:
    """A compact per-card label: the device index with its trimmed model name (or backend kind).

    Drops the common ``NVIDIA GeForce`` / ``NVIDIA`` vendor prefix so a 4090 reads as ``RTX 4090`` rather
    than eating the column, and falls back to the accelerator kind when the device map carried no name.
    """
    if device_name:
        trimmed = device_name
        for prefix in ("NVIDIA GeForce ", "NVIDIA "):
            if trimmed.startswith(prefix):
                trimmed = trimmed[len(prefix) :]
                break
        return f"{device_index} · {trimmed}"
    return f"{device_index} · {kind}"


def mini_bar(fraction: float, width: int) -> str:
    """Render a fixed-width filled/unfilled block bar (no percentage label).

    Shared by the live view's per-process progress and the overview's inline lanes so a single
    fill convention is used everywhere. Uses ASCII characters (#/-) when low-fidelity mode is
    active (see configure_fidelity).
    """
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * width))
    if _low_fidelity:
        return "#" * filled + "-" * (width - filled)
    return "█" * filled + "░" * (width - filled)


def sparkline(values: Sequence[float]) -> str:
    """Render a sequence of values as a sparkline, scaled to its own min/max.

    An empty sequence yields an empty string; a flat sequence renders as a low, even baseline so a
    steady (but non-zero) signal still reads as present rather than blank. Uses ASCII characters
    when low-fidelity mode is active (see configure_fidelity).
    """
    if not values:
        return ""
    ticks = _SPARK_TICKS_ASCII if _low_fidelity else _SPARK_TICKS
    lowest = min(values)
    highest = max(values)
    span = highest - lowest
    if span <= 0:
        # Flat line: show a mid-low baseline so a steady signal is visibly "there".
        return ticks[2] * len(values)
    last_index = len(ticks) - 1
    return "".join(ticks[int((value - lowest) / span * last_index)] for value in values)
