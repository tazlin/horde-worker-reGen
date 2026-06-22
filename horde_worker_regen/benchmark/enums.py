"""Closed value sets for the benchmark ramp: tiers, stages, axes, outcomes, and findings.

Centralising these as :class:`~enum.StrEnum` keeps the ladder, criteria, controller, and report
free of the magic strings they previously compared against (``level.axis == "controlnet"``), so a
typo is a load-time error and the valid set is discoverable. Every member is a ``str`` at runtime,
so the enums serialize to their value in JSON and round-trip through the pydantic models that store
them (``RampLevel``, ``LevelReport``, ``Finding``) without custom encoders.
"""

from __future__ import annotations

from enum import StrEnum, auto


class BenchTier(StrEnum):
    """A model family the ramp can benchmark, conservative-to-demanding."""

    SD15 = auto()
    SDXL = auto()
    FLUX = auto()
    QWEN = auto()
    ZIMAGE = auto()


class BenchStage(StrEnum):
    """A phase of the ramp; the single-letter value is the prefix of every level id in the stage."""

    BASELINE = "A"
    """Tier baseline at the most conservative configuration; establishes the it/s reference."""
    CONCURRENCY = "B"
    """Queue depth, thread count, and batch size."""
    FEATURES = "C"
    """hires-fix, post-processing, controlnet, and the QR-code workflow."""
    ALCHEMY = "D"
    """Alchemy forms, on both execution lanes, solo and concurrent with image jobs."""
    DOWNLOADS = "E"
    """Ad-hoc network fetches (loras)."""
    VALIDATION = "V"
    """Post-ramp sustained-load soak of the synthesized recommendation."""


class BenchAxis(StrEnum):
    """What a level ramps; a failure on an axis skips that axis's higher rungs only."""

    BASELINE = auto()
    QUEUE_SIZE = auto()
    THREADS = auto()
    BATCH = auto()
    HIRES_FIX = auto()
    POST_PROCESSING = auto()
    CONTROLNET = auto()
    """Classic preprocessor controlnet (canny/hed/depth/openpose); SD1.5-only in hordelib."""
    QR_CODE = auto()
    """The QR-code controlnet workflow; the real SDXL controlnet capability (also runs on SD1.5)."""
    ALCHEMY_CLIP = auto()
    """Caption/interrogation/NSFW forms, which run on the safety process (the CLIP lane)."""
    ALCHEMY_GRAPH = auto()
    """Upscaler/face-fixer/strip-background forms, which run on the inference processes (graph lane).

    A separate axis from the CLIP lane so a failure in one lane does not skip the other."""
    ALCHEMY_CONCURRENT = auto()
    """Alchemy forms run concurrently with image generation jobs (headroom-gated)."""
    DOWNLOADS = auto()
    VALIDATION = auto()


class LevelOutcome(StrEnum):
    """The terminal verdict of a single level run."""

    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CRASHED = auto()
    CRASHED_HANG = auto()


class FindingKind(StrEnum):
    """A class of robustness problem surfaced into the report's remediation queue."""

    OOM = auto()
    HANG = auto()
    CRASH = auto()
    LOST_JOB = auto()
    DOUBLE_SUBMIT = auto()
    PROCESS_RECOVERY = auto()
    DOWNLOAD_STALL = auto()
    SWALLOWED_ERROR = auto()


SELECTABLE_AXES: tuple[BenchAxis, ...] = (
    BenchAxis.QUEUE_SIZE,
    BenchAxis.THREADS,
    BenchAxis.BATCH,
    BenchAxis.HIRES_FIX,
    BenchAxis.POST_PROCESSING,
    BenchAxis.CONTROLNET,
    BenchAxis.QR_CODE,
    BenchAxis.ALCHEMY_CLIP,
    BenchAxis.ALCHEMY_GRAPH,
    BenchAxis.ALCHEMY_CONCURRENT,
)
"""The axes an operator can individually deselect (CLI ``--exclude-axis``, TUI per-axis switches).

Ordered by stage (concurrency, then features, then alchemy) for stable presentation. BASELINE,
DOWNLOADS, and VALIDATION are deliberately excluded: they are governed by other flags (the tier set,
``--include-downloads``, and ``--no-validate``) rather than by per-axis selection."""


__all__ = [
    "SELECTABLE_AXES",
    "BenchAxis",
    "BenchStage",
    "BenchTier",
    "FindingKind",
    "LevelOutcome",
]
