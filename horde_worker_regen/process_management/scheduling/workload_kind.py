"""The leaf home for :class:`WorkloadKind`, importable without the flow machinery."""

from __future__ import annotations

from strenum import StrEnum


class WorkloadKind(StrEnum):
    """A distinct kind of work reGen pops, runs, and submits as its own flow.

    Audio and video generation are the intended next entries; they are reserved here (commented) rather
    than declared so an unhandled member cannot be routed before its flow exists.
    """

    IMAGE_GENERATION = "image_generation"
    ALCHEMY = "alchemy"
    TEXT_GENERATION = "text_generation"
    # AUDIO_GENERATION = "audio_generation"  # reserved: the next flow to add
    # VIDEO_GENERATION = "video_generation"  # reserved
