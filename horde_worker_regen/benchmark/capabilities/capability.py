"""The vocabulary of the capability engine: what a machine is being asked to prove.

A :class:`Capability` is the thing a probe proves or disproves: a (tier, kind, magnitude) triple,
e.g. "SD1.5 can run batch size 4" or "SDXL can run the QR-code controlnet workflow". The
:class:`CapabilityKind` enumerates the closed set of things worth proving, and
:class:`CapabilityVerdict` the closed set of outcomes. Capabilities are frozen (hashable), so the
supervisor can hold proven/disproven sets and a probe can name its prerequisites as a tuple of
capabilities.

This module is pure: it imports only the shared :class:`~enum.StrEnum` tiers/findings and pydantic,
so it is safe to import anywhere (TUI, progress, pytest collection) without dragging the harness or
torch in.
"""

from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict

from horde_worker_regen.benchmark.enums import BenchTier


class CapabilityKind(StrEnum):
    """A class of thing a probe can prove a machine can (or cannot) do.

    These replace the old ``BenchAxis`` lattice members one-for-one, with two deliberate renames the
    report's suggestion mapping depends on: ``LORA_DOWNLOAD`` (the ad-hoc CivitAI lora fetch, which
    proves ``allow_lora``) and ``QR_CODE`` (the QR-code controlnet workflow, the genuine SDXL
    controlnet capability, which proves ``allow_sdxl_controlnet`` on SDXL). ``SUSTAINED`` is the
    post-ramp soak (the old ``validation``).
    """

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
    """Upscaler/face-fixer/strip-background forms, which run on the inference processes (graph lane)."""
    ALCHEMY_CONCURRENT = auto()
    """Alchemy forms run concurrently with image generation jobs (headroom-gated)."""
    LORA_DOWNLOAD = auto()
    """Ad-hoc lora fetches from CivitAI; proves ``allow_lora`` and measures download bandwidth."""
    SUSTAINED = auto()
    """Post-ramp sustained-load soak of the synthesized recommendation."""


class CapabilityVerdict(StrEnum):
    """The terminal outcome of one capability probe."""

    PROVEN = auto()
    """The probe ran and met its criteria: the capability is grounded in a real result."""
    DISPROVEN = auto()
    """The probe ran and failed its criteria: the capability does not hold on this machine."""
    SKIPPED = auto()
    """The probe never ran: a prerequisite was not proven, the machine could not host it, or the
    run was aborted by an earlier catastrophe."""
    CRASHED = auto()
    """The probe's worker crashed or hung without producing a usable result."""


_KIND_LABELS: dict[CapabilityKind, str] = {
    CapabilityKind.BASELINE: "baseline",
    CapabilityKind.QUEUE_SIZE: "queue depth",
    CapabilityKind.THREADS: "threads",
    CapabilityKind.BATCH: "batch size",
    CapabilityKind.HIRES_FIX: "hires-fix",
    CapabilityKind.POST_PROCESSING: "post-processing",
    CapabilityKind.CONTROLNET: "controlnet",
    CapabilityKind.QR_CODE: "qr-code controlnet",
    CapabilityKind.ALCHEMY_CLIP: "alchemy (CLIP lane)",
    CapabilityKind.ALCHEMY_GRAPH: "alchemy (graph lane)",
    CapabilityKind.ALCHEMY_CONCURRENT: "alchemy (concurrent)",
    CapabilityKind.LORA_DOWNLOAD: "ad-hoc lora download",
    CapabilityKind.SUSTAINED: "sustained load",
}


class Capability(BaseModel):
    """One provable property of a machine: a model tier, a kind, and the quantity that kind proves.

    ``magnitude`` is the numeric quantity the probe proves the machine can do, so the recommendation can
    read it straight off the result: the batch size (2 versus 4), the thread count, the queue depth, or
    the post-processing maximum resolution. It is 0 for a boolean capability (the kind either holds or it
    does not: baseline, controlnet, the alchemy lanes), and it also disambiguates the rungs of a
    quantitative kind that build on one another (batch 2 then 4; the post-processing sweep at 0 then the
    resolution-scaling probe at its max resolution). The triple is the identity, so the model is frozen
    and hashable: the supervisor keeps proven/disproven *sets* of capabilities and a probe names its
    prerequisites as a tuple of them.
    """

    model_config = ConfigDict(frozen=True)

    tier: BenchTier
    kind: CapabilityKind
    magnitude: int = 0

    @property
    def slug(self) -> str:
        """A stable, filesystem- and ``pytest -k``-friendly identifier (``sd15-controlnet``).

        The magnitude is appended only when it disambiguates (``sd15-batch-4``), so single-probe kinds
        keep the clean ``{tier}-{kind}`` form the test catalog selects on.
        """
        base = f"{self.tier}-{self.kind}"
        return f"{base}-{self.magnitude}" if self.magnitude else base

    @property
    def label(self) -> str:
        """A human-readable description for reports and the TUI (``SDXL batch size 4``)."""
        kind_label = _KIND_LABELS[self.kind]
        suffix = f" {self.magnitude}" if self.magnitude else ""
        return f"{self.tier.upper()} {kind_label}{suffix}"


__all__ = [
    "Capability",
    "CapabilityKind",
    "CapabilityVerdict",
]
