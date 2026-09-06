"""The actions an admission decision returns for the executor to run.

A decision over the snapshot never faults a job, replaces a child or sends a message; it names the action here
and the executor performs it against the live worker. The vocabulary is deliberately small: only actions a
decision has to return travel as commands, while ledger bookkeeping and log lines stay direct calls at the act
site.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse


class FaultCause(enum.Enum):
    """Why an admission decision faults a queued job before any child touches VRAM for it."""

    UNSERVICEABLE = enum.auto()
    """No serving card can ever host the model's minimum footprint."""
    QUARANTINED = enum.auto()
    """The model's load is quarantined; loading it again re-arms the crash loop it was quarantined to stop."""


@dataclass(frozen=True)
class FaultJob:
    """Fault a queued job terminally so the horde reissues it elsewhere."""

    job: ImageGenerateJobPopResponse
    cause: FaultCause
    reason: str


@dataclass(frozen=True)
class ReplaceProcess:
    """Cycle an inference child before its slot takes a different model."""

    process_id: int


SchedulerCommand = FaultJob | ReplaceProcess
