"""Resource-governance decision units for the inference scheduler.

The package separates *deciding* a resource remedy from *acting* on it: pure decision functions consume
an immutable snapshot of the host's state and return typed remedy commands, and the scheduler executes
those commands through a single dispatcher. Multi-tick bookkeeping lives with the governor that owns it
rather than scattered across the scheduler.

Modules:

* [`snapshots`][horde_worker_regen.process_management.scheduling.governance.snapshots]: immutable inputs.
* [`actions`][horde_worker_regen.process_management.scheduling.governance.actions]: typed remedy commands.
* [`ram_governor`][horde_worker_regen.process_management.scheduling.governance.ram_governor]: host system-RAM
  governance (danger floor, per-process ceiling, pop hold, shed/restore).
"""

from horde_worker_regen.process_management.scheduling.governance.actions import (
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    MarkProcessDraining,
    PausePops,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    RestoreCardProcess,
    SetPopHold,
    StopTrackingShedCard,
)
from horde_worker_regen.process_management.scheduling.governance.ram_governor import (
    RAM_PRESSURE_PAUSE_SECONDS,
    RamGovernorState,
    decide_degrade_response,
    decide_over_ceiling_reclaim,
    decide_pop_hold,
    decide_pressure_governance,
    decide_process_reduction,
    decide_shed_card_restore,
)
from horde_worker_regen.process_management.scheduling.governance.snapshots import (
    CardProcessSnapshot,
    HostMemorySnapshot,
    InferenceSlotSnapshot,
)

__all__ = [
    "RAM_PRESSURE_PAUSE_SECONDS",
    "CardProcessSnapshot",
    "ClearProcessDraining",
    "EvictIdleModels",
    "GovernanceAction",
    "HostMemorySnapshot",
    "InferenceSlotSnapshot",
    "MarkProcessDraining",
    "PausePops",
    "RamGovernorState",
    "RecycleProcess",
    "ReduceCardProcesses",
    "ReduceWorkerProcesses",
    "RestoreCardProcess",
    "SetPopHold",
    "StopTrackingShedCard",
    "decide_degrade_response",
    "decide_over_ceiling_reclaim",
    "decide_pop_hold",
    "decide_pressure_governance",
    "decide_process_reduction",
    "decide_shed_card_restore",
]
