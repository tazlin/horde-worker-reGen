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
    RestoreWorkerProcess,
    SetPopHold,
    StopTrackingShedCard,
    StopTrackingWorkerShed,
)
from horde_worker_regen.process_management.scheduling.governance.governor import (
    GovernanceHost,
    ResourceGovernor,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import (
    AdmissionDecision,
    AdmissionResult,
    PreloadSlotSnapshot,
    RamReclaimOutcome,
    ReclamationExecutor,
    VramGateResult,
    VramReclaimOutcome,
    card_preload_order,
    compute_preload_disallowed_processes,
    decide_ram_reclaim_outcome,
    decide_vram_reclaim_outcome,
    preload_concurrency_blocked,
    select_head_room_process_id,
)
from horde_worker_regen.process_management.scheduling.governance.ram_governor import (
    RAM_PRESSURE_PAUSE_SECONDS,
    RamGovernorState,
    WorkerProcessShedState,
    decide_degrade_response,
    decide_draining_followthrough,
    decide_over_ceiling_reclaim,
    decide_pop_hold,
    decide_pressure_governance,
    decide_process_reduction,
    decide_shed_card_restore,
    decide_shed_restore,
)
from horde_worker_regen.process_management.scheduling.governance.snapshots import (
    CardProcessSnapshot,
    HostMemorySnapshot,
    InferenceSlotSnapshot,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    WholeCardPhase,
    WholeCardResidency,
    WholeCardResidencyLedger,
    WholeCardResidencyMachine,
    max_coresident_for_peak,
)

__all__ = [
    "AdmissionDecision",
    "AdmissionResult",
    "RAM_PRESSURE_PAUSE_SECONDS",
    "CardProcessSnapshot",
    "ClearProcessDraining",
    "EvictIdleModels",
    "GovernanceAction",
    "GovernanceHost",
    "HostMemorySnapshot",
    "InferenceSlotSnapshot",
    "MarkProcessDraining",
    "PausePops",
    "PreloadSlotSnapshot",
    "RamGovernorState",
    "RamReclaimOutcome",
    "ReclamationExecutor",
    "RecycleProcess",
    "ReduceCardProcesses",
    "ReduceWorkerProcesses",
    "ResourceGovernor",
    "RestoreCardProcess",
    "RestoreWorkerProcess",
    "SetPopHold",
    "StopTrackingShedCard",
    "StopTrackingWorkerShed",
    "VramGateResult",
    "VramReclaimOutcome",
    "WholeCardPhase",
    "WholeCardResidency",
    "WholeCardResidencyLedger",
    "WholeCardResidencyMachine",
    "WorkerProcessShedState",
    "card_preload_order",
    "compute_preload_disallowed_processes",
    "decide_degrade_response",
    "decide_draining_followthrough",
    "decide_over_ceiling_reclaim",
    "decide_pop_hold",
    "decide_pressure_governance",
    "decide_process_reduction",
    "decide_ram_reclaim_outcome",
    "decide_shed_card_restore",
    "decide_shed_restore",
    "decide_vram_reclaim_outcome",
    "max_coresident_for_peak",
    "preload_concurrency_blocked",
    "select_head_room_process_id",
]
