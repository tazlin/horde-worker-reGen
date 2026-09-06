"""The scheduler's act site: governance decisions and arbiter verdicts executed against the live worker.

The resource governor decides over a host-memory snapshot and the VRAM arbiter over a frozen device snapshot;
neither acts. This executes what they return, through the host (the inference scheduler) that owns the process
map, the worker state and the lifecycle manager, so the decision layers stay pure functions of their inputs and
every remedy the worker takes passes through one place.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import PopPauseOwner, WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.reclaim_ladder import VerifiedReclaimLadder
from horde_worker_regen.process_management.resources.resource_budget import RamPressureVerdict
from horde_worker_regen.process_management.resources.run_metrics import ChurnKind
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    HeadReclaimContext,
    VramActuator,
)
from horde_worker_regen.process_management.scheduling.governance import (
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    MarkProcessDraining,
    PausePops,
    RamGovernorState,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    RestoreCardProcess,
    RestoreWorkerProcess,
    SetPopHold,
    StopTrackingShedCard,
    StopTrackingWorkerShed,
    WorkerProcessShedState,
)
from horde_worker_regen.process_management.scheduling.ledgers.ram_reclaim import RamReclaimLedger
from horde_worker_regen.utils.config_coercion import config_number


class ExecutionHost(VramActuator, Protocol):
    """The worker surface the executor acts through: the VRAM actuators plus the RAM remedies and their state.

    Implemented by the inference scheduler. The RAM remedies (component eviction, idle-model unload, the
    stale-slot cycle) stay scheduler methods because they read and mutate the model map and the reserve
    ledger; the executor sequences them and records their measured result in the governor's bookkeeping.
    """

    _state: WorkerState
    _process_map: ProcessMap
    _process_lifecycle: ProcessLifecycleManager
    _runtime_config: RuntimeConfig
    _max_inference_processes: int
    _last_ram_verdict: RamPressureVerdict | None

    @property
    def _ram_governor_state(self) -> RamGovernorState:
        """The RAM governor's multi-tick bookkeeping."""
        ...

    @property
    def ram_reclaim(self) -> RamReclaimLedger:
        """The host RAM reclaim ledger."""
        ...

    def _evict_unprotected_components_under_pressure(self) -> bool:
        """Drop idle, unprotected staged components from the budgeted RAM cache."""
        ...

    def unload_models(self, *, under_pressure: bool = False, for_head_of_queue: bool = False) -> bool:
        """Unload an idle resident model from system RAM."""
        ...

    def _replace_stale_ram_unload_process(self, *, protect_process_id: int | None = None) -> bool:
        """Cycle the slot whose allocator kept freed pages, returning them to the OS."""
        ...

    def _record_churn(self, kind: ChurnKind) -> None:
        """Count one churn event of ``kind`` for the run metrics."""
        ...


class PlanExecutor:
    """Executes governance actions and arbiter verdict commands against the host, one action each."""

    def __init__(self, host: ExecutionHost) -> None:
        """Bind the executor to the host it acts through."""
        self._host = host

    def execute_actuations(
        self,
        commands: tuple[ActuatorCommand, ...],
        *,
        device_index: int | None,
        for_head_of_queue: bool,
        head: HeadReclaimContext | None = None,
    ) -> tuple[ActuatorCommand, ...]:
        """Run the pressure-relief commands a verdict described, returning the ones whose actuator acted.

        Routed through :meth:`VerifiedReclaimLadder.execute_arbiter_commands` so a verdict's DEFER path and
        the governor's SATURATED verified ladder share one reclaim execution surface: the two triggers can never
        become two mechanisms evicting the same card by different rules. ``head`` names the head an eviction or
        context reduction acts on behalf of.
        """
        return VerifiedReclaimLadder.execute_arbiter_commands(
            commands,
            self._host,
            device_index=device_index,
            for_head_of_queue=for_head_of_queue,
            head=head,
        )

    def execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
        """Execute governance decisions against the live worker: the single act site for RAM remedies.

        The governor's multi-tick bookkeeping (draining marks, shed-card tracking) is mutated here, at
        execution time and with the measured result of each remedy (a card is only recorded as shed when
        its count actually fell), so the decision layer stays a pure function of its snapshot.
        """
        governor_state = self._host._ram_governor_state
        for action in actions:
            match action:
                case SetPopHold(active=hold_active):
                    if hold_active != self._host._state.ram_pressure_pop_hold:
                        # The soft hold has no other trace than a skipped-pop counter; name the reading behind
                        # each edge so a log reader can see what "ram_pressure" was measuring.
                        reading = (
                            self._host._last_ram_verdict.reason()
                            if self._host._last_ram_verdict is not None
                            else "no reading"
                        )
                        margin = config_number(self._host._runtime_config.bridge_data.ram_reserve_mb)
                        logger.info(
                            f"Host RAM pop hold {'engaged' if hold_active else 'released'}: {reading}, hold margin "
                            f"{margin:.0f} MB above the floor while work is in flight; in-flight jobs continue.",
                        )
                    self._host._state.ram_pressure_pop_hold = hold_active
                case PausePops(
                    until_time=until_time,
                    pause_seconds=pause_seconds,
                    reason=reason,
                    available_mb=available_mb,
                    floor_mb=floor_mb,
                ):
                    prior_owner = self._host._state.self_throttle_pause_owner
                    pause_reason = f"host RAM pressure: {reason}"
                    self._host._state.self_throttle_paused = True
                    self._host._state.self_throttle_paused_until = until_time
                    self._host._state.self_throttle_pause_owner = PopPauseOwner.RAM_PRESSURE
                    self._host._state.self_throttle_pause_reason = pause_reason
                    self._host._process_lifecycle.action_ledger.record(
                        LedgerEventType.POP_PAUSE_ARMED,
                        reason=pause_reason,
                        detail={
                            "owner": PopPauseOwner.RAM_PRESSURE.value,
                            "duration_seconds": round(pause_seconds, 1),
                            "available_ram_mb": round(available_mb, 1) if available_mb is not None else None,
                            "floor_ram_mb": round(floor_mb, 1) if floor_mb is not None else None,
                        },
                    )
                    # A still-standing pause from a different backstop is only superseded here when this
                    # RAM deadline is the later one (the decision layer emits PausePops only then), so name
                    # the transition rather than silently relabelling the shared deadline.
                    takeover = (
                        f" (superseding a standing {prior_owner.value} pause)"
                        if prior_owner is not None and prior_owner is not PopPauseOwner.RAM_PRESSURE
                        else ""
                    )
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>System RAM below the danger floor ({reason}); pausing job pops for "
                        f"{pause_seconds:.0f}s{takeover} and shedding idle footprint so the host is not driven "
                        "into an OS OOM kill. In-flight jobs finish; pops resume once RAM recovers.</>",
                    )
                case EvictIdleModels():
                    # First reclaim the cheap, targeted way: drop idle, unprotected staged components from the
                    # budgeted RAM cache, keeping a queued job's staged model resident. Only when nothing
                    # unprotected can be evicted (or the budgeted cache is off) does the coarser whole-RAM
                    # unload run: unload an idle resident model, and when none remains, cycle the slot whose
                    # allocator kept the freed pages (only a process cycle returns them to the OS). Mirrors the
                    # preload reclaim path so sustained pressure with a drained queue still reclaims RAM.
                    if not self._host._evict_unprotected_components_under_pressure() and not self._host.unload_models(
                        under_pressure=True,
                    ):
                        self._host._replace_stale_ram_unload_process()
                case ReduceWorkerProcesses(
                    target_count=target_count,
                    planned_count=planned_count,
                    pressure_shortfall_mb=pressure_shortfall_mb,
                ):
                    current = self._host._process_map.num_loaded_inference_processes()
                    planned = planned_count if planned_count > 0 else self._host._max_inference_processes
                    after = self._host._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=None,
                        pressure_shortfall_mb=pressure_shortfall_mb,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        # The record is the live shortfall below plan, not an accumulation of reductions: a
                        # whole-card residency restore can regrow the pool between reductions, and a running
                        # total would over-count every cycle without bound while the pool is back at plan.
                        governor_state.worker_shed = WorkerProcessShedState(
                            planned_process_count=planned,
                            shed_process_count=max(0, planned - after),
                        )
                        shortfall_note = (
                            f", shortfall ~{pressure_shortfall_mb:.0f} MB" if pressure_shortfall_mb is not None else ""
                        )
                        logger.opt(ansi=True).info(
                            f"<fg #ff8c69>RAM pressure reduced worker inference contexts "
                            f"({current} -> {after} of {planned}{shortfall_note}); the pool will be "
                            "restored incrementally once RAM has headroom.</>",
                        )
                case ReduceCardProcesses(device_index=device_index, target_count=target_count):
                    current = self._host._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._host._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        governor_state.shed_cards.add(device_index)
                case MarkProcessDraining(
                    process_id=process_id,
                    resident_ram_mb=resident_ram_mb,
                    ceiling_mb=ceiling_mb,
                ):
                    governor_state.draining_process_ids.add(process_id)
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling) while the host is under its RAM floor; "
                        "draining it (no new work) so it can be recycled once its in-flight job finishes.</>",
                    )
                case ClearProcessDraining(process_id=process_id):
                    governor_state.draining_process_ids.discard(process_id)
                case RecycleProcess(process_id=process_id, resident_ram_mb=resident_ram_mb, ceiling_mb=ceiling_mb):
                    process_info = self._host._process_map.get(process_id)
                    if process_info is None:
                        # The process exited between snapshot and execution; nothing to reclaim.
                        governor_state.draining_process_ids.discard(process_id)
                        continue
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling); "
                        "recycling it to return the retained RAM to the OS.</>",
                    )
                    governor_state.draining_process_ids.discard(process_id)
                    self._host._process_lifecycle._replace_inference_process(process_info, intentional_reclaim=True)
                    self._host.ram_reclaim.note_cycle()
                    self._host._record_churn("process_cycle")
                case RestoreCardProcess(device_index=device_index, target_count=target_count, planned_count=planned):
                    current = self._host._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._host._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring an inference context on device "
                        f"{device_index} ({current} -> {after} of {planned}) so the card resumes serving.</>",
                    )
                    if after >= planned:
                        governor_state.shed_cards.discard(device_index)
                case RestoreWorkerProcess(target_count=target_count, planned_count=planned):
                    current = self._host._process_map.num_loaded_inference_processes()
                    after = self._host._process_lifecycle.scale_inference_processes(target_count, device_index=None)
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring a worker inference context "
                        f"({current} -> {after} of {planned}).</>",
                    )
                    if after >= planned:
                        governor_state.worker_shed = None
                    elif governor_state.worker_shed is not None and after > current:
                        governor_state.worker_shed.shed_process_count = max(
                            0,
                            governor_state.worker_shed.shed_process_count - (after - current),
                        )
                case StopTrackingShedCard(device_index=device_index):
                    governor_state.shed_cards.discard(device_index)
                case StopTrackingWorkerShed():
                    governor_state.worker_shed = None
