# Recovery and safety responsibility map

This page lists the one owner of each decision about replacing a process, judging the worker wedged, placing the
safety check and holding a lane down for a waiting job. The reasoning behind each rule is in
[Resilience and recovery](../explanation/resilience_and_recovery.md),
[Process lifecycle](../explanation/process_lifecycle.md), [VRAM arbiter](../explanation/vram_arbiter.md) and
[Process lanes and chaining](../explanation/process_lanes_and_chaining.md).

Paths are relative to `horde_worker_regen/process_management/`. The lifecycle is `ProcessLifecycleManager` in
`lifecycle/process_lifecycle.py`, the coordinator is `WorkerRecoveryCoordinator` in
`lifecycle/worker_recovery_coordinator.py`, and the scheduler is `InferenceScheduler`.

## Decisions and owners

| Decision | Owner | Readers | Rule in one sentence |
| --- | --- | --- | --- |
| A dead child is replaced | lifecycle `_reap_if_crashed` | every process type | Death is definitive, so the reap is not debounced and does not read boot state. |
| A live inference slot is hung or broken | lifecycle `replace_hung_processes` (silence timeout widened by `_effective_inference_step_timeout`, stuck-step watchdog, state timeouts through `_check_and_replace_process`, all-processes-timed-out), `_replace_faulting_inference_lanes` | the main loop | Each branch has its own timer, and the branches share the `_recently_recovered` debounce. |
| A crawling sampler is killed as the last rung | lifecycle `_replace_if_paged_and_slow`, `SATURATION_KILL_MIN_SECONDS`; clock `HordeWorkerProcessManager._device_saturation_duration_seconds` | the reclaim ladder's end | An `INFERENCE_STARTING` slot that crawls is replaced once the card has been SATURATED for 10 s with the ladder unresolved. |
| A slot is replaced for another owner's reason | scheduler `clearance_admit_process` (refused outright), `unload_models_from_vram` (send failure), `_replace_stale_ram_unload_process`; `PlanExecutor.execute_commands`, `execute_governance_actions`; `HordeWorkerProcessManager._control_loop_tick` (maintenance), `_apply_supervisor_command` (operator) | lifecycle `_replace_inference_process` | Each calls `_replace_inference_process` directly, whose only gate is the recovery park. |
| The inference pool is rebuilt | coordinator `perform_soft_reset`, `maybe_reset_stuck_governance_hold`; lifecycle `rebuild_inference_pool` | `RecoverySupervisor` | A rebuild replaces every live inference slot, one in `PROCESS_STARTING` included. |
| The worker is wedged | coordinator `assess_wedge`, `structural_queue_wedge_active`; the dispatcher's queue-deadlock latch, `_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS` | `RecoverySupervisor.evaluate` | One assessor over five inputs, each with its own clock, and a queue deadlock counts once 20 s have passed since the latest of the latch's start, the last excuse lifting and the last inference-slot boot ending. |
| The save-our-ship action | `lifecycle/recovery_supervisor.py` `RecoverySupervisor.evaluate` | the coordinator | A wedge held 1.5 s earns one soft reset, and both the soft reset and give-up are held under `live_boot_holds`, a live boot short of the boot hard cap. |
| The safety process is replaced | lifecycle `_reap_if_crashed`; coordinator `reconcile_orphaned_safety_jobs`; `SafetyOrchestrator.start_evaluate_safety` (dispatch failure); the soft reset; `pause_safety_on_gpu`, `restore_safety_on_gpu` | lifecycle `safety_processes_should_be_replaced` | The orphaned-verdict watchdog replaces safety only when `is_safety_capacity_available` is false, or flags it when a job is faulted after `SAFETY_REQUEUE_MAX` re-checks while a process is live, and never while `safety_placement_transition_pending` holds. |
| How long a job may await a verdict | coordinator `_orphan_safety_grace_seconds`, `ORPHAN_SAFETY_GRACE_SECONDS` | `reconcile_orphaned_safety_jobs` | The 45 s baseline holds on the GPU or with no live safety process, and the grace scales with measured checks and the backlog while a process is alive off the GPU. |
| Whether safety may take a check | coordinator `is_safety_pool_ready` | intake | An idle process able to accept a check, read for intake and never as a replacement trigger. |
| Safety leaves the GPU | scheduler `_reconcile_runtime_safety_placement` (runtime demotion), reclaim requests from the governor ladder, the arbiter and the SOS remedy (`RECLAIM_LADDER`), whole-card residency (`WHOLE_CARD`), lifecycle deferred start escalation, permission withdrawn on reload | lifecycle `pause_safety_on_gpu` with a `PauseOwner` | Every process move goes through the one actuator under a named pause owner. |
| A single check runs on the CPU | scheduler `safety_check_window`, `safety_check_patience_seconds` | `SafetyOrchestrator` | A check the arbiter cannot seat within the patience runs on the CPU, with no process move. |
| Safety returns to the GPU | scheduler `_reconcile_runtime_safety_placement`; weights through `_promote_safety_weights_if_room` | lifecycle `restore_safety_on_gpu` | The reconciler alone restores the process, after `SafetyPlacementInputs.restore_headroom_fits`, the restore dwell and the arbiter, while weight promotion needs only an unpressured card the arbiter admits. |
| The restore dwell | scheduler `_safety_placement_restore_dwell_seconds`; `scheduling/ledgers/safety_placement.py` `SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR`, `MEMORY_PRESSURE_PAUSE_OWNERS` | the reconciler | Twice the measured safety readiness latency (`safety_readiness_latency_seconds`, at least 30 s), for pauses whose owner is a memory-pressure owner. |
| The restore headroom's marginal need | scheduler `_sampling_peak_jobs_for_card`, `_resident_committed_mb_for_model` | `restore_headroom_fits` | The heaviest learned peak among in-progress and eligible queued jobs, less what a lane already holds for that model. |
| A lane stays paused for a beneficiary | `resources/reclaim_ladder.py` `ReclaimLadder._unwind_restore_obligations`, `lane_pause_beneficiaries`; `HordeWorkerProcessManager._beneficiary_waiting` | the ladder's restore | The ladder asks `beneficiary_waiting(job_id, kind)` with the paused lane's `ReclaimRungKind`, and a job waiting for post-processing never holds the post-processing lane. |
| A post-processing job waits for room | `workers/post_process_orchestrator.py` `PostProcessOrchestrator._age_out_unfittable`, `_ADMISSION_PATIENCE_SECONDS` | the job tracker | A chain that cannot be admitted for 90 s is faulted, and the clock runs under a reclaim-ladder pause. |
| A child ends on a handler exception | `lifecycle/horde_process.py` `receive_and_handle_control_messages` | the parent's reap | Any exception from a supported handler ends the child. |
| A LoRA is on disk | `JobTracker._cached_lora_keys`; hordelib's ad-hoc cache in the download process; pins from `AuxPrefetchCoordinator._current_pins` applied by the download process's `_apply_eviction_pins` | aux preparation | The parent's set is "seen present this session", dropped only on a resolve failure, and every queued job's files are pinned. |

## Fences

- **One soft reset on a 1.5 s wedge** (`RecoverySupervisor`). A transient wedge, such as a bad model load or a
  one-off deadlock, recovers at this rung.
- **`live_boot_holds` and the boot hard cap** (`RecoverySupervisor.evaluate`). A progressing replacement protects
  queued work, and a rebuild under it replaces the replacement. The hard cap keeps a stuck boot from masking the wedge.
- **The 20 s structural-wedge minimum** (`_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`). Transient all-idle gaps and preload
  windows are not definitive wedges. The minimum restarts when an excuse lifts or an inference slot finishes booting
  (`structural_queue_wedge_active`), so a hold's end is not read as a sustained wedge.
- **The latch held while a process starts, re-armed only for a booting inference slot.** A respawning slot must not
  clear a real deadlock, and remedies that cycle safety or service lanes must not hold the wedge clock at zero.
- **The scaled orphan grace and the capacity test** (`_orphan_safety_grace_seconds`,
  `is_safety_capacity_available`). The safety lane is single-serial, so the one process is unready for the whole of
  every check. A busy process is progressing, and replacing it discards the check it holds.
- **The flag after `SAFETY_REQUEUE_MAX` re-checks.** The watchdog is the only reader of safety-check progress, since
  the lifecycle hang watchdog does not keep safety-check state.
- **The restore dwell.** Safety leaves promptly and comes back once room is durable, so a restore does not re-trip.
- **The restore forecast charges queued peaks** (`_sampling_peak_jobs_for_card`). Readmitting safety on room the next
  peak takes back is how one restore buys the next eviction.
- **Weight promotion has no dwell** (`_promote_safety_weights_if_room`). A flip that keeps the process and the CUDA
  context costs too little to earn a forecast.
- **The post-processing clause in `_beneficiary_waiting`.** Without it, a lane paused for a chain would be
  restored on the next HEALTHY sample, which the pause itself produces. The rung kind keeps that clause off the
  post-processing lane.
- **The post-processing age-out under a ladder pause.** The ladder restores a lane only when the card reaches
  HEALTHY, which may never come.
- **10 s of SATURATED before the kill** (`SATURATION_KILL_MIN_SECONDS`). A card a softer rung could still rescue is
  never killed.
- **A child exits on a handler exception.** A handler that failed mid-action leaves the process state unknown, so
  the parent recovers it. A message the child does not support is dropped instead, since no handler ran.

## Friction

- **Eleven replacement paths, one debounce.** Only the `replace_hung_processes` branches share `_recently_recovered`.
  The supervisor keeps its own action spacing, and the other paths call `_replace_inference_process`, which does not
  guard against a slot already in `PROCESS_STARTING`. A new path has to read boot and busy state itself.
- **Two pricers of safety's room.** `restore_headroom_fits` charges the heaviest queued peak through
  `_sampling_peak_jobs_for_card`, while the arbiter prices the same load on measured free. A queued job of a model no
  lane holds is charged its whole peak whether or not it would sample beside safety, so on a busy card the forecast
  can hold safety on the CPU for a whole session.
- **A memory-pressure dwell for any ladder pause.** `MEMORY_PRESSURE_PAUSE_OWNERS` includes `RECLAIM_LADDER` whatever
  filed the request, so an SOS rung for a wedge unrelated to memory earns the same forecast as a pressure eviction.
- **Remedy demotion against promotion.** The SOS `DEMOTE_SAFETY_WEIGHTS` rung frees room, and
  `_promote_safety_weights_if_room` takes it back on the next cycle because the card is not pressured. The promoter
  does not read a settling remedy, as `WorkerRecoveryCoordinator.holds_lane_pause` does for lane pauses.
- **Two readings of saturation in one ladder.** The kill's clock in `HordeWorkerProcessManager` is set only while the
  governor reads SATURATED, while the ladder it ends runs under SATURATED or the per-step floor's force. The ladder
  can run to unresolved under PRESSURE with the kill's clock at zero.
- **Known-lost verdicts cover retired launches only.** `MessageDispatcher.take_safety_verdicts_known_lost` is filled
  when a retired launch's verdict is dropped. `_reap_if_crashed` knows a safety child died mid-check but does not mark
  its jobs, so they wait out the full grace.
- **The per-check CPU route reads VRAM only.** `safety_check_window` sends a check to the CPU on the arbiter's
  verdict while the host RAM danger floor may be engaged, and `receive_and_handle_control_messages` then ends the
  child on the allocation failure.
- **LoRA presence has two owners.** hordelib's ad-hoc cache evicts without telling the parent, so
  `_cached_lora_keys` reports a file present until a resolve fails. Pins cannot hold a file evicted before its job
  was popped.

## Not determined

- Whether the restore forecast should charge queued jobs at all, or only in-progress jobs and models a lane holds.
- Whether a `RECLAIM_LADDER` pause filed by SOS or for a starved head should earn the memory-pressure dwell.
- Which state timeouts in `_check_and_replace_process` read boot state for the safety and post-processing lanes.
- Whether a rung-2 residency restores safety without the dwell on release.
- Whether the kill's clock should count the episode the ladder runs, and what horizon then applies.
- Whether a CPU allocation failure inside a safety check should fail that check and keep the child.

## Where to read next

- [Resilience and recovery](../explanation/resilience_and_recovery.md#layer-3-save-our-ship-sos-escalation) for the
  supervisor and its boot backstops, and
  [Stranded safety-check jobs](../explanation/resilience_and_recovery.md#stranded-safety-check-jobs).
- [Process lifecycle](../explanation/process_lifecycle.md#process-replacement) for the replacement paths.
- [Runtime safety placement](../explanation/vram_arbiter.md#runtime-safety-placement) and
  [the kill as last rung][kill-rung].
- [Process lanes and chaining](../explanation/process_lanes_and_chaining.md) for the restore rule on paused lanes.
- [Logs](logs.md#runtime-safety-placement-diagnostics) and
  [Supervisor stall forgiveness](logs.md#supervisor-stall-forgiveness). The `safety_stage_stall`,
  `safety_start_escalated_to_cpu`, `inference_lane_replaced`, `inference_slot_quarantined` and
  `service_lane_restart_churn` findings in [Log findings](log_findings.md) apply.

[kill-rung]: ../explanation/vram_arbiter.md#verified-reclaim-the-per-step-floor-and-the-kill-as-last-rung
