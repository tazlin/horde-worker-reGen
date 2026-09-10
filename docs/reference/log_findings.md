# Log findings

Every finding `horde-log diagnose` can emit, with the id it prints, the severity it carries, what makes
it fire, and what to do about it. The id is the stable handle: it is what the JSON output keys on, what
the dashboard's Diagnostics tab shows, and what a `see also:` line points at.

The ids are declared as `FindingKind` members in `horde_worker_regen/analysis/finding_kinds.py`, one
per entry in the `FINDING_SPECS` table. A contract test (`tests/analysis/test_log_findings_doc.py`)
walks that table in both directions, so a new kind cannot ship without an entry here and a row cannot
outlive the kind it describes. The "Fires when" and "Remedy" columns are written for a reader, not
lifted from the detector's own remediation text, so they are maintained here rather than generated.

Severity is the sort order of the report, not a queue: **critical** findings come first, then
**warning**, then **info**. Some detectors pick their severity from the evidence (a one-off versus a
sustained pattern); those are marked *varies*. Findings are for problems, so a healthy subsystem emits
nothing at all: the absence of `parent_loop_stall` means the parent loop was fine.

For how the detectors, the log lines they read, and the dashboard stay in step, see
[Log diagnostics contract](../explanation/log_diagnostics_contract.md). For the commands, see
[CLI → `horde-log`](cli.md#horde-log).

## Startup and process lifecycle

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `crash_on_start_loop` | critical | Inference children crash before reaching readiness, repeatedly. The child's own exception is lifted across the process boundary from its startup log. | Fix what the named exception says. A git clone/checkout failure points at the shared ComfyUI environment directory, not at torch. |
| `preload_kills_child_loop` | critical | One model ends every slot it is loaded onto: repeated load-failure recoveries naming the same model. | Remove that model from the offered set (or repair its weights) rather than raising timeouts. |
| `empty_model_pop_cascade` | varies | Pops arrive with no model name, and the blank name propagates into preloads and slot quarantines. | An upstream reference/pop problem; check the model reference and the offered model list. |
| `doomed_pool_no_giveup` | critical | The recovery ladder flapped through soft resets and quarantines without ever abandoning ship. | The pool cannot be restored; the worker should exit so something restarts it. Fix the underlying crash. |
| `gave_up_clean` | info | The worker abandoned an unrecoverable pool deliberately. The healthy end of the ladder, recorded so it is not mistaken for a crash. | Nothing; find the cause in the crash finding above it. |
| `stuck_inference_step` | warning | The stuck-step watchdog reaped a lane reporting the same sampling step without advancing. | Usually a driver/ComfyUI wedge on that card; check for VRAM overcommit and driver state. |
| `post_processing_vram_stall` | varies | The post-processing lane was reaped mid-stage, or ran co-resident with sampling on an over-committed card. | Give post-processing its own headroom, or move the lane off the contended card. |
| `orphan_wedge` | warning | The orphaned-job watchdog punted in-progress jobs that had no live inference slot. | A downstream symptom of slot loss; take the process-lifecycle finding above it first. |

## Memory and residency

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `oom` | critical | Explicit CUDA out-of-memory faults appear in the log. | Reduce max_power / max_threads / batch, or free VRAM. The named model is the one that did not fit. |
| `swallowed_oom` | warning | Jobs faulted with a bare "no images produced" and no classified cause: the shape an OOM takes when it is caught and reworded downstream. | Treat as an OOM until proven otherwise; the remedy is the same. |
| `file_descriptor_exhaustion` | critical | An inference process hit its file-descriptor limit (EMFILE). | Raise the process file-descriptor limit; the faults will otherwise recur on every large job. |
| `scheduler_starvation_wedge` | varies | The head of the queue was budget-deferred on an idle, ample-VRAM device until the force-admit backstop broke the wedge. | An over-conservative per-process overhead or an unsatisfiable budget for that model; reduce process churn or relax the budget. |
| `unsatisfiable_head_starvation` | varies | The same head model was deferred repeatedly over a long window with no verified progress and nothing clearing it. | Confirm the model actually fits the device; a head that never admits starves the whole queue. |
| `residency_reconciliation_holds` | varies | Dispatch was held to reconcile residency after an idle-VRAM eviction, repeatedly. | Swap churn: check `unload_models_from_vram_often` and the eviction thresholds. |
| `whole_card_convergence_wedge` | critical | A whole-card residency could not reach sole residency because a sibling or service lane pinned the teardown. | Move the pinning lane off that card, or disable whole-card residency for the model. |
| `whole_card_nonhead_residency_starvation` | varies | A whole-card residency held for a non-head model while the queue head waited. | The residency policy is serving the wrong model; check the retention and pop-claim rules. |
| `whole_card_residency_churn` | warning | Whole-card residency was reserved and restored repeatedly in one session. | Reservation churn: the pool is being torn down and rebuilt; look at process recoveries and safety placement. |
| `whole_card_pop_claim_episodes` | info | A whole-card residency claimed the pop offer at least once. Recorded so the claim windows are visible. | Nothing on its own; context for the monopoly finding. |
| `whole_card_pop_claim_monopoly` | warning | Claims repeatedly ran to their maximum hold with other models' heads parked behind them. | The claim window is too generous for this traffic mix; shorten it or narrow the claiming model set. |
| `model_churn` | varies | Preloads approach a third of dispatches, or preloads are cleared before they run. Both ratios are against the session's own dispatch count. | The resident set is turning over faster than the work. Check `unload_models_from_vram_often`, narrow the offered model set, or fix whatever prevents same-model jobs from batching onto one lane. |

## Dispatch and throughput

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `multi_card_dispatch_serialization` | varies | A multi-card worker averaged well under half its cards holding an in-flight job while the queue sat near the intake budget, with idle lanes already holding the models the queue was asking for. | A scheduling problem, not a per-card config one: look at head-of-queue ordering that will not seat a non-head job on a free card, whole-card residency holds, and per-card clearance leases. `horde-log jobs` shows the split and the concurrency histogram. |
| `head_dispatch_stall` | varies | The head of the queue was parked and not dispatching, either behind a known gate (warning) or with no gate that explains it (critical). | The named gate is the lever. A stall with no gate is a scheduler bug worth reporting. |
| `slow_generation_drop_spiral` | varies | The horde aborted generations as too slow. The verdict is attributed from the lifecycle split: pop->dispatch aging points at scheduling, finished->submit aging at pipeline balance, and slow generation itself at GPU/config. A capture without dispatch/finish lines says the wait could not be located. | Follow the attribution, not the abort: lowering max_power only helps the generation case. |
| `post_processing_deferral_starvation` | varies | The post-processing admission gate deferred the same job every scheduling tick with no lane completion afterwards. | The gate's VRAM reserve cannot be met; free headroom on that card or move the lane. |
| `safety_stage_stall` | varies | The safety stage lost verdicts or backed up enough to requeue jobs. | Enable `safety_on_gpu` so safety is not CPU-bound, or add safety capacity. |
| `pop_liveness_full_queue` | critical | The local job queue was full and stopped draining: the worker held work and served none of it. | The most severe liveness signal there is. Take the dispatch or process-lifecycle finding alongside it. |
| `parent_loop_stall` | varies | The parent drained no child IPC messages for longer than the session's own status-print cadence allows. Nothing dispatches, completes or submits while that loop is stopped. | Find what blocked the asyncio thread: a synchronous disk scan, model-reference read, or an untimed network call. `horde-log timeline` over the gap shows what came just before. |
| `lane_placement` | warning | An auxiliary lane (post-processing, utilities, safety) re-spawned onto a different card than it held, or stacked onto card 0 alongside GPU safety while other cards were free. | Read that card's duty and free VRAM against the rest of the fleet before blaming the GPU. Pin the auxiliary lanes if the placement was not intended. |

## Pops, faults, and the horde

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `forced_maintenance` | varies | The horde put the worker into maintenance (critical), or the operator did (info). | Clearing maintenance without fixing the drop cause just re-triggers it; take the spiral finding first. |
| `consecutive_failure_pause` | warning | The worker self-paused pops after consecutive faults. | A downstream symptom; the fault census names what actually failed. |
| `pop_api_error_dominance` | warning | The horde repeatedly refused this worker's pops. The horde's verbatim message is quoted. | Act on the horde's message; most refusals name a config or account condition. |
| `pop_governor_dominance` | info | One pop governor spell held the offer for a large share of the session. | Expected for a large-model cooldown; a surprise otherwise. |
| `faulted_job_census` | warning | Any job faulted this session. Lists each one's model and cause. | Take the largest cause first; the causes map to the findings above. |
| `model_reference_sample_fault` | warning | A running sample faulted on a model reference that could not be read or parsed. | A stale or partially-written reference cache; refresh it. |

## Session context

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `session_summary` | varies | Always, once per session. Carries how the session ended, its duration, the peak recovery count, the worker version and model count, and which rotated archives were folded into the parse. | Nothing; it is the header the other findings are read against. |
