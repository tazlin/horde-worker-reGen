# Runs and residency responsibility map

This page lists the one owner of each decision about dominant-model runs (rung 1), whole-card teardown residencies
(rung 2) and the line skips that seat work past a queue head. The reasoning behind each rule is in
[Resource governance](../explanation/resource_governance.md), [VRAM arbiter](../explanation/vram_arbiter.md),
[Admission pipeline](../explanation/admission_pipeline.md) and
[Resilience and recovery](../explanation/resilience_and_recovery.md).

## Decisions and owners

Module paths are under `horde_worker_regen/process_management/`. `InferenceScheduler` is in
`scheduling/inference_scheduler.py`, and "the ledger" is `WholeCardResidencyLedger` in
`scheduling/governance/whole_card.py`.

| Decision | Owner | Readers | Rule |
| --- | --- | --- | --- |
| Which rung a model is on | `InferenceScheduler.card_dominance`, from `pricing.model_sampling_peak` | the run grant, `StreamForecast.needs_exclusive_residency`, `pricing.pair_co_samples` | The rung is computed per card from the offered set, the learned peaks and the foreign floor, and changes only when the slack crosses zero by more than `pricing.DOMINANCE_HYSTERESIS_MB`. |
| A rung-1 run starts | `InferenceScheduler._grant_dominant_runs_at_first_step`, `_dominant_run_grantable`, `_note_dominant_run`; ledger `record_grant` at `DominanceRung.CANNOT_CO_SAMPLE` | the preload pass, each cycle | A run is granted when a rung-1 job reaches its first step and the card holds no other grant. |
| A run's copy stays on the card | `pricing.copy_belongs_to_dominant_run` | `pricing.sweep_spares_idle_copy`, `InferenceScheduler._issue_retention_eviction`, `preload.lanes_committed_elsewhere`, the retention governor gate through `InferenceScheduler._weights_belong_to_dominant_run` | The copy belongs to the run while the ledger holds the model on that card and the run has a next job, either a queued job for the model or the standing pop claim. |
| The pop offer narrows to the held model | ledger `pop_claim`, applied last in `jobs/job_popper.py` through `offer_under_pop_claim` | `JobPopper` | Every held grant of either rung claims the offer until `pop_claim_retention_ended`, which reads the max hold, three empty pops inside 10 s (`note_pop_outcome`) and a terminal fault of the model (`note_terminal_fault`). |
| How long the claim lasts | setting `whole_card_residency_max_hold_seconds` (`bridge_data/data_model.py`), read through `InferenceScheduler._whole_card_max_hold_seconds` | `pop_claim_retention_ended` | The cap runs from the establishment and ends the claim and retention only, so work already accepted still drains. |
| A run job passes a held head | `InferenceScheduler._select_dominant_run_job`, line-skip reason `dominant_run` | `get_next_job_and_process` | A run job is seated on the lane holding the run's weights when the head was declined this cycle, and does not spend the head's skip budget. |
| Lanes holding a held model are preferred | `InferenceScheduler._residency_promoted_lanes`, through ledger `holder_for_model` | dispatch placement | A grant of either rung promotes the lanes holding its model. |
| What a run does not claim | ledger `record_grant` (rung 1) and `teardown_held` | safety and ladder restores, `_residency_should_pause_safety`, `residency_bypass_models` | A run does not tear the card down, open a grace window, count toward the establish-rate limiter or carry a minimum hold, so a reader whose premise is a teardown reads `teardown_held()`. |
| A rung-1 run ends | `InferenceScheduler._restore_siblings_after_whole_card`, ledger `record_run_end` | every run reader, through the ledger | A run ends once the undispatched head is a different model past `_head_aged_past_anti_starvation`, even with its own work queued, and otherwise once that work drains and the cooldown passes, a different-model head waits or claim retention ends. |
| Large-model re-entry cooldown opens | `InferenceScheduler.is_whole_card_residency_active` (ledger `any_held`) | `jobs/large_model_pop_governor.py` | The cooldown starts when no card holds a grant of either rung. |
| A rung-2 residency is demanded | `StreamForecast.needs_exclusive_residency` (`resources/resource_budget.py`), ledger `residency_demanded` | `InferenceScheduler._decide_whole_card_demand` (preload admission), `_resident_whole_card_head_ready` (dispatch) | Only the queue head demands the card, and the dispatch path runs only for a job dispatched without a line skip. |
| A rung-2 residency ends | `InferenceScheduler._restore_siblings_after_whole_card`, `whole_card._MIN_HOLD_SECONDS`, setting `whole_card_residency_cooldown_seconds` | sibling regrowth, safety restore | A teardown holds for the minimum hold, then releases once the model's pending work has drained and the cooldown passes, retention has ended, or a different-model head is ready. |
| A new residency is deferred for churn | ledger `governor_hold` (`establish_rate_exceeded`, `_ESTABLISH_WINDOW_LIMIT` per `_ESTABLISH_WINDOW_SECONDS`) | `_decide_whole_card_demand` when the card does not hold a grant, `_resident_whole_card_head_ready` when the held model differs | Only the establish-rate limiter defers an establishment, and the deferred head waits for the card without being admitted co-resident. |
| What a governor deferral does to the rest of the worker | `AdmissionDecision.DEFER_RESIDENCY_GOVERNOR` (`scheduling/governance/preload_admission.py`), mapped to `PreloadPassControl.NEXT_JOB` by `preload.pass_control_for`; ledger `note_governor_defer`, `governor_deferred_head` | the preload pass, `InferenceScheduler._displaced_head_outstanding_mb`, `whole_card_governor_defer_active` (read by `lifecycle/worker_recovery_coordinator.py`), the `defer_residency_governor` entry in `liveness/gate_registry.py` | The pass goes on to later jobs, and the record releases head protection and excuses the recovery supervisor for at most one establish-rate window. |
| Whether an establishment or restore opens a grace window | ledger `grace_seconds_to_open` (`_GRACE_BUDGET_SECONDS` per `_GRACE_BUDGET_WINDOW_SECONDS`), charged by `record_grant` and `record_restore`, withdrawn by `close_restore_window` | `InferenceScheduler.whole_card_residency_grace_active`, read by the recovery coordinator | A card with allowance left opens the whole window and a card that has spent it does not open one, and the establishment or restore proceeds either way. |
| Which seat paths may pass the head | `LineSkip.reason` (`jobs/job_models.py`); seats in `get_next_job_and_process`: the resident-bypass loop, `_select_dominant_run_job`, `_select_idle_thread_diversity_job`, the post-processing hold, `_select_cross_card_job` | dispatch, the affinity window | Only `resident_bypass` spends the head's skip budget. |
| How far a resident-model job may pass a cold head | `scheduling/dispatch_affinity.py` (the skip count and the 0.2 x ttl window from the first skip) | the resident-bypass loop, the placement reorder | A bypass past either bound is refused. |
| When a head's age overrides a hold | `dispatch_affinity.ANTI_STARVATION_TTL_FRACTION` (0.3 x ttl from pop), through `InferenceScheduler._head_aged_past_anti_starvation` | the resident bypass, the placement reorder, `_head_starts_within_ttl_share`, preload slot sparing, the VRAM unload, the run end | A head past the bound withdraws the affinity budget and ends a rung-1 run holding the card for another model. |
| Teardown-resident jobs pass a cold head | `residency_bypass_models` in `get_next_job_and_process`, built from ledger `teardown_held()` | the resident-bypass loop | Queued jobs of a rung-2 resident model pass the head without spending its budget. |
| Other holds on a head | `HEAD_PROTECTION_MAX_STARVE_SECONDS`, `HEAD_RAM_DEFER_BARRIER_SECONDS` (`scheduling/ledgers/head_admission.py`), `CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS` (`scheduling/clearance_lease.py`), `POP_LIVENESS_FROZEN_QUEUE_SECONDS` (`process_manager.py`), the establish-rate window, the claim cap | their own gates | Each runs a fixed clock that does not read the horde deadline. |
| Large-model pop limiters | `JobPopper._is_large_model` (the EXTRA_LARGE size tier) | the switch throttle, the re-entry cooldown, the VRAM-pressure narrowing | These stages key on the size tier, which is a different predicate from the measured rung. |

## Fences

- **Run grant at the first step** (`_grant_dominant_runs_at_first_step`). The claim narrows intake, and a claim taken
  at the dispatch commit would narrow it for the whole load. The run paragraph of the four-layers section in
  `vram_arbiter.md` explains it.
- **The `dominant_run` seat** (`_select_dominant_run_job`). On one card a held head ends the dispatch pass, so a run
  job queued behind it would keep the run and the head waiting on each other. The seat does not check the head's
  age itself, because the run end carries the bound.
- **The run end on head age** (`_restore_siblings_after_whole_card`). The horde deadline outranks run continuity.
  Every run reader goes through the ledger, so the one end withdraws the copy spare, the seat, the promotion and the
  claim together.
- **`residency_bypass_models` reads `teardown_held()`.** A teardown bars the head from loading until it drains, and
  only the resident model's own queued jobs drain it. A run does not tear the card down, so its jobs reach the
  card through the `dominant_run` seat instead. The residency-claims-intake section of `resource_governance.md`
  explains it.
- **The affinity budget and the anti-starvation override** (`dispatch_affinity.py`). An unbounded bypass could
  starve the head behind a steady stream of resident work, and the budget alone does not bound the head's age
  since pop.
- **Two establishments per 240 s** (`_ESTABLISH_WINDOW_LIMIT`). Two lets a retry follow a failed first
  establishment at once, and a third inside the window is the signature of a pricing oscillation. The window stays
  under the 900 s pop-gate structural-wedge backstop, so a deferred head never accrues toward it.
- **The grace allowance** (`_GRACE_BUDGET_SECONDS`). Back-to-back churn would otherwise re-arm the supervisor's
  excuse faster than it expires. The spend never refuses the card, because a card held off for want of an excuse
  sits idle with the head unserved.
- **`close_restore_window`.** A restore that regrows to the ceiling while no lane restarts withdraws its window, so
  the supervisor is not told to ignore a wedge for a teardown that did not happen.
- **The minimum hold** (`_MIN_HOLD_SECONDS`). Releasing a teardown seconds after its grant buys one lighter job for
  two pool rebuilds. The hold is shorter than the establish grace, so it never keeps a residency alive past the
  supervisor's window.
- **A governor deferral continues the preload pass** (`pass_control_for`). The deferred head does not run an
  escalation, so a later load does not take anything from it. Other single-card stops still end the pass, since
  the head's escalations outrank later loads (`stop_is_card_scoped`, `_HEAD_ESCALATION_STOPS`).
- **Head protection released for a governor-deferred head** (`_displaced_head_outstanding_mb`). Reserving the
  deferred head's demand against smaller ready work would idle the card against a claim nobody is making.
- **Rung-2 entry is head-only** (`residency_demanded`). Only the head may claim the card. The dispatch path's
  exemption for line-skipped jobs was written for resident fills that are already size-appropriate.

## Friction

- **Rung-2 entry and the line-skip seats.** `_resident_whole_card_head_ready` runs only when no line skip chose the
  job, and `_select_idle_thread_diversity_job` and the resident-bypass loop check degraded retries only, never the
  candidate's rung. A non-head rung-2 job preloaded co-resident can therefore be dispatched by a `diversity` skip
  without a teardown.
- **The rung-2 bypass and the deadline.** `_head_aged_past_anti_starvation` clears the affinity budget and leaves
  `residency_bypass_models` and pin-wait untouched, so a rung-2 resident model's jobs can pass a head past its
  deadline. `whole_card_residency_max_hold_seconds` runs from the establishment and the ttl from the head's pop,
  and neither reads the other. The setting's docstring sizes the cap so an earlier-queued job drains inside its
  deadline, which holds for rung 1 only through the run end.
- **The run copy and the reclaim ladder.** `resources/reclaim_ladder.py` `build_reclaim_ladder` emits
  `UNLOAD_IDLE_MODEL` from `InferenceScheduler.build_reclaim_ladder_candidates` without asking
  `copy_belongs_to_dominant_run`. The sweep, retention and preload do ask, so the ladder alone can evict a run's copy.
- **A run's next job may be one that cannot dispatch.** `copy_belongs_to_dominant_run` counts any queued job for the
  model through `queued_job_naming`. A run keeps its copy for a job that cannot run until the run end's head-age
  condition fires.
- **Size tier against the measured rung.** `JobPopper._is_large_model` drives the switch throttle, the re-entry
  cooldown and the VRAM-pressure narrowing, while `card_dominance` decides runs and teardowns. The narrowing runs
  before `_apply_whole_card_pop_claim`, so it can empty a claim's offer, and the claim then withholds the pop.
- **Re-entry pacing.** The claim cap, `governor_hold` and the large-model re-entry cooldown each pace a model's
  return to the card without a shared view of demand. A burst that outlasts the cap re-establishes, and each
  re-establishment spends the limiter's window.

## Not determined

- Whether a rung-2 teardown's budget exemption should yield to the anti-starvation bound.
- Whether the claim cap should derive from the job ttl, with the setting kept as an override.
- Whether the pop-time limiters should key on the card's rung instead of the size tier.
- Whether the VRAM-pressure narrowing has emptied a claim's offer in a live session. The pop-claim gate stamp is
  not in the parent log text.
- Whether a later preload, admitted while a governor deferral continues the pass, can take the slot a deferred
  rung-2 head needs once the window lets it through.
- Why a pop can stay held at `no_inference_process` through a sole residency.

## Where to read next

- [Resource governance](../explanation/resource_governance.md): whole-card residency state, bounding residency
  churn and the residency claims intake.
- [VRAM arbiter](../explanation/vram_arbiter.md): the four layers (rungs and runs) and cross-job retention.
- [Admission pipeline](../explanation/admission_pipeline.md): dispatch selection and continuing the preload pass.
- [Resilience and recovery](../explanation/resilience_and_recovery.md): the SOS layer and what counts as progress.
- [Log findings](log_findings.md): `whole_card_residency_churn`, `whole_card_pop_claim_episodes`,
  `whole_card_pop_claim_monopoly`, `whole_card_nonhead_residency_starvation`, `whole_card_convergence_wedge`,
  `head_dispatch_stall` and `slow_generation_drop_spiral`.
- [Clearance and child loads responsibility map](responsibility-map-clearance-and-child-loads.md): what admits and
  clears a job once it holds the card.
