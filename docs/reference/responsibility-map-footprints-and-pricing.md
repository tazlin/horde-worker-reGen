# Footprint and pricing responsibility map

This page lists the one owner of each decision about what a job or a model costs a card, from the learned reading
a child reports to the price dispatch, clearance and the dominance verdict charge. The reasoning behind each rule is
in [VRAM arbiter](../explanation/vram_arbiter.md) and [Resource governance](../explanation/resource_governance.md).

Paths are relative to `horde_worker_regen/process_management/`. `pricing` is `scheduling/admission/pricing.py`, the
store is `LearnedFootprintStore` in `resources/vram_footprints.py`, and the scheduler is `InferenceScheduler` in
`scheduling/inference_scheduler.py`.

## Decisions and owners

| Decision | Owner | Readers | Rule in one sentence |
| --- | --- | --- | --- |
| Which key a monolithic sampling reading is filed under | `ipc/message_dispatcher.py` `MessageDispatcher._observe_footprint_peak`, `_JobSamplingPeak`, `_commit_job_sampling_peak` | the store's `SAMPLE` and `SAMPLE_ACTIVATION` keys | The key and the baseline come from `job.model`, a reading is dropped while the lane holds another model or is `INFERENCE_PRIMED`, and one peak per job is committed under the key of the job's first attributed reading. |
| Which key a disaggregated or at-rest reading is filed under | scheduler `observe_disaggregated_sampling_peak`; store `observe_job_footprint` | the same store | Both file under the job's model, the rule the monolithic feeder follows. |
| Whether a sampling reading is plausible | `resources/vram_footprints.py` `plausible_sampling_peak_mb`, checked per reading in the feeder and again at commit | every feeder | A reading above the key's bound is dropped, and a `SAMPLE_ACTIVATION` key is bounded by the band's heaviest sampler-only seed. |
| Which stored readings survive a load | `FOOTPRINT_STORE_SCHEMA_VERSION` (5), `_MISKEYED_SAMPLING_SCHEMA_VERSIONS`, `LearnedFootprintStore.sanitize_sampling_observations` (called by the scheduler with `max_batch`) | every reader of the store | A file of schemas 2 to 4 loads without its `SAMPLE` and `SAMPLE_ACTIVATION` entries, and the sanitiser clamps `SAMPLE`, `SAMPLE_ISOLATED` and `SAMPLE_ACTIVATION` windows to their bounds. |
| A key's price | `LearnedFootprintStore.estimate_mb`, `measured_estimate_mb`, `measured_job_estimate_net_of_context_mb` | pricing, the scheduler | `estimate_mb` is the seed raised by the raise-only watermark, and the measured figure is the second-highest of the last 20 readings x 1.10 plus context once a key has 5, read for the job class else the pooled key. |
| The figure the dominance verdict prices a model at | `pricing.model_sampling_peak` | `pricing.decide_card_dominance`, scheduler `card_dominance` | The pooled `SAMPLE` measured figure at the most-observed band, else the seed of a plain batch-1 job, and never the watermark. |
| The verdict's rung and its hysteresis | `pricing.dominance_rung`, `decide_card_dominance`, `DOMINANCE_HYSTERESIS_MB` | the whole-card machinery, `forecast_streaming` | A model is rung 1 or rung 2 per card from its peak, the offered set and the foreign floor, and a held rung moves only when the slack crosses a 256 MB band. |
| What a dispatch or preload prices a job at | `pricing.candidate_delta_mb` through `pricing.learned_sampling_peak_mb`; the scheduler's copy `_learned_sampling_peak_mb` | dispatch, preload | The seed raised by the watermark, lowered by the class-else-pooled measured figure plus the LoRA delta, floored at the weights. |
| What a staged job is priced at clearance | `scheduling/admission/clearance.py` `staged_sampling_peak_mb` with `pricing.sampling_reserve_mb` | `ClearanceController` | Diffusion weights plus the learned activation reserve, raised to ComfyUI's inference reserve, else `candidate_delta_mb` for a disaggregation-class job. |
| What a residency head needs | scheduler `_residency_head_need_mb` through `_sampling_peak_mb` | the whole-card residency | The head's whole-job `SAMPLE` figure through the scheduler's learned-peak copy. |
| Whether a lane keeps its weights | scheduler `_retention_fit` through `VramBudget.check_job` | `_should_keep_model_resident` | A static whole-job seed less the current foreign floor, with no learned figure. |
| A rung-2 model's seat | `pricing.partial_seat_mb` | clearance, the partial-load grant | The model's weight share plus the class-else-pooled activation. |
| Whether a model wants the whole card | `pricing.forecast_streaming` (`wants_whole_card`) | the residency forecast | True for an operator pin or a rung-2 verdict. |
| A checkpoint's at-rest figure | `pricing.learned_resident_footprint_mb`, `measured_resident_footprint_mb`; scheduler copies `_learned_resident_footprint_mb`, `_measured_resident_footprint_mb` | reclaim pricing, retention, seats | The learned `RESIDENT` figure once the model has produced a result. |
| The foreign floor | `resources/foreign_vram_floor.py` `ForeignVramFloorTracker`, `floor_with_known_tenant_mb`; scheduler `_sustained_foreign_floor_mb`, `current_foreign_floor_mb` | arbiter ceiling, `achievable_ceiling_mb`, the verdict and its recompute key, retention, `unpausable_tenancy_mb`, the whole-card claim, safety restore headroom and safety card choice | A 120 s trailing minimum of positive `total - free - worker footprint` readings, withheld until the window fills and raised to a managed tenant's footprint. |
| Unattributed VRAM for serviceability | scheduler `serviceability_baseline_mb` from `HordeWorkerProcessManager.latest_baseline_estimate_mb`; the arbiter's `baseline_mb` | `model_serviceability_verdicts`, the arbiter | The reconciler's baseline estimate, a second measure of memory the worker does not hold. |
| A staged job's dispatch booking | `pricing.staging_charge_mb`, `dispatch_staging_charge_mb` | the arbiter's outstanding units | Under the lease a dispatch books the family's support weights plus 348 MB, at least 2,048 MB. |
| Who nets a staged sibling's booking | `clearance.sibling_staging_relief_mb` | clearance | A staged sibling not yet cleared is netted for what its encode has already allocated, and a sibling holding a grant, a stage grant included, keeps its whole charge. |

## Fences

- **The per-job fold** (`_JobSamplingPeak`). A child sends several memory reports per job, so filing each one
  made the 20-entry window span several jobs and counted readings outside a job's sample beside it.
- **The job's model as the key, and the mid-swap drop** (`_observe_footprint_peak`). A lane still swapping reports
  the previous model's allocator high-water. Because the first attributed reading fixes the job's key, a reading
  taken in that gap would file a large model's peak under a small model's key.
- **The primed-lane skip** (`_observe_footprint_peak`). A primed lane holds the encode working set alone, and a
  reading there would lower the job's price.
- **The per-key bound** (`plausible_sampling_peak_mb`). Stores have held card-sized figures under small models'
  keys, which a child reports when it overflows the card or caches a checkpoint.
- **Second-highest x 1.10 at five entries** (`measured_estimate_mb`). One outlier cannot price a key. Two can, which
  is why the feeder's key rule matters more than the estimator.
- **Activation from whole-weight jobs only.** A partial load's peak less the load's resident share reads streamed
  weights as activation, and a seat priced from it pushes every later load deeper into partial mode.
- **The schema 5 load rule** (`_MISKEYED_SAMPLING_SCHEMA_VERSIONS`). Older feeders filed under the lane's loaded
  model, and a clamp cannot remove a misfiled figure that sits under a card-capped bound. Dropping the keys costs
  every model five jobs of seed pricing, while `SAMPLE_ISOLATED` and the at-rest stages load as written.
- **No watermark in the verdict or the activation reserve** (`model_sampling_peak`, `sampling_reserve_mb`). The
  watermark carries card-sized child peaks and would price a model off the card on one reading.
- **The 256 MB verdict band** (`DOMINANCE_HYSTERESIS_MB`). A peak drifting about a threshold would otherwise flip a
  model in and out of the teardown residency.
- **The 2,048 MB staging floor** (`staging_charge_mb`). SDXL's dual-CLIP encode measures near that figure, and a
  family with lighter support weights still allocates the encode working set. A cleared sibling keeps its whole
  charge in `sibling_staging_relief_mb` because it is about to load the weights.
- **The foreign floor as a trailing minimum** (`ForeignVramFloorTracker`). A spike never lowers the ceiling, and a
  reading at or below zero is the worker's own over-charged footprint, so it does not lower the floor.

## Friction

- **The verdict prices two populations.** `model_sampling_peak` reads the pooled `SAMPLE` key, which holds batched
  and hires-fix jobs, while the seed branch prices a plain batch-1 job. One batched job can move every model of a
  baseline to a higher rung, though `pricing.candidate_delta_mb` already reads the job-class key.
- **Three prices for one job.** `candidate_delta_mb` prices the whole-job `SAMPLE` figure, `staged_sampling_peak_mb`
  prices weights plus the activation reserve, and `_residency_head_need_mb` reads `SAMPLE` through the scheduler's
  copy. The residency head need and clearance both answer whether the head can sample now, so one can hold a head
  the other would grant.
- **The scheduler keeps private copies of `pricing.py`'s learned-peak helpers.** `_learned_sampling_peak_mb`,
  `_learned_resident_footprint_mb` and `_measured_resident_footprint_mb` re-implement
  `pricing.learned_sampling_peak_mb`, `learned_resident_footprint_mb` and `measured_resident_footprint_mb`. The
  scheduler copy gates the measured at-rest figure on `JobTracker.has_model_produced_result`, the pricing copy on
  `SchedulingSnapshot.models_with_results`. A change to one rule has to land in both.
- **Two measures of non-worker VRAM.** The foreign floor feeds the arbiter ceiling, the verdict and retention, and
  `serviceability_baseline_mb` feeds serviceability and the arbiter's `baseline_mb`. They are computed
  separately, and whether they agree on a card with a foreign tenant is listed below as undetermined.
- **The dispatch price still reads the watermark.** `learned_sampling_peak_mb` returns `estimate_mb`, the seed raised
  by the watermark, for a key under five entries and whenever the measured figure is not lower.

## Not determined

- Whether `has_model_produced_result` and `snapshot.models_with_results` are the same set.
- Whether the foreign floor and the reconciler baseline agree on a card with a multi-gigabyte foreign tenant.
- Whether the verdict should price the batch-1 job class, so a model that is rung 1 only for batched jobs does not
  take the run levers.
- Whether a stage-granted waiter that is itself held at sample clearance should keep its whole booking against a
  sibling's stage.

## Where to read next

- [VRAM arbiter](../explanation/vram_arbiter.md): the learned-footprint feeder and schema passages and the
  dominance verdict under "The four layers", then
  [The partial-load seat](../explanation/vram_arbiter.md#the-partial-load-seat) and
  [Cross-job retention](../explanation/vram_arbiter.md#cross-job-retention).
- [Resource governance](../explanation/resource_governance.md#whole-card-residency-state): what the verdict's
  rungs open.
- [Logs](logs.md#vram-admission-diagnostics) for the hold lines that carry each price, and the
  `whole_card_residency_churn`, `residency_reconciliation_holds` and `vram_pressure` findings in
  [Log findings](log_findings.md).
