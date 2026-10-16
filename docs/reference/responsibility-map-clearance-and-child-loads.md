# Clearance and child loads responsibility map

This page lists the one owner of each decision between a staged job's dispatch and the end of its sample: the
clearance grant and the figures it carries, the text encoder's room, and what the child loads against those
figures in hordelib and ComfyUI. The reasoning behind each rule is in [VRAM arbiter](../explanation/vram_arbiter.md),
[Admission pipeline](../explanation/admission_pipeline.md) and
[Resource governance](../explanation/resource_governance.md).

## Decisions and owners

Worker paths are under `horde_worker_regen/process_management/`, and `InferenceScheduler` is in
`scheduling/inference_scheduler.py`. hordelib paths are under `hordelib/execution/` in the hordelib repository, and
ComfyUI is the copy hordelib vendors.

| Decision | Owner | Readers | Rule |
| --- | --- | --- | --- |
| Whether a staged child may stage or sample | `ClearanceController.step`, `decide_clearances`, `_clear`, `_clear_stage` (`scheduling/clearance_lease.py`); verdicts `decide_clearance_admit` and `decide_clearance_stage_admit` (`scheduling/admission/clearance.py`); `VramArbiter.evaluate` | `InferenceScheduler.clearance_admit_process`, `clearance_stage_admit_process` | An idle child with a stage permit is granted the stage and sample permits in one clear. |
| When the parent may grant | the staged waiter (`ClearanceWaiter`), built for a process `is_staged_short_of_sampling` (`lifecycle/process_info.py`) | `ClearanceController.step` | A staged child is a waiter from dispatch, so its grant can reach it before the child's job start. |
| The free figure a grant carries | `InferenceScheduler._measured_device_free_mb`, netted per step by `ClearanceController._grant_charge_mb` | `ClearanceLeaseProxy`, hordelib `sampling_lease.py` | Each child cleared in one step gets the device reading less the charges of the children cleared before it in that step. |
| The reserve a grant carries | `pricing.sampling_reserve_mb`, which mirrors hordelib `vram_planning.compute_inference_reserve_mb` through `resource_budget.effective_inference_reserve_mb` | the staged waiter, `staged_sampling_peak_mb` | The reserve is at least ComfyUI's inference reserve whenever the card total is known. |
| Figures of a grant not yet consumed | `ClearanceController._refresh_unconsumed_grant_figures`, through `ClearanceLeaseProxy._parent_refresh_grant_figure` and `_parent_refresh_stage_figure` | the child's acquire | Each step rewrites the device free and the reserve of every unconsumed grant from that step's reading and the staged waiter's price. |
| Figures at the child's job start | `ClearanceLeaseProxy.begin_job`, called from `workers/inference_process.py` | the child's acquire | The four shared figures and both consumed flags reset, so a grant issued before the job start carries its figures from the next step. |
| A cleared grant that retires unsampled | `ClearanceLeaseProxy._parent_reclaim_unconsumed_permit`, from `ClearanceController._retire_grant` | the next job's acquire | The parent takes the permit back, so the next job does not acquire a stale grant. |
| The clearance price of a staged job | `staged_sampling_peak_mb`, `staged_materialization_delta_mb` (`scheduling/admission/clearance.py`) | `InferenceScheduler._upgrade_dispatch_reservation_to_full` | The window is priced at the diffusion weights plus the reserve, without the text encoder or the decode tail. |
| The encoder's room before clearance | `pricing.dispatch_staging_charge_mb` and `overlap_staging_charge_mb` (both through `staging_charge_mb`); sibling relief in `scheduling/admission/clearance.py` | dispatch admission, the overlap gate | The dispatch booking carries the encode until clearance re-books the entry, and a waiting sibling is charged only while its `clearance_wait_entered` is false. |
| The encoder leaves the card before the wait | hordelib `_release_text_encoders_before_wait`, `_text_encoder_still_needed` (`sampling_lease.py`) | `staged_sampling_peak_mb` assumes it | The encoder returns to its CPU origins whenever a sampling lease is set and no later node of the prompt needs it. |
| CPU-origin retention | hordelib `cpu_weight_retention.py` (`stash_cpu_origins`, `restore_cpu_origins`, `release_copied_back_weights`) | `comfy_patches.py`, `sampling_lease.py` | A released component restores from host RAM for the next encode. |
| Weight prefetch | worker `InferenceScheduler.warm_idle_lanes` (child `warm_model`); hordelib `_prefetch_sampling_model_weights` at the hook | the child's load | The worker warms idle lanes before dispatch, and the hook touches the sampling model's pages before the upload. |
| What the lease brackets | hordelib `install_sampling_lease_hook` on `CFGGuider.sample`, and the stage lease in `sampling_lease.py` | every sampling path | The lease brackets the stage upload and the denoise, and the encoder load, the encode and the VAE decode run outside it. |
| The reserve the child keeps free | hordelib `_bound_by_sampling_reserve` | ComfyUI `load_models_gpu`, which adds `extra_reserved_memory()` on top | The child raises `minimum_memory_required` to the granted reserve. |
| The free view the child sizes a load against | hordelib `free_memory_view_clamped`, rebased on each grant by `rebase_free_memory_view_clamp` (`comfy_patches.py`) | ComfyUI partial-load sizing | The rebase takes the allocator reservation at that moment as the baseline and the grant figure as the truth, with ComfyUI's own view as the ceiling. |
| A partial load at sampling becomes full | hordelib `_load_models_gpu_hijack`, `_force_full_load_would_overflow_vram`; `_partial_load_allowed` for the staging call | ComfyUI `load_models_gpu` | The load is forced full unless the overflow check, charging the held reserve, finds it would overflow the card. |
| The partial-load seat | `VramArbiter._partial_seat_admission`, `pricing.partial_seat_mb`; booking capped at the available room in `scheduling/admission/clearance.py` | the clearance verdict | The seat is an admission threshold, and the child's load is bounded by the grant figure less the reserve. |
| Decode working memory | hordelib `_free_memory_capped` and the support-model clamp (`comfy.small_model_working_clamp` line) in `comfy_patches.py`; ComfyUI's tiled decode fallback | the child's decode | The VAE's working estimate is zeroed and `free_memory` capped near the VAE weights, so the decode does not unload the job's weights and falls back to tiled decoding on a true OOM. |
| How long a child waits for a grant | `CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS` (`scheduling/clearance_lease.py`) | hordelib acquire, `ClearanceController._reconcile_grants` | Past the timeout the child samples unpriced and the parent flags the window. |

## Fences

- **`begin_job` unsets the figures.** A permit released after the previous job timed out stays available to the
  next acquire, and the figure written with it describes the card before the previous job sampled. The permit
  reclaim on a retired unsampled grant guards the same case.
- **`_refresh_unconsumed_grant_figures`.** The child consumes a grant seconds after it, once its encoder is off the
  card and siblings have unloaded, so the figure frozen at the grant tick understates the card.
- **Encoder release before the wait** (`_release_text_encoders_before_wait`). The diffusion load then has no encoder
  to partially unload and copy back, and the encoder's room is free while the job waits. On a card that cannot
  hold a Flux transformer, its encoder, the foreign floor and every lane context at once, keeping the encoder
  resident is not an option.
- **The staged price excludes the encoder** (`staged_sampling_peak_mb`). It follows from the release above and
  holds when the clear comes after the child's encode.
- **The lease sits on `CFGGuider.sample`.** Every ComfyUI sampling path reaches that call, and a hook on either
  `comfy.sample` entry point would leave one path sampling outside the lease.
- **Two permits in one clear** (`_clear`). A single clear lets an idle child stage and sample without a second wait.
- **Same-step netting** (`ClearanceController.step`). Children cleared in one step would otherwise each read the
  whole free figure and load into the same room.
- **The decode clamp** (`_free_memory_capped`). Freeing ComfyUI's full decode estimate evicts the co-resident
  diffusion weights, and ComfyUI degrades to tiled decoding on a true OOM anyway.

## Friction

- **The encoder booking at clearance.** `_upgrade_dispatch_reservation_to_full` re-books the dispatch entry at
  `staged_materialization_delta_mb`, which excludes the encoder. When the clear precedes the child's encode, the
  encoder then loads without a booking. `clearance_wait_entered` reports that the encode is done, and only the
  sibling relief reads it.
- **The sample-grant rebase double-counts the staged upload.** Under two-phase clearance the sample permit is
  released with the stage permit, so the figure `_refresh_unconsumed_grant_figures` last wrote can predate the
  child's staged upload. `rebase_free_memory_view_clamp` takes the current reservation as the baseline, so upload
  after the reading appears in neither term, and ComfyUI sizes the remaining load from the inflated view.
- **The refresh undoes same-step netting.** `ClearanceController.step` hands each child cleared in one step the
  figure less earlier children's charges, and the next step's `_refresh_unconsumed_grant_figures` writes the raw
  device reading to every unconsumed grant. A child that consumes more than one control loop after its grant reads
  the un-netted figure.
- **Who funds the decode.** The `staged_sampling_peak_mb` docstring and the decision-pipeline section of
  `vram_arbiter.md` say ComfyUI funds the decode tail by unloading the job's weights. hordelib's `_free_memory_capped`
  exists so the decode does not unload them, and relies on free room and the tiled fallback.
- **The encoder's room in a rung-1 run.** `pricing.copy_belongs_to_dominant_run` treats the run's diffusion copy as
  kept, and the clearance prices the reserve alone because the weights are resident. The child's encoder reload
  for each job pages part of that copy out of the card through ComfyUI's `free_memory`, and no owner prices the
  re-read.
- **Margins stack on the reserve.** `pricing.sampling_reserve_mb` already includes ComfyUI's inference reserve,
  `_bound_by_sampling_reserve` raises the minimum to it, and ComfyUI adds `extra_reserved_memory()` again. The
  child keeps a known constant more free than the price assumes.

## Not determined

- Whether the encode should be leased, or the clear withheld until `clearance_wait_entered`. No ruling records it.
- Which owner should make the sample-grant rebase exact under two-phase: the controller, by delaying or annotating
  the figure, or the clamp in hordelib.
- Whether the decode tail should be priced, or the documentation corrected to name free room and the tiled
  fallback as its funding.
- Whether the same-step netting loss has occurred live. It needs two same-step clears whose children both appear
  in lane logs.
- Whether a child that acquires within one control loop of its job start reads the unset figures before the
  refresh restores them.
- Why the parent figure at a sample grant can reflect only part of a staged upload: NVML lag under WDDM or a parent
  loop stall.

## Where to read next

- [VRAM arbiter](../explanation/vram_arbiter.md): the decision pipeline, admission authority and the partial-load
  seat.
- [Admission pipeline](../explanation/admission_pipeline.md): where each clearance decision lives and what it reads.
- [Resource governance](../explanation/resource_governance.md): dispatch bookings and how they decay.
- [Logs](logs.md): VRAM admission diagnostics.
- [Log findings](log_findings.md): `unpriced_sampling_windows`, `stage_outlived_sample_grant`, `stage_refusals`,
  `oom` and `swallowed_oom`.
- [Runs and residency responsibility map](responsibility-map-runs-and-residency.md): what holds the card before a
  job reaches clearance.
