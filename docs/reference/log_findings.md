# Log findings

Every finding `horde-log diagnose` can emit, and every one the dashboard's Insights tab reads off the
running worker, with the id it prints, the severity it carries, what makes it fire, and what to do about it. The id is the stable handle: it is what the JSON output keys on, what
the dashboard's Diagnostics tab shows, and what a `see also:` line points at.

Findings run over every session `horde-log sessions` lists. That includes each `horde-benchmark` run
read from `bridge_harness.log`, which is tagged `[benchmark run: bridge_harness.log]` in the text output
and carries `"log_kind": "harness"` in the JSON. A finding under that tag describes the benchmark, not the
worker, even when the two ran over the same hours. Worker sessions and benchmark runs are numbered from #0
within their own kind, so a session is identified by its `log_kind` and index together. In a support
bundle, each `diagnose.json` entry carries `session_index` (that per-kind number), `log_kind`, and
`session_label`, the heading the text output prints for the session (`#0`, or
`#0  [benchmark run: bridge_harness.log]`).

The ids are declared as `FindingKind` members in `horde_worker_regen/analysis/finding_kinds.py`, one
per entry in the `FINDING_SPECS` table. A contract test (`tests/analysis/test_log_findings_doc.py`)
walks that table in both directions, so a new kind cannot ship without an entry here and a row cannot
outlive the kind it describes. The "Fires when" and "Remedy" columns are written for a reader, not
lifted from the detector's own remediation text, so they are maintained here rather than generated.

Severity is the sort order of the report, not a queue: **critical** findings come first, then
**warning**, then **suggestion**, then **info**. The reader sees an action word rather than the level
name: `Fix now` (critical: the worker is losing work or will be paused), `Check` (warning: something is
wrong or wasteful), `Try` (suggestion: nothing is wrong, a change would likely earn more) and `Note`
(info: context, no action). The JSON output carries both, as `severity` and `badge`. Some detectors pick
their severity from the evidence (a one-off versus a sustained pattern); those are marked *varies*. Findings are for problems, so a healthy subsystem emits
nothing at all: the absence of `parent_loop_stall` means the parent loop was fine.

Where acting on a remedy needs background the finding cannot carry inline, its spec names a deep-dive
page (`FindingSpec.reference_page`), printed under the remedy as a `see:` line and carried in the JSON
output as `reference_page`. A test asserts every such path is a page that exists, so the column here
stays a summary and the link stays live.

For how the detectors, the log lines they read, and the dashboard stay in step, see
[Log diagnostics contract](../explanation/log_diagnostics_contract.md). For the commands, see
[CLI → `horde-log`](cli.md#horde-log).

## Startup and process lifecycle

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `crash_on_start_loop` | critical | Image processes crash before they are ready, repeatedly. The error is lifted from the process's own start-up log. | Fix the error it names. A git clone failure points at the shared ComfyUI environment directory, not at torch: delete that directory and let one process rebuild it. |
| `preload_kills_child_loop` | critical | One model crashes every process that loads it: repeated process deaths naming the same model. | Remove the model from your list and download it again before adding it back. |
| `empty_model_pop_cascade` | varies | The horde sent job offers with no model name. On an old worker the blank name was loaded, crashed processes and got blocked; a current worker hands the offer back. | Update the worker. If it keeps happening, report it to the horde; your models are not at fault. |
| `doomed_pool_no_giveup` | critical | The worker kept rebuilding processes that could not recover instead of stopping itself. | Fix the crash the crash-on-start finding names, then restart the worker. |
| `gave_up_clean` | info | The worker stopped itself after its processes could not be restored. The intended outcome, recorded so it is not mistaken for a crash. | Nothing here; fix the crash the crash-on-start finding names. |
| `stuck_inference_step` | warning | An image process repeated one sampling step without finishing and was restarted. | Check the process log before the restart for a LoRA shape error and stop pairing that LoRA with that model. |
| `post_processing_vram_stall` | varies | Post-processing ran out of room on the card: it stalled, or it shared the card with sampling and nothing stalled (info). | Turn on the VRAM budget; as a stopgap lower `max_threads` or `queue_size`, or turn off post-processing on this card. |
| `orphan_wedge` | warning | Jobs were dropped because the process running each one had disappeared. | A symptom of process loss; fix the crash, hang or memory finding above it first. |
| `inference_slot_retired` | varies | A card served with fewer image processes than planned because a start could not get room on it: the worker waited without progress, or the card can never free what a start needs beside what else uses it (a structural shortfall, stated with the card's room and the requirement). One finding per card, with how long it ran short, to its restore or the session end. A warning while a retirement is never restored, a note when every one was. | For a structural shortfall the card cannot hold its planned processes: lower `max_threads`, or move what else uses the card (`text_gpu_device_index` when it is the text backend the worker launches). Otherwise close the program holding the card's memory; the worker plans the process again once the card has held room for a few minutes. |
| `safety_start_escalated_to_cpu` | warning | A safety start could not get room on its card, after a wait without progress or at once for a structural shortfall, so the worker started the safety check on the CPU. It stays there until its card has held room for a while; the evidence says whether it went back before the session ended. | Close the program holding the card's memory, or for a structural shortfall lower `max_threads` or move the text backend with `text_gpu_device_index`. Checks on the CPU are slower, so each job takes longer meanwhile. |
| `inference_lane_replaced` | warning | The worker replaced an image process because of its own failed jobs. It replaces one at once after a job failed with a CUDA runtime error, or after a run of failed jobs across more than one model. The evidence says, for each CUDA error, whether Windows ran out of memory to commit in the minute before it. | A CUDA error right after the host ran out of memory comes from the host condition, and the worker recovered on its own. Any other CUDA error, or a run of failures across models, is a worker or driver defect. Report it with `horde-log bundle`. |
| `inference_slot_quarantined` | critical | The worker stopped restarting an image process, because it was replaced too often within five minutes or failed to start several times in a row. The evidence lists the replacements of that process in the window, which name what drove it. | The worker runs with one fewer image process until it rebuilds its processes or restarts. Fix the cause the replacement reasons name, and report it with `horde-log bundle` when they are CUDA errors or runs of failed jobs. |
| `lane_fault_rule_misfire` | warning | A process replacement may not have helped. Either a process replaced for mostly out-of-memory faults failed again within ten minutes, which points to an over-committed card, or a process was taken out of service after a CUDA error that came within a minute of Windows running out of memory to commit. It exists so the worker maintainers can judge whether the replacement rules need changing. | Send `horde-log bundle` output to the worker maintainers. |

## Memory and residency

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `oom` | critical | Jobs failed with out-of-memory errors. The allocator's message names the model that faulted and the other processes holding memory at that moment. When at least 80% of three or more faults came from one process while another process finished jobs over the same span, the finding names that process as broken instead of the card as full. A `torch.AcceleratorError: CUDA error: out of memory` is named as a failed CUDA call, which can leave the process unusable. | For a broken process, a current worker replaces it by itself. Restart an older one and report it with `horde-log bundle`. Otherwise the evidence favours too much work on the card. Lower `max_threads` or `queue_size`, or turn on the VRAM budget, and run fewer models at once when several share a nearly full card. |
| `swallowed_oom` | warning | Jobs failed with a bare "no images were produced" and no reason. ComfyUI can report an out-of-memory error this way. | Check free VRAM around the failures. If the card was full, treat them as out-of-memory faults. |
| `file_descriptor_exhaustion` | critical | A process ran out of file handles. It then fails every job while still sending heartbeats. | Raise the open-file limit as a stopgap and report it; the leak is in the worker. Lowering memory settings will not help. |
| `pagefile_exhaustion` | critical | Windows refused memory to a worker process with error 1455, "the paging file is too small". The parent sees only a process stuck starting. The error is lifted from the process's start-up or loop log. | Let Windows manage the paging file size, or set a larger fixed size, then restart the worker. As a stopgap run fewer processes by lowering `max_threads` or `queue_size`. |
| `scheduler_starvation_wedge` | varies | The VRAM budget held the next job back on a card with plenty free. Critical if the wait ran to a rebuild and dropped jobs. | Relax the VRAM budget, or turn off `unload_models_from_vram_often` and `high_performance_mode` so processes stop cycling. |
| `unpriced_sampling_windows` | varies | A process began sampling without a recorded clearance grant. Each such line is paired with the last `Starting inference for job` line for the same process, and the wait between them is banded: under 3 s, 3 s to under 30 s, and 30 s or more. The evidence gives the band counts, the median wait and the first line of the costliest band present. Warning when three or more waited 30 s or more, info otherwise. | A short wait is a grant that was issued, with its `cleared process N` line, but missing from the parent's ledger. It costs nothing. A long wait is the process running out its 60 s lease-acquire timeout with the card idle, so the clearance pricing or reclaim held it. Read the `Clearance held for process N (model): defer; ...` line for that process. |
| `unsatisfiable_head_starvation` | varies | The same model kept reaching the front of the queue and being refused a load while the card sat idle, and nothing cleared it. | Check that the model fits the card with room to run. If it does, relax the budget or reduce swapping. If not, remove it. |
| `residency_reconciliation_holds` | varies | The worker paused to unload an idle neighbour before running the next job, repeatedly. | Each pause is a swap paid for holding more models than the card fits at peak. Check `unload_models_from_vram_often` and the eviction thresholds. |
| `whole_card_convergence_wedge` | critical | A model that needs the whole card could not get it: an idle process or a service process never left the card. | Capture the log around the stall and report it. As a stopgap lower `queue_size`, or do not serve a whole-card model beside a deep queue of others. |
| `whole_card_nonhead_residency_starvation` | varies | The card was reserved for a job that was not next in line, so the next job had nowhere to run. | Capture the log around the stall and report it. The worker should only reserve the card for the next job. |
| `whole_card_residency_churn` | warning | The whole card was reserved and released over and over in one session. | Thrash: each round trip costs a model reload and a safety restart. Usually a model that does not need the whole card is being given it. |
| `whole_card_pop_claim_episodes` | info | The worker asked the horde for one model only while it held the card. Recorded so the claim windows are visible. | Nothing if that is the heavy work this worker is for. Otherwise trim the model list or lower `whole_card_residency_max_hold_seconds`. |
| `whole_card_pop_claim_monopoly` | warning | Claims kept running to the hold cap while other models' jobs sat waiting. | Decide whether this worker should specialise. If it should serve a mix, lower `whole_card_residency_max_hold_seconds` or drop the whole-card model. |
| `host_ram_starvation` | warning | System RAM holds new jobs, repeatedly defers model loads, or forces process reclaim. Reports thresholds, holds per hour, drain/recycle counts and queued-work concurrency. | Update the worker, check other programs' RAM usage, and bundle continued holds or repeated restarts. |
| `model_churn` | varies | Models were loaded and unloaded far more often than jobs ran, or loads were cleared before they ran. | Turn off `unload_models_from_vram_often`. Serve fewer models, or use the model pool, and raise `queue_size` so same-model jobs run back to back. |

## Dispatch and throughput

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `multi_card_dispatch_serialization` | varies | A multi-card worker ran well under half its cards while jobs waited with their model already loaded on an idle process. | A scheduling problem, not a per-card setting. Run `horde-log jobs` for the per-job split and the busy-cards histogram, then report it with the log. |
| `head_dispatch_stall` | varies | The next job kept being held back, behind a named rule (warning) or with no rule that explains it (critical). | The named rule is the lever. A hold with no rule named is a scheduler fault worth reporting. |
| `slow_generation_drop_spiral` | varies | The horde dropped jobs for taking too long. The finding says where the time went: waiting to start, generating, or waiting after generation. | Follow where the time went. Lowering `max_power` only helps when generation itself was slow. |
| `post_processing_deferral_starvation` | varies | Post-processing waited for room on the card that never came, and the finished image behind it was never sent. | Report it with the log around the wait. As a stopgap serve fewer models on that card or turn off post-processing. |
| `safety_stage_stall` | varies | The safety check fell behind or lost results. Critical when jobs were dropped with no image. | Turn on `safety_on_gpu` so the check is not bound to the CPU. If the worker's own reservations keep restarting the check, report it. |
| `safety_stage_capacity` | varies | The cards finished jobs faster than the single safety process could check them, and jobs waited well past one check for it. Critical once the wait exceeds the median generation time. | Turn on `safety_on_gpu`. Lowering `max_power` or `queue_size` will not help: every image still goes through the same single checker. |
| `pop_liveness_full_queue` | critical | The queue was full and nothing moved: no job started and none finished. | The most severe stall signal there is. Read it with the dispatch or process finding beside it, and report it with the log around the freeze. |
| `parent_loop_stall` | varies | The main process stopped reading its workers' messages for longer than the status cadence allows. Nothing starts, finishes or is sent back meanwhile. | Run `horde-log timeline` over the gap to see what the main process was doing just before it went quiet, then report it. |
| `lane_placement` | warning | A helper process (safety, post-processing, utilities) restarted onto a different card, or several stacked onto one card while others were free. | Compare that card's duty and free VRAM with the rest before reading its low duty as a GPU problem. Pin the helpers if the move was not intended. |
| `utilities_lane_bringup_timeout` | warning | The image-utilities lane's capability service did not answer its health check before the launcher's startup budget ran out, so the lane was killed and started again. | The worker retries by itself and a later attempt usually wins once the file cache is warm. If every attempt timed out, read the lane's own console log. `enable_image_utilities` turns the lane off. |
| `supervisor_channel_lost` | warning | The worker's channel to its supervisor (the dashboard or the attach runner) closed. The worker keeps serving jobs but sends no more reports, so the supervisor may restart it as frozen. The evidence says whether Windows ran out of memory to commit in the minute before. | Report it with `horde-log bundle` if it repeats. |

## Pops, faults, and the horde

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `forced_maintenance` | varies | The horde put the worker into maintenance after it dropped too many jobs (critical), or the worker was in maintenance for another reason such as the operator setting it (info). | Fix what is dropping jobs before clearing maintenance, or it comes back. The slow-generation or scheduler findings name the cause. |
| `consecutive_failure_pause` | warning | The worker paused asking for work after three failed jobs in a row. | A symptom of whatever failed the jobs; the failed-jobs finding names them. The pause clears on its own. |
| `text_backend_wedged` | critical | The text backend refused every job as busy while generating nothing. A current worker logs this itself when it holds its text jobs, with the backend's own count of finished generations at the decision, which the headline states (or that the backend reported no counts); for an older worker it fires on at least five busy-refused text jobs faulted in a row over ten minutes or more with no text job submitted between them. | Look in the backend output (`logs/text_backend.log` for a backend the worker launches) for `find_slot` or `failed to find a memory slot`. Parallel requests share one context pool, which must hold `text_threads` × `max_context_length`; a current worker sizes its own launch that way, and a backend you run yourself needs `--contextsize` set the same way. `text_threads: 1` only confirms the cause. |
| `pop_api_error_dominance` | warning | The horde kept refusing this worker's requests for work with the same message, quoted in the evidence. | Do what the message says. Most refusals name a condition on the account or the worker registration that will not clear on its own. |
| `pop_governor_dominance` | info | One of the worker's own waits (a whole-card reservation, a large-model cooldown, the wait for the safety check) took a large share of the session. | Nothing if throughput was as expected. Otherwise this says where the time went, and the details say which setting each wait follows. |
| `post_processing_offer_withheld` | varies | Post-processing went out switched off in the worker's job requests for at least a quarter of the observed session, or for one stretch of 15 minutes or more, while the config had it on. Critical at three quarters. Time before the first logged change of the offer is not counted, nor is any silence of more than five minutes between log lines. The evidence names each reason with how often it turned the offer off. | Follow the reason that held it off longest: update a worker at 18.8.1 or older, fix the post-processing stall behind the fault breaker, serve fewer models on a card that has no room for the peak, or let the post-processing models finish downloading. Too many post-processing jobs in hand needs nothing. |
| `idle_fill_offer_stuck` | critical | The idle-fill breaker armed and no disarm line followed for 3 minutes or more, counting an arm still open at the session end. A healthy arm closes within seconds of the 5 second default `idle_fill_threshold_seconds`. The evidence names the arm, its disarm, the last ladder offer line while armed, how many arms ran long, and how far the advertised offer narrowed. A log without arm lines is read from the `Advertising models in pop request` lines instead: an offer under half the session's widest that lasted as long, while an empty pop reported jobs skipped for `models`, fires the same finding. | Send the log with `horde-log bundle` to the worker maintainers. |
| `service_lane_restart_churn` | warning | The reclaim ladder stopped a service lane and restarted it within 5 seconds at least 3 times in a session. A lane cold start costs 6 to 27 seconds, so a restart that quick freed nothing a job could use. The evidence gives, per lane, the stop and restart pairs, how many were quick, the median interval and how many had a hold line between them. | Send the log with `horde-log bundle` to the worker maintainers. |
| `lane_double_dispatch_signature` | critical | Any line saying an inference result was lost while its job was in progress, or a result arrived for a job no longer tracked. The evidence counts each, plus the ended-job ownership releases as context. A process that sampled without a recorded clearance grant is `unpriced_sampling_windows` instead. | Send the log with `horde-log bundle` to the worker maintainers. |
| `faulted_job_census` | warning | Any job failed this session. The evidence lists each one's model and cause. | Take the largest cause first; each maps to one of the other findings. |
| `model_reference_sample_fault` | warning | A job failed because the model list could not be read while it ran. | The job is retried on its own. If it repeats, report it: the model list was being refreshed while the job ran, which is a worker bug. |

## Session context

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `model_economics` | info | Any job was submitted. Read from each `Submitted generation` line: the headline gives the session's kudos, kudos per sampling second (the job's generation time) and per wall second (popped to submit), and the share of wall time taken by the model paying least per wall second among those with 20 or more jobs. The evidence has one row per model for the eight that earned most, plus any model with 20 or more jobs earning under half the session's rate per wall second. | Nothing to do by itself. The horde sets the kudos price, so a model that pays little is not the worker's fault; the share of wall time it takes is the worker's scheduling. Read it beside `whole_card_pop_claim_episodes` and the duty findings. |
| `session_summary` | varies | Always, once per session. Says how the session ended and how long it ran, with the worker version, model count, recovery counts and any rotated archives folded into the parse in the evidence. | Nothing; it is the header the other findings are read against. |

## Live dashboard

These are read from the running worker's state by the dashboard's Insights tab, never from a log, so
`horde-log diagnose` does not emit them. The Insights tab also shows `consecutive_failure_pause` when
the worker has paused itself and `forced_maintenance` (as a note) while maintenance is on. They use the
same card, badge words and copy rules as the log findings, and their thresholds live in
`horde_worker_regen/tui/recommendations.py`.

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `fault_rate` | warning | More than a tenth of this session's finished jobs failed, over at least ten jobs. | Check the Logs tab for the cause. Drop the models that fail most, or lower `max_power` or `max_batch`. |
| `vram_pressure` | warning | A process's peak VRAM use is within a few percent of the card's total. | Lower `max_batch` or `max_power`, or turn off `safety_on_gpu`, so the card keeps room. |
| `low_duty_cycle` | suggestion | The GPU was sampling under half the time while work was waiting. | Raise `max_threads` or `queue_size` so a second job can load while one runs; otherwise put models on an SSD and keep the list short. |
| `low_demand_idle` | suggestion | More than ten minutes of the session had no jobs available. | Offer more models or raise `max_power` to be offered more jobs. |
| `extra_slow_batching` | suggestion | `extra_slow_worker` is on with `max_batch` above 1. | Set `max_batch` to 1. |
| `model_pool_off_swaps` | suggestion | The pool is off and the session has recorded several model loads that pushed another model out. | Try the model pool or its demand-following preset; leave it off if variety is the point. |
| `model_pool_stale_demand` | warning | The pool's demand reading is over fifteen minutes old. | Check the worker's connection to the horde; the pool holds its seats until a fresh reading arrives. |
| `model_pool_unproductive_seats` | suggestion | A seat keeps getting empty requests, or has matched nothing in ten minutes since seating. | Review the pinned models, or turn on the ranker in the `model_pool` settings. |
| `model_pool_resident_matches` | info | A seat matched a job while its model was already loaded in the last few minutes, and no pool issue is outstanding. | Nothing; the pool is doing its job. |
