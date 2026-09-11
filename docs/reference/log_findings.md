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

## Memory and residency

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `oom` | critical | The card ran out of memory. The allocator's message names the model that faulted and the other processes holding memory at that moment. | Lower `max_threads` or `queue_size`, or turn on the VRAM budget. Many processes with almost nothing free means too many models at once, not a model that is too large. |
| `swallowed_oom` | warning | Jobs failed with a bare "no images were produced" and no reason. ComfyUI can report an out-of-memory error this way. | Check free VRAM around the failures. If the card was full, treat them as out-of-memory faults. |
| `file_descriptor_exhaustion` | critical | A process ran out of file handles. It then fails every job while still sending heartbeats. | Raise the open-file limit as a stopgap and report it; the leak is in the worker. Lowering memory settings will not help. |
| `scheduler_starvation_wedge` | varies | The VRAM budget held the next job back on a card with plenty free. Critical if the wait ran to a rebuild and dropped jobs. | Relax the VRAM budget, or turn off `unload_models_from_vram_often` and `high_performance_mode` so processes stop cycling. |
| `unsatisfiable_head_starvation` | varies | The same model kept reaching the front of the queue and being refused a load while the card sat idle, and nothing cleared it. | Check that the model fits the card with room to run. If it does, relax the budget or reduce swapping. If not, remove it. |
| `residency_reconciliation_holds` | varies | The worker paused to unload an idle neighbour before running the next job, repeatedly. | Each pause is a swap paid for holding more models than the card fits at peak. Check `unload_models_from_vram_often` and the eviction thresholds. |
| `whole_card_convergence_wedge` | critical | A model that needs the whole card could not get it: an idle process or a service process never left the card. | Capture the log around the stall and report it. As a stopgap lower `queue_size`, or do not serve a whole-card model beside a deep queue of others. |
| `whole_card_nonhead_residency_starvation` | varies | The card was reserved for a job that was not next in line, so the next job had nowhere to run. | Capture the log around the stall and report it. The worker should only reserve the card for the next job. |
| `whole_card_residency_churn` | warning | The whole card was reserved and released over and over in one session. | Thrash: each round trip costs a model reload and a safety restart. Usually a model that does not need the whole card is being given it. |
| `whole_card_pop_claim_episodes` | info | The worker asked the horde for one model only while it held the card. Recorded so the claim windows are visible. | Nothing if that is the heavy work this worker is for. Otherwise trim the model list or lower `whole_card_residency_max_hold_seconds`. |
| `whole_card_pop_claim_monopoly` | warning | Claims kept running to the hold cap while other models' jobs sat waiting. | Decide whether this worker should specialise. If it should serve a mix, lower `whole_card_residency_max_hold_seconds` or drop the whole-card model. |
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

## Pops, faults, and the horde

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `forced_maintenance` | varies | The horde put the worker into maintenance after it dropped too many jobs (critical), or the worker was in maintenance for another reason such as the operator setting it (info). | Fix what is dropping jobs before clearing maintenance, or it comes back. The slow-generation or scheduler findings name the cause. |
| `consecutive_failure_pause` | warning | The worker paused asking for work after three failed jobs in a row. | A symptom of whatever failed the jobs; the failed-jobs finding names them. The pause clears on its own. |
| `pop_api_error_dominance` | warning | The horde kept refusing this worker's requests for work with the same message, quoted in the evidence. | Do what the message says. Most refusals name a condition on the account or the worker registration that will not clear on its own. |
| `pop_governor_dominance` | info | One of the worker's own waits (a whole-card reservation, a large-model cooldown, the wait for the safety check) took a large share of the session. | Nothing if throughput was as expected. Otherwise this says where the time went, and the details say which setting each wait follows. |
| `faulted_job_census` | warning | Any job failed this session. The evidence lists each one's model and cause. | Take the largest cause first; each maps to one of the other findings. |
| `model_reference_sample_fault` | warning | A job failed because the model list could not be read while it ran. | The job is retried on its own. If it repeats, report it: the model list was being refreshed while the job ran, which is a worker bug. |

## Session context

| Id | Severity | Fires when | Remedy |
|----|----------|------------|--------|
| `session_summary` | varies | Always, once per session. Says how the session ended and how long it ran, with the worker version, model count, recovery counts and any rotated archives folded into the parse in the evidence. | Nothing; it is the header the other findings are read against. |
