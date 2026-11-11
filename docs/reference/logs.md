# Logs

The worker writes logs to the `logs/` directory under its run root: the working directory, unless
`HORDE_WORKER_RUN_ROOT` names another (see [the `.abort` file](../explanation/shutdown_and_faults.md#the-abort-file)
for what else lives under it). The dashboard's **Logs** tab reads these same files,
and the **Diagnostics** tab runs the [`horde-log diagnose`](cli.md#horde-log) detectors over
them to surface what went wrong, without you needing a shell. See
[How the diagnostics stay in sync](../explanation/log_diagnostics_contract.md) for how a detector, the
log line it reads, and the dashboard are kept from drifting apart.

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info). |
| `bridge_n.log` | Per-process log. |
| `trace.log` | Errors, criticals, and `TRACE`-level lines (including the suppressed repeats described below). |
| `trace_n.log` | The same, per process. |
| `bridge_tui.log` / `bridge_host.log` | The supervisor (parent) process's own log: TUI dashboard or `--host` wrapper. Captures worker launch, crash-loop, and TUI-process crash diagnostics that never reach `bridge.log`. |
| `text_backend.log` | A managed text backend's own stdout and stderr, appended across launches. Written by the backend, not the worker. |

## Download timing

The download child records `Download task timing`, including total task time, measured pause/throttle
wait and the bandwidth share at completion. Its engine records `Download transfer` separately from
`Download verification`, so an extra checksum pass is visible independently of the HTTP transfer.
Debug-level range-probe and response lines include the final hostname, without URL queries or credentials.
See [Diagnosing slow downloads](../explanation/model_downloads.md#diagnosing-slow-downloads) for how
to compare one and four connections on the same file while idle and during inference.

## Repeating telemetry is time-boxed

Some lines describe a condition that is re-evaluated on every control-loop tick: a child's memory report,
or a reason the parent skipped a periodic reconciliation. Emitting each one at `DEBUG` puts several lines
a second into `bridge.log` and crowds out the job-flow story an operator is actually reading.

Such lines are time-boxed per subject (per process, per device): at most one emission per 30 seconds lands
at `DEBUG`, and the repeats in between are emitted at `TRACE` instead. Nothing is dropped. `bridge.log`
includes a periodic sample and `trace.log` includes the full series, so a per-tick reconstruction is still
possible after the fact.

If you are adding a log line that fires on a timer rather than on an event, use
`throttled_log_level()` from `horde_worker_regen/process_management/_internal/util.py` and log at the level
it returns.

## Deadlock snapshot free-VRAM figures

The deadlock diagnostics include two free-VRAM figures and they are not interchangeable:

- `device_free_vram=` is derived from the children's torch view of the card. Under WDDM demand-paging that
  view counts paged-out blocks as free and has been seen overstating a card by gigabytes, which reads as a
  worker refusing to dispatch onto an empty card. On a multi-GPU host the leading figure is the tightest
  card and a parenthetical names each card's own reading (`(card0=6750MB, card1=7580MB)`).
- `measured_device_free_vram=` is the parent's own device-level (NVML) reading, the same figure admission
  prices against. When the two disagree, this is the one the card is actually offering.

A snapshot carrying only the first figure is from a worker with no device-level reading wired in.

## Rotation and retention

The supervisor logs (`bridge_tui.log` / `bridge_host.log`) and the benchmark run logs rotate at a **25 MB**
size cap, are compressed to `.zip` once rotated, and keep a bounded number of older files (20 for the
supervisor, 10 for benchmark runs). This keeps total disk use bounded under a heavy or long-running session,
and keeps any single file small enough that the **Logs** tab can tail it without buffering a multi-GB file.
The dashboard reads only the trailing window of a log, so a large file scrolls quickly to the latest lines.

## Headless model-pool status

The headless worker's periodic status block always names the model-pool mode. With the pool off, it says that
normal eligible-model advertising has no persistent seat bias. With the pool on, one compact line lists the
first six seats (plus an overflow count), their source (`M`, `R`, or `S`) and readiness (`resident` with GPU,
`cold`, or `downloading`), the most recent fixed/free advertising lane, each lane's matched/popped rate and
resident-match count, bench size, demand-reading age, and charged/session download admission budget. The charge is the
reference-declared model size booked when a request starts, not measured disk use or bandwidth. A pending seat
names its target model and reads `downloading`; reaching the limit does not cancel that already-started transfer.

## Which process is which

The numbered logs map to the worker's child processes:

- `bridge_0.log` is the **safety** process.
- `bridge_1.log` and higher are **inference** processes.

For why the worker runs separate inference and safety processes, see
[Architecture](../explanation/architecture.md).

## What the worker reads out of `text_backend.log`

A managed text backend writes this file itself; the worker only appends its launches to it and reads three
lines back out. Once a launch becomes ready, the worker scans what that launch wrote for llama.cpp's
`model buffer size`, `KV buffer size` and `compute buffer size` lines, sums each kind over the devices that
printed it, and shows the totals on the backend's dashboard row. They are display detail only: no admission
or pricing decision reads them, and a backend that prints none (a quiet debug level, or a program that is
not llama.cpp-based) simply leaves them blank. The patterns are registered as log signatures with samples
pinned to a real backend log, the same contract the worker's own parsed lines are held to.

## Tailing a log live

```powershell
# Windows PowerShell
Get-Content bridge_1.log -Wait
```

```bash
# Linux/macOS
less +F bridge_1.log
```

## Startup context cost

At startup `bridge.log` has one INFO line per card from the accelerator probe:

```text
Accelerator 0 context cost: baseline 3600MB, after context 4087MB, after holder 4575MB, first 487MB, marginal 488MB. The two figures agree (1 probe run).
```

The three readings are device-wide VRAM used on the card: before the probe created any context, after it
created its own, and after a second process started holding a context on the same card. `first` is the
first-context cost (after context less baseline) and `marginal` is the cost of each further context (after
holder less after context). The worker holds both figures for the whole session. A clean launch reads them
within a few tens of MB of each other. `unread` marks a reading the probe could not take, as on a card
without NVML.

The sentence after the figures is the check's result:

- `The two figures agree` means the line's run was accepted. A count above one means an earlier run on some
  card disagreed and the probe ran again. The count is the number of probe runs for the whole host, so it also
  reads above one on a card that agreed on its first run. The line shows the accepted run only. The readings
  of an earlier run that disagreed are not logged, and the count is the only sign of them.
- `The two figures disagree on every run` means no run agreed within the tolerance (a quarter of the smaller
  figure, at least 64 MB) after three runs. The figures are from the run with the median first-context
  figure, and a WARNING beginning `Accelerator N first-context and marginal figures disagree on every probe
  run` lists every run's readings. Another program allocating or releasing memory on the card while the worker
  starts is the usual cause, and the baseline readings show whether the card was already busy.
- `Not checked` means the first run's two figures could not be compared, because the card had no baseline
  reading or the marginal was not measured as an NVML delta. The figures are the first run's, and the card is
  not measured again.

`vram_per_process_overhead_mb` in the config, when above 0, replaces the first-context figure regardless of
this line. The
figures in use appear on every stream forecast line as `overhead/proc=` and `marginal/ctx=`.

The dashboard's Stats tab shows the same result per card, and lists every run's readings for a card whose figures
disagree on every run, as the WARNING does (see
[Startup memory measurement](../how-to/use-the-dashboard.md#startup-memory-measurement)).

## VRAM admission diagnostics

A warning beginning `Deferring post-processing for job` records a lane-admission miss. Its candidate is the
post-processing chain's marginal VRAM requirement. The available-room expression reports the same arithmetic
the arbiter decided from: current device-free VRAM minus outstanding reservations and the proportional noise
margin. Outstanding means memory not yet materialized in the device-free sample; already-realized commitments
are not subtracted again.

The stable prefix is part of the diagnostics contract used by `horde-log diagnose`. The numeric wording may
become more precise without invalidating older logs, so the detector accepts both the current measured-room
format and the former free-after-commitments format. Repeated warnings for one job, with no later PP completion,
surface as `post_processing_deferral_starvation` in the dashboard's Diagnostics tab.

## Runtime safety-placement diagnostics

Two info lines in `bridge.log` record every move of the safety process the runtime placement policy makes, and
both state the evidence rather than a cycle count:

```text
Runtime safety placement: moving safety off card 0 after 34s of sustained memory pressure there (card 0:
measured free 812MB, marginal need 1091MB, noise buffer 512MB, safety footprint 3044MB, governor HEALTHY).
Restoring it costs about 30s of safety unavailability, which is the dwell this pressure had to outlast.

Runtime safety placement: restoring safety to card 0 after 71s of measured free at or above the 4647MB it needs
to survive the peak that card is committed to (card 0: measured free 5210MB, ...).
```

`marginal need` is what the card still has to find for the heaviest sampling peak it is committed to, net of
what the process holding that model already has on the device; it is the figure both sides forecast against.
A `Runtime safety placement inputs:` line at DEBUG includes the same evidence on every change to it, so a capture
can attribute a flip to what the policy saw; unchanged repeats drop to TRACE. For the policy itself see
[VRAM arbiter](../explanation/vram_arbiter.md#runtime-safety-placement).

A deferred safety start whose card never drains is started on the CPU once, with a warning of this shape:

```text
Deferred SAFETY process start on device 0 has waited 601s with no headroom progress (677MB free, governor
pressure; the start needs at least 1631MB and a healthy governor). Starting the safety process on the CPU so
finished images can be checked; runtime safety placement returns it to a GPU once a permitted card shows
durable room.
```

The later restore is the ordinary `Runtime safety placement: restoring the safety process to the GPU.` line.

A check that finds the device full is rerun on the CPU by the same process, which writes one warning per
fallback to its log (`bridge_0.log`). The stage is `CLIP staging`, `NSFW check` or `aesthetic score`, and the
free figure is the one the allocator stated:

```text
Safety job 9c46c418-81a8-4727-b8c5-495da36f9efd ran out of device memory during NSFW check (0 bytes free). The
check was rerun on the CPU in 5.73 seconds and the process stays up.
```

See [Resilience and recovery](../explanation/resilience_and_recovery.md#a-safety-check-that-runs-out-of-device-memory).

A deferred inference start on such a card is retired instead, while another inference lane is serving, with
a warning of this shape (a structural shortfall names the card's achievable room in place of the wait):

```text
Deferred INFERENCE process 1 start on device 0 has waited 601s with no headroom progress (600MB free, governor
pressure; the start needs at least 1631MB and a healthy governor). Retiring the slot: device 0 now plans 0
inference process(es) and the worker plans 1. It is planned again once the card has held room for a start
for 300s.
```

Once the card has held room for the whole dwell, one info line records the restore:

```text
Restoring a retired INFERENCE slot on device 0 as process 1: the card has held room for a start for 300s
(9000MB free, the start needs 1631MB). Device 0 now plans 1 inference process(es) and the worker plans 2.
```

Both lines have a matching action-ledger event, `inference_start_retired` and `inference_slot_restored`.

The `Withholding job pops: post-inference safety backlog` warning's closing advice depends on where safety is
running: it suggests enabling `safety_on_gpu` only while that is off, and names resource governance when the
setting is on but the placement policy has moved safety to the CPU.

On a multi-GPU worker, a safety process pinned to a permitted GPU has fixed residency: memory pressure,
whole-card residency, and the reclaim ladder do not move it off that card or demote its weights. A deep backlog
uses dispatch pressure instead. These edge-triggered info lines mark the durable exclusion arming above two
pending checks and clearing only after the backlog reaches zero:

```text
Safety backlog of 3 checks exceeds 2 on card 1, which hosts the safety process: holding new inference
dispatches off that card until the backlog drains. Jobs already running there are left to finish.
Safety backlog on card 1 is at 0: new inference dispatches may land there again.
```

While the exclusion stands, a job that only that card can serve remains queued and its dispatch-hold record is
classified as `safety_backlog_exclusion`. Other cards continue accepting work. Single-GPU workers retain the
runtime demote/restore policy and use the pop self-throttle as their backlog relief valve.

## Lines that say a mechanism acted

Each of these is logged once per edge, so a session can be read for whether one mechanism acted without
inferring it from throughput. All are registered in `analysis/log_signatures.py`.

| Line | Level | Logged when |
| ---- | ----- | ----------- |
| `hordelib: <version> (<source>)` | INFO | Once at start. `<source>` is `site-packages`, or `editable <checkout> g<commit>` with `.modified` on a dirty tree. The same string is in the status block's Worker Info line and the stats `session_start` row. |
| `Idle-fill armed: head job model <model> waited <n>s for dispatch` | INFO | The idle-fill breaker arms. |
| `Idle-fill disarmed: ...` | INFO | It disarms: the head dispatched, left the queue, or no sibling is free. |
| `Idle-fill offer: rung <i> of <n>, <k> model(s) at max_power <p>` | INFO | A pop is shaped by the idle-fill ladder and the shaped offer changed. |
| `Released process <n>'s ownership of job <id>: the job ended without a result from that lane.` | INFO | The control loop retired a lane's ownership of a job that ended some other way than that lane's result. Rare by design. |
| `Reclaim ladder: keeping the <lane> lane down until job <id> finishes sampling.` | INFO | A lane pause booked for a job is held because that job still waits. Once per pause. |
| `Post-processing finished for job <id> in <t> seconds on process <n> (chain <a>s, encode <b>s, <w>x<h>, <operations>).` | INFO | A post-processing job returns with the lane's measurements. The short form without the parenthesis is logged when the lane sent none. |
| `Upload encode for job <id>: <t>s in a worker thread.` | DEBUG | Submit encoded the image for upload. |
| `Upload for job <id>: bytes already in the upload encoding, not re-encoded.` | DEBUG | Submit uploaded the bytes as the lane produced them. |
| `Clearance lease on device <d>: restored the <n>MB activation reserve of process <p>'s unconsumed grant after its job start.` | DEBUG | A grant reached the child before its job start, the job start unset the grant's figures, and the next clearance step wrote the reserve back. Once per grant. The session count is `ClearanceController.reserves_restored_after_job_start`. |
| `Clearance booking for job <id> on process <p> went from <a> MB to <b> MB at the <stage grant\|sample grant\|encode boundary>. <detail>` | DEBUG | A clearance grant re-booked the job's dispatch reservation, or the job's encode boundary report arrived after the grant and the reservation was booked against it. At a grant the detail shows whether the boundary report had arrived. Before it arrives, the grant's charge is held whole. Once per grant and once per boundary. The same figures are in the clearance decision's `booking_before_mb`, `booking_after_mb` and `encode_boundary_arrived` inputs. |
| `Whole-card governor deferral of <model> on card <d> cleared after <n>s held.` | INFO | The establish-rate limiter let a head through that it had deferred, at preload admission or at dispatch. `<n>` is the time since the first deferral of that head. `<d>` is `None` on a single-GPU worker, whose one residency state is keyed worker-wide. |
| `Whole-card pop claim engaged for <model>: advertising that model alone while it holds the card, for at most <n>s. Earning on card <d>, in kudos per card second: <m> for the claimed model, <r> for the rest of the card, over <j> jobs of the claimed model. <verdict>` | INFO | A claim over the pop offer engaged. `<m>` and `<r>` are the model rate and the rest rate of `CardEarningAccumulator.rate` over the trailing day, `unmeasured` when a rate's denominator is zero. `<j>` counts the claimed model's finished jobs on the card, faulted jobs included. `<d>` is the lane's device index, 0 on a single-card host. The verdict sentence is read below the table. A claim whose model earns below the rest stands only when the policy or an exemption keeps it, and the line then ends with `The card is held for it because <reason>.` The reason is `whole_card_claim_policy is always`, `it is listed in whole_card_claim_always_models`, `it is pinned to a model pool seat`, or `the card fits it only at a reduced size, and without the hold it is asked for on one pop in several`. Once per engage. |
| `Not holding the card for <model>, since it earns below the rest of the card. On card <d> it earns <m> kudos per card second, and the rest of the card's work earns <r>. The worker asks the horde for the other models, and asks for <model> again <n>s after its jobs finish (whole_card_low_earning_return_seconds). To ask the horde only for <model> while it holds the card, set whole_card_claim_policy to always, or add the model to whole_card_claim_always_models.` | INFO | A whole-card grant holds the card for a model whose earning verdict is below the rest, and the default `whole_card_claim_policy` withholds its pop claim. `<m>`, `<r>` and `<d>` are read as on the engage line. `<n>` is `whole_card_low_earning_return_seconds`. Once per model, at the end of a residency release pass, while the grant holds the card. A claim that stood and is then withheld for the same model logs this line and does not log a release. |
| `Not holding the card for <model>, since the worker is not asking the horde for it now. Asking the horde only for that model would leave nothing to ask for. The worker asks the horde for its other models.` | INFO | A whole-card grant holds the card for a model left out of the offer, so its pop claim is withheld. The `Not offering <model>` warning gives the reason the model is left out. Once per model, at the same edge as the line above. |
| `Not asking the horde for <model> for now, since it earns below the rest of the card. The jobs for it already taken finish first. The worker asks for it again <n>s after they finish (whole_card_low_earning_return_seconds), or sooner when the horde has no work for the other models or the worker has no work at all.` | INFO | The large-model pop governor's paced return took the model out of the offer, because its claim is withheld. Once per pacing, on the pop cycle that starts it. `<n>` is `whole_card_low_earning_return_seconds`. |
| `Asking the horde for <model> again, because <cause>.` | INFO | The paced return ended. `<cause>` is `<n>s have passed since its last job finished`, `the horde had no work for the other models` (an answered pop came back with no job for an offer that left the model out), `the worker has no work at all` (the idle escape), or `whole_card_claim_policy, whole_card_claim_always_models or an exemption now holds the card for it`. Once per return. After an empty answer the model is not paced again until a job for another model is popped. |
| `Released the pending exclusive admits for <model> on card <d>, since the card does not hold a residency for that model.` | INFO | A residency release pass found a pending job marked exclusive for a card whose residency has ended, for example after a soft reset. While the mark stands, every other job's preload onto that card is refused. Once per release. `<d>` is `None` on a single-GPU worker. |
| `Foreign floor on device <d> moved from <a> to <b>: total <t>, device free <f>, worker footprint <w>MB from <source> (<n> pids listed), ledger <l>MB, reading <r>.` | DEBUG | The card's sustained foreign floor changed. The floor is the minimum reading over a trailing window, so it rises when the lowest samples age out and falls on a new low. The worker footprint is the figure subtracted from the card's use, and `<source>` is the branch that produced it: the driver's per-process usage, or the ledger when the driver reports no per-process usage or lists no worker pid. A rise with the ledger figure far below the card's use shows memory held by the worker's processes being learned as foreign. Once per move, per card. |
| `Save-our-ship: holding the soft reset while a replacement child is still alive and booting (boot elapsed <n>s, hard cap <c>s). Rebuilding the pool now would replace the replacement.` | INFO | The recovery supervisor would rebuild the pool, but a replacement child is still booting inside the hard cap. Once per hold. |
| `Reclaim ladder: restoring the <lane> lane while job <id> still waits, since the job waits on that lane.` | DEBUG | A lane pause booked for a job is restored although the job still waits, because the job waits on the paused lane itself. In practice the post-processing lane. |
| `Alchemy form <id> was already submitted on the horde. Recorded as delivered with no reward. Horde response: <message>` | WARNING | The horde refused an alchemy submit because it already holds a result for the form. The form counts as submitted. No reward is recorded, because the response does not show whether an earlier attempt or the horde settled the form. |

The engage line ends with one of these verdict sentences. Each rate has a 95% band on each
side (see [Persisted card earning](stats.md#persisted-card-earning)).

| Sentence | Meaning |
| -------- | ------- |
| `The claimed model is not below the rest of the card by more than the spread of the readings.` | The model's band reaches the rest's band or lies above it. The model may still read lower than the rest. The difference is then within the noise in both rates. The stats stream and the dashboard report this verdict as `pays`. |
| `The claimed model earns below the rest of the card.` | The top of the model's band is under the bottom of the rest's band. |
| `The claim's value is unmeasured until 20 jobs are counted.` | Fewer than 20 of the model's jobs are counted on the card, so one unusual job could set its rate. |
| `The claim's value is unmeasured until both rates have readings from two time buckets.` | A rate has no reading, or its figures fall in fewer than two 15-minute buckets, so it has no standard error. |
| `Card earning is not measured on this worker. The claim's value is unmeasured.` | The scheduler has no earning accumulator. The line then has no figures. |

Under the default `whole_card_claim_policy` a verdict of below the rest withholds the claim, unless the model is
exempt. The other verdicts leave the claim standing. The rules are in
[Pop intake](responsibility-map-runs-and-residency.md#pop-intake) ("Whether a claim is withheld").

## Pop gates in the parent log

A pop cycle that ends at a gate stamps `WorkerState.last_pop_gate`. For most gates it does not write a log
line. The gates are listed in order under
[Pop intake](responsibility-map-runs-and-residency.md#pop-intake).

| Gate | What the parent log shows |
| ---- | ------------------------- |
| The pop governors: `whole_card_residency`, `large_model_switch`, `large_model_reentry`, `low_earning_return`, `post_inference_backpressure`, `unservable_model_holdback`, `consecutive_failure_pause`, `pop_error_backoff`, `lora_pop_backoff`, `self_throttle_pause`, `megapixelstep_wait`, `model_stickiness` | `Pop governor ENTER: <name>` and `Pop governor EXIT: <name>` at each spell's edges |
| The whole-card pop claim | `Whole-card pop claim engaged for <model>` and `Whole-card pop claim released for <model>: <reason>`. A claim withheld for a low-earning model logs `Not holding the card for <model>` instead, and its pops are not held at this gate. |
| The large-model limits (`large_model_limits`), which include the paced return | the `large_model_switch`, `large_model_reentry` and `low_earning_return` governor spells. The `low_earning_return` reading names the paced model and gives the time left once its jobs finish. The paced return also logs `Not asking the horde for <model> for now` and `Asking the horde for <model> again, because <cause>` |
| The safety and submit backlogs | `Withholding job pops: ...`, the safety form repeated on a timer |
| An offer emptied by every stage | `Every advertisable model was narrowed out of this pop's offer`, on the first such cycle |
| No configured models | an error on every cycle |
| Queue full, warm-up, no serving safety process, no available inference process, the pop frequency, and a claim whose offer an earlier stage emptied | no line |

A sent pop logs `Advertising models in pop request` (DEBUG), and an empty reply logs `No job available`. When a
span has both lines absent, its pops were held at a gate. The 60 s pop-liveness line (`Pop liveness: no pop attempt has
reached the horde`) reports the gate when the silence lasts that long. The live status snapshot's `pop_gate` and
`pop_gate_since` fields (`WorkerStateSnapshot`) also report it while it holds.

## Time a job waits for auxiliary files

A job's wait for its LoRA or TI files starts at `Requested aux prefetch for job <id>` (DEBUG) and ends at the
first of `Aux prefetch complete for job <id>` (DEBUG), `Aux prefetch deadline reached for job <id>`,
`Aux prefetch budget for job <id>`, `prefetch failed terminally for ...; dispatching job(s) <ids>` that lists the job, or the job's
`Starting inference for job <id>`. A job whose files were all present when it was popped does not have a request line
and does not appear in this measure. The per-job `lora_wait_seconds` on `job_completed` is the same wait,
clamped into the job's queue wait ([Stats export](stats.md)).

## Action-ledger events for runs and safety replacement

`action_ledger.jsonl` does not have a row for a model unload, a finished model load, a model-map change or a process
state change. A lane's residency between dispatches therefore cannot be read from the ledger. The nearest rows
are `preload_requested` and `inference_dispatched`, and the lane logs and the stats stream's
`process_state_summary` hold the rest. The ledger records these two events with the fields below in `detail`.

| Event | Fields | Recorded when |
| ----- | ------ | ------------- |
| `dominant_run_ended` | `device_index`, `run_model`, `cause` (`drained`, `starved_head` or `superseded`); for `starved_head` also `head_model`, `head_seconds_since_pop`, and the head's id as `job_id`; for `superseded` also `superseding_model` | A dominant-model run ends because its jobs drained, because a different-model head waited past the anti-starvation bound, or because a rung-2 residency for a different model took the card. |
| `safety_replacement_requested` | `safety_requeues`, `safety_process_alive`, and the faulted job's id as `job_id` | The orphaned-verdict watchdog faults a job that ran out of safety requeues and asks for the safety pool to be replaced. The replacement itself is recorded as `process_replaced`. |

## Supervisor stall forgiveness

Two lines in `bridge_tui.log` / `bridge_host.log` describe the *supervisor's* own liveness rather than the
worker's. The supervisor only observes the worker when it ticks, so a gap of more than 30 seconds between
two ticks is time it could not see and must not charge to the worker; it moves the wedge baseline forward
instead. That forgiveness is bounded (three re-graces, or five minutes of already-forgiven time, per
rolling hour), because an unbounded one silently disables the wedge backstop.

- `INFO` `Supervisor tick gap of Ns (it was likely descheduled or the host slept). Resetting the worker
  wedge baseline …` records one forgiven gap. The line includes a running counter (`Re-grace 2/3 in the last
  60 minutes`, plus session totals), so a repeating pattern reads as one condition rather than as unrelated
  events. A single line after a laptop resume is expected and does not need action. A rising counter means the
  host is starving the dashboard/host process.
- `WARNING` `Supervisor tick gap of Ns, but its stall-forgiveness budget is spent …` means further gaps are
  now charged to the worker so wedge detection can proceed. Emitted once when the budget runs out, not per
  gap, so its absence on later gaps does not mean the condition cleared. A worker that is still advancing
  is unaffected by this; a silent one will be force-killed and relaunched.
- `WARNING` `Worker (pid=…) survived the prior wedge tree-kill attempt. Retrying …` means the same frozen PID
  remained alive after the orphan-proof kill. The supervisor retries every ten seconds until process death
  reaches the ordinary bounded relaunch path. Real loop progress or a replacement clears the retry state.

The same counters are on the dashboard status bar as a `stalls n/3` segment (yellow while forgiving, red
once the budget is spent), which appears only when there is something to report. See
[Frontend and durable state](../explanation/frontend_and_state.md) for the mechanism.

## Sharing logs

Do **not** post `.log` files in public channels. Send them to a maintainer directly: we cannot
guarantee your API key is not present in a log. See [Troubleshooting](../how-to/troubleshoot.md).

Host-RAM hold edges name available RAM, the hard floor, and the shared soft/preload/restore requirements.
`host_ram_starvation` correlates those edges with RAM load refusals, ceiling drains, creep cycles and
queued-work concurrency. The parser accepts the older reserve-margin hold wording for historical bundles.
