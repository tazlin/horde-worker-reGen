# Stats export

Besides the human-readable [logs](logs.md), the worker can write a machine-readable **stats stream** for
offline analysis of a run: a structured record of what it did, decided, and observed. The
[`horde-duty-report`](cli.md#horde-duty-report) command and the dashboard's history views read this stream,
and it is the data source a timeline of a session's notable events is built from.

The export is **opt-in**. Toggle it per session from the dashboard, or set
[`stats_export_enabled`](../explanation/bridge_config.md#stats-export-and-retention) to export it on every
start. It is off by default.

## Location and filenames

Files are written under `.horde_worker_regen/stats/` in the worker's working directory:

```
stats-v{worker_version}-{stamp}-{index:03d}.jsonl
```

A session opens `-000` and rolls to `-001`, `-002`, ... once the active file passes the size cap
(5 MiB). Each line is one JSON object discriminated by its `event` field; a reader that does not
recognize an `event` value should skip that line, so the schema can grow without breaking older tools.

## Rotation, retention, and autozip

Within a session the exporter rotates by size (a new `-NNN` file). Across sessions, the
[startup lifecycle](../explanation/bridge_config.md#stats-export-and-retention) manages the directory:
with `stats_autozip_enabled` it compresses inactive prior-session files to `.jsonl.gz`, then a
fail-closed purge ages out (`stats_purge_max_age_days`) and size-caps (`stats_purge_max_total_gb`) the
directory. Only files the exporter itself writes (`stats-v*.jsonl`/`.jsonl.gz`) are ever eligible for
deletion; a foreign file, a leftover `.tmp`, or a nested folder is never touched. The
[`horde-stats`](cli.md#horde-stats) command runs the same compress/downsample operations by hand.

## Event schema

| `event`          | Emitted | Carries |
| ---------------- | ------- | ------- |
| `session_start`  | Once, when export begins | `worker_version`, `timestamp`, and a flat `config` snapshot of the throughput-relevant resolved bridge_data (max_power, max_threads, queue_size, residency and post-processing flags, disaggregation, model count, ...). The anchor for attributing a behavioural change to the configuration it ran under. |
| `session_end`    | Once, as the last line of the session | Terminal `reason`, `duration_seconds`, `jobs_submitted`, `jobs_faulted`, `process_recoveries`. |
| `job_completed`  | Per finished job or alchemy form | The full [`JobMetricsRecord`][horde_worker_regen.process_management.resources.run_metrics.JobMetricsRecord] (stage timings, queue-wait/e2e/sampling seconds, model, resolution, sampler/scheduler/cfg_scale, post-processing, VRAM high-water) and its resolved `baseline`. |
| `stats_sample`   | At most once per second | A periodic [`StatsSample`][horde_worker_regen.process_management.ipc.supervisor_channel.StatsSample] (throughput, kudos/hr, VRAM/RAM, duty cycle). |
| `decision`       | On an admission/dispatch/reclaim verdict (coalesced) | `decision_kind`, `subject`, `verdict`, `reason`, and a flat `inputs` map of the quantities the arbiter decided from. |
| `resource_state` | On a device/overflow transition (edge-triggered) | `state_kind` (governor / WDDM paging / saturation-unresolved), `state`, `device_index`, and flat `inputs`. |

A `job_completed` record pairs the priced request with what it cost to serve: alongside resolution, steps
and batch count it carries `sampler_name` (as the horde advertised it, uncanonicalized), `scheduler`
(the schedule sampled on, `karras`/`normal` when the request only carried the legacy karras bool) and
`cfg_scale`, against measured `sampling_seconds` and the horde's `kudos_reward`. Records written before
these fields existed simply omit them.

### Worker-condition fields on `job_completed`

Two jobs of identical shape can cost very different amounts depending on what the worker was doing around
them, so a record also carries the conditions it was served under. All of these are optional: a record
that could not measure one omits it (or writes `null`), which is distinct from a measured zero.

| Field | Meaning |
| ----- | ------- |
| `model_load_seconds` | Seconds spent loading *this job's own* checkpoint inside its metrics window, summed over the disk-to-RAM and RAM-to-VRAM phases the child reported for that model. `0.0` when the model was already resident, so the field separates the jobs that paid a model switch from those that did not. A load performed under a preceding job's window (a preload staged ahead of dispatch) is attributed to that window. |
| `lora_wait_seconds` | Seconds the job spent blocked on its LoRA/TI files being placed on disk, measured from the pop to the moment its auxiliary set was marked prepared and clamped into `queue_wait_seconds`. A residual wait, not a download duration: a fetch that overlapped other waiting contributes only the part the job actually waited on. Absent for a job carrying no LoRAs and for one whose auxiliary readiness was never stamped (no prefetch pipeline ran). Without it, a cache-miss LoRA appears only as an inflated `queue_wait_seconds`. |
| `queue_depth_at_dispatch` | How many *other* jobs were queued for or running inference at the moment this job was dispatched. |
| `post_processing_depth_at_dispatch` | How many *other* jobs were queued for or running post-processing at that same moment. A job that requests post-processing behind a busy lane pays a tail its own generation did not cause. |
| `whole_card` | Whether a [whole-card exclusive residency](../explanation/resource_governance.md) for this job's model was held on its card at dispatch. Such a job ran with the card to itself and carries the amortized cost of establishing that residency. |
| `process_age_seconds` | How long the inference process serving the job had been alive at dispatch. A young process has cold component and RAM caches, so its first jobs pay loads a long-lived process does not; a mid-session respawn contaminates the jobs around it. |

A re-dispatched job (one whose first attempt failed and was retried) reports the dispatch conditions of the
attempt that actually ran, not of its first.

The `baseline` on a `job_completed` record is the model's baseline as the loaded model reference states it
(`stable_diffusion_1`, `stable_diffusion_xl`, `flux_1`, ...), or `null` when no record for the model was
available. A harness or benchmark session resolves it from the same reference a production worker uses.

### Reconciling `session_end` with the stream

`session_end` closes the session's file: the worker writes it only once every loop that can still finalize
a job (the submitter and the control loop's post-inference drain) has returned, so no `job_completed`
record can follow it. Its two job counters partition exactly those records: `jobs_submitted` counts the
`job_completed` records for work that did not fault and `jobs_faulted` those that did, both covering image
jobs and alchemy forms, so the pair sums to the number of `job_completed` lines above the marker. Records
finalized while the export was switched off are absent from both, because they are absent from the file.
A session that ends without a `session_end` line was killed rather than shut down.

### Scenario provenance on `session_start`

A session driven by the harness or the benchmark (rather than by live horde traffic) additionally carries
`scenario_id` and `scenario_revision` in its `config` snapshot, naming the workload definition it ran and
that definition's revision. They identify which workload produced the session's records, so streams
gathered on different machines can be paired by workload rather than only by configuration, and a workload
edit does not silently blend two different job mixes under one name. A production worker runs no scenario
and its snapshot omits both keys.

The driver publishes them through the `HORDE_WORKER_SCENARIO_ID` and `HORDE_WORKER_SCENARIO_REVISION`
environment variables rather than through `bridgeData.yaml`: the workload belongs to whatever is driving
the worker, not to the operator's configuration, so it stays off the user-facing config surfaces.

### The `decision` record and its coalescing contract

A `decision` captures an arbitration and the already-computed quantities it decided from, so a post-mortem
reads the decision arithmetic directly instead of reconstructing it from prose. `inputs` is a flat map that,
depending on `decision_kind`, includes figures such as `device_free_mb`, `available_mb`,
`outstanding_reservations_mb`, `noise_buffer_mb`, `candidate_delta_mb`, and the governor state.

Decision points are re-evaluated every scheduling tick, so a naive per-evaluation emission would repeat the
same line many times per second. The exporter therefore **coalesces** a decision that holds steady, keyed by
`(decision_kind, subject)`:

- The first evaluation, or any change of `(verdict, reason)`, emits one record.
- While the same verdict holds, a further record is emitted only once every ~30 seconds, carrying
  `repeat_count` (how many evaluations it stands in for) rather than one record per tick.
- When the condition clears (the subject is admitted or its memory freed), a final record is emitted with
  `resolved` set to `true`.

So a sustained hold reads as one opening record, occasional heartbeats, and one resolution, not a flood.
`first_seen_ts` marks when the current unresolved condition first appeared.

`decision_kind` is one of `vram_admission`, `inference_dispatch`, `pp_deferral`, or `reclaim_rung`;
`verdict` is one of `admit`, `defer`, `deny`, `withhold`, `freed`, or `no_op` (the last three of which are
*resolving*).

## Stats tab: the Model pool section

When the [fixed model pool](../explanation/model_pool.md) is enabled, the dashboard's **Stats** tab grows a
**Model pool** section (it is absent entirely when the pool is off, so a worker that does not run the pool
sees an unchanged tab). It reports pool readiness and pop outcomes split by lane:

- **Seats** and **Bench**: how many seats are currently resident, logically seated, and configured in total,
  plus how many models are cooling down on the bench.
- **Last routed lane**: the advertising lane the most recent pool-routed pop used (`FIXED` for the seated-model
  offer, `FREE` for the wider offer).
- **Fixed lane** and **Free lane**: each lane's session-cumulative `matched / pops`, the match rate, and the
  number of matches whose model was already resident at acceptance time. A lane with no pops yet shows `-`.
  These are pop outcomes rather than completed-job throughput. A low fixed-lane match rate
  points to seated models the horde is not feeding; a low free-lane rate points to a thin wider offer.

In the same tab, the **By model totals** table marks any model that currently holds a pool seat with a `◆`
suffix (the panel subtitle carries the legend), so a seated model's jobs, megapixelsteps, and latency read in
place rather than in a duplicate per-seat table.

## See also

- [Logs](logs.md): the human-readable side of the same run.
- [Bridge configuration](../explanation/bridge_config.md#stats-export-and-retention): the export and
  retention config fields.
- [Command-line reference](cli.md#horde-stats): the `horde-stats` and `horde-duty-report` commands.
