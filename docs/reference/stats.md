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
| `session_end`    | Once, on clean shutdown | Terminal `reason`, `duration_seconds`, `jobs_submitted`, `jobs_faulted`, `process_recoveries`. |
| `job_completed`  | Per finished job or alchemy form | The full [`JobMetricsRecord`][horde_worker_regen.process_management.resources.run_metrics.JobMetricsRecord] (stage timings, queue-wait/e2e/sampling seconds, model, resolution, post-processing, VRAM high-water) and its resolved `baseline`. |
| `stats_sample`   | At most once per second | A periodic [`StatsSample`][horde_worker_regen.process_management.ipc.supervisor_channel.StatsSample] (throughput, kudos/hr, VRAM/RAM, duty cycle). |
| `decision`       | On an admission/dispatch/reclaim verdict (coalesced) | `decision_kind`, `subject`, `verdict`, `reason`, and a flat `inputs` map of the quantities the arbiter decided from. |
| `resource_state` | On a device/overflow transition (edge-triggered) | `state_kind` (governor / WDDM paging / saturation-unresolved), `state`, `device_index`, and flat `inputs`. |

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

## See also

- [Logs](logs.md): the human-readable side of the same run.
- [Bridge configuration](../explanation/bridge_config.md#stats-export-and-retention): the export and
  retention config fields.
- [Command-line reference](cli.md#horde-stats): the `horde-stats` and `horde-duty-report` commands.
