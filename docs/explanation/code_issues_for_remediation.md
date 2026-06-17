# Code Issues for Remediation

This document catalogs potential bugs, rough edges, and code smells discovered
during a documentation QA pass. These are **code** issues (not documentation
issues); they should be evaluated and either fixed or intentionally documented
in the codebase.

---

## 1. `WorkerState.too_many_consecutive_failed_jobs_wait_time` is misleading

**File:** `horde_worker_regen/process_management/worker_state.py`

The field defaults to `60 * 10` (600 seconds) and is passed to the status
reporter for display. However, the **actual** wait duration used by
`_handle_consecutive_failures` is the module-level constant
`CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS = 180` (from `pop_throttler.py`).

The status display will show "waiting 600 s" while the code actually waits only
180 s. The field should either be removed (in favor of the constant) or the
constant should be replaced by the field.

---

## 2. `_select_models_for_pop` stickiness can produce an empty model set

**File:** `horde_worker_regen/process_management/job_popper.py`

When model stickiness activates and **all** inference processes are busy:

```python
free_models = {
    process.loaded_horde_model_name
    for process in process_map.values()
    if not process.is_process_busy() and process.loaded_horde_model_name is not None
}
if len(loaded_models) >= 1:
    models = free_models  # ← empty set when all processes are busy
```

`free_models` will be empty (no free processes), `len(loaded_models) >= 1` is
True (processes have models loaded even if busy), so `models` becomes the empty
set. This causes `_select_models_for_pop` to return `None`, skipping the pop.
This may be intentional (don't pop if no process can accept work), but the
intent is unstated and the `len(loaded_models) >= 1` guard looks like it was
meant to prevent this exact scenario.

**Suggested fix:** Either add a comment explaining the intent, or fall back to
all loaded models (not just free ones) when `free_models` is empty.

---

## 3. `LRUCache` type annotation is too broad

**File:** `horde_worker_regen/process_management/lru_cache.py`

```python
def append(self, key: str) -> object:
```

The return type `object` is overly broad. The method can only ever return `None`
or a `str` (the bumped key). Also, `cache` is typed as
`OrderedDict[str, ModelInfo | None]` but the values are always set to `None`;
this suggests the cache is used as an ordered set rather than a key-value store.
Consider renaming or re-typing.

---

## 4. Dead code / unused field in `LRUCache`

**File:** `horde_worker_regen/process_management/lru_cache.py`

`self.cache[key] = None`: the value `None` is written but never read back
meaningfully. The cache is effectively an ordered set. The `ModelInfo | None`
value type is misleading.

---

## 5. `_bridge_data_loop` hot-reload does not detect mid-read modifications

**File:** `horde_worker_regen/process_management/process_manager.py`

```python
self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)
if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
    ...
    self._last_bridge_data_reload_time = time.time()
```

If the YAML file is modified while `BridgeDataLoader.load()` is reading it, the
mtime might advance past `time.time()` (set after the load completes), causing
the next poll cycle to miss the change. In practice the 1-second poll interval
makes this extremely unlikely, but it is a theoretical race.

**Suggested fix:** Capture `os.path.getmtime()` immediately _after_ the load
succeeds and use that as `_last_bridge_data_reload_time`, rather than
`time.time()`.

---

## 6. Post-processing overlap check references `post_process_job_overlap` but the config field is likely `allow_post_processing`

**File:** `horde_worker_regen/process_management/process_manager.py` (property
`post_process_job_overlap_allowed`)

```python
@property
def post_process_job_overlap_allowed(self) -> bool:
    bd = self._runtime_config.bridge_data
    return (bd.moderate_performance_mode or bd.high_performance_mode) and bd.post_process_job_overlap
```

The config field accessed is `post_process_job_overlap`, but the documentation
refers to `allow_post_processing`. Verify that these are the same field (or
different fields with confusingly similar names). If they are different, one may
be a bug.

---

## 7. `is_time_for_shutdown` calls `all()` on a potentially empty iterable

**File:** `horde_worker_regen/process_management/shutdown_manager.py`

```python
if all(
    inference_process.last_process_state == HordeProcessState.PROCESS_ENDING
    or inference_process.last_process_state == HordeProcessState.PROCESS_ENDED
    or inference_process.last_process_state == HordeProcessState.PROCESS_STARTING
    for inference_process in self._process_map.get_inference_processes()
):
    return True
```

Python's `all([])` returns `True`. If `get_inference_processes()` returns an
empty list (e.g., before any inference processes have been started), this guard
passes immediately. The upstream checks (no pending jobs, no in-progress jobs)
may make this harmless, but the implicit truthiness of an empty iterable is a
readability trap.

**Suggested fix:** Add an explicit `if not inference_processes: return False`
guard, or add a comment explaining why the empty case is safe.

---

## 8. `_handle_consecutive_failures` returns `True` on first detection without waiting

**File:** `horde_worker_regen/process_management/job_popper.py`

When `consecutive_failed_jobs >= 3` is first detected, the method sets
`too_many_consecutive_failed_jobs = True`, records the time, and returns `True`
(skip pop). On the _next_ call (1 second later), the first check
`cur_time - too_many_consecutive_failed_jobs_time > 180` will be False (only ~1
second elapsed), so it returns `True` again. This is correct behavior for a
1-second poll loop. However, the logic is spread across two conditionals
(`too_many_consecutive_failed_jobs` first, then `consecutive_failed_jobs >= 3`)
which makes the flow harder to follow than a single state-machine approach.

---

## 9. `safety_orchestrator.py` has unreachable code after `critical_fault` return

**File:** `horde_worker_regen/process_management/safety_orchestrator.py`

After the `if critical_fault:` block returns early, the subsequent
`if completed_job_info.sdk_api_job_info.id_ is None: raise ValueError` checks
are unreachable for the critical-fault path. The redundant `None` checks after
the early return appear to be defensive leftovers. They are harmless but
misleading.

---

## 10. `fake_worker_processes.py` is referenced in docs but path is uncertain

**File:** Referenced in `architecture.md` as `fake_worker_processes`

The module `fake_worker_processes.py` exists at
`horde_worker_regen/process_management/fake_worker_processes.py` but its
contents and interface are not documented. The dry-run documentation would
benefit from a brief explanation of what this module provides.

---

## 11. `allow_post_processing` vs `post_process_job_overlap` naming confusion

**Files:**

- `horde_worker_regen/bridge_data/data_model.py`
- `horde_worker_regen/process_management/process_manager.py`

Two similarly-named boolean fields control different things:

| Field                      | Source                                | Purpose                                                                                                  |
| -------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `allow_post_processing`    | `CombinedHordeBridgeData` (inherited) | Advertised to the API at pop time; tells the horde this worker accepts post-processing jobs             |
| `post_process_job_overlap` | `reGenBridgeData`                     | Controls whether a new inference job can start while the previous job's post-processing is still running |

The names are confusingly similar and the distinction is easy to miss when
reading the code. Consider renaming `post_process_job_overlap` to
`overlap_inference_with_post_processing` or similar.

---

## 12. `SIGTERM` is never registered despite docstrings claiming it is

> **✅ Resolved (2026-06-12).** `SIGTERM` is now registered alongside `SIGINT` in
> both the main process (`process_manager.py:1130`) and child processes
> (`horde_process.py:277`). `docker stop` / `systemd` / `kill` now enter the
> graceful-shutdown path. Retained here for history; the original report follows.

**Files:**

- `horde_worker_regen/process_management/process_manager.py`
- `horde_worker_regen/process_management/shutdown_manager.py`
- `horde_worker_regen/process_management/horde_process.py`

Both `HordeWorkerProcessManager.signal_handler` and
`ShutdownManager.signal_handler` are documented as
`"Handle SIGINT and SIGTERM."`, but only `SIGINT` is ever installed:

```python
signal.signal(signal.SIGINT, self.signal_handler)   # process_manager.py
signal.signal(signal.SIGINT, signal_handler)          # horde_process.py (children)
```

There is no `signal.signal(signal.SIGTERM, ...)` anywhere. As a result, a
`SIGTERM` (the default signal sent by `docker stop`, `systemd`, `kill`, etc.)
bypasses the graceful-shutdown path entirely and terminates the process with
Python's default behavior; in-progress jobs are not finalized.

**Suggested fix:** Register `SIGTERM` alongside `SIGINT` for both the main
process and child processes, or correct the docstrings if SIGTERM handling is
intentionally omitted.

> **Docs note:** `shutdown_and_faults.md` currently documents the _intended_
> behavior (SIGINT **and** SIGTERM initiate graceful shutdown), assuming this is
> fixed.

---

## 13. The `.abort` file is written and cleared but never read as a trigger

> **✅ Resolved (2026-06-12).** The control loop now watches for an
> externally-created `.abort` file each tick (`process_manager.py:898`) and aborts
> immediately when found, so `.abort` is a real external trigger as documented.
> Retained here for history; the original report follows.

**Files:**

- `horde_worker_regen/process_management/shutdown_manager.py`
- `horde_worker_regen/run_worker.py`

`ShutdownManager.abort()` writes an empty `.abort` file, and `init()` in
`run_worker.py` removes a stale `.abort` on startup:

```python
with logger.catch(), open(".abort", "w") as f:   # shutdown_manager.py
    f.write("")
...
if os.path.exists(".abort"):                       # run_worker.py (startup only)
    os.remove(".abort")
```

Nothing reads `.abort` at runtime. Creating the file externally therefore has
**no effect**; it is only ever a breadcrumb the worker writes when it aborts
itself, then clears on the next start. This contradicts the natural reading of
`.abort` as an external "please abort" trigger for process managers that cannot
send signals easily.

**Suggested fix:** Decide the intended contract and make the code match it:
either add a runtime check for an externally-created `.abort` file (e.g. in the
control loop), or drop the file write if it is not meant to be observable.

> **Docs note:** `shutdown_and_faults.md` currently documents `.abort` as an
> external abort trigger (the apparent intent), assuming a runtime watcher is
> added.

---

## 14. `max_download_processes` is vestigial

**File:** `horde_worker_regen/process_management/process_manager.py`

`max_download_processes` (default `1`) is summed into `num_total_processes`:

```python
def num_total_processes(self) -> int:
    return self.max_inference_processes + self.max_safety_processes + self.max_download_processes
```

However, there is no download process type (`HordeProcessType` has only
`INFERENCE` and `SAFETY`), and no code path starts a download process (only
`start_inference_processes` and `start_safety_processes` exist).
`num_total_processes` itself also appears to have no callers. The parameter
therefore has no runtime effect and inflates an unused total, likely reflecting
a planned-but-unimplemented (or removed) feature.

**Suggested fix:** Remove `max_download_processes` (and `num_total_processes` if
confirmed unused), or implement download processes if that was the original
intent.

---

_Document generated 2026-06-11 during docs/ QA pass; reviewed 2026-06-12 (issues
12 and 13 since resolved)._
