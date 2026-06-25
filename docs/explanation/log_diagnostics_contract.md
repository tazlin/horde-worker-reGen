# Log diagnostics contract

- [Log diagnostics contract](#log-diagnostics-contract)
    - [The three layers and the seams between them](#the-three-layers-and-the-seams-between-them)
    - [One facade, two front-ends](#one-facade-two-front-ends)
    - [The TUI renders findings generically](#the-tui-renders-findings-generically)
    - [The logging-to-detector contract test](#the-logging-to-detector-contract-test)
    - [Adding a detector](#adding-a-detector)
    - [See also](#see-also)

The worker turns a heap of append-across-restarts logs into plain-language, actionable findings:
"the horde forced you into maintenance because you dropped 8 jobs", not "grep for `WorkerMaintenance`".
That capability spans three pieces of code that are easy to let drift apart, because nothing in the
type system ties them together:

1. **The logging layer** (`process_management/`): the worker emits a line like
   `Failed to pop job (Maintenance Mode): ...`.
2. **The detector layer** (`horde_worker_regen/analysis/detectors.py`): a regex recognizes that line
   and produces a [`Finding`][horde_worker_regen.analysis.detectors.Finding].
3. **The presentation layer**: the [`horde-log`](../reference/cli.md#horde-log) CLI and the dashboard's
   **Diagnostics** tab show those findings.

A reworded log line can silently retire a detector; a new detector can silently fail to appear in the
dashboard. This page describes the design choices and the one test that keep those failure modes from
happening quietly.

## The three layers and the seams between them

There are two seams, and they are deliberately treated differently:

| Seam | Risk | How it is contained |
| ---- | ---- | ------------------- |
| logging &harr; detector | A reworded emit stops matching a detector's regex; the detector goes dead with no error | A [contract test](#the-logging-to-detector-contract-test) pins each detector to a representative real log line |
| detector &harr; presentation | A new detector is added but the dashboard does not know to show it | The TUI [renders findings generically](#the-tui-renders-findings-generically), so it shows whatever the detectors produce |

The logging layer itself is intentionally left untouched by the diagnostics code: the worker logs for
operators first, and the detectors adapt to it, not the other way around. That keeps log messages free
to read naturally instead of being constrained to a machine-parseable schema.

## One facade, two front-ends

Both front-ends call the same entry point,
[`diagnose()`][horde_worker_regen.analysis.diagnose.diagnose], which loads a log path, segments it into
per-launch sessions, and runs every detector:

```
diagnose(path) -> list[SessionDiagnosis]   # each: a WorkerSession + its ranked Findings
```

The CLI's `diagnose` subcommand renders the result as text or JSON; the Diagnostics tab renders it as
panels. Neither contains its own copy of the "load, segment, run detectors" orchestration, so the two
can never disagree about what a log says. This is also why the tab does not shell out to `horde-log`:
it calls the facade directly, in-process, on a background thread.

## The TUI renders findings generically

The Diagnostics tab does **not** have per-incident display code. It iterates a `Finding`'s fields
(`severity`, `title`, `verdict`, `evidence`, `remediation`, `see_also`) and renders them the same way
regardless of which detector produced them, exactly as the Insights tab renders its recommendations.
The only thing the tab knows about the analysis layer is the *shape* of `Finding` and the `Severity`
enum used to colour the badge.

The practical consequence: **a new detector appears in the dashboard with no change to the TUI.** The
detector &harr; presentation seam has no per-detector surface to maintain.

## The logging-to-detector contract test

`tests/analysis/test_detector_contract.py` is the guard for the other seam. It holds one *golden* log
line per detector (reusing the line builders in `tests/analysis/test_detectors.py`, which mirror the
real worker emits) and asserts two things:

- **Each detector fires on its golden line** at the expected severity. If the worker rewords an emit so
  a detector's regex no longer matches, that detector's golden line stops firing and this test names
  exactly which detector broke.
- **Every detector has a fixture** (the no-orphan guard): the set of functions in
  [`DETECTORS`][horde_worker_regen.analysis.detectors.DETECTORS] must equal the set of contract
  fixtures. Adding a detector without recording its log signature fails the suite.

This test earned its keep on its first run: it found that the in-progress orphan watchdog's emit
(`...punting it so the queue can drain (orphaned-job watchdog).`) did not match the orphan detector's
regex, so `detect_orphan_wedge` had never been able to fire on a real log. Pinning the detector to the
real emit forced the regex to be corrected.

The contract test also leans on a naming convention: a detector named `detect_X` emits a finding whose
`id` is `X`. Keeping the id derivable from the function name (rather than a second hand-maintained
string) is what lets the contract test compute the expected id and the no-orphan guard compare sets.

## Adding a detector

The contract reduces the work of a new incident class to three touches, all in the analysis layer:

1. Write `detect_<name>(context) -> list[Finding]` in `detectors.py`, returning a finding with `id`
   `<name>`.
2. Add it to the `DETECTORS` list.
3. Add a golden-line fixture for it in `test_detector_contract.py`.

The CLI and the Diagnostics tab pick it up with no changes. The third step is the single manual
duplication the design accepts, and the no-orphan guard makes forgetting it a hard failure rather than
a silent gap.

## See also

- [Command-line reference: `horde-log`](../reference/cli.md#horde-log)
- [Logs](../reference/logs.md)
- [Resilience and recovery](resilience_and_recovery.md): the incidents most detectors recognize.
