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
   and produces a [`Finding`][horde_worker_regen.analysis.finding_kinds.Finding] of a declared
   [`FindingKind`][horde_worker_regen.analysis.finding_kinds.FindingKind].
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

Detector signatures are anchored to emitted lifecycle events, not merely to distinctive words that may
also appear inside serialized payloads, configuration dumps, or exception context. For example, a
post-processing stall detector accepts the process-state event and the stable post-processing job event
forms; a payload field containing the same state name is evidence only when an actual event line accompanies
it. Negative contract rows preserve that boundary.

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

The Diagnostics tab does **not** have per-incident display code. It reads a `Finding`'s `severity`,
`title`, `verdict`, `evidence`, `remediation` and `see_also` and renders them the same way regardless of
which detector produced them, exactly as the Insights tab renders its recommendations. Three of those
are properties resolved against the finding's kind, so the tab needs no knowledge of the spec table: the
only thing it knows about the analysis layer is the *shape* of `Finding` and the `Severity` enum used to
colour the badge.

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

## The log lines are a contract, and the registry is where it lives

Every regex the analysis package uses to read a worker log line is registered in
`horde_worker_regen/analysis/log_signatures.py`, with the `module:function` that emits the line and a
literal sample copied from a real log. **Before changing a log message in
`horde_worker_regen/process_management/`, grep the registry for it**; if it is registered, update the
pattern and the sample in the same change and re-run the analysis contract tests
(`uv run pytest tests/analysis/test_log_signatures.py` and
`uv run pytest -m slow tests/analysis/test_log_contract_dry_run.py`).

Three tests hold the two ends of that contract:

- **The literal pin** (`test_log_signatures.py`): every registered pattern still matches its recorded
  sample, and no pattern in the lifecycle parser is missing from the registry, so a new parser cannot
  be added without a sample and an emitter.
- **The live half** (`test_log_contract_dry_run.py`, marked `slow`): the worker is run through the
  dry-run harness, and every pattern the run can exercise must match a line that run actually wrote.
  This is the test that goes red when an emitting f-string is reworded. A line a dry run structurally
  cannot produce (a fault, the submit success line under `dry_run_skip_api`) carries a `dry_run_reason`
  in the registry and is covered by the literal pin alone.
- **The catalogue guard** (`test_log_findings_doc.py`): every finding id a detector constructs appears
  in [Log findings](../reference/log_findings.md).

## Detectors read the shared lifecycle model, not the raw lines

Job-shaped facts come from one parse per session:
[`job_lifecycle_for(context)`][horde_worker_regen.analysis.job_lifecycle.job_lifecycle_for], cached on
the `SessionContext`. It carries per-job records (pop, dispatch, generation, safety, submit, fault, and
the batch that joined several horde jobs to one dispatch), the process-to-card map with its re-spawn
history, the card count and intake budget, the parsed status prints, the line-skip census, the
model-movement counts, and the parent's IPC drain timestamps; on top of those sit the derived views (the
three wait segments with their medians and p90s, the time-weighted sampling-concurrency histogram, the
seating census).

**A detector that needs any of those reads the model rather than regexing job lines of its own.** Three
reasons: the parse is linear in a session that can run to hundreds of thousands of lines and paying it
once matters; two detectors regexing the same line will drift apart; and a wait split into segments is
what stops a detector from attributing a delay to whatever stage it happens to measure. That
misattribution is not hypothetical: the drop-spiral finding used to report a long pop-to-submit latency
as "aged in the post-inference queue" and blame the safety stage, when on the bundle that motivated this
work the wait was almost entirely pop-to-dispatch.

## Adding a detector

The contract reduces the work of a new incident class to four touches, all in the analysis layer, and
the diagnosis is **declared before it is detected**:

1. Add a `FindingKind` member in `finding_kinds.py` whose value is the printed id `<name>`, and a
   `FindingSpec` for it in `FINDING_SPECS`: the catalogue title, the remediation that holds for every
   emit of the kind, and the `see_also` kind a reader should turn to next. Both cross-references and
   catalogue coverage are checked against this table, so a link that goes nowhere fails a test rather
   than reaching an operator.
2. Write `detect_<name>(context) -> list[Finding]` in `detectors.py`, returning a finding with
   `kind=FindingKind.<NAME>`. Read job-shaped facts through `job_lifecycle_for(context)`; if you need a
   log line the model does not carry, register its pattern in `log_signatures.py` first.
3. Add it to the `DETECTORS` list.
4. Add a golden-line fixture for it in `test_detector_contract.py`, and an entry in
   [Log findings](../reference/log_findings.md).

What the detector writes at the emit site is only what the session produced: the `verdict` (which
narrates the measurement and is never templated), the evidence, the severity, and — where the fix
genuinely depends on what was measured — a `remediation_addendum` appended to the spec's advice. A kind
whose severities read as different incidents may pass a `title_override`; the spec's title stays the
catalogue name. Where every emit words its own fix, the spec's `remediation` is empty and the addendum
carries it.

The CLI and the Diagnostics tab pick it up with no changes. Step 4's two halves are the manual
duplication the design accepts, and the no-orphan guard and the catalogue guard make forgetting either
a hard failure rather than a silent gap.

Thresholds derive from the session's own shape (its card count, its intake budget, its status cadence,
its dispatch count), never from a constant that encodes one machine: the same detector has to be right
on a one-card volunteer host and a sixteen-lane fleet.

## See also

- [Command-line reference: `horde-log`](../reference/cli.md#horde-log)
- [Log findings](../reference/log_findings.md): every finding id, its trigger, and its remedy.
- [Logs](../reference/logs.md)
- [Resilience and recovery](resilience_and_recovery.md): the incidents most detectors recognize.
