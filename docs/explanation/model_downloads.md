# Model Downloads and Availability

- [Model Downloads and Availability](#model-downloads-and-availability)
    - [Two kinds of "download"](#two-kinds-of-download)
    - [The dedicated download process](#the-dedicated-download-process)
    - [Pop-time auxiliary prefetch](#pop-time-auxiliary-prefetch)
    - [Model availability and the pop gate](#model-availability-and-the-pop-gate)
    - [Feature readiness: deps plus models on disk](#feature-readiness-deps-plus-models-on-disk)
    - [Planning: what a config implies for disk](#planning-what-a-config-implies-for-disk)
    - [Pause and bandwidth controls](#pause-and-bandwidth-controls)
    - [Parallel downloads by host](#parallel-downloads-by-host)
    - [Segmented downloads (connections per file)](#segmented-downloads-connections-per-file)
    - [Download-only mode and the model picker](#download-only-mode-and-the-model-picker)
    - [Standalone download CLI](#standalone-download-cli)
    - [The gated R2 mirror for auxiliary models](#the-gated-r2-mirror-for-auxiliary-models)
    - [Benchmark downloads: one coherent picture](#benchmark-downloads-one-coherent-picture)
    - [See also](#see-also)

Model weights are large and slow to fetch, and a worker should be able to start
serving the models it *already* has while the rest download in the background.
This page describes how the worker downloads models, tracks what is on disk, and
keeps popping aligned with availability.

## Two kinds of "download"

The worker downloads two distinct things, owned by different places:

- **The model reference**: the JSON model database (which models exist, their
  baselines, files, and checksums). The **parent process** owns all reference
  downloading and writes the converted files to disk;
  [`reference_helper`][horde_worker_regen.reference_helper] hands every subprocess
  an *offline* (read-only, never-download) reference manager so a child can never
  trigger a network fetch (which would otherwise be possible under a `fork` start
  method). All canonical on-disk knowledge (the weights root, per-category
  folders, component routing) lives in `horde_model_reference.on_disk_layout`,
  not in a worker-local bridge.
- **The model weights**: the actual checkpoints. These are fetched by a dedicated
  background download process (below).

## The dedicated download process

[`download_process.py`][horde_worker_regen.process_management.workers.download_process]
runs a hordelib `SharedModelManager` **without** a full ComfyUI init (listing and
downloading checkpoints needs only the model managers, not the inference stack).
It reports a rich, labelled status so the TUI and console can show exactly when,
how, where, and why models download, moving through phases:

- **`INITIALIZING`**: `load_model_managers()` reads the (already-downloaded)
  reference from disk; retried with backoff on failure.
- **`SCANNING`**: the first on-disk scan, which may SHA256-hash large files, so it
  never looks hung.
- **`DOWNLOADING`**: live per-file progress for every download in flight (several
  can run at once, one per source host), plus the queue and any failures.

This process lives **outside** the main `ProcessMap`: it serves no jobs and must
not be swept up by the inference/safety hung-process logic. A missing or failed
download process can never wedge startup: the worker can still run inference on
whatever is already present. It is started and stopped by the
`ProcessLifecycleManager` (`start_download_process` / `end_download_process`).

Every per-file fetch (image **and** auxiliary) goes through the same validated
download in
[`model_download_core`][horde_worker_regen.model_download_core]: after the fetch it
verifies the on-disk file (sha256 when the record carries a checksum, otherwise a
presence check) and re-downloads once on a mismatch, via
[`ensure_aux_model_present`][horde_worker_regen.model_download_core.ensure_aux_model_present].
Previously only image checkpoints were validated and the auxiliary path reported
success unconditionally, so a truncated ControlNet/post-processing file was trusted
and a job that used it would fault.

The ControlNet annotators are **first-class models**, not an opaque side-channel.
`horde_model_reference` models each detector checkpoint as a `controlnet_annotator`
record and hordelib exposes a `controlnet_annotator` model manager, so the worker
fetches each annotator file as its own per-file aux download: it gets the same size,
progress, checksum verification, and on-disk presence reporting as every other model,
instead of a single opaque "annotators" line that froze through a full ComfyUI init.

Once the files are present, a single **exclusive** `ANNOTATOR_VERIFY` task runs each
preprocessor once (a ComfyUI init) to confirm they actually load. That verify is the
one place the otherwise-offline download process boots a full ComfyUI/torch/CUDA
stack, so it is **gated on a persistent marker**: hordelib records a marker (keyed to
the pinned `comfyui_controlnet_aux` commit) once every preprocessor has run, and the
worker reads it *before* booting (the read needs neither `hordelib.initialise` nor a
GPU). A warm marker means a prior session already verified this pin, so the verify is
skipped outright and no boot is paid. The verify therefore runs only when it is
genuinely due: a fresh install, or an annotator pin bump. (File integrity across
sessions is covered separately by the per-file `.sha256` sidecars, so a warm marker
never trusts changed bytes: a corrupt file fails its checksum and withholds the
feature before the verify is ever considered.) A verify failure is
recovered, not ignored: the detector checkpoints are re-downloaded once (a corrupt
file is the likely cause) and re-verified; if it **still** fails, ControlNet is
disabled for the session and the operator is notified with a remediation hint, rather
than leaving the worker to fault every ControlNet job. Because that verify can wedge
inside hordelib, the
[scheduler][horde_worker_regen.process_management.models.download_scheduler.HostAwareDownloadScheduler]
bounds how long an exclusive task may block the queue: past the bound it relaxes
exclusivity so the image-model downloads a worker needs to serve jobs proceed, while
the stuck task harmlessly keeps running. The relaxation is logged once.

(A hordelib without the first-class `controlnet_annotator` manager falls back to the
legacy single exclusive preload that both fetches and verifies in one ComfyUI init.)

## Pop-time auxiliary prefetch

A job's LoRAs and textual inversions are network fetches that must land before the
job can sample. Running them on an inference lane occupies a GPU-feeding process for
the length of the transfer, so the worker instead prefetches them off the inference
lanes: at pop the parent's
[`AuxPrefetchCoordinator`][horde_worker_regen.process_management.models.aux_prefetch_coordinator.AuxPrefetchCoordinator]
computes the job's not-yet-cached auxiliary files and asks the dedicated download
process to place them on disk while the job stays `PENDING_INFERENCE`. Each file is
fetched at most once even when several jobs share it, an already-present file
short-circuits without a CivitAI round-trip, and the request carries an eviction-pin
set (every auxiliary file any tracked job still references) so a file one pending job
needs is not evicted before that job dispatches.

The prefetch pipeline is the only preparation path. Until it clears a job's gate the job
is invisible to both dispatch selection and model preload: it is never chosen as the
dispatch head, never preloaded or staged, and holds no lane and no VRAM reservation, so
a fitting sibling behind it is preloaded and sampled instead of idling. Before the
enforced disk floor a missing LoRA is fetched, the download process constrains the shared
ad-hoc cache to its free-space floor, evicting least-recently-used ad-hoc LoRAs through the
manager's own pin-aware eviction so a file another pending job still references is never
evicted to make room.

The download process reports a per-file outcome back to the parent. A success marks
the session auxiliary cache and, once every LoRA and textual inversion a pending job
needs is present, clears that job's dispatch gate. Failure handling is kind-agnostic
across LoRAs and textual inversions. A terminal *rejection* (a file the fetch API
permanently refuses: a LoRA that is invalid, too large, NSFW on an SFW-only worker, or
definitively not found upstream; a textual inversion the API rejects or reports missing)
is recorded as skipped so the job's auxiliary set counts as ready without it and the job
dispatches rather than faulting, and a later job referencing the same file is neither
re-requested nor re-faulted. A rejection arms no backoff (it is a property of that one
file, not a sick download path). The fetch layer surfaces a definitive upstream not-found
(a metadata `401`/`404`) as its own terminal reason, distinct from a transient outage
(an exhausted retry budget), so a reference that can never exist is skipped once rather
than retried into a fault storm.

A plain *failure* (a transient transfer error) faults the affected pending job promptly
(it holds no lane or reservation, so nothing in flight is disturbed) and arms that file's
per-class download backoff, exactly as an inference-side aux failure does: once a class
incident is active a fresh failure of that class is classified terminal rather than
requeued into the same failing path. When a plain failure is classified *terminal* (out
of retries, or arriving while its class incident is active), the coordinator also memoizes
the reference as skipped: retrying it is futile, so co-queued and later jobs referencing
the same file dispatch without it rather than each faulting in turn. This bounds a single
doomed reference to at most one terminal fault per incident even when the root cause could
not classify it as a rejection. Unlike a surfaced rejection, whose skip is permanent (the
file can never become usable), a plain-failure skip is incident-scoped: it lapses with the
backoff's decay window, so a reference that merely failed during an outage is retried once
the incident passes instead of being silently omitted for the rest of the session.

Each auxiliary class holds its own escalating download backoff. The LoRA backoff also
gates pop-time LoRA advertising, because the pop request carries an `allow_lora`
capability flag the worker can withhold. The pop request has no per-request
textual-inversion capability flag (no textual-inversion analogue to `allow_lora`), so
the textual-inversion backoff influences fault classification only: it cannot suppress
textual-inversion traffic at the pop, and the popper's advertising logic gates on the
LoRA backoff alone.

The worker-wide LoRA-advertising suppression (`background_download_active`, which stops
new LoRA pops while the download subsystem is transferring) is scoped away from this
pipeline: it counts only non-prefetch downloads (bulk/default seeding, image and aux
model fetches), never the job-driven ad-hoc LoRA/TI prefetches. Counting a prefetch here
would suppress the very LoRA pops the prefetch exists to make dispatchable, self-serializing
LoRA intake the moment one job's file began downloading. The disk-floor and strike-backoff
suppressions are separate mechanisms and are unaffected by this scoping.

Two mechanisms keep the pipeline live. A per-job deadline derived from the configured
download timeout, checked in the periodic scheduling scan, faults a job whose prefetch
never resolves, so a job is never left pending forever: this is the backstop. The
deadline is a backstop, not the primary detector (the download process reports a real
failure through the outcome path), so it defers rather than punishing a slow-but-alive
transfer: when a deadline expires while the downloader still shows the job's file in
flight, the coordinator extends that job's deadline by one download-timeout budget
instead of faulting. Deferral is bounded at two extra budgets (three total) so a wedged
transfer still faults, and short-circuits earlier when the download reports byte progress
that stops advancing between expiries (a file reporting no bytes cannot show progress, so
it defers up to the cap). That byte-stall check compares a file's reported count across the
consecutive expiries of the job waiting on it, and the remembered count is keyed by that
live deadline rather than by whatever the downloader happens to report on any single tick:
a tick on which the in-flight view momentarily reports nothing can no longer wipe the memory
and reset the file to "progressing by default", so frozen bytes defer at most once. A dead
download process makes the in-flight view empty outright (a process that has exited cannot be
advancing any transfer), so its frozen final snapshot never defers a deadline against a
corpse and the job faults on its first budget; the parent's
[download-process liveness sweep](resilience_and_recovery.md#the-background-download-process)
then restarts the downloader and re-requests the pending job. The first deferral per job is
logged once; later deferrals of the same job stay silent. The download process itself keeps a slow-but-alive transfer
running well past any per-job deadline (its own fetch bound exists only to reclaim the
executor slot from an implausibly long transfer): abandoning it would discard the
progress, and the completed file serves the faulted job's retry, and every later job,
from cache. In the same
periodic step a reconcile sweep re-requests any pending auxiliary-bearing job that has no
request in flight (a retryable-failure requeue, a lost result message, or a download-process
restart that dropped its in-flight map) and marks a job prepared outright when its files are
already cached: this is the liveness that heals. The tracker's retry policy bounds the total
attempts, so a permanently failing job still faults rather than looping. Should an inference
child later fail to resolve a prefetched file on disk (a raced eviction or disk error), its
retryable fault withdraws the job's prepared state and forgets the contradicted session-cache
entries, so the reconcile sweep re-requests the files instead of the scheduler re-dispatching
the job into the same failure. The coordinator also
pushes a pins-only eviction update whenever the set of still-referenced auxiliary files
changes (for example when a job completes), coalesced so an unchanged set is never re-sent.

## Model availability and the pop gate

[`ModelAvailability`][horde_worker_regen.process_management.models.model_availability.ModelAvailability]
holds the set of image models currently present on disk plus the live download
status. It is single-writer (the message dispatcher, on download-process reports)
and many-reader (the job popper, process lifecycle, snapshot builder).

The present-set is `None` until the download process makes its first report.
While unknown, readers treat **every** configured model as present, preserving the
legacy behaviour of workers that pre-download everything and run without a
download process (tests, harness, dry-run). `scan_complete` distinguishes the
authoritative post-scan reports from early partial ones.

The popper uses this to keep advertised models aligned with reality:
`_select_models_for_pop` filters the pop request down to models actually present
(`filter_present`), so the worker never accepts a job for a model still
downloading. Once the first authoritative scan completes, `ModelDownloadCoordinator` kicks off
the initial download of any configured-but-missing models and lazily starts
inference/safety processes when their model gates clear.

[`ModelDownloadCoordinator`][horde_worker_regen.process_management.models.download_coordinator.ModelDownloadCoordinator]
owns this parent-side reconciliation and startup gating. The configured
image-model set is **authoritative on every config reload**, in
both directions. Adding a model fetches it in the background without a restart;
removing one prunes it from the download queue and aborts it if it is the
in-flight download (`request_downloads(..., desired_image_models=...)` carries the
whole desired set, which the download process reconciles against its pending and
current work). This only stops *in-progress* and *queued* fetches: weights already
fully on disk are kept (removing a model from config has never deleted weights;
that remains the job of the purge / `only_models_on_disk` controls). Required
safety and auxiliary downloads are never pruned by a model removal.

## Feature readiness: deps plus models on disk

Image models are not the only thing a worker advertises. ControlNet,
SDXL-ControlNet, and post-processing each need **two** things before the worker
can actually serve a job that asks for them: the Python packages that back them
(`onnxruntime` for the ControlNet annotators, `rembg` for background removal), and
the models/annotators themselves on disk. Advertising a feature before either is
in place only earns a job the worker will fault.

The two halves are tracked separately and fused parent-side:

- The **deps** half is enforced at config load by
  [`coerce_bridge_data_to_capabilities`][horde_worker_regen.capabilities.coerce_bridge_data_to_capabilities],
  which turns the opt-in flag off (with a loud, actionable hint) when the packages
  are missing. So by the time a flag is still on, the deps are present.
- The **presence** half is reported up from the download process, the only
  torch-free authority on what is on disk. It probes the loaded managers
  (the same per-model availability the aux download builder uses, including the
  `controlnet_annotator` manager's per-record on-disk presence) and reports a
  tri-state per feature: present, not-yet-present, or unknown.

[`feature_readiness.py`][horde_worker_regen.process_management.models.feature_readiness]
is a pure function that combines these into a per-feature state: `offered`,
`waiting` (enabled, deps present, models still downloading), `missing_deps`,
`disabled`, or `failed`. The job popper withholds a gated feature from the pop
request until it is `offered`, so the worker never advertises a capability whose aux
downloads are still in flight. Mirroring image-model availability, an **unknown**
presence (no download process, or no report yet) never withholds a feature, so a
worker that pre-downloaded everything keeps its long-standing behaviour.

`failed` is distinct from `waiting`: it is the terminal state for a feature whose
models are present but cannot run (the annotator verify above), so it withholds
ControlNet until the operator restarts, rather than implying it will recover on its
own. `waiting` clears itself once a download lands; `failed` does not.

The same readiness drives the display, so the table can never disagree with what
the worker advertises: the Downloads tab shows the full per-feature table
(with the install hint when deps are missing), and the Overview health panel
carries a compact one-line summary of the engaged features. Ad-hoc LoRA and textual
inversion prefetch (fetched per job by the download process, see
[pop-time auxiliary prefetch](#pop-time-auxiliary-prefetch)) and the safety models
keep their own gating and appear as read-only rows.

## Planning: what a config implies for disk

[`model_download_plan.py`][horde_worker_regen.model_download_plan] answers, without
importing torch/ComfyUI, the questions the TUI, console, and model picker need:
which configured models are already on disk, how much disk the configuration will
consume, and whether the target volume can hold it. Sizes come from the model
record itself, and all on-disk-layout knowledge is delegated to
`horde_model_reference.on_disk_layout` (there is no worker-local
category/folder/size bridge to keep in step with hordelib).

## Pause and bandwidth controls

Downloads honour live controls so an operator can throttle or pause fetching
without stopping the worker:

- `download_rate_limit_kbps` (bridge config) caps download bandwidth.
- The TUI's Downloads tab can pause/resume and adjust the rate live; these arrive
  as supervisor control messages and are applied mid-download by a dedicated
  control thread inside the download process (the worker loop is otherwise blocked
  inside the download).

`download_file` exposes a per-chunk progress callback but no native pause or
rate-limit; the worker implements both inside that callback (block while paused;
sleep to cap kB/s).

## Parallel downloads by host

The download process fetches several models at once rather than one at a time,
parallelizing across **distinct source hosts** (e.g. `civitai.com` ‖
`huggingface.co` ‖ the aux R2 mirror). A small executor-thread pool drains a
host-aware scheduler
([`download_scheduler.py`][horde_worker_regen.process_management.models.download_scheduler]):
every pending download (generation checkpoints *and* the aux models: CLIP/BLIP,
controlnets, post-processors) is tagged with the hostname of its download URL, and
the scheduler admits work under two live limits:

- `download_max_parallel_downloads` (default 4): the global ceiling across all
  hosts. Set it to 1 to restore fully-sequential downloading.
- `download_per_host_concurrency` (default 1): how many downloads to the *same*
  host may run at once. Left at 1, a single host is never hit by more than one
  download; raise it to also parallelize within a host (useful when many aux models
  share one host, such as the R2 mirror below).

Job-driven ad-hoc LoRA/TI prefetches are exempt from the per-host limit (they
neither wait for nor consume a host slot; the global ceiling still applies). A
pending job's dispatch gate is waiting on exactly that fetch, so it must not queue
behind an unrelated slow transfer to the same host, and the exemption cannot
stampede a host: the fetches are single-stream (the host limit exists to be polite
about multi-connection segmented downloads) and the ad-hoc engine's own worker pool
bounds them again downstream.

Both limits are honoured at startup and retuned live on config reload (raising the
global limit grows the executor-thread pool so the new ceiling is actually used; it
is never shrunk, so a later lower limit simply leaves threads idle). The shared
`download_rate_limit_kbps` cap is an **aggregate**: it is divided across the
downloads in flight, so N parallel downloads still respect it together. A download
removed from config mid-flight is dropped from the queue and aborted if running;
required safety and aux work is never aborted by an image-model removal. The
Downloads tab lists every concurrent download, grouped by host.

Two correctness invariants underpin the threading. First, the hordelib calls that
mutate one model manager's shared state (its on-disk model lists) are serialized
**per manager**, so two downloads on different managers run truly in parallel while
two on the same manager never corrupt that shared state; the annotator **verify**
(`ANNOTATOR_VERIFY`), which needs a full ComfyUI init, runs **exclusively** (no other
download alongside it), while the annotator *files* download per-file like any other
aux model. Second, a per-file fetch (image or aux) that fails transiently is re-queued a
bounded number of times rather than abandoned until the next config reload, and a
recorded failure is cleared once a later attempt for that model succeeds.

## Segmented downloads (connections per file)

Host-level parallelism speeds up fetching *several* files at once but does nothing
for the rate of a *single* large checkpoint, and that single-file rate is what an
operator with one big model and a fat link actually feels: a lone TCP stream to a
CDN is usually window/RTT-limited well below the link. So the engine can fetch one
large file over several concurrent **ranged** connections at once and reassemble it.

- `download_connections_per_file` (default 4; 1 = single stream) is the per-file
  connection count, honoured at startup and retuned live on config reload, and also
  surfaced in the Config tab. It is independent of the host limits above: it
  parallelizes the bytes *within* one file, while they parallelize *across* files.

The engine ([`download_engine.download_file`][] in `horde_model_reference`) treats
segmentation as a fast path with a guaranteed fallback. It first sends a one-byte
ranged probe: only a `206` with a known total over the segmentation threshold
(~64 MiB) is split into per-connection byte ranges written to a sparse `.spart` at
their offsets; anything else (a small file, or a server that answers the probe with
a `200` because it ignores `Range`) drops straight to the original single-stream,
resumable `.part` path. Segmented downloads do **not** resume an interruption (the
sparse partial is discarded and refetched), which is why the resumable single stream
is kept for the cases that benefit from it. The gated R2 mirror is left single-stream
(it is already the fast accelerator); only the origin fetch is segmented.

## Download-only mode and the model picker

An operator can fetch models from the TUI *without* committing the GPU. The
Downloads tab's **Download only** button (and the first-run start modal's "Download
models only" choice) starts the worker far enough to run the download process and
the supervisor channel, but holds inference, the safety process, and job popping in
a **download-only hold** (`SupervisorCommand.DOWNLOADS_ONLY_HOLD`). The worker keeps
reporting availability while it fetches; pressing **Go live**
(`SupervisorCommand.GO_LIVE`) clears the hold and the normal availability gate then
starts inference and popping. Going live mid-download is safe with no special casing:
the present-set pop gate never advertises a model that is still downloading, so a
half-fetched checkpoint is simply not offered until it lands.

**Choose models…** opens a picker
([`download_picker.py`][horde_worker_regen.tui.widgets.download_picker]) seeded from
the configured model set (the same resolution the Config tab shows), with the
not-yet-present models pre-selected, so its default is exactly "download what this
configuration is missing". The operator may narrow or broaden the selection and
optionally include the auxiliary models; confirming issues a
`SupervisorCommand.DOWNLOAD_MODELS` for the chosen names (entering the hold first
when the worker was not already running, so the picker can start a stopped worker
purely to download).

The picker does not fetch a parallel, throwaway list: the chosen names are folded
into the worker's *one* authoritative desired-on-disk set (the
[`DesiredState`][horde_worker_regen.process_management.models.desired_state] held by the
process manager, the union of the configured models and the operator's picker
additions). Every download trigger reconciles against that one set, so a config
reload no longer cancels a picker-added download. Removing a model from the desired
set only prunes its queued or in-flight download (it stops being fetched and
offered); on-disk files are left in place, since reclaiming disk is a separate,
explicit action. Picker additions are in-memory only: a model the picker fetched
stays on disk across a restart, but the desired set then reverts to whatever the
configuration resolves to.

## Standalone download CLI

`download_models.py` is a standalone entry point to pre-fetch the configured
models without starting the worker, useful for provisioning a machine or a
container image ahead of time.

## The gated R2 mirror for auxiliary models

Generation checkpoints are fetched from their origin hosts as always, but the
smaller *auxiliary* models (controlnets and annotators, CLIP/BLIP, the safety
checker, and post-processors like upscalers and face restorers) can come from a
project-hosted Cloudflare R2 mirror that is faster and free to serve. Access is
gated: the worker's configured `api_key` is exposed to the download path as the
`AIHORDE_API_KEY` environment variable (inherited by the download process), and
`hordelib` passes it to the model-reference download engine, which tries the
content-addressed mirror first.

This is purely an accelerator. If the key is anonymous or not eligible under the
mirror's policy, if a file is not mirrored, or if the mirror is unreachable, the
engine transparently falls back to the model's original host, so downloads never
depend on the mirror. The mirror and its eligibility policy are described in the
horde-model-reference docs (the gated R2 model mirror); nothing here needs
configuration beyond a real (non-anonymous) `api_key`.

The same path now also covers the **controlnet annotators** (the
`comfyui_controlnet_aux` detector weights). They used to be fetched directly from
HuggingFace by the annotator package on first use; hordelib now pre-fetches them
through this unified engine (gated mirror, then HuggingFace origin) into the exact
paths the package expects, so the detectors find them present and skip their own
download. They are a first-class `controlnet_annotator` model-reference category
(verified in `horde_model_reference.annotator_catalog`, bridged to records by
`annotator_records`), and hordelib registers an `AnnotatorModelProvider` so the set
is also queryable as `source="comfyui_controlnet_aux"` which is no longer an opaque
side-channel.

## Benchmark downloads: one coherent picture

The benchmark needs the same models on disk that the worker does, so its "Download
models" surface is folded into this subsystem rather than scanning disk on its own.

**What a feature needs is reckoned in full.** The benchmark plan accounts for every
file a selected level exercises, not just the image checkpoint: the ControlNet
*model checkpoints* for the control types it sweeps (canny, depth, openpose), the
ControlNet annotators, and the post-processing models (upscalers, face restorers).
Presence for each is resolved torch-free from the model reference (the same
`horde_model_reference` on-disk-layout helpers the worker uses, plus the annotator
catalog), so the dry-run preview never pays a cold inference-stack import and a
machine whose ControlNet files are only partly present is correctly told what is
still missing rather than that nothing is.

**A running worker is the source of truth.** When a worker is live, the benchmark's
download view reflects the worker's own snapshot: a model the worker reports present
reads present, and one it is actively fetching reads *downloading* (not "ready", and
not offered for a second, redundant fetch). The fetch itself is delegated to that
worker's download process (it keeps serving; a download takes no GPU), so there is
never a second downloader contending for the same files. With no worker running, the
benchmark fetches the missing models itself through the shared download core.

**Starting a benchmark is never a silent teardown.** A benchmark needs the GPU to
itself, so launching one stops a running worker and pauses its in-flight downloads.
When a worker is serving, the TUI asks first: an explicit confirm explains the
takeover, and cancelling leaves the worker serving untouched.

## See also

- [Add custom models](../how-to/add-custom-models.md): configuring extra models
- [Frontend and durable state](frontend_and_state.md): the TUI Downloads tab and
  supervisor control channel
- [Performance and Backpressure](performance_and_backpressure.md): how popping is
  gated, including by model availability
- [`ModelAvailability`][horde_worker_regen.process_management.models.model_availability.ModelAvailability]
