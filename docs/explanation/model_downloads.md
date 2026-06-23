# Model Downloads and Availability

- [Model Downloads and Availability](#model-downloads-and-availability)
    - [Two kinds of "download"](#two-kinds-of-download)
    - [The dedicated download process](#the-dedicated-download-process)
    - [Model availability and the pop gate](#model-availability-and-the-pop-gate)
    - [Planning: what a config implies for disk](#planning-what-a-config-implies-for-disk)
    - [Pause and bandwidth controls](#pause-and-bandwidth-controls)
    - [Parallel downloads by host](#parallel-downloads-by-host)
    - [Download-only mode and the model picker](#download-only-mode-and-the-model-picker)
    - [Standalone download CLI](#standalone-download-cli)
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

[`download_process.py`][horde_worker_regen.process_management.download_process]
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

## Model availability and the pop gate

[`ModelAvailability`][horde_worker_regen.process_management.model_availability.ModelAvailability]
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
downloading. Once the first authoritative scan completes, the manager kicks off
the initial download of any configured-but-missing models.

The configured image-model set is **authoritative on every config reload**, in
both directions. Adding a model fetches it in the background without a restart;
removing one prunes it from the download queue and aborts it if it is the
in-flight download (`request_downloads(..., desired_image_models=...)` carries the
whole desired set, which the download process reconciles against its pending and
current work). This only stops *in-progress* and *queued* fetches: weights already
fully on disk are kept (removing a model from config has never deleted weights;
that remains the job of the purge / `only_models_on_disk` controls). Required
safety and auxiliary downloads are never pruned by a model removal.

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
([`download_scheduler.py`][horde_worker_regen.process_management.download_scheduler]):
every pending download (generation checkpoints *and* the aux models: CLIP/BLIP,
controlnets, post-processors) is tagged with the hostname of its download URL, and
the scheduler admits work under two live limits:

- `download_max_parallel_downloads` (default 4): the global ceiling across all
  hosts. Set it to 1 to restore fully-sequential downloading.
- `download_per_host_concurrency` (default 1): how many downloads to the *same*
  host may run at once. Left at 1, a single host is never hit by more than one
  download; raise it to also parallelize within a host (useful when many aux models
  share one host, such as the R2 mirror below).

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
two on the same manager never corrupt that shared state; the ControlNet annotators,
which need a full ComfyUI init, run **exclusively** (no other download alongside
them). Second, a per-file fetch (image or aux) that fails transiently is re-queued a
bounded number of times rather than abandoned until the next config reload, and a
recorded failure is cleared once a later attempt for that model succeeds.

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
download. The set is cataloged in `horde_model_reference.annotator_catalog`.

## See also

- [Add custom models](../how-to/add-custom-models.md): configuring extra models
- [Frontend and durable state](frontend_and_state.md): the TUI Downloads tab and
  supervisor control channel
- [Performance and Backpressure](performance_and_backpressure.md): how popping is
  gated, including by model availability
- [`ModelAvailability`][horde_worker_regen.process_management.model_availability.ModelAvailability]
