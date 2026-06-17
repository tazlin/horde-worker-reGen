# Model Downloads and Availability

- [Model Downloads and Availability](#model-downloads-and-availability)
    - [Two kinds of "download"](#two-kinds-of-download)
    - [The dedicated download process](#the-dedicated-download-process)
    - [Model availability and the pop gate](#model-availability-and-the-pop-gate)
    - [Planning: what a config implies for disk](#planning-what-a-config-implies-for-disk)
    - [Pause and bandwidth controls](#pause-and-bandwidth-controls)
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
- **`DOWNLOADING`**: per-file progress (current download, queue, failures).

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

## Standalone download CLI

`download_models.py` is a standalone entry point to pre-fetch the configured
models without starting the worker, useful for provisioning a machine or a
container image ahead of time.

## See also

- [Add custom models](../how-to/add-custom-models.md): configuring extra models
- [Frontend and durable state](frontend_and_state.md): the TUI Downloads tab and
  supervisor control channel
- [Performance and Backpressure](performance_and_backpressure.md): how popping is
  gated, including by model availability
- [`ModelAvailability`][horde_worker_regen.process_management.model_availability.ModelAvailability]
