# Install the worker

You do not need Python installed first; the installer fetches its own Python and PyTorch. It also needs
`git`: it uses an existing one on your PATH if present, fetches a portable copy on Windows if you have
none, and on Linux/macOS prompts you to install it (a one-line package install). Before installing, it
shows a notice of what it will download and from where, and asks you to confirm. Before you start, check
that your machine is supported on the
[project README](https://github.com/Haidra-Org/horde-worker-reGen#readme), and grab a free
[AI Horde API key](https://aihorde.net/register).

If this is your first time, the [Getting started tutorial](../tutorials/getting-started.md) walks
through install and first run end to end. This page is the reference for each install method.

## Windows

### Download and double-click (easiest)

No command line needed. Download
**[HordeWorker-Setup.exe](https://github.com/Haidra-Org/horde-worker-reGen/releases/latest/download/HordeWorker-Setup.exe)**,
double-click it, and click through the installer. It shows what it will install and the third-party
licenses and asks you to accept, then installs per-user (no administrator rights) and opens the dashboard
when it finishes. A **Start Menu** shortcut is created by default (untick it to skip it); a **desktop**
shortcut is offered as an unticked checkbox. The install folder also gets a browser-rendering
**`_Start_Here.html`** that explains what to run and what every file is for.

If Windows SmartScreen shows "Windows protected your PC", the installer is not code-signed yet. Click
**More info**, then **Run anyway**.

### Scripted install (advanced)

Prefer the command line, or want an unattended install? Paste the one-liner into PowerShell:

```powershell
irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
```

The `irm ... | iex` one-liner may trigger SmartScreen ("Windows protected your PC"). Click
**More info**, then **Run anyway**, the same step used to install tools like `uv`.

## Linux and macOS

Paste into a terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
```

On macOS this installs a CPU-only build, which is not practical for serving real jobs. For usable
performance you need an NVIDIA GPU on Windows or Linux, or AMD on Linux via
[ROCm](run-on-amd-rocm.md).

## What happens on first run

However you installed, the first run builds the environment (it pulls Python and PyTorch and can take
several minutes), then opens the **dashboard in your browser**. A short wizard collects your API key
and worker name, helps you pick models, optionally benchmarks your machine, and starts the worker. See
[Use the dashboard](use-the-dashboard.md) for the wizard and the tabs.

After you start, your chosen models download in the background. The first run can take 30 to 60
minutes depending on your selection and connection; the worker serves each model as it finishes, so
keep the window open. Re-run the same install command any time to [update](update-the-worker.md).

## What the scripted install does, and does not do

It installs the worker into a folder in your current directory, and puts the larger, reusable artifacts
(downloaded Python, the `uv` package cache, and your chosen models) in a **peered** sibling folder named
`<worker>-data` (for example `HordeWorker-data` next to `HordeWorker`). That data folder is preserved if
you ever delete or reinstall the worker folder, so a fresh install never re-downloads your models. It does
**not** change any system-wide settings. It shows a notice and asks before installing, and asks (default
No) before creating any shortcut.

- Accept the install notice without the prompt with `HORDE_WORKER_ASSUME_YES=1` (required when piped with
  no terminal, e.g. `curl -LsSf .../install.sh | HORDE_WORKER_ASSUME_YES=1 sh`).
- Create shortcuts without being asked with `HORDE_WORKER_SHORTCUTS=1`, or skip them entirely with
  `HORDE_WORKER_NO_SHORTCUTS`.
- Skip the auto-launch with `HORDE_WORKER_NO_LAUNCH`.
- Override the install location with `HORDE_WORKER_DIR`.
- Override the data folder (models, cache, Python) with `HORDE_WORKER_DATA_DIR`, for example to put it on
  another drive. By default it sits next to the worker folder as `<worker>-data`.

Keep the install path free of spaces. The one-liner installs into a `HordeWorker` (Windows) or
`horde-worker` (Linux/macOS) folder in your **current directory**, so `cd` to the drive you want
before running it.

## Disk space

This is a large install. Even before any models, the GPU (CUDA) environment needs roughly **10 to
15 GB** (the `.venv` alone is around 7 to 10 GB; PyTorch and its bundled NVIDIA CUDA libraries are the
bulk, and that floor is unavoidable on a GPU install). The CPU-only build is around 3 to 5 GB.

**Models are separate and much larger.** Each is around 2 to 8 GB, and a useful selection runs to tens
or hundreds of GB. The one-liner keeps everything on the drive you run it from, with the models (and the
uv cache and Python) in the peered `<worker>-data` folder; set `HORDE_WORKER_DATA_DIR` before installing
to place that folder on another drive.

## Manual install (git or zip)

Prefer to clone or download a zip?

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

No git? Download the
[latest zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip) and extract
it (use a path without spaces).

Then run `horde-worker.cmd` (Windows) or `./horde-worker.sh` (Linux/macOS): it installs dependencies on
first run and opens the dashboard. To run without a UI instead, use the non-interactive scripts: run
`update-runtime.cmd` to install, copy `bridgeData_template.yaml` to `bridgeData.yaml` and fill in your
details, then `horde-bridge.cmd`. See [Run headless](run-headless.md).

For a from-scratch virtualenv install (your own Python, manual `uv`/`pip`), see
[Choose a PyTorch build](choose-a-pytorch-build.md).

## Next steps

- [Configure the worker for your GPU](configure-for-your-gpu.md)
- [Use the dashboard](use-the-dashboard.md)
- [Run headless](run-headless.md)
- [Troubleshooting](troubleshoot.md)
