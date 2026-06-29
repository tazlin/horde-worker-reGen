# Install from scratch and choose a PyTorch build

Most people should use the [installer](install.md), which detects your GPU and picks the right PyTorch
build automatically. This page is for power users who manage their own virtualenv, or who need to
force a specific build (a particular CUDA version, CPU-only, or ROCm).

## Prerequisites

- Install [git](https://git-scm.com/).
- Install your GPU stack (CUDA or ROCm) if you have not already.
- Install Python 3.12. If you use the official Python installer and do not already use Python
  regularly, tick **Add python.exe to PATH** on the first screen.
- Configure at least 8 GB (preferably 16 GB+) of swap. This applies to Linux too.
- Clone the worker:

  ```bash
  git clone https://github.com/Haidra-Org/horde-worker-reGen.git
  cd horde-worker-reGen
  ```

## Set up a virtualenv

```bash
python -m venv regen          # first time only
# certain Windows setups: py -3.11 -m venv regen
```

Activate it:

- Windows (cmd): `regen\Scripts\activate.bat`
- Windows (PowerShell): `regen\Scripts\Activate.ps1`
- Linux/macOS: `source regen/bin/activate`

You should now see `(regen)` at the start of your shell prompt. If you do not, activation did not work;
try again or ask for help before continuing.

## Install dependencies

Match the PyTorch wheel index to your driver's CUDA version:

```bash
# CUDA 13.0/13.1 driver
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# AMD ROCm 6.4
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.4
```

Use `cu132` for a CUDA 13.2+ driver, `cu130` for CUDA 13.0/13.1, and `cu126` for a CUDA 12.6+ driver
(the only CUDA 12 build of torch 2.12.0). Always install `torch` **and** `torchvision` from the same
index; they must share one CUDA build.

Your driver's CUDA version is only the upper bound. The build also has to contain kernels for your
**GPU architecture**, or every job dies at the first CUDA call with `no kernel image is available for
execution on the device` (which ComfyUI hides behind a generic "no images produced" fault). The two
build families cover different, overlapping architecture ranges:

| Build | Architectures (compute capability) | GPUs |
| --- | --- | --- |
| `cu126` | `sm_50`–`sm_90` | Maxwell through Hopper (incl. Ada/Ampere) |
| `cu130` / `cu132` | `sm_75`–`sm_120` | Turing through Blackwell |

So a **Blackwell** card (RTX 50-series, `sm_120`) must use `cu130`+ even though it would otherwise accept
a CUDA 12 build, and a **pre-Turing** card (Maxwell/Pascal/Volta, e.g. GTX 10-series) must stay on
`cu126` because CUDA 13 dropped those architectures. The installer handles this for you (it reads the
GPU's compute capability via `nvidia-smi --query-gpu=compute_cap`, not just the driver version); only
override the build by hand if you have confirmed your card is in the range above. A Blackwell card on a
CUDA 12.x driver needs a **driver update** (to a CUDA 13 driver) before `cu130` can load.

This check is not one-time. Every `update-runtime` / sync re-reads the live GPU and clamps the resolved
build into the card's architecture window before installing, so a build that cannot run is never put on
disk. A worker that recorded `cu126` before a Blackwell card was installed (or before this clamp
existed) therefore self-heals to `cu130` on its next update, and even a hand-forced
`HORDE_WORKER_BACKEND=cu126` on a Blackwell card is corrected upward (an unrunnable build helps nobody)
with a note explaining the swap. The corrected build is re-recorded so the fix sticks.

> **Audio (torchaudio) is not installed.** It has no `+cu132` wheel and audio generation is currently
> unsupported, so the worker omits it (a missing torchaudio is stubbed at runtime; image/video work is
> unaffected). If you specifically need it, install a build matching your torch index ad hoc, e.g.
> `pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu130` (cu126/cu130/cpu
> only; there is no cu132 build).

## Selecting a PyTorch build

The worker locks the latest torch (2.12.0) as one thin `uv` extra per build: `cu126`, `cu130`,
`cu132`, or `cpu`. You normally never set this by hand; the build is detected from your GPU driver and
the `update-runtime` / `install` scripts run `uv sync --locked --extra <build>` for you.

To force one, set `HORDE_WORKER_BACKEND`:

```bash
HORDE_WORKER_BACKEND=cu132 ./update-runtime.sh   # CUDA 13.2+ build (auto-selected on 13.2+)
HORDE_WORKER_BACKEND=cu130 ./update-runtime.sh   # CUDA 13.0/13.1 build
HORDE_WORKER_BACKEND=cu126 ./update-runtime.sh   # CUDA 12.6+ build
HORDE_WORKER_BACKEND=cpu   ./update-runtime.sh   # no GPU
HORDE_WORKER_BACKEND=rocm  ./update-runtime.sh   # Linux ROCm runtime detected/installed
```

torch 2.12.0 has no `cu128` wheel, so a CUDA 12.x driver uses `cu126` (a legacy `cu128` request is
remapped to `cu126` automatically). The full list of build extras is in `pyproject.toml`.

Only the CUDA and CPU builds are locked. **ROCm** and **older torch versions** are installed ad hoc
(not from the lockfile), which is easy to mix in but not pinned:

```bash
./update-runtime-rocm.sh                                            # torch 2.9.1 on ROCm 6.4 (override: HORDE_WORKER_ROCM_TORCH)
UV_TORCH_BACKEND=auto uv pip install torch torchvision              # let uv auto-detect your GPU
uv pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128   # an older line
```

On Windows AMD, the installer detects supported Radeon/Ryzen AI devices and installs the `rocm-windows`
profile with AMD's official ROCm Windows wheels:

```powershell
$env:HORDE_WORKER_BACKEND = "rocm-windows"
.\update-runtime.cmd
```

## Run the worker

```bash
# Copy the template and set at least an API key and worker name
cp bridgeData_template.yaml bridgeData.yaml

python download_models.py    # critical: run before the worker, every time
python run_worker.py
```

`Ctrl+C` stops the worker after it finishes any in-progress jobs.

## Keep it updated

If you manage the venv yourself, re-run the install command every time you `git pull`, with `-U`:

```bash
python -m pip install -r requirements.txt -U --extra-index-url https://download.pytorch.org/whl/cu130
```

Swap the index URL to match your build (`cu132`, `cu126`, or `rocm6.4`). See
[Update the worker](update-the-worker.md).
