# Install from scratch and choose a PyTorch build

Most people should use the [installer](install.md), which reads your GPU and installs the right PyTorch
build without being asked. This page is for driving that yourself: managing the environment by hand, or
forcing a particular build (a specific CUDA version, CPU-only, or ROCm).

In brief:

- The build has to carry kernels for your GPU's **architecture**, not merely satisfy your driver's CUDA
  version. Get it wrong and every job dies at the first CUDA call.
- `cu126` covers `sm_50` to `sm_90` (Maxwell through Hopper). `cu130` and `cu132` cover `sm_75` to
  `sm_120` (Turing through Blackwell). A Blackwell card needs CUDA 13; a pre-Turing card cannot use it.
- `HORDE_WORKER_BACKEND=<build>` forces the choice, and the worker corrects a build that cannot run the
  card that is present.
- `cu126`, `cu130`, `cu132`, and `cpu` are installed from the lockfile. ROCm, Intel XPU, and older torch
  lines are installed ad hoc.

## Before you start

- Install [git](https://git-scm.com/).
- Install your GPU stack (CUDA or ROCm) if you have not already.
- Configure at least 8 GB of swap, 16 GB or more preferred. This applies to Linux too.
- Clone the worker:

  ```bash
  git clone https://github.com/Haidra-Org/horde-worker-reGen.git
  cd horde-worker-reGen
  ```

You do not need to install Python for the script path: `runtime.sh` and `runtime.cmd` fetch `uv`, and
`uv` fetches its own managed CPython. Install Python 3.12 yourself only if you intend to point `uv` at
an environment you built (see [Install into an environment you manage](#install-into-an-environment-you-manage)).

## Pick the build that matches your GPU

Your driver's CUDA version is only the upper bound. The wheel also has to contain kernels for your GPU
architecture, or every job dies at the first CUDA call with `no kernel image is available for execution
on the device`, which ComfyUI reports as a generic "no images produced" fault.

The two build families cover overlapping but different architecture windows:

| Build | Architectures (compute capability) | GPUs |
| --- | --- | --- |
| `cu126` | `sm_50` to `sm_90` | Maxwell through Hopper, including Ampere and Ada |
| `cu130` / `cu132` | `sm_75` to `sm_120` | Turing through Blackwell |

Two consequences:

- A **Blackwell** card (RTX 50-series, `sm_120`) must use `cu130` or newer, even though its driver would
  accept a CUDA 12 build. On a CUDA 12.x driver it needs a driver update first.
- A **pre-Turing** card (Maxwell, Pascal, Volta, for example the GTX 10-series) must stay on `cu126`.
  CUDA 13 dropped those architectures.

Find your card's compute capability, which is what the choice turns on:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Then see what the worker would pick for this machine:

```bash
./runtime.sh detect      # Windows: runtime.cmd detect
```

These are the build names it can resolve to:

| Build | For |
| --- | --- |
| `cu132` | An NVIDIA driver at CUDA 13.2 or newer. Auto-selected there. |
| `cu130` | An NVIDIA driver at CUDA 13.0 or 13.1. |
| `cu126` | An NVIDIA driver at CUDA 12.6 or newer. The only CUDA 12 build of the locked torch line, and it also runs on CUDA 13. |
| `cpu` | No usable GPU. Roughly 100x slower; for testing and alchemist-only work. |
| `rocm` | Linux with a detected or installed ROCm runtime. |
| `rocm-windows` | Windows with a supported Radeon or Ryzen AI device, using AMD's official ROCm Windows wheels. |

There is no `cu128` wheel in the locked torch line, so a CUDA 12.x driver uses `cu126` and a legacy
`cu128` request is remapped to it. The extras themselves are declared in `pyproject.toml`.

## Install it

### With the worker's scripts

`update-runtime.sh` (or `update-runtime.cmd`) installs and updates the environment. Set
`HORDE_WORKER_BACKEND` to override what detection would have chosen:

```bash
HORDE_WORKER_BACKEND=cu132 ./update-runtime.sh   # CUDA 13.2+ build
HORDE_WORKER_BACKEND=cu130 ./update-runtime.sh   # CUDA 13.0/13.1 build
HORDE_WORKER_BACKEND=cu126 ./update-runtime.sh   # CUDA 12.6+ build
HORDE_WORKER_BACKEND=cpu   ./update-runtime.sh   # no GPU
HORDE_WORKER_BACKEND=rocm  ./update-runtime.sh   # Linux ROCm
```

On Windows AMD:

```powershell
$env:HORDE_WORKER_BACKEND = "rocm-windows"
.\update-runtime.cmd
```

To go back to automatic selection, run the script again without the variable set.

### Install into an environment you manage

The project is a `uv` project with a committed lockfile. There is no `requirements.txt`. What the
scripts run, and what you can run directly, is:

```bash
uv sync --locked --extra cu130
```

Swap `cu130` for the build you picked. `--locked` installs exactly the wheels the lockfile names, which
is what keeps `torch` and `torchvision` on the same CUDA build. Add `--extra controlnet` and
`--extra post-processing` for the optional feature dependencies.

To put the environment somewhere other than `.venv`, point `uv` at it:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/env uv sync --locked --extra cu130
```

### Verify the build can run your card

```bash
./runtime.sh run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

The first line names the installed build (for example `2.12.1+cu132 13.2`). The second is the list of
architectures the wheel was compiled for:

```
2.12.1+cu132 13.2
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

Your card can run that wheel if the list holds an `sm_<n>` of the same major version as your compute
capability with a minor at or below it (a `sm_89` card is served by `sm_86`, because binary kernels are
forward-compatible within a major version), or any `compute_<n>` at or below your capability, which is
JIT-compiled at load. An `sm_120` card against the list above is fine; an `sm_61` card is not. When
nothing matches, install a build from the right family above rather than reinstalling the same one.

> **torchaudio is not installed.** It has no `+cu132` wheel and audio generation is unsupported, so the
> worker omits it and stubs it at runtime. Image and video work are unaffected. If you specifically need
> it, install a build matching your torch index ad hoc, for example
> `pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu130`. Only `cu126`,
> `cu130`, and `cpu` have one.

## Why the worker may override your build

Build selection is a prediction: the installer has to choose a wheel before torch exists, so it maps
your GPU's compute capability onto a build. Two backstops check that prediction against reality, and
either one can change what gets installed.

**Every sync clamps the build into your card's architecture window.** A resolved or hand-forced build
that has no kernels for the card present is corrected before anything is installed, so an unrunnable
wheel never reaches disk. This is why `HORDE_WORKER_BACKEND=cu126` on a Blackwell card installs `cu130`
instead, with a note explaining the swap, and why a worker that recorded `cu126` before a Blackwell card
was fitted moves itself to `cu130` on its next update. The corrected build is re-recorded, so the fix
sticks.

**Every launch checks the installed wheel against the live GPU.** A matching lockfile proves the right
package versions are installed, not that the installed torch has kernels for the card actually in the
machine: a GPU swap, or an environment copied from another machine, leaves an up-to-date venv whose
torch still cannot launch a kernel. The check is stamped per lock and card, so a healthy start stays
instant, and when the installed wheel cannot run the card and a different locked build can, the launch
re-syncs to the runnable one first.

Every selection and clamp is recorded to `bin/backend-decision.json`, so a support bundle shows why a
given build was chosen.

If a sync warns that the installed wheel still cannot run your GPU, the selection table is out of date
for your card. That is a worker bug rather than unsupported hardware, and reinstalling will not fix it:
please [file an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues), and force a newer build
with `HORDE_WORKER_BACKEND` as a stopgap.

## ROCm, Intel XPU, and older torch lines

Only the CUDA and CPU builds are locked. The others pull a triton sidecar
(`pytorch-triton-rocm`, `pytorch-triton-xpu`) that has no wheel under the locked torch range, so they
are installed ad hoc: easy to mix in, and not pinned.

```bash
./update-runtime-rocm.sh                                 # ROCm 6.4 (override the torch version with HORDE_WORKER_ROCM_TORCH)
UV_TORCH_BACKEND=auto uv pip install torch torchvision   # let uv detect your GPU
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/xpu   # Intel XPU
```

Install `torch` and `torchvision` from the same index every time. They must share one build.

For the AMD path end to end, see [Run on AMD ROCm](run-on-amd-rocm.md).

## Run the worker

```bash
cp bridgeData_template.yaml bridgeData.yaml   # then set at least an API key and worker name
./preload-models.sh                           # download and verify models before the first run
./horde-bridge.sh                             # start the headless worker
```

On Windows use `preload-models.cmd` and `horde-bridge.cmd`. `Ctrl+C` stops the worker once it finishes
any in-progress jobs. For the dashboard instead, run `horde-worker.sh` or `horde-worker.cmd` (see
[Use the dashboard](use-the-dashboard.md)).

## Keep it updated

Re-run the install command after every `git pull`, so the environment matches the lockfile the release
expects:

```bash
./update-runtime.sh      # Windows: update-runtime.cmd
```

It reuses the recorded build, and re-checks it against the card as described above. Set
`HORDE_WORKER_BACKEND` again only when you want to change build. See
[Update the worker](update-the-worker.md).

## See also

- [Compute backends](../explanation/compute_backends.md): which accelerators the worker supports, the
  optional feature extras, the utilities venv, and CPU-only mode.
- [Command line](../reference/cli.md): every script, verb, and environment variable.
- [Troubleshoot](troubleshoot.md): what a failed start usually means.
