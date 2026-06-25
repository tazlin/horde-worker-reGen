# Compute backends (NVIDIA, ROCm, XPU, MPS, CPU)

The worker runs on `hordelib`, which wraps ComfyUI, and **ComfyUI's `model_management` is already
backend-agnostic**: it detects and drives NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon
(MPS), Ascend/NPU, Cambricon/MLU, DirectML and CPU, and exposes safe wrappers (`get_torch_device`,
`get_total_memory`, `get_free_memory`, `soft_empty_cache`, `get_all_torch_devices`). The worker's job
is to **not bypass that layer** with NVIDIA-only calls.

## The single source of accelerator truth

All device discovery, VRAM accounting and device-cache management route through
`hordelib.utils.torch_memory` (re-exported from `hordelib.api`):

- `enumerate_accelerators() -> list[AcceleratorInfo]` - every device on the machine, each tagged with
  an `AcceleratorKind` (`cuda`/`rocm`/`xpu`/`npu`/`mlu`/`mps`/`directml`/`cpu`).
- `get_torch_total_vram_mb()` / `get_torch_free_vram_mb()` - device VRAM, falling back to system RAM
  for CPU/MPS where there is no separate device memory.
- `clear_accelerator_cache()` - the backend-aware cache release (never a bare
  `from torch.cuda import empty_cache`).

Each helper has two paths:

1. **ComfyUI loaded** (the inference process): it delegates to `comfy.model_management`, detected via
   `sys.modules` so the call never *triggers* a ComfyUI import. This matters because the safety
   process must never import ComfyUI.
2. **ComfyUI not loaded** (the parent, the TUI wizard, the safety process): a plain-torch fallback
   that checks each backend in turn (`torch.cuda` for CUDA/ROCm, `torch.xpu`, `torch.backends.mps`,
   else CPU) instead of assuming `torch.cuda`. `enumerate_accelerators()` always returns at least one
   device (a CPU pseudo-device), so callers never get the empty inventory a bare
   `torch.cuda.device_count()` loop produces on a non-CUDA backend.

## Why this matters

`torch.cuda.device_count()` returns `0` and `torch.cuda.is_available()` is `False` on MPS/XPU/
DirectML, so any code that enumerates devices or reads VRAM directly through `torch.cuda` would see
*no hardware at all* on those backends. The worker's hardware probe (`SystemResources.detect`), the
inference process's cache clearing, the TUI setup wizard's VRAM sizing, and the benchmark's machine
info all consume the abstraction above instead.

## NVML is optional enrichment, not a requirement

`pynvml`/NVML provides GPU **utilization, temperature and power** telemetry (the benchmark duty-cycle
sampler in `utils/gpu_monitor.py` and the rich stats in `hordelib.utils.gpuinfo`). There is no
portable cross-backend torch equivalent, so this stays **NVIDIA-only and strictly optional**: when
NVML is absent the sampler collects nothing and reports `None`, and VRAM totals still come from the
backend-agnostic path. A missing `nvidia-smi`/NVML never breaks startup.

## What still limits non-NVIDIA backends

The remaining constraints are **packaging, not worker code** (see the README's "Why fewer GPU types
than ComfyUI?"): DirectML and *locked* ROCm/XPU builds are blocked upstream in PyTorch (the
`pytorch-triton-rocm` / `pytorch-triton-xpu` sidecars are not published for the torch line the worker
pins), so those torch builds are installed ad-hoc rather than from the lockfile. The runtime itself no
longer hard-codes NVIDIA, and the heavy optional features are no longer in the base install (see
below), so a base worker installs and runs on the accelerators ComfyUI supports.

## Optional features and their dependencies

Two features depend on native packages that have no wheels for some accelerators, so they are **not in
the base install**. They live in worker install-time extras that re-export the corresponding
`horde-engine` extras:

| Worker extra | Enables | Native dependency | Wheels missing on |
| ------------ | ------- | ----------------- | ----------------- |
| `post-processing` | Upscale, face-fix, **and background removal** (one atomic bucket) | `rembg` (CPU); `lpips` rides along for the bundled face-fix node | `rembg` typically present on x86; absent on some accelerators |
| `controlnet` | ControlNet preprocessing / annotators | `onnxruntime` (DWPose) | Intel XPU, Apple MPS, Ascend |

Only `rembg` is the accelerator-gating dependency; `lpips` is a pure-Python package shipped in the same
extra only because horde-engine's vendored `facerestore_cf` node imports it but declares it nowhere.

Everything else (core image generation, the NSFW/CSAM safety classifier, ESRGAN upscalers, the
CodeFormer/GFPGAN face-fixers, LoRA, img2img) is pure PyTorch and runs on every backend. The extra
upscaler architectures (`spandrel-extra-arches`, horde-engine's `upscale-extra`) also ship in **base**
on every backend: it is a pure-Python universal wheel with no accelerator-specific blocker, so it is not
a per-backend feature extra; it only widens which upscaler models load.

### Install profiles ("NVIDIA/CPU full, others lean")

The bootstrap (`bootstrap.py sync`, used by every installer) chooses extras from the resolved torch
build:

- **NVIDIA (`cu126`/`cu130`/`cu132`) and CPU**: install `post-processing` and `controlnet` by default,
  so existing installs are unchanged.
- **ROCm / XPU / MPS and other ad-hoc backends**: install **lean** (base only).

Override per install with `HORDE_WORKER_FEATURES` (comma or space separated):

```bash
# Opt a ROCm box into both features (their CPU wheels do exist on x86 Linux):
HORDE_WORKER_FEATURES="post-processing,controlnet" ./update-runtime.sh
# Force a lean NVIDIA install (skip the optional features):
HORDE_WORKER_FEATURES=none ./update-runtime.sh
```

For Intel XPU, install the torch build ad-hoc first (no locked build exists):

```bash
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/xpu
```

### Config coercion (advertised == runnable)

Whatever is installed, the worker reads hordelib's capability registry
(`hordelib.api.available_features`) at startup and on every config reload, and coerces the bridge data
so it never advertises a feature it cannot run (`horde_worker_regen/capabilities.py`):

- If `rembg` is absent, `allow_post_processing` is coerced **off** with a warning. Post-processing is
  one atomic bucket because the AI Horde API cannot accept upscale/face-fix while refusing
  background-removal per job, so the whole option is disabled even though the upscalers/face-fixers
  themselves would run. (Alchemy forms are enumerated per-form, so alchemy still offers the pure-torch
  upscalers/face-fixers and drops only `strip_background`.)
- If `onnxruntime` is absent, `allow_controlnet` and `allow_sdxl_controlnet` are coerced **off**.

This means an operator can leave their existing config untouched when moving to a lean backend: the
worker self-limits to what it can serve rather than popping jobs it would fault.

> Verification note: the backend-agnostic paths are tested on **NVIDIA and CPU** directly, and on the
> other backends by code-path inspection plus mocked-backend unit tests
> (`tests/test_torch_memory.py` in hordelib, `tests/process_management/resources/test_system_resources.py` in
> the worker). They are expected to work on ROCm/XPU/MPS but are not yet hardware-verified there.

## Release coupling

The accelerator abstraction lives in `hordelib` (`horde-engine`). The worker depends on a **published
`horde-engine`** release, so these helpers reach the worker only once a `horde-engine` version that
includes them is published and the worker's pin is bumped.
