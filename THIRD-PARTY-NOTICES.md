# Third-party notices

The AI Horde Worker itself is licensed under the **GNU AGPL-3.0** (see [`LICENSE`](LICENSE)).

Installing and running the worker downloads and uses the third-party components listed below. Each is the
property of its respective authors and is distributed under its own license. This file names the license
and points to the authoritative text for each; the worker does not modify these components' licenses.

## Tooling and runtime (downloaded at install time)

| Component | Role | Source | License |
|-----------|------|--------|---------|
| uv | Package manager | https://github.com/astral-sh/uv | Apache-2.0 OR MIT |
| CPython (python-build-standalone) | Private Python runtime | https://github.com/astral-sh/python-build-standalone | PSF License |
| PyTorch | Deep-learning runtime | https://github.com/pytorch/pytorch | BSD-3-Clause |
| torchvision | Vision ops for PyTorch | https://github.com/pytorch/vision | BSD-3-Clause |
| MinGit (git-for-windows) | Portable git, **only** fetched on Windows when no system git is present | https://github.com/git-for-windows/git | GPLv2 |

## Image-generation engine and nodes (cloned on first run)

These are cloned from GitHub by `hordelib` the first time the worker runs a job, pinned to specific
commits recorded in `hordelib`'s manifest.

| Component | Source | License |
|-----------|--------|---------|
| ComfyUI | https://github.com/comfyanonymous/ComfyUI | GPL-3.0 |
| comfyui_controlnet_aux | https://github.com/Fannovel16/comfyui_controlnet_aux | see repository |
| ComfyQR | https://github.com/coreyryanhanson/ComfyQR | see repository |

## Python package dependencies

The worker's Python dependencies (resolved and pinned by `uv.lock`) are installed into the local `.venv`
on first run. The complete, authoritative license text for every installed package ships **inside that
package**, in its `*.dist-info/` directory under `.venv`. To produce a single aggregated file of those
texts from an installed worker, run from the install folder:

```
uv run python packaging/collect-licenses.py
```

This writes `THIRD-PARTY-LICENSES-FULL.txt` next to it. It reads only the license metadata already present
in your installed environment, so it always reflects exactly the versions you have.

## AI models

Any AI models you choose to serve are downloaded later from CivitAI and Hugging Face. Each model carries
its own license (many use the
[CreativeML OpenRAIL license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)); review a
model's license before use.
