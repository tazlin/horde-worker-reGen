# Utilities venv requirements pins

This directory holds the CI-compiled dependency pins for the worker's **second virtual environment**, the
image-utilities capability venv. That venv runs the `horde-image-utilities` capability service in an
environment separate from the worker's own `.venv`, so its native, accelerator-gated dependencies never
share a resolution with the worker's dependencies.

## Naming convention

One file per locked torch-build token:

```
requirements/utilities.<backend_token>.txt
```

`<backend_token>` is one of the locked torch-build tokens the bootstrap resolves (`cu126`, `cu130`,
`cu132`, `cpu`). The bootstrap looks up `utilities.<token>.txt` for the resolved backend when it decides
whether to provision the utilities venv; a token with no committed file simply has nothing to provision
yet (the bootstrap stays a no-op for it).

## How these files are generated

Each file is a fully-resolved, **hashed** pin compiled in CI with `uv pip compile`, of:

```
horde-image-utilities[server,annotators_<X>,rembg_<X>,mediapipe]==<pinned version>
```

where the `annotators_<X>` / `rembg_<X>` extras are selected to match the accelerator the backend token
targets. Do not hand-edit these files: regenerate them through the CI pipeline so the resolution and the
hashes stay consistent.

When a pin carries `--hash=` lines, the bootstrap installs it with `--require-hashes`; a not-yet-hashed
placeholder still installs (without hash enforcement) so early bring-up is possible before the hashed
pins land.

## Provisioning

The bootstrap (`worker_bootstrap`) provisions the utilities venv after a successful worker sync, when the
resolved feature set is non-empty and a requirements pin for the backend exists. See
`docs/explanation/compute_backends.md` (the "Image-utilities capability venv" section) for the full flow
and the `HORDE_WORKER_FEATURES` / `HORDE_WORKER_SKIP_UTILITIES` controls.

`utilities.example.txt` in this directory is an illustrative skeleton only (the `example` token is never a
real backend), showing the header a CI-compiled file carries.
