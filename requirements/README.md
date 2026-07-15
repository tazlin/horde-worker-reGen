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

Each file starts as a checked-in bootstrap seed, then is promoted to a fully-resolved, **hashed** pin in
the release pipeline with `uv pip compile`. The seed is intentionally installable: it pins the service and
the matching torch/torchvision build and includes a PyTorch wheel index. This means a fresh clone can
create its utilities venv before the first release-pipeline promotion.

The service requirement is:

```
horde-image-utilities[server,annotators,rembg_<cpu|cuda|rocm>]==<pinned version>
```

`annotators` is accelerator-neutral; `rembg_cpu`, `rembg_cuda`, or `rembg_rocm` is selected for the
backend. Do **not** add `mediapipe` to the annotator seed: the current service release declares an OpenCV
wheel conflict between those extras. Promote a seed only after resolving that conflict or adding a deliberate
deployment override.

When a pin carries `--hash=` lines, the bootstrap installs it with `--require-hashes`; an unhashed seed
installs without hash enforcement. A promoted pin must preserve the backend's explicit torch and torchvision
versions and wheel source, not let the resolver choose an unrelated PyPI torch build.

## Promotion checklist

1. Bump the `horde-image-utilities` version and/or backend torch pair in the appropriate seed.
2. Compile the seed in a clean environment using the backend's PyTorch wheel source and `--generate-hashes`.
3. Verify the generated file with `uv pip install --require-hashes --dry-run -r requirements/utilities.<token>.txt`.
4. Test `bootstrap.py sync --backend <token>` from an empty data directory, then start the capability server
   with the resulting `utilities-venv` interpreter.
5. Commit the generated pin and its matching worker/docs changes together.

## Provisioning

The bootstrap (`worker_bootstrap`) provisions the utilities venv after a successful worker sync, and also
repairs it on a launch whose main venv is already current. It acts when the resolved feature set is non-empty
and a requirements pin for the backend exists. See `docs/explanation/compute_backends.md` (the
"Image-utilities capability venv" section) for the full flow and the `HORDE_WORKER_FEATURES` /
`HORDE_WORKER_SKIP_UTILITIES` controls.

`utilities.example.txt` in this directory is an illustrative skeleton only (the `example` token is never a
real backend), showing the header a CI-compiled file carries.
