# Testing

How to verify a change to this worker, from the unit sweep to a live run against the horde.

The worker runs on NVIDIA, AMD (ROCm on Linux and Windows), Intel XPU, Apple MPS and plain CPU, and
starts its text backend on CUDA, Vulkan or CPU. The unit and simulation suites run on all of them,
including a machine with no GPU. The install, the bands that can run, and the manual checks differ per
backend, so each section below states which backends it applies to.

Contents:

- [Prerequisites](#prerequisites)
- [Install and verify](#install-and-verify)
- [Run the tests](#run-the-tests)
- [Lint and types](#lint-and-types)
- [Continuous integration](#continuous-integration)
- [Test on a non-NVIDIA or CPU-only machine](#test-on-a-non-nvidia-or-cpu-only-machine)
- [Test the installer and bootstrap](#test-the-installer-and-bootstrap)
- [Test on real hardware](#test-on-real-hardware)
- [Test a text generation change](#test-a-text-generation-change)
- [Test the dashboard](#test-the-dashboard)
- [Run a live worker in maintenance mode](#run-a-live-worker-in-maintenance-mode)
- [Read a run](#read-a-run)

## Prerequisites

- `uv` at the version pinned in `pyproject.toml`. Nothing else is needed for the default sweep.
- A GPU for the `gpu` band, the benchmark and a live run. The GPU decides which install to make.
- A `bridgeData.yaml` with your API key, for a live run only.

## Install and verify

The torch build is an extra. Pick the one for your card:

| Extra | Machine | Locked |
|---|---|---|
| `cu132` | NVIDIA, driver at CUDA 13.2 or newer | yes |
| `cu130` | NVIDIA, driver at CUDA 13.0 or 13.1 | yes |
| `cu126` | NVIDIA, driver at CUDA 12.6 or newer, or a pre-Turing card | yes |
| `cpu` | No GPU. Image generation is off; alchemy and the text worker still run | yes |
| ROCm, XPU, MPS | AMD, Intel, Apple. Installed on top of `cpu` from the vendor wheel index | no |

A Blackwell card needs `cu130` or newer. A pre-Turing card must stay on `cu126`. [Choose a PyTorch
build](docs/how-to/choose-a-pytorch-build.md) has the architecture table and the ROCm and XPU commands.

Development install, NVIDIA or CPU:

```bash
uv sync --locked --all-packages --extra cu132 --dev --extra post-processing --extra controlnet
```

Swap `cu132` for your extra. The two feature extras add ControlNet and post-processing. Some non-NVIDIA
backends have no wheels for them, so leave them off there unless
[Compute backends](docs/explanation/compute_backends.md#optional-features-and-their-dependencies) lists
them as available for yours.

Verify the installed build:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The version suffix is the build (`+cu132`, `+rocm6.4`, `+cpu`). On NVIDIA and ROCm the second line
must be `True`. On XPU check `torch.xpu.is_available()` instead, on MPS `torch.backends.mps.is_available()`.

Two rules about the environment:

- Run Python through `uv run --no-sync`, from the project directory. Outside it, `uv run` uses a bare
  interpreter without the package. A plain `uv run` may re-sync and replace your torch build.
- A sync without any torch extra installs the CPU build. The worker then logs "CPU-only torch build",
  disables image generation, and the pop gate reports `torch_unusable`.

## Run the tests

| Band | Command | When | Needs |
|---|---|---|---|
| Default sweep | `uv run pytest` | Every change | Nothing |
| Slow | `uv run pytest -m slow` | The worker lifecycle, the TUI, anything that spawns processes | Nothing |
| GPU | `uv run pytest -m gpu` | Inference, VRAM handling, the benchmark probes | A CUDA device; see [Test on real hardware](#test-on-real-hardware) for other backends |
| Network | In the default sweep. `-m "not network"` excludes it | The server capability probe | The live horde API |
| Chaos sweep | `uv run pytest -m chaos_sweep` | Admission, reclaim, residency, scheduling, lifecycle. Pre-release gate | Nothing |
| Golden traces | `uv run pytest tests/process_management/liveness/test_golden_traces.py` | Scheduler and retention changes. In the default sweep | Nothing |
| Log contract | `uv run pytest -m slow tests/analysis/test_log_contract_dry_run.py tests/analysis/test_log_signatures.py` | A registered log line or `analysis/log_signatures.py` | Nothing |
| Session replay | `uv run pytest tests/process_management/liveness/test_session_replay.py` (one window, in the default sweep); `-m slow` adds the whole session | A scheduling, retention, clearance or lane change that claims a duty effect | Nothing |

The default sweep takes about 15 minutes. `tests/process_management` is most of it, so run one package
at a time to bisect. [CONTRIBUTING.md](CONTRIBUTING.md#the-generated-chaos-sweep-pre-release-gate)
describes the chaos sweep and its seed replay.

Markers:

- `slow`, `gpu`, `chaos_sweep` and `golden_regen` run only when `-m` names them. `-m "slow or gpu"`
  runs two at once.
- `network` marks the few tests that fetch the production API schema. They are in the default sweep
  and fail offline. Exclude them with `-m "not network"`.
- `golden_regen` rewrites the committed scheduler traces under `tests/process_management/liveness/golden/`
  instead of comparing against them. Review the diff it produces.
- `closed_loop` marks the simulator rows under `tests/process_management/liveness/`. They are in the default
  sweep and cost seconds each. While iterating on a scheduling change, run `-m "not closed_loop"` plus the
  rows your change touches, then the whole band once before committing.
- `e2e` marks the whole-worker dry-run flows under `tests/e2e/`. Most are in the default sweep. The ones
  that spawn real child processes are also `slow`.
- The session replay drives the dispatch-world simulator with a recorded worker session:
  `tests/process_management/liveness/replay/` holds the job stream (one row per job with its parameters,
  stage timings, measured VRAM peaks and kudos, no prompts) and the learned footprint store the worker had,
  with the platform key neutralised. The rows are derived from a support bundle's stats `job_completed` rows by
  `uv run horde-log replay-fixture <bundle> --session N <out dir>` (N as `horde-log sessions` numbers it), never
  written by hand, and carry no host or worker names. A replay row states its calibration band against
  the live session in its docstring; it is a calibration, not a gate. A scheduling change is judged by its
  outcome against the replay at the previous commit, and a window that runs far from live is a simulator
  finding to report before anything is tuned.
- A marker decorator must be on the test function itself. On a helper beside it, `-m` deselects the test
  silently.

Parallel runs. `pytest-xdist` is installed and off by default:

```bash
uv run pytest tests/process_management/liveness -m closed_loop -n auto
uv run pytest -m "slow or chaos_sweep" -n auto --dist loadgroup
```

`--dist loadgroup` keeps the rows marked `xdist_group("serial_timing")`, whose assertions are wall-clock
measurements, on one worker. Parallelise a band on its own selection: the fastest-first order in
`tests/conftest.py` only applies to a serial run, so a mixed selection under `-n` no longer fails fast on
a cheap break. Concurrent runs in one checkout are safe. Each test has its own run root for logs, state
and the `.abort` sentinel.

Other switches:

- `uv run pytest --shuffle-seed 1` randomises the order inside each phase and prints the seed. Run it
  once before relying on a parallel band.
- Every run prints its thirty slowest tests.
- `-p no:cacheprovider` keeps the checkout free of cache directories.

Rerun a failure by its node id, quoted verbatim when parametrized. A chaos failure prints its seed.
Replay it with `HORDE_CHAOS_SEEDS=<seed>`.

## Lint and types

```bash
uv run ruff format . && uv run ruff check . --fix
uv run pyrefly check
uv run prek run --all-files
```

The hooks use the same ruff and pyrefly versions as the `dev` group, so a clean local run is a clean CI run.

## Continuous integration

Three workflows under `.github/workflows/`:

| Workflow | Trigger | Runner | Steps |
|---|---|---|---|
| Unstable Tests (`prtests.yml`) | A pull request to `main` that touches Python, JSON, the lock, the requirements or the workflows | Ubuntu, Python 3.12, `cpu` extra | Hooks; Dockerfile lint; lock drift; default sweep; `slow` band |
| Tests (`maintests.yml`) | A push to `main` with the same path filter | Same | Same, minus the Dockerfile lint |
| Chaos Sweep (`chaos-sweep.yml`) | Nightly, or by hand with a seed range | Same | Both chaos sweep tiers |

What this means for a contributor:

- A green PR proves the default and `slow` bands pass on Linux with the CPU build. That is all it proves.
- The `lock-drift` job fails when the root lock and the image-utilities lock disagree. After a torch bump
  in the root lock, re-pin and re-lock `requirements/utilities` as `requirements/README.md` describes.
- A change to markdown, YAML config or shell scripts does not trigger the test workflows. Run the hooks
  locally.
- Nothing in CI has a GPU, so the `gpu` band, the benchmark and every accelerator path are only ever
  exercised on a contributor's machine. Run them before opening a PR that touches inference, VRAM, the
  bootstrap detection or the text backend launch.
- Nothing in CI runs Windows or macOS. The tray icon, the installer wizard, the PowerShell detection
  script and the Windows ROCm profile are only ever exercised on a contributor's machine.
- The chaos sweep does not run on a PR. Run it locally before a change to admission, reclaim, residency,
  scheduling or lifecycle. A regression that slips through shows as a red nightly run, so check the
  workflow's history the day after such a change merges.

## Test on a non-NVIDIA or CPU-only machine

The default and `slow` bands run in full on any machine, because the suites fake the inference
processes and the text backend. What follows is the additional manual check per machine.

**CPU only.** Install with `--extra cpu`. Start the worker once against a real config and confirm:

- The log states that image generation is disabled for a CPU-only build and lists the alchemy forms
  as available.
- `bin/backend` contains `cpu`. The orchestrator learns its compute mode from this file, not from torch,
  so a wrong token is a wrong mode. See
  [CPU / alchemist-only mode](docs/explanation/compute_backends.md#cpu--alchemist-only-mode-running-without-a-usable-gpu).
- With `scribe: true`, the provisioner downloads the no-CUDA koboldcpp build and starts it on the CPU
  backend.

**AMD, ROCm.** Install `cpu` first and then the ROCm torch stack, either through `runtime-rocm.sh` or by
hand from the ROCm wheel index. `bin/backend` contains `rocm` on Linux and `rocm-windows` or a
`rocm-gfx*` token on Windows. Then:

- Run the `gpu` band. Its skip gate is `torch.cuda.is_available()`, which is `True` under ROCm torch, so
  the band runs. Several probes assume NVML, so a failure there is a finding about ROCm support rather
  than a broken test.
- Confirm the provisioner downloaded the no-CUDA koboldcpp build and that the backend log shows a Vulkan
  device. The ROCm fork of koboldcpp is a separate project and is never chosen automatically. To test it,
  point `text_backend_executable` at it.
- Feature extras default to off on a lean backend. To test them where wheels exist, set
  `HORDE_WORKER_FEATURES=controlnet,post-processing` before syncing, then confirm the startup log has no
  capability coercion warning.

**Intel XPU, Apple MPS.** As for ROCm, minus the `gpu` band, which skips without a CUDA device. The
accelerator probe and a live run are the test. Confirm the probe reports the device kind and that the
inference processes start on it. `bin/backend` contains `cpu` on these installs, so a managed text
backend starts on the CPU path. Set `text_backend_accelerator: vulkan` to test the Vulkan build on an
Intel card.

**Every non-NVIDIA backend.** Two regressions have no test and need a manual check:

- The probed device kind must match the build token. A mismatch (a GPU install whose driver is broken,
  a CPU install on a machine with a GPU) is a startup warning. Grep the bridge log for it after any
  change to `compute_mode.py`, `accelerator_probe.py` or the bootstrap.
- The worker must only advertise what it can run. After a change to capabilities or the feature extras,
  start the worker with every feature enabled in config and confirm the coercion warnings name exactly
  the features whose dependencies are missing.

## Test the installer and bootstrap

`worker_bootstrap/` is the Python package behind `runtime.sh`, `runtime.cmd` and the platform shims. It
runs before the project venv exists, so it uses only the standard library and its tests import it from
the repo root rather than from the installed package.

Its unit tests are in `tests/bootstrap/`, in the default sweep. They cover backend detection and
precedence, the sync plan, the koboldcpp provisioner, the updater and the standard-library-only guard.
Two depend on the platform:

- `test_detect_parity.py` runs the installer wizard's PowerShell detection and the Python detector
  against the same stubbed `nvidia-smi` output and requires the same token from both. Windows only.
- `test_utilities_env.py` covers the provisioning plan for the image-utilities venv under
  `requirements/utilities/`. CI checks that project's lock in a separate step.

To test detection on a real machine without installing anything:

```bash
./runtime.sh detect          # Windows: runtime.cmd detect
```

The output is the token it would choose and the reasoning: the driver's CUDA version, the compute
capability, and whether the build was clamped to the card's architecture window. To test a path your
card would not choose, force a build with `--backend <token>` or the `HORDE_WORKER_BACKEND` environment
variable. An existing `bin/backend` outranks detection on later runs, so delete it or pass the flag to
switch builds.

To test a sync without touching your development environment, run the shim in a scratch clone. It keeps
its cache, its managed Python and its models in a sibling `<clone>-data` directory, or wherever
`HORDE_WORKER_DATA_DIR` points, so the clone can be deleted afterwards.

## Test on real hardware

The `gpu` band starts real inference processes and runs real generations. Each test takes minutes. It is
skipped unless `-m gpu` is given and a CUDA device is present.

```bash
uv run pytest -m gpu -k canary                # one SD 1.5 job through a warm worker, the cheapest signal
uv run pytest -m gpu -k sd15-baseline         # one capability probe by its slug
uv run pytest -m gpu tests/gpu                # the whole capability catalog on one warm worker
```

The capability probes under `tests/gpu/` are the same probes `horde-benchmark run` executes, so a
passing probe here and a passing benchmark tier agree by construction. Each probe skips itself when the
machine does not fit its tier, and a probe that needs a CivitAI token skips without one.

The benchmark is a manual check on the whole machine rather than a test:

```bash
uv run horde-benchmark plan          # what would run on this machine and why
uv run horde-benchmark run           # the capability tiers
uv run horde-benchmark soak          # sustained throughput
uv run horde-benchmark report        # the last run, rendered
```

Its unit tests in `tests/benchmark/` run in the default sweep against a fake executor.

Run only one of the `gpu` band, the benchmark and a live worker at a time on a card. Each one budgets
VRAM from the free memory it measures at start and assumes nothing else will take it, so a second tenant
causes out-of-memory faults that look like worker bugs. The benchmark also measures throughput, which a
second tenant halves.

## Test a text generation change

The text worker communicates with koboldcpp over HTTP. Most of the path is testable without a model.

**Fake backend, fake horde.** In the default sweep, no hardware:

- `tests/process_management/jobs/test_text_generation_coordinator.py`: pop contract, wedge proof, relaunch, dry run
- `tests/text_backends/test_kobold_api.py`: the client against an aiohttp stand-in
- `tests/process_management/lifecycle/test_text_backend_supervisor.py`: readiness, footprint, buffer parsing
- `tests/process_management/resources/test_foreign_vram_floor.py`: the managed backend as a standing VRAM floor
- `tests/text_backends/test_launch.py`: the koboldcpp command line rendered from the backend-independent settings
- `tests/bootstrap/test_koboldcpp_bin.py`: which release asset each platform and accelerator gets

**Real koboldcpp, opt-in.** Needs the provisioned binary under `bin/`, which the worker downloads on the
first text-enabled start, and a GGUF model:

```bash
HORDE_TEXT_SMOKE_GGUF=<path to a GGUF> uv run pytest -m slow tests/text_backends/test_kobold_api_smoke.py tests/process_management/lifecycle/test_text_backend_supervisor.py
```

The real-backend rows skip when the binary or the variable is missing. A 3B Q4 model runs the smoke in
under a minute. This is the test to run on AMD, Intel and CPU machines: only NVIDIA gets the CUDA build,
so on any other host this proves the no-CUDA build starts and generates on Vulkan or CPU.

**Whole worker, fake backend, fake horde.** The harness runs a scribe end to end under a faked API:

```bash
uv run pytest -m "slow and e2e" tests/e2e/test_harness.py::test_text_jobs_run_through_the_fake_backend_with_no_api
```

`HarnessConfig(text_jobs=make_text_scenario(n))` enables the role and the outcomes are counted in
`HarnessResult.num_text_jobs_*`. Dry-run mode never starts koboldcpp, so a real backend under a fake
horde is not available. For that, use a live run.

**Your own backend.** [Run a text worker](docs/how-to/run-a-text-worker.md#attach-to-a-backend-you-already-run)
describes attaching the worker to a koboldcpp you run yourself. That is how to test a fork, a ROCm build,
or a model the catalogue does not include.

## Test the dashboard

`tests/tui/` has the pilot tests for the Textual app, the web host, the worker host and the attach
protocol. They are in the `slow` band and take about 10 minutes:

```bash
uv run pytest -m slow tests/tui
```

Two of them wait one tick of the app clock and can fail on a loaded machine. Rerun the failed node id
on its own before treating it as a failure. For a visual change, run the app:

```bash
uv run horde-worker            # terminal dashboard
uv run horde-worker-web        # browser dashboard; --no-browser to just serve it
```

[Use the dashboard](docs/how-to/use-the-dashboard.md) lists the tabs and what each should show.

## Run a live worker in maintenance mode

A worker in maintenance mode only receives requests made with the owner's API key, so you can control
the workload. This is the test for anything the fake backend and fake horde do not cover: a real backend
under real pops, kudos, the worker record on the horde, and every accelerator-specific path.

1. Make a run root outside the checkout and copy `bridgeData.yaml` into it. The worker writes `logs/`,
   `.horde_worker_regen/` and `.abort` there.

2. Start the worker through the attach supervisor, which exposes it as three JSONL files:

   ```bash
   uv run horde-worker-attach --session-dir <run root>/session --config <run root>/bridgeData.yaml
   ```

   [Attach a supervisor](docs/how-to/attach-a-supervisor.md) describes the files. `state.jsonl` has one
   line per poll. Watch `worker_up`, `server_maintenance`, `popped`, `submitted` and `kudos_hr`. The
   pop gate and its reasons are in `logs/bridge.log`.

3. Enter maintenance through the worker, so the flag counts as deliberate and the worker's automatic
   maintenance clearing does not undo it. Append to `commands.jsonl`:

   ```json
   {"command": "SET_SERVER_MAINTENANCE", "server_maintenance_enabled": true}
   ```

   A role's worker is registered with the horde only after its first pop, so send the command again once
   every enabled role has registered. `server_maintenance` in `state.jsonl` becomes true. To check the
   horde's side, `GET /v2/workers/{id}` on the horde API. The horde rejects requests without `Client-Agent`
   and `User-Agent` headers.

4. Send requests pinned to your worker id (`"workers": [<id>]`) under your key.

   | Kind | Submit | Poll |
   |---|---|---|
   | Text | `POST /v2/generate/text/async` | `GET /v2/generate/text/status/{id}` |
   | Image | `POST /v2/generate/async` | `GET /v2/generate/check/{id}`, then `/status/{id}` |

   Every parameter must fit what the worker advertises. The "advertising" log line has `max_length`
   and `max_context_length`. A request that does not fit is accepted with "No available workers can
   fulfill" and queues until it expires.

5. Read the result in `<run root>/logs/bridge.log`: the backend ready line with its VRAM footprint,
   pops and submits with kudos, and the retention and restore lines. `logs/text_backend.log` is
   koboldcpp's output. Then:

   ```bash
   uv run horde-log diagnose <run root>/logs
   uv run horde-log bundle <run root>/logs
   ```

6. Stop with `Ctrl+C`, or append `{"command": "GRACEFUL_SHUTDOWN"}` to the inbox. The worker drains for
   up to about two minutes. Wait for its process to exit before relaunching. With
   `remove_maintenance_on_init` set, the next start clears maintenance itself.

## Read a run

`horde-log` triages a logs directory: `sessions`, `diagnose`, `timeline`, `jobs`, `job <id>` and
`bundle`. `--last` and `--session N` select a worker session; `--last-benchmark` and `--benchmark N`
select a benchmark run. To add a detector, see [Write a finding](docs/how-to/write-a-finding.md). Every
registered log line is gated by the log contract band above.

When reporting a problem, attach the bundle. It is redacted and contains the logs, the text backend's
output, and a machine summary: worker version, OS, CPU, RAM, disk, and on NVIDIA the driver's CUDA
ceiling and each card's compute capability. `--probe-gpu` adds a full accelerator inventory.
