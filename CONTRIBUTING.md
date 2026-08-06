# Contributing to horde_worker_reGen

## Code Quality Tools

* [uv](https://docs.astral.sh/uv/)
    * A modern Python package manager and virtual environment tool
    * Run `uv sync --extra <extra>` to install dependencies and `uv run <command>` to run commands in the virtual environment.
        * See `pyproject.toml` for the dependencies and extras used in this project - you need to specify your GPU specific torch.
    * The required uv version is pinned in `pyproject.toml`; CI's setup action reads that same pin.
* [prek](https://github.com/j178/prek)
    * Pre-commit compatible hooks for code quality and formatting
    * Run `prek run --all-files` or see `.pre-commit-config.yaml` for more info.
    * The hooks run Ruff's linter and formatter plus Pyrefly. Ruff and Pyrefly hook revisions match the exact versions in the `dev` dependency group. The Pyrefly hook enters the project environment through `uv run --no-sync`, so an unrelated global installation cannot change results or trigger an unsafe environment sync.
* [ruff](https://github.com/astral-sh/ruff)
    * Linting rules from a wide variety of selectable rule sets
    * `ruff format` is used for formatting, and `ruff check` is used for linting.
    * The vendored `docs/haidra-assets` submodule is excluded; format it only in its owning repository.
    * See `pyproject.toml` for the rules used.
    * See all rules (but not necessarily used in the project) availible in rust [here](https://beta.ruff.rs/docs/rules/).
* [Pyrefly](https://pyrefly.org/)
    * Static type safety

## Code Style

* See the [haidra python style guide](docs/haidra-assets/docs/meta/python.md) for more details on code style and best practices.

## Testing

* Run the suite with `uv run pytest`. The default sweep is fast because three bands are **opt-in** and skipped unless you ask for them:
    * `-m slow` runs the tests that spawn real OS subprocesses (the end-to-end worker-lifecycle family) or take multiple seconds. Run this before pushing a change that touches the worker lifecycle.
    * `-m gpu` runs the tests that need a real accelerator; they auto-skip when no CUDA device is present.
    * `-m chaos_sweep` runs the generated wedge-liveness chaos sweep. See below; the default sweep runs its representative core slice instead.
* `-m "slow or gpu"` runs both opt-in bands at once. CI runs the fast sweep and the `slow` band as separate steps, so the full-lifecycle coverage is exercised on every push.

### The generated chaos sweep (pre-release gate)

Seeded scenarios compose heterogeneous per-job demand, model ordering, arrival shape, initial residency, a production-resolved worker topology, and a schedule of disturbances. Each is judged for end-to-end completion of all the work it queues, with no job given up on and no job waiting past a bound derived from its own shape. Run both tiers before a release, and after any change to admission, reclaim, residency, scheduling, or lifecycle:

```sh
uv run pytest tests/process_management/liveness/test_chaos_generated.py -m chaos_sweep
uv run pytest tests/e2e/test_chaos_generated_e2e.py -m "chaos_sweep and slow"
```

The first drives the scheduling loop on a fake clock (minutes). Its committed range is a covering array: a meta-test requires every pair of card, queue grammar, arrival, thread/queue request, demand shape, initial residency, and sequence-length values to appear. Requested modelled-card disturbances must produce an effective receipt; a no-op injection fails the row. The second boots a worker with real child processes per scenario (longer, and it needs both marker names because it is also in the `slow` band). It uses production-derived lane counts, verifies the real manager resolved the same topology, and restricts child fault scripts to a single lane and one event so process-local job ordinals remain unambiguous. Both print how many scenarios they ran and which axes the generated space does not explore. The same pair runs nightly, and on demand, through the `Chaos Sweep` workflow.

The fast configuration-space contract exhaustively crosses the semantic boundaries for `max_threads`, `queue_size`, configured model count, workload role, sampling-lease slots, and tail overlap. It runs in the ordinary test suite; the stateful sweep then spends its budget on behavioral paths rather than repeating cheap resolver arithmetic.

Set `HORDE_CHAOS_SEEDS` to replay or widen: `HORDE_CHAOS_SEEDS=1063` for a single seed, `HORDE_CHAOS_SEEDS=2000:2500` for a range, `HORDE_CHAOS_SEEDS=7,19,23` for a list. Every failure prints the seed and the whole scenario, so a red run replays from its message.

## Pull Requests

* We welcome community contributions to horde_worker_reGen! If you have an idea for a new feature, bug fix, or improvement, please feel free to submit a pull request.
* Before submitting a pull request, please ensure that your code follows the project's coding standards and that you have added appropriate tests for your changes.
* When submitting a pull request, please provide a clear description of the changes you have made and the problem you are trying to solve.
* We will review your pull request as soon as possible and provide feedback or merge it if it meets our standards. Thank you for contributing to horde_worker_reGen!
