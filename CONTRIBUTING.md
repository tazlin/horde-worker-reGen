# Contributing to horde_worker_reGen

## Code Quality Tools

* [uv](https://docs.astral.sh/uv/)
    * A modern Python package manager and virtual environment tool
    * Run `uv sync --extra <extra>` to install dependencies and `uv run <command>` to run commands in the virtual environment.
        * See `pyproject.toml` for the dependencies and extras used in this project - you need to specify your GPU specific torch.
* [prek](https://github.com/j178/prek)
    * Pre-commit compatible hooks for code quality and formatting
    * Run `prek run --all-files` or see `.pre-commit-config.yaml` for more info.
* [ruff](https://github.com/astral-sh/ruff)
    * Linting rules from a wide variety of selectable rule sets
    * `ruff format` is used for formatting, and `ruff check` is used for linting.
    * See `pyproject.toml` for the rules used.
    * See all rules (but not necessarily used in the project) availible in rust [here](https://beta.ruff.rs/docs/rules/).
* [Pyrefly](https://pyrefly.org/)
    * Static type safety

## Code Style

* See the [haidra python style guide](docs/haidra-assets/docs/meta/python.md) for more details on code style and best practices.

## Testing

* Run the suite with `uv run pytest`. The default sweep is fast because two bands are **opt-in** and skipped unless you ask for them:
    * `-m slow` runs the tests that spawn real OS subprocesses (the end-to-end worker-lifecycle family) or take multiple seconds. Run this before pushing a change that touches the worker lifecycle.
    * `-m gpu` runs the tests that need a real accelerator; they auto-skip when no CUDA device is present.
* `-m "slow or gpu"` runs both opt-in bands at once. CI runs the fast sweep and the `slow` band as separate steps, so the full-lifecycle coverage is exercised on every push.

## Pull Requests

* We welcome community contributions to horde_worker_reGen! If you have an idea for a new feature, bug fix, or improvement, please feel free to submit a pull request.
* Before submitting a pull request, please ensure that your code follows the project's coding standards and that you have added appropriate tests for your changes.
* When submitting a pull request, please provide a clear description of the changes you have made and the problem you are trying to solve.
* We will review your pull request as soon as possible and provide feedback or merge it if it meets our standards. Thank you for contributing to horde_worker_reGen!
