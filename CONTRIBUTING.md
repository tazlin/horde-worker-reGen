# Contributing to horde_worker_reGen

## Code Quality Tools

* [uv](https://docs.astral.sh/uv/)
    * A modern Python package manager and virtual environment tool
    * Run `uv sync --extra <extra>` to install dependencies and `uv run <command>` to run commands in the virtual environment.
        * See `pyproject.toml` for the dependencies and extras used in this project - you need to specify your GPU specific torch.
* [pre-commit](https://pre-commit.com/)
    * Creates virtual environments for formatting and linting tools
    * Run `pre-commit run --all-files` or see `.pre-commit-config.yaml` for more info.
* [ruff](https://github.com/astral-sh/ruff)
    * Linting rules from a wide variety of selectable rule sets
    * `ruff format` is used for formatting, and `ruff check` is used for linting.
    * See `pyproject.toml` for the rules used.
    * See all rules (but not necessarily used in the project) availible in rust [here](https://beta.ruff.rs/docs/rules/).
* [Pyrefly](https://pyrefly.org/)
    * Static type safety

## Things to know

* The `AI_HORDE_DEV_URL` environment variable overrides `AI_HORDE_URL`. This is useful for testing changes locally.
* pytest files which end in `_api_calls.py` run last, and never run during the CI. It is currently incumbent on individual developers to confirm that these tests run successfully locally. In the future, part of the CI will be to spawn an AI-Horde and worker instances and test it there.
