"""Standard-library-only bootstrap brain for the AI Horde Worker.

This package is deliberately free of third-party imports at module load time. The platform shims run it
through ``uv run --python 3.12 --no-project --script bootstrap.py`` *before* the project virtual
environment exists, so nothing here (or anything it imports) may depend on the worker's installed
dependencies. It owns GPU detection, torch-build selection, config seeding, ``uv sync``, and launching the
worker, replacing orchestration that used to be duplicated across the ``.cmd``/``.ps1``/``.sh`` scripts.
"""
