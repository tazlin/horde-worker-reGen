"""Backend (torch build) selection: precedence, the legacy cu128 remap, and locked-extra validation."""

from __future__ import annotations

import tomllib
from pathlib import Path

# Builds whose torch wheels are pinned in uv.lock and installable with ``uv sync --locked --extra <build>``.
# Used only as fallback guidance text; the authoritative list is read from pyproject.toml at runtime. ROCm
# and amd-unsupported are intentionally absent (ROCm installs ad-hoc; amd-unsupported has no build).
_LOCKED_HINT = ("cu126", "cu130", "cu132", "cpu")


def remap_legacy(token: str) -> str:
    """Map a retired build token onto its current equivalent.

    torch 2.12.0 publishes no cu128 wheel, so an existing install whose ``bin/backend`` still says cu128 is
    routed to cu126 (which runs on any CUDA 12.6+ driver). Other tokens pass through unchanged.
    """
    cleaned = token.strip()
    if cleaned.lower() == "cu128":
        return "cu126"
    return cleaned


def read_backend_file(path: Path) -> str | None:
    """Return the persisted backend token from ``bin/backend``, or None when absent/empty."""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def write_backend_file(path: Path, token: str) -> None:
    """Persist the chosen backend token to ``bin/backend`` (no trailing newline), creating ``bin/``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")


def resolve_backend(
    *,
    cli_flag: str | None = None,
    env_value: str | None = None,
    file_value: str | None = None,
    detected: str | None = None,
    default: str = "cu126",
) -> str:
    """Resolve the backend by precedence: CLI flag > env var > ``bin/backend`` > detection > default.

    The first non-empty source wins, then the result is passed through :func:`remap_legacy` so a stale
    cu128 becomes cu126. Detection is an optional input so callers that must not probe hardware (e.g.
    ``sync``) can omit it.
    """
    for candidate in (cli_flag, env_value, file_value, detected, default):
        if candidate and candidate.strip():
            return remap_legacy(candidate)
    return default


def locked_extras(pyproject_path: Path) -> list[str]:
    """Read the build extras uv.lock can install, from ``[project.optional-dependencies]``."""
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    extras = data.get("project", {}).get("optional-dependencies", {})
    return sorted(extras.keys())


def validate_locked_extra(token: str, pyproject_path: Path) -> None:
    """Raise ValueError with actionable guidance when ``token`` is not a locked build extra."""
    available = locked_extras(pyproject_path) or list(_LOCKED_HINT)
    if token in available:
        return
    raise ValueError(
        f"'{token}' is not a locked build. The lockfile provides the latest torch for: "
        f"{', '.join(available)}. For AMD/ROCm or an older torch line, install ad-hoc, e.g.: "
        "UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio",
    )
