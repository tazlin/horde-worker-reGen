"""Backend (torch build) selection: precedence, the legacy cu128 remap, and locked-extra validation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

# Builds whose torch wheels are pinned in uv.lock and installable with ``uv sync --locked --extra <build>``.
# Used only as fallback guidance text; the authoritative list is read from pyproject.toml at runtime. ROCm
# and amd-unsupported are intentionally absent (ROCm profiles install ad-hoc; amd-unsupported has no build).
_LOCKED_HINT = ("cu126", "cu130", "cu132", "cpu")

# Optional feature extras (see pyproject ``[project.optional-dependencies]``). These re-export
# horde-engine feature extras (onnxruntime/mediapipe for controlnet; rembg for post-processing) and are
# NOT torch builds, so they must be excluded from build-token validation and added separately.
FEATURE_EXTRAS: tuple[str, ...] = ("controlnet", "post-processing")

# Builds whose wheels include the feature deps, so they default to the full feature set (preserving the
# pre-split behaviour where these deps were always installed). Lean backends (rocm/xpu/mps/...) default
# to none and opt in via the HORDE_WORKER_FEATURES env var instead.
_FULL_FEATURE_BUILDS = frozenset({"cu126", "cu130", "cu132", "cpu"})


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
    """Read the torch-build extras uv.lock can install, from ``[project.optional-dependencies]``.

    Feature extras (:data:`FEATURE_EXTRAS`) are excluded: they are not torch builds and must not be
    accepted as a backend token.
    """
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    extras = data.get("project", {}).get("optional-dependencies", {})
    return sorted(key for key in extras if key not in FEATURE_EXTRAS)


def desired_feature_extras(token: str, *, env_value: str | None = None) -> tuple[str, ...]:
    """Return the feature extras to install for *token*, honouring a ``HORDE_WORKER_FEATURES`` override.

    Full builds (NVIDIA/CPU, :data:`_FULL_FEATURE_BUILDS`) default to every feature extra so existing
    installs keep post-processing/controlnet with no change. Lean backends (ROCm/XPU/MPS) default to
    none. The override is a comma or space separated list of extra names; the literal ``none`` (or an
    empty value) forces the lean set even on a full build.

    Raises:
        ValueError: When the override names an extra that is not a known feature extra.
    """
    if env_value is not None and env_value.strip():
        if env_value.strip().lower() == "none":
            return ()
        names = tuple(name for name in re.split(r"[,\s]+", env_value.strip()) if name)
        unknown = [name for name in names if name not in FEATURE_EXTRAS]
        if unknown:
            raise ValueError(
                f"unknown feature extra(s) in HORDE_WORKER_FEATURES: {', '.join(unknown)}. "
                f"Known feature extras: {', '.join(FEATURE_EXTRAS)}.",
            )
        return names
    if token in _FULL_FEATURE_BUILDS:
        return FEATURE_EXTRAS
    return ()


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
