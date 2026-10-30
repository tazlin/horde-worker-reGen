"""Backend (torch build) selection: precedence, provenance, the legacy cu128 remap, and locked-extra validation."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from worker_bootstrap import detect

_CPU_TOKEN = "cpu"

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


class BackendSource(StrEnum):
    """Where a backend token came from."""

    CLI = "cli"
    """A ``--backend`` flag (or one of its shortcuts)."""
    ENV = "env"
    """The ``HORDE_WORKER_BACKEND`` environment variable."""
    FILE = "file"
    """The persisted ``bin/backend`` token."""
    DETECTED = "detected"
    """Hardware detection."""
    DEFAULT = "default"
    """The fallback build, used when every other source was empty."""
    INTERACTIVE = "interactive"
    """The operator's answer to the CPU-or-GPU prompt."""


DETECTION_SOURCES: frozenset[BackendSource] = frozenset({BackendSource.DETECTED, BackendSource.DEFAULT})
"""Sources whose persisted token a sync re-picks against the live driver and cards. Every other source is explicit."""


@dataclass(frozen=True)
class ResolvedBackend:
    """Represents a resolved backend token and the source that won the precedence."""

    token: str
    source: BackendSource


def resolve_backend(
    *,
    cli_flag: str | None = None,
    env_value: str | None = None,
    file_value: str | None = None,
    detected: str | None = None,
    default: str = "cu126",
) -> ResolvedBackend:
    """Resolve the backend by precedence: CLI flag > env var > ``bin/backend`` > detection > default.

    The first non-empty source wins, then its token is passed through :func:`remap_legacy` so a stale
    cu128 becomes cu126. Detection is an optional input so a caller that cannot or should not probe
    hardware can omit it (the default then applies). A persisted ``file_value`` outranks ``detected``, so
    an install path re-picks a detection-derived file token itself and reconciles the result against the
    live GPU (see :func:`worker_bootstrap.detect.reconcile_backend_for_gpu`).
    """
    candidates = (
        (cli_flag, BackendSource.CLI),
        (env_value, BackendSource.ENV),
        (file_value, BackendSource.FILE),
        (detected, BackendSource.DETECTED),
    )
    for candidate, source in candidates:
        if candidate and candidate.strip():
            return ResolvedBackend(token=remap_legacy(candidate), source=source)
    return ResolvedBackend(token=default, source=BackendSource.DEFAULT)


@dataclass(frozen=True)
class InstallRecord:
    """Represents the provenance of the persisted ``bin/backend`` token (the breadcrumb's ``install`` entry)."""

    token: str
    source: BackendSource
    replaced_token: str | None = None
    """The token this one replaced, when a reconcile or re-pick changed it."""
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serializable breadcrumb entry."""
        return {
            "stage": "install",
            "final_token": self.token,
            "source": str(self.source),
            "replaced_token": self.replaced_token,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, entry: object) -> InstallRecord | None:
        """Parse a breadcrumb entry, or return None when it is missing, malformed, or lists an unknown source."""
        if not isinstance(entry, dict):
            return None
        token = entry.get("final_token")
        source = entry.get("source")
        if not isinstance(token, str) or not token or source not in set(BackendSource):
            return None
        replaced = entry.get("replaced_token")
        reason = entry.get("reason")
        return cls(
            token=token,
            source=BackendSource(source),
            replaced_token=replaced if isinstance(replaced, str) else None,
            reason=reason if isinstance(reason, str) else None,
        )


def persisted_source(token: str, record: InstallRecord | None) -> BackendSource:
    """Return where a persisted ``bin/backend`` token came from.

    The install record sets the source when it is for this token. A token with no matching record (an install
    from before provenance was recorded, or one the graphical installer wrote) counts as detected when it
    is a CUDA build and as an explicit file choice otherwise. The only choice the installer ever offered
    was CPU versus GPU, so a persisted ``cpu`` beside a GPU was deliberate.
    """
    if record is not None and record.token == token:
        return record.source
    return BackendSource.DETECTED if token in detect.CUDA_BUILDS else BackendSource.FILE


def choose_backend_interactively(
    detected_token: str,
    *,
    explicitly_chosen: bool,
    interactive: bool,
    prompt: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> str:
    """Offer an interactive user the choice of CPU / alchemist-only mode when a GPU build was detected.

    Returns the backend token to install. The choice exists so a user can deliberately run the worker
    without a GPU (CPU build, image generation disabled, alchemy-only) even on a machine that has one, for
    example to leave the GPU free for other work. A CPU-only build installs no CUDA and is ~100x slower.

    The detected token is returned unchanged when the user already chose a backend explicitly (a CLI flag
    or ``HORDE_WORKER_BACKEND``), when the run is non-interactive, or when no GPU was detected at all
    (the token is already ``cpu`` and there is nothing to choose). Only an auto-detected GPU build prompts.

    Args:
        detected_token: The backend token detection resolved (e.g. ``cu132`` or ``cpu``).
        explicitly_chosen: Whether the backend came from a CLI flag or env var (so do not second-guess it).
        interactive: Whether a terminal prompt is possible.
        prompt: Input function (injectable for tests).
        emit: Output function (injectable for tests).
    """
    if explicitly_chosen or not interactive or detected_token == _CPU_TOKEN:
        return detected_token

    emit(
        f"A GPU backend ({detected_token}) was detected. You can instead install the CPU-only build to run "
        "in alchemist-only mode (no image generation, only upscaling, face-fixing and interrogation). "
        "CPU is ~100x slower and mainly useful without a usable GPU, or to leave the GPU free for other work.",
    )
    answer = prompt("Install for [G]PU (recommended) or [C]PU only / alchemist-only? [G/c] ").strip().lower()
    if answer in ("c", "cpu"):
        return _CPU_TOKEN
    return detected_token


def choose_mixed_card_policy_interactively(
    builds: dict[detect.MixedCardPolicy, str],
    capabilities: tuple[tuple[int, int], ...],
    *,
    prompt: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> detect.MixedCardPolicy:
    """Alert an interactive user that the cards need different builds and ask which policy to apply.

    Args:
        builds: The build each policy picks (see :func:`worker_bootstrap.detect.mixed_card_builds`).
        capabilities: Every card's compute capability.
        prompt: Input function (injectable for tests).
        emit: Output function (injectable for tests).

    Returns:
        The chosen policy. An empty or unrecognised answer is ``all``, the non-interactive default.
    """
    cards = ", ".join(detect.format_capability(cap) for cap in sorted(set(capabilities)))
    emit(
        f"This machine has NVIDIA cards that need different PyTorch builds (compute capability {cards}). "
        "A card the installed build does not carry kernels for cannot run jobs.",
    )
    labels = {
        detect.MixedCardPolicy.ALL: "[A]ll cards",
        detect.MixedCardPolicy.HIGHEST: "[H]ighest card",
        detect.MixedCardPolicy.LOWEST: "[L]owest card",
    }
    for policy, label in labels.items():
        build = builds[policy]
        stranded = detect.cards_without_kernels(build, capabilities)
        note = (
            f", no kernels for compute capability {', '.join(detect.format_capability(cap) for cap in stranded)}"
            if stranded
            else ", runs every card"
        )
        emit(f"  {label}: installs {build}{note}")
    answer = prompt("Install for [A]ll cards (recommended), the [H]ighest card or the [L]owest card? [A/h/l] ")
    choice = answer.strip().lower()
    if choice in ("h", "highest"):
        return detect.MixedCardPolicy.HIGHEST
    if choice in ("l", "lowest"):
        return detect.MixedCardPolicy.LOWEST
    return detect.MixedCardPolicy.ALL


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


def expects_utilities_lock(token: str) -> bool:
    """Return whether the committed image-utilities dependency lock is expected to cover *token*.

    The full builds (:data:`_FULL_FEATURE_BUILDS`) default the utilities lane on and are each a build extra
    of the utilities uv project (``requirements/utilities/``, whose lock is pinned by
    ``tests/bootstrap/test_utilities_env.py``), so for them a *missing* lock is a broken or incomplete
    install rather than the benign "not covered" case a feature-opted-in lean backend hits. This lets the
    provisioner distinguish the two and warn only on the former.
    """
    return token in _FULL_FEATURE_BUILDS


def validate_locked_extra(token: str, pyproject_path: Path) -> None:
    """Raise ValueError with actionable guidance when ``token`` is not a locked build extra."""
    available = locked_extras(pyproject_path) or list(_LOCKED_HINT)
    if token in available:
        return
    raise ValueError(
        f"'{token}' is not a locked build. The lockfile has torch builds for: "
        f"{', '.join(available)}. For AMD/ROCm or an older torch version, install torch by hand, for example: "
        "UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio",
    )
