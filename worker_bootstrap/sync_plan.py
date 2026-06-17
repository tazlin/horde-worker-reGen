"""Type-safe planning of what a dependency sync would download, and whether a big upgrade is skippable.

The managed install leaves torch/torchvision unbounded but pins exact versions in ``uv.lock``, so a
torch version only advances when a new *release* ships a new lock. Each bump downloads ~1.5 GB+, so
before running the real ``uv sync`` we show the user what would change and let them "limp along" on the
currently-installed torch when nothing actually forces the upgrade.

Whether a hold is allowed (OPTIONAL) or forbidden (REQUIRED) is decided by uv's own resolver, not by us
parsing constraints: ``uv.lock`` records original version specifiers only for the root project (where
torch is unbounded), never the transitive floor a dependency like horde-engine might declare. The
authoritative test is a held dry-run (``runner.uv_sync_held(..., dry_run=True)``); this module consumes
its boolean result via the ``holdable`` flag and never tries to reconstruct the floor itself.

This module is part of the bootstrap brain, so it must stay standard-library only (see
``tests/bootstrap/test_stdlib_only.py``): version comparison is a small PEP 440 subset, not ``packaging``.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "ChangeKind",
    "PackageChange",
    "SyncPlan",
    "build_plan",
    "decide",
    "format_sync_plan",
    "held_overrides_text",
    "installed_versions",
    "is_held_upgrade",
    "parse_dry_run",
    "version_at_least",
]

# Packages we pin in the override when the user opts to limp along. Holding torch holds its CUDA runtime
# deps transitively; torchvision is held alongside it so the pair always shares one build.
_HELD_PACKAGES = ("torch", "torchvision")

# Rough per-package download sizes, used only to decide whether to prompt and to show an approximate
# total. They are deliberately coarse constants (real wheels vary by build); unknown packages contribute
# nothing and flip the plan's ``sizes_complete`` flag so the total is shown as a lower bound.
_GiB = 1024**3
_MiB = 1024**2
_SIZE_EXACT: dict[str, int] = {
    "torch": 2 * _GiB,
    "torchvision": 8 * _MiB,
    "torchaudio": 8 * _MiB,
    "xformers": 200 * _MiB,
}
_SIZE_PREFIX: tuple[tuple[str, int], ...] = (
    ("nvidia-", 400 * _MiB),
    ("pytorch-triton", 250 * _MiB),
    ("triton", 250 * _MiB),
)

# Lines uv prints for an install plan: "+ name==version" (added) and "- name==version" (removed). An
# upgrade shows both for the same name. Tolerant on purpose: unrecognized lines are ignored so a uv
# output-format change degrades to "no changes parsed" (and the caller falls back to a normal sync).
_CHANGE_LINE_RE = re.compile(r"^\s*([+\-])\s+([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")


class ChangeKind(StrEnum):
    """How a single package would change during a sync."""

    INSTALL = "install"
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    REMOVE = "remove"


@dataclass(frozen=True)
class PackageChange:
    """One package's change in a planned sync."""

    name: str
    from_version: str | None
    to_version: str | None
    kind: ChangeKind
    est_download_bytes: int | None
    is_large: bool
    required: bool
    """For a held-candidate upgrade: True when uv could not resolve the hold, so it is mandatory."""


@dataclass(frozen=True)
class SyncPlan:
    """The aggregate picture of what a sync would download."""

    changes: tuple[PackageChange, ...]
    total_download_bytes: int
    cache_dir: str
    cache_is_owned: bool
    free_disk_bytes: int | None
    holdable: bool
    """True when at least one large upgrade exists AND uv confirmed a hold resolves."""

    @property
    def large_upgrades(self) -> tuple[PackageChange, ...]:
        """Return the held-candidate upgrades (torch/torchvision moving up with a known prior version)."""
        return tuple(c for c in self.changes if is_held_upgrade(c))

    @property
    def large_changes(self) -> tuple[PackageChange, ...]:
        """Return every change to a large package (for emphasis in the rendered table)."""
        return tuple(c for c in self.changes if c.is_large)

    @property
    def has_skippable_upgrade(self) -> bool:
        """Return whether the user may limp along (a large upgrade exists and the hold resolves)."""
        return self.holdable and bool(self.large_upgrades)

    @property
    def sizes_complete(self) -> bool:
        """Return whether every downloading change contributed a known size (else total is a lower bound)."""
        return all(
            c.est_download_bytes is not None
            for c in self.changes
            if c.kind in (ChangeKind.INSTALL, ChangeKind.UPGRADE, ChangeKind.DOWNGRADE)
        )

    @property
    def fits(self) -> bool:
        """Return whether the download fits in free space (True when free space is unknown)."""
        if self.free_disk_bytes is None:
            return True
        return self.total_download_bytes <= self.free_disk_bytes


def _normalize(name: str) -> str:
    """Return the PEP 503-ish normalized package name (lowercase, runs of -_. collapsed to a single -)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _public_version(version: str) -> str:
    """Strip a local segment (the ``+cu132`` in ``2.12.1+cu132``); the index supplies the right build."""
    return version.split("+", 1)[0]


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a PEP 440-subset release into a tuple of ints for comparison (local/pre segments dropped)."""
    release = _public_version(version).split("!", 1)[-1]
    parts: list[int] = []
    for piece in release.split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts) or (0,)


def version_at_least(installed: str, target: str) -> bool:
    """Return whether *installed* is >= *target* comparing the PEP 440-subset release (local segment ignored)."""
    return _version_tuple(installed) >= _version_tuple(target)


def is_large(name: str) -> bool:
    """Return whether a package is one of the large downloads worth calling out / sizing specially."""
    normalized = _normalize(name)
    if normalized in _SIZE_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix, _ in _SIZE_PREFIX)


def is_held_upgrade(change: PackageChange) -> bool:
    """Return whether a change is a torch/torchvision upgrade we could hold (a known prior version)."""
    return (
        _normalize(change.name) in _HELD_PACKAGES
        and change.kind == ChangeKind.UPGRADE
        and change.from_version is not None
    )


def estimate_download_bytes(name: str) -> int | None:
    """Return a coarse download-size estimate for a package, or None when it is not in the size table."""
    normalized = _normalize(name)
    if normalized in _SIZE_EXACT:
        return _SIZE_EXACT[normalized]
    for prefix, size in _SIZE_PREFIX:
        if normalized.startswith(prefix):
            return size
    return None


def installed_versions(venv_dir: Path) -> dict[str, str]:
    """Return ``{normalized_name: version}`` for packages installed in ``venv_dir`` (no imports).

    Reads ``*.dist-info`` directory names from the venv's site-packages without importing anything, so it
    is safe to call before/without torch. Versions keep any local segment (e.g. ``2.12.1+cu132``).
    """
    versions: dict[str, str] = {}
    for dist_info in venv_dir.rglob("*.dist-info"):
        stem = dist_info.name[: -len(".dist-info")]
        name, sep, version = stem.rpartition("-")
        if not sep or not name:
            continue
        versions[_normalize(name)] = version
    return versions


def parse_dry_run(output: str) -> list[PackageChange]:
    """Parse ``uv sync --dry-run`` output into a list of package changes.

    uv prints ``+ name==version`` for additions and ``- name==version`` for removals; an upgrade shows
    both lines for one package. We reconcile them per package into INSTALL / UPGRADE / DOWNGRADE / REMOVE.
    Lines that do not match are ignored, so an unexpected uv format yields an empty list and the caller
    falls back to a normal sync.
    """
    added: dict[str, str] = {}
    removed: dict[str, str] = {}
    for line in output.splitlines():
        match = _CHANGE_LINE_RE.match(line)
        if not match:
            continue
        sign, raw_name, version = match.groups()
        target = added if sign == "+" else removed
        target[_normalize(raw_name)] = version

    changes: list[PackageChange] = []
    for name in sorted(added.keys() | removed.keys()):
        to_version = added.get(name)
        from_version = removed.get(name)
        if to_version is not None and from_version is not None:
            kind = (
                ChangeKind.UPGRADE
                if _version_tuple(to_version) >= _version_tuple(from_version)
                else ChangeKind.DOWNGRADE
            )
        elif to_version is not None:
            kind = ChangeKind.INSTALL
        else:
            kind = ChangeKind.REMOVE
        downloads = kind in (ChangeKind.INSTALL, ChangeKind.UPGRADE, ChangeKind.DOWNGRADE)
        changes.append(
            PackageChange(
                name=name,
                from_version=from_version,
                to_version=to_version,
                kind=kind,
                est_download_bytes=estimate_download_bytes(name) if downloads else 0,
                is_large=is_large(name),
                required=False,
            ),
        )
    return changes


def held_overrides_text(changes: Iterable[PackageChange], installed: dict[str, str]) -> str | None:
    """Return the uv override file body pinning held packages at their installed version, or None.

    Returns None when there is nothing to hold (no torch/torchvision upgrade), in which case limping
    along is not possible. The public version is used (e.g. ``torch==2.12.1``); the per-extra index in
    ``pyproject.toml`` supplies the matching ``+cuXXX`` build.
    """
    lines: list[str] = []
    for change in changes:
        if not is_held_upgrade(change):
            continue
        held_version = installed.get(_normalize(change.name)) or change.from_version
        if held_version is None:
            return None
        lines.append(f"{change.name}=={_public_version(held_version)}")
    return "\n".join(lines) + "\n" if lines else None


def build_plan(
    changes: Sequence[PackageChange],
    *,
    holdable: bool,
    cache_dir: str,
    cache_is_owned: bool,
    free_disk_bytes: int | None,
) -> SyncPlan:
    """Assemble a :class:`SyncPlan` from parsed changes and the (uv-confirmed) hold feasibility.

    ``holdable`` is the result of the held dry-run: when False, any large upgrade is marked REQUIRED.
    """
    finalized = tuple(
        PackageChange(
            name=c.name,
            from_version=c.from_version,
            to_version=c.to_version,
            kind=c.kind,
            est_download_bytes=c.est_download_bytes,
            is_large=c.is_large,
            required=is_held_upgrade(c) and not holdable,
        )
        for c in changes
    )
    total = sum(c.est_download_bytes or 0 for c in finalized)
    return SyncPlan(
        changes=finalized,
        total_download_bytes=total,
        cache_dir=cache_dir,
        cache_is_owned=cache_is_owned,
        free_disk_bytes=free_disk_bytes,
        holdable=holdable and any(is_held_upgrade(c) for c in finalized),
    )


def free_bytes(path: Path) -> int | None:
    """Return free bytes on the volume holding ``path``, or None when it cannot be determined."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def human_bytes(num: int | None) -> str:
    """Format a byte count as a short human string (e.g. ``1.8 GiB``); ``?`` when unknown."""
    if num is None:
        return "?"
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def format_sync_plan(plan: SyncPlan) -> str:
    """Render the plan as a console table plus a summary footer."""
    header = ("PACKAGE", "FROM", "TO", "SIZE", "CLASS")
    rows: list[tuple[str, str, str, str, str]] = [header]
    for change in plan.changes:
        if change.kind == ChangeKind.REMOVE:
            klass = "remove"
        elif is_held_upgrade(change):
            klass = "REQUIRED" if change.required else "optional"
        else:
            klass = "-"
        size = "-" if change.kind == ChangeKind.REMOVE else human_bytes(change.est_download_bytes)
        rows.append(
            (
                change.name,
                change.from_version or "-",
                change.to_version or "-",
                size,
                klass,
            ),
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]

    total = human_bytes(plan.total_download_bytes)
    approx = "" if plan.sizes_complete else " (lower bound; some sizes unknown)"
    owned = "owned" if plan.cache_is_owned else "shared/external"
    lines.append("")
    lines.append(f"Total download ~{total}{approx}; free {human_bytes(plan.free_disk_bytes)}.")
    lines.append(f"Cache: {plan.cache_dir} ({owned}).")
    return "\n".join(lines)


def decide(
    plan: SyncPlan,
    *,
    hold_requested: bool,
    headless: bool,
    headless_policy: str,
    confirm_threshold_bytes: int,
    interactive: bool,
    prompt: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> str:
    """Decide what to do about a planned sync: ``"proceed_full"``, ``"hold"``, or ``"abort"``.

    Args:
        plan: The computed sync plan (carries uv-confirmed hold feasibility).
        hold_requested: The user explicitly asked to limp along (``--hold-torch`` / env).
        headless: No interactive terminal, or consent was captured upstream (installer/CI).
        headless_policy: ``"proceed"`` or ``"hold"`` for big optional upgrades in headless runs.
        confirm_threshold_bytes: Download size above which an interactive run must confirm.
        interactive: Whether a terminal prompt is possible.
        prompt: Input function (injectable for tests).
        emit: Output function (injectable for tests).
    """
    if not plan.large_upgrades:
        return "proceed_full"

    if not plan.has_skippable_upgrade:
        if hold_requested:
            floor = ", ".join(c.name for c in plan.large_upgrades)
            emit(f"This {floor} upgrade is mandatory (a dependency requires the newer version); cannot limp along.")
        return "proceed_full"

    if hold_requested:
        return "hold"

    if plan.total_download_bytes < confirm_threshold_bytes:
        return "proceed_full"

    if headless:
        return "hold" if headless_policy == "hold" else "proceed_full"

    if not interactive:
        return "proceed_full"

    answer = (
        prompt(
            f"This update downloads ~{human_bytes(plan.total_download_bytes)}. "
            "[U]pgrade now / [H]old current versions / [C]ancel? ",
        )
        .strip()
        .lower()
    )
    if answer in ("h", "hold"):
        return "hold"
    if answer in ("c", "cancel"):
        return "abort"
    return "proceed_full"
