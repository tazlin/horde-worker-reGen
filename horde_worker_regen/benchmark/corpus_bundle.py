"""Package one pricing-corpus run as a self-describing bundle.

A run leaves two things behind that only mean something together: the definition artifact and the
stats parts the worker wrote while the run's jobs flowed. Both are named by a local-clock stamp, neither
name carries the machine, and the two stamps differ, so on a maintainer's disk holding several
contributors' runs the pairing is guesswork. The bundle removes the guesswork: one directory named by
machine, tier and a UTC stamp, holding copies of exactly those files plus a manifest with their hashes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from horde_worker_regen.benchmark.pricing_corpus import SCENARIO_NAME, PricingCorpusDefinition

BUNDLE_MANIFEST_NAME = "bundle.json"
BUNDLE_FORMAT_VERSION = "1"

SESSION_SEARCH_WINDOW_SECONDS = 3600.0
"""How long after the definition was written its stats session may have started."""

_STATS_PART_RE = re.compile(r"^(?P<stem>stats-v.+-\d{8}-\d{6})-(?P<index>\d+)\.jsonl(?:\.gz)?$")
"""The worker's stats file name: one session stamp, rotated into numbered parts at a size limit."""


class BundleError(RuntimeError):
    """Raised when a run's stats session cannot be paired with its definition."""


@dataclass(frozen=True)
class BundledFile:
    """One file copied into the bundle."""

    name: str
    role: str
    sha256: str
    bytes: int


def utc_stamp(epoch_seconds: float) -> str:
    """Render epoch seconds as a compact UTC stamp, sortable across machines and timezones."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch_seconds))


def bundle_name(definition: PricingCorpusDefinition) -> str:
    """Return the bundle directory name for a stamped definition."""
    if definition.machine is None or definition.created_at is None:
        raise BundleError("only a definition stamped with a machine and a write time can be bundled")
    return f"corpus-{definition.machine.machine_id}-{definition.tier}-{utc_stamp(definition.created_at)}"


def _session_start(path: Path) -> dict[str, object] | None:
    """Read the ``session_start`` event of a stats part, or None when it has none."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if event.get("event") == "session_start":
                return event
            if event.get("event") == "job_completed":
                return None
    return None


def find_session_parts(stats_dir: Path, definition: PricingCorpusDefinition) -> list[Path]:
    """Find the stats parts of the session this definition describes, in rotation order.

    The stats stream opens a few seconds after the artifact is written, so the session is the one whose
    ``session_start`` carries this corpus's scenario id and revision and the nearest timestamp at or
    after ``created_at`` within the search window. A session that started earlier belongs to another run.
    """
    if definition.created_at is None:
        raise BundleError("the definition carries no created_at, so its stats session cannot be located")
    candidates: list[tuple[float, str]] = []
    for path in stats_dir.iterdir():
        match = _STATS_PART_RE.match(path.name)
        if match is None or int(match.group("index")) != 0:
            continue
        start = _session_start(path)
        if start is None:
            continue
        config = start.get("config")
        if not isinstance(config, dict):
            continue
        if config.get("scenario_id") != SCENARIO_NAME or str(config.get("scenario_revision")) != str(
            definition.scenario_revision
        ):
            continue
        raw_timestamp = start.get("timestamp", 0.0)
        timestamp = float(raw_timestamp) if isinstance(raw_timestamp, (int, float)) else 0.0
        if definition.created_at <= timestamp <= definition.created_at + SESSION_SEARCH_WINDOW_SECONDS:
            candidates.append((timestamp, match.group("stem")))
    if not candidates:
        raise BundleError(
            f"no {SCENARIO_NAME} revision {definition.scenario_revision} stats session in {stats_dir} started "
            f"within {SESSION_SEARCH_WINDOW_SECONDS:.0f}s after the definition was written",
        )
    _timestamp, stem = min(candidates)
    parts: dict[int, Path] = {}
    for path in stats_dir.iterdir():
        match = _STATS_PART_RE.match(path.name)
        if match is not None and match.group("stem") == stem:
            parts[int(match.group("index"))] = path
    return [parts[index] for index in sorted(parts)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    definition: PricingCorpusDefinition,
    *,
    definition_path: Path,
    stats_dir: Path,
    out_root: Path,
) -> Path:
    """Copy a run's definition and stats parts into a bundle directory and write its manifest.

    Args:
        definition: The stamped definition the run wrote.
        definition_path: Where the run wrote it.
        stats_dir: The worker's stats directory holding the run's session.
        out_root: The directory the bundle directory is created under.

    Returns:
        The bundle directory.

    Raises:
        BundleError: If the definition is unstamped or its session cannot be found.
    """
    parts = find_session_parts(stats_dir, definition)
    bundle_dir = out_root / bundle_name(definition)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files: list[BundledFile] = []
    for source, role in ((definition_path, "definition"), *((part, "stats") for part in parts)):
        target = bundle_dir / source.name
        shutil.copyfile(source, target)
        files.append(BundledFile(name=source.name, role=role, sha256=_sha256(target), bytes=target.stat().st_size))

    machine = definition.machine
    manifest = {
        "bundle_format": BUNDLE_FORMAT_VERSION,
        "machine": machine.model_dump(mode="json") if machine is not None else None,
        "tier": definition.tier,
        "scenario_name": definition.scenario_name,
        "scenario_revision": definition.scenario_revision,
        "created_at": definition.created_at,
        "created_at_utc": utc_stamp(definition.created_at) if definition.created_at is not None else None,
        "files": [file.__dict__ for file in files],
    }
    (bundle_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bundle_dir


__all__ = [
    "BUNDLE_MANIFEST_NAME",
    "BundleError",
    "BundledFile",
    "bundle_name",
    "find_session_parts",
    "utc_stamp",
    "write_bundle",
]
