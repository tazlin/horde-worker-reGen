# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Entry point for the AI Horde Worker bootstrap.

The platform shims run this as ``uv run --python 3.12 --no-project --script bootstrap.py <subcommand>``.
It is deliberately tiny and standard-library only: uv provisions a Python and runs this *before* the
project virtual environment exists, so the heavy worker dependencies are not importable yet. All logic
lives in the sibling ``worker_bootstrap/`` package (also standard-library only) so it can be unit-tested.
"""

import contextlib
import json
import shutil
import sys
from pathlib import Path


def _remove_recovery_path(path: Path) -> None:
    """Remove one recovery target without following directory symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _copy_recovery_path(src: Path, dst: Path) -> None:
    """Restore one recovery target from its durable backup."""
    _remove_recovery_path(dst)
    if src.is_dir() and not src.is_symlink():
        shutil.copytree(src, dst, symlinks=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst, follow_symlinks=False)


def _recover_interrupted_update(root: Path) -> bool:
    """Restore source after a terminated overlay, before importing any bootstrap package code."""
    marker = root / "bin" / "update-transaction.json"
    backup = root / "bin" / "update-backup"
    if not marker.exists():
        return False
    transaction = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(transaction, dict) or transaction.get("schema") != 1:
        raise OSError("The update recovery marker has an unsupported format.")
    targets = transaction.get("targets")
    if not isinstance(targets, list):
        raise OSError("The update recovery marker has no target list.")
    validated: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for entry in targets:
        if not isinstance(entry, dict) or not isinstance(entry.get("existed"), bool):
            raise OSError("The update recovery marker contains an invalid target entry.")
        name = entry.get("name")
        candidate = Path(name) if isinstance(name, str) else None
        if (
            not isinstance(name, str)
            or not name
            or candidate is None
            or candidate.name != name
            or name in (".", "..", "bin", ".venv", ".horde-sync-stamp", "bridgeData.yaml", "logs")
            or candidate.suffix.lower() in (".cmd", ".sh", ".ps1", ".bat")
        ):
            raise OSError(f"Invalid update recovery target {name!r}.")
        if name in seen:
            raise OSError(f"The update recovery marker repeats target {name!r}.")
        seen.add(name)
        existed = entry["existed"]
        if existed:
            saved = backup / name
            if not (saved.exists() or saved.is_symlink()):
                raise OSError(f"The update recovery backup is missing {name!r}.")
        validated.append((name, existed))

    # Preflight the entire backup before deleting any partial source. If the backup itself is damaged,
    # leave the install untouched and direct the operator to the idempotent installer recovery path.
    for name, existed in validated:
        target = root / name
        _remove_recovery_path(target)
        if existed:
            saved = backup / name
            _copy_recovery_path(saved, target)
    marker.unlink()
    with contextlib.suppress(OSError):
        _remove_recovery_path(backup)
    return True


# When uv runs this script, sys.path[0] is already this file's directory, so the sibling package imports
# cleanly. Insert it explicitly too, so the script also works when invoked by other means.
_ROOT = Path(__file__).resolve().parent
try:
    if _recover_interrupted_update(_ROOT):
        print("Recovered an interrupted worker update; the previous version is intact. Re-run update to try again.")
except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
    print(
        f"ERROR: Could not recover an interrupted worker update: {error} "
        "Re-run the latest installer over this folder without deleting worker data.",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

sys.path.insert(0, str(_ROOT))

from worker_bootstrap.cli import main  # noqa: E402  (must follow the sys.path bootstrap above)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
