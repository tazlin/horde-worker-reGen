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

import sys
from pathlib import Path

# When uv runs this script, sys.path[0] is already this file's directory, so the sibling package imports
# cleanly. Insert it explicitly too, so the script also works when invoked by other means.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker_bootstrap.cli import main  # noqa: E402  (must follow the sys.path bootstrap above)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
