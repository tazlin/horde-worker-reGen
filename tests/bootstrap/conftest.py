"""Make the bundled (un-installed) ``worker_bootstrap`` package importable for these tests.

``worker_bootstrap`` ships as a root-level bundle directory, not a wheel package, so it is not on
``sys.path`` via the editable install the way ``horde_worker_regen`` is. Insert the repo root here so the
bootstrap unit tests can import it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
