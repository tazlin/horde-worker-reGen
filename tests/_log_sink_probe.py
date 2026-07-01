"""Subprocess helper for ``test_log_file_registry``.

Triggers a real logging setup in isolation and reports the loguru file-sink basenames it registers.

Run as a subprocess (never imported as a test) so the destructive side effects of the real setups (the
child mode reassigns ``sys.stdout``/``sys.stderr`` and registers atexit sinks, hordelib emits startup
noise) land in a throwaway working directory instead of the test runner. Invoked as::

    python tests/_log_sink_probe.py <mode> <workdir> <out.json>

where ``mode`` selects which setup to run. The discovered basenames are written to ``out.json`` (not
stdout, which some modes redirect to a file).
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    """Run the requested logging setup and dump its registered file-sink basenames to a JSON file."""
    mode, workdir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    os.chdir(workdir)
    os.makedirs("logs", exist_ok=True)

    if mode == "hordelib-main":
        from hordelib.utils.logger import HordeLog

        HordeLog.initialise(setup_logging=True, process_id=None, verbosity_count=3)
    elif mode == "hordelib-child":
        from hordelib.utils.logger import HordeLog

        HordeLog.initialise(setup_logging=True, process_id=0, verbosity_count=3)
    elif mode == "supervisor-tui":
        from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging

        setup_supervisor_file_logging("tui", quiet_console=True)
    elif mode == "supervisor-host":
        from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging

        setup_supervisor_file_logging("host")
    else:
        raise SystemExit(f"unknown probe mode: {mode}")

    from horde_worker_regen.log_file_registry import discover_registered_file_sink_basenames

    names = sorted(discover_registered_file_sink_basenames())
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(names, handle)


if __name__ == "__main__":
    main()
