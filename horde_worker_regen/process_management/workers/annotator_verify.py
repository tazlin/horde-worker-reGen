"""Run every ControlNet preprocessor once and exit with the verdict.

The download process runs this module in a child interpreter (``python -m ...``) when a verify is due: the
check needs a full ComfyUI/torch boot plus every detector's weights, which would otherwise stay resident in
the download process for the rest of the session. The exit status is the whole interface: ``0`` when every
preprocessor ran, ``1`` otherwise.

hordelib's logger is not set up here. Its setup with no process id installs the *main* process's sinks
(``logs/bridge.log`` and the console), so a child that let it run would write into the orchestrator's log
and emit the main-process startup marker the session splitter keys off, cutting one launch into two sessions.
loguru's default stderr handler stays, and the parent's stderr log captures it.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Verify the annotators; return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--directml", type=int, default=None, help="DirectML device index (Windows AMD GPUs).")
    args = parser.parse_args(argv)

    import hordelib
    from hordelib.api import SharedModelManager

    extra_comfyui_args = [f"--directml={args.directml}"] if args.directml is not None else []
    hordelib.initialise(setup_logging=False, extra_comfyui_args=extra_comfyui_args)
    return 0 if SharedModelManager.preload_annotators() else 1


if __name__ == "__main__":
    raise SystemExit(main())
