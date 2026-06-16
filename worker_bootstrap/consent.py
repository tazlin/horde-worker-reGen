"""Install-time disclosure and consent, shared by every bootstrap entry point.

The one-line installers and the graphical ``.exe`` each capture consent in their own native way (a console
prompt, or the wizard's license page) and signal it with an environment flag or a persisted marker. This
module is the one place that decides, for a given run, whether to print the notice, prompt, or proceed, so
the policy is identical no matter which front-end invoked the bootstrap.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Any of these being set means consent was already captured upstream: a front-end console prompt
# (HORDE_WORKER_ASSUME_YES), a deliberate automation/CI choice (HORDE_WORKER_NONINTERACTIVE), or a launcher
# re-entering the bootstrap (HORDE_WORKER_FROM_LAUNCHER). In all three we must not prompt again.
_CONSENT_ENV_VARS = ("HORDE_WORKER_ASSUME_YES", "HORDE_WORKER_NONINTERACTIVE", "HORDE_WORKER_FROM_LAUNCHER")

_FALLBACK_NOTICE = (
    "This installs the AI Horde Worker. It will download a private Python runtime, PyTorch, ComfyUI and\n"
    "supporting components from the internet into this folder, and later the AI models you choose. See\n"
    "https://github.com/Haidra-Org/horde-worker-reGen for the full notice and licenses."
)


def consent_env_var() -> str | None:
    """Return the name of the first set consent-granting env var, or ``None`` when none is set."""
    return next((name for name in _CONSENT_ENV_VARS if os.environ.get(name)), None)


def is_interactive() -> bool:
    """Return True only when we can meaningfully prompt: both stdin and stdout are real terminals."""
    return bool(sys.stdin) and sys.stdin.isatty() and sys.stdout.isatty()


def read_notice(notice_path: Path) -> str:
    """Return the bundled install notice, falling back to a short built-in summary if it is missing."""
    try:
        text = notice_path.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_NOTICE
    return text or _FALLBACK_NOTICE


def _write_marker(marker_path: Path) -> None:
    """Record that consent was established (best-effort; a missing marker only means we may re-ask)."""
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("consent recorded\n", encoding="utf-8")
    except OSError:
        pass


def ensure_consent(
    *,
    notice_path: Path,
    marker_path: Path,
    detail_lines: Sequence[str] = (),
    interactive: bool | None = None,
    consent_env: str | None = None,
    prompt: Callable[[str], str] = input,
) -> bool:
    """Show the install notice and decide whether the install may proceed.

    Returns True to proceed and False to abort. Once consent is established (a flag, a console "yes", or a
    headless run) a persistent marker is written so later runs (dependency updates, the ``.exe``'s deferred
    first launch) do not prompt again.

    Args:
        notice_path: The bundled ``INSTALL_NOTICE.txt`` to display.
        marker_path: Where to record that consent was captured (``bin/install-consent``).
        detail_lines: Run-specific lines appended to the notice (chosen GPU backend, git handling).
        interactive: Whether a prompt is possible; defaults to :func:`is_interactive`.
        consent_env: A set consent env var name; defaults to :func:`consent_env_var`.
        prompt: The input function (injectable for tests).
    """
    if interactive is None:
        interactive = is_interactive()
    if consent_env is None:
        consent_env = consent_env_var()

    if marker_path.exists():
        # Consent already recorded (earlier install, or the .exe's license page wrote the marker). Stay
        # quiet so updates and the deferred first launch don't re-print the whole notice.
        return True

    if consent_env is not None:
        # A front-end already captured consent and showed the notice (the one-line installer's prompt, the
        # .exe's license page, or a deliberate automation flag), so do not reprint the whole thing here.
        print(f"Install consent acknowledged ({consent_env}).")
        _write_marker(marker_path)
        return True

    # Nothing upstream captured consent, so we own the disclosure: show the full notice and run details.
    print(read_notice(notice_path))
    for line in detail_lines:
        print(line)
    print()

    if not interactive:
        # No terminal to prompt and no explicit flag: an automated/headless context (e.g. Docker, CI).
        # Proceed so established automation keeps working; the notice above stands as the disclosure.
        print(
            "No interactive terminal detected and no consent flag set; proceeding. "
            "Set HORDE_WORKER_ASSUME_YES=1 to make this explicit.",
        )
        _write_marker(marker_path)
        return True

    answer = prompt("Proceed with installation? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        _write_marker(marker_path)
        return True
    print("Installation cancelled. Nothing was installed.")
    return False
