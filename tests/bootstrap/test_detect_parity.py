"""Parity guard: the Inno wizard's detect-backend.ps1 and worker_bootstrap.detect must agree.

The graphical installer still detects the GPU with detect-backend.ps1 at wizard time (before uv, and thus
bootstrap.py, exists), while every other channel uses worker_bootstrap.detect. If the two drift, a user
could be warned one thing by the wizard and get another build at first launch. This runs both against the
same stubbed nvidia-smi and asserts they pick the same token. Windows-only (detect-backend.ps1 needs
Windows PowerShell).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from worker_bootstrap import detect

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DETECT_PS1 = _REPO_ROOT / "packaging" / "detect-backend.ps1"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="detect-backend.ps1 parity is Windows-only")


def _run_ps1_detector() -> str:
    """Run detect-backend.ps1 with the current PATH and return its single token."""
    powershell = shutil.which("powershell") or "powershell"
    completed = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(_DETECT_PS1)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


_UMD_HEADER = "| NVIDIA-SMI 610.88  KMD Version: 610.88  CUDA UMD Version: 13.3 |"


@pytest.mark.parametrize(
    ("cuda_header", "compute_caps", "expected"),
    [
        ("CUDA Version: 13.2", ["8.6"], "cu132"),
        ("CUDA Version: 13.1", ["8.6"], "cu130"),
        ("CUDA Version: 13.0", ["8.6"], "cu130"),
        ("CUDA Version: 12.8", ["8.6"], "cu126"),
        # Drivers that print "CUDA UMD Version" in place of "CUDA Version".
        (_UMD_HEADER, ["8.9"], "cu132"),
        # Blackwell (sm_120) on a CUDA 12.x driver: the architecture floor lifts both detectors to cu130.
        ("CUDA Version: 12.8", ["12.0"], "cu130"),
        # Pre-Turing (Pascal sm_61) on a CUDA 13.2 driver: the ceiling holds both detectors at cu126.
        ("CUDA Version: 13.2", ["6.1"], "cu126"),
        # Mixed cards take the non-interactive default, which is every card when one build covers them all ...
        (_UMD_HEADER, ["6.1", "7.5"], "cu126"),
        ("CUDA Version: 12.8", ["8.9", "12.0"], "cu130"),
        # ... and the highest card when one build cannot cover them all.
        (_UMD_HEADER, ["6.1", "12.0"], "cu132"),
    ],
)
def test_detect_parity_nvidia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_header: str,
    compute_caps: list[str],
    expected: str,
) -> None:
    """With a stub nvidia-smi on PATH, both detectors return the same NVIDIA build token.

    The stub prints the CUDA header for the bare ``nvidia-smi`` call and, for the ``--query-gpu`` call (any
    args) with one compute-capability line per card, so both the driver-version and architecture-clamp
    paths are exercised identically by each detector.
    """
    stub = tmp_path / "nvidia-smi.bat"
    # cmd.exe reads a bare "|" as a pipe, so the header's table borders are escaped.
    header_line = cuda_header.replace("|", "^|")
    cap_lines = "".join(f"echo {cap}\r\n" for cap in compute_caps)
    stub.write_text(
        f'@echo off\r\nif "%~1"=="" (\r\necho {header_line}\r\n) else (\r\n{cap_lines})\r\n',
        encoding="ascii",
    )
    # Prepend the stub dir so both detectors resolve nvidia-smi to it rather than a real driver.
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    python_token = detect.detect_backend()
    powershell_token = _run_ps1_detector()

    assert python_token == expected, f"bootstrap.detect picked {python_token}, expected {expected}"
    assert powershell_token == expected, f"detect-backend.ps1 picked {powershell_token}, expected {expected}"
    assert python_token == powershell_token
