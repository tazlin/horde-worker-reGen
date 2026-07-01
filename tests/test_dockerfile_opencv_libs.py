"""Guard that the container images install the system libraries opencv-python needs at import.

horde-engine pulls ``opencv-python``, which several bundled ComfyUI nodes (e.g. layer diffusion's
``lib_layerdiffusion``) import at load time. On a slim base image ``import cv2`` needs both ``libGL``
(``libgl1``) and ``libgthread``/``libglib`` (``libglib2.0-0``); if either is missing the import raises,
ComfyUI silently skips registering the node, and it surfaces only much later as a ``KeyError`` for that
node class at generation time. These parse the Dockerfiles so that gap cannot regress unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_OPENCV_LIBS = ("libgl1", "libglib2.0-0")


@pytest.mark.parametrize("dockerfile", ["Dockerfile.cuda", "Dockerfile.rocm"])
def test_dockerfile_installs_opencv_runtime_libs(dockerfile: str) -> None:
    """Each container image must apt-install the shared libraries opencv-python imports against."""
    text = (_REPO_ROOT / "Dockerfiles" / dockerfile).read_text(encoding="utf-8")
    missing = [lib for lib in _REQUIRED_OPENCV_LIBS if lib not in text]
    assert not missing, (
        f"Dockerfiles/{dockerfile} does not install {missing}; opencv-python's `import cv2` will fail at "
        f"runtime, silently dropping the ComfyUI nodes that import it (e.g. layer diffusion)."
    )
