"""Exercise file-descriptor exhaustion (``RLIMIT_NOFILE`` / ``EMFILE``) at its real victim call sites.

When an inference process runs its descriptor table to the per-process ``RLIMIT_NOFILE`` ceiling, every
subsequent ``open()`` is refused with ``EMFILE`` (errno 24, "Too many open files"). The routine calls that
then fail are far from whatever leaked the descriptors: the free-RAM log line the sampler emits on each
tqdm redraw ends in ``psutil.virtual_memory()`` opening ``/proc/meminfo``, and model/LoRA loading opens
``.safetensors``. So a full table makes an inference process fault every job until it is recycled.

These tests drive the process to that ceiling and assert the exact victim calls fail with ``EMFILE`` whose
text is what the ``file_descriptor_exhaustion`` detector matches, tying the detector to the string the OS
actually produces. They need no GPU and no particular RAM/VRAM figure, so they run on any POSIX host.

``RLIMIT_NOFILE`` / ``EMFILE`` is POSIX-only: Windows has no ``resource`` module, no ``/proc/meminfo``, and
a handle ceiling high enough that this class of exhaustion does not arise, so these tests skip there. The
detector's recognition of the log line is exercised separately, without a descriptor ceiling, in
``test_detectors.py`` / ``test_detector_contract.py``.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import psutil
import pytest

from horde_worker_regen.analysis.detectors import _FD_EXHAUSTION_RE

resource = pytest.importorskip(
    "resource",
    reason="RLIMIT_NOFILE / EMFILE is POSIX-only; the fault cannot manifest on Windows (no resource module).",
)


@contextmanager
def _descriptor_ceiling_reached() -> Iterator[None]:
    """Drive the process to its file-descriptor ceiling, then hand control back with zero headroom.

    Clamps the soft ``RLIMIT_NOFILE`` low (so exhaustion is quick and deterministic regardless of the
    host's real ceiling), consumes the remaining headroom with throwaway opens, and yields with the table
    full so the *next* ``open()`` anywhere is refused with ``EMFILE``. Restores the limit and closes every
    throwaway descriptor on exit, so the clamp cannot leak into the rest of the test session.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    opened: list[int] = []
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 128), hard))
        while True:
            try:
                opened.append(os.open(os.devnull, os.O_RDONLY))
            except OSError as exc:
                if exc.errno != errno.EMFILE:
                    raise
                break  # Headroom is now zero: the ceiling is reached.
        yield
    finally:
        for fd in opened:
            with suppress(OSError):
                os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_free_ram_probe_fails_with_emfile_at_ceiling() -> None:
    """The free-RAM probe (``psutil.virtual_memory``) is the first victim once the table is full.

    The sampler's free-RAM log line ends in ``log_free_ram`` -> ``virtual_memory`` -> open
    ``/proc/meminfo``. Under a full descriptor table that open raises ``EMFILE``, and its text is what the
    detector matches, so the log line the worker emits is recognised.
    """
    with _descriptor_ceiling_reached(), pytest.raises(OSError) as exc_info:
        psutil.virtual_memory()
    assert exc_info.value.errno == errno.EMFILE
    assert _FD_EXHAUSTION_RE.search(str(exc_info.value)), str(exc_info.value)


def test_model_file_open_fails_with_emfile_at_ceiling(tmp_path: Path) -> None:
    """A checkpoint/LoRA-style file open is refused once the table is full, stranding model loads.

    Stands in for a ``.safetensors`` load: the file exists and is readable, but the process has no
    descriptor left to open it with, so the load faults with the same errno-24 text.
    """
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"\x00")

    with _descriptor_ceiling_reached(), pytest.raises(OSError) as exc_info:
        os.close(os.open(checkpoint, os.O_RDONLY))
    assert exc_info.value.errno == errno.EMFILE
    assert _FD_EXHAUSTION_RE.search(str(exc_info.value)), str(exc_info.value)


def test_detector_regex_matches_real_kernel_text() -> None:
    """The regex the detector keys off matches the string the OS actually produces (loop-closing check).

    Guards against the detector drifting away from real ``EMFILE`` text: the message here is the kernel's,
    not a hand-written fixture, so a reworded detector regex that stopped matching real exhaustion would
    fail this even though the synthetic-fixture tests still passed.
    """
    with _descriptor_ceiling_reached():
        try:
            os.open(os.devnull, os.O_RDONLY)
        except OSError as exc:
            real_text = str(exc)
    assert _FD_EXHAUSTION_RE.search(real_text), real_text
    # The system-wide ENFILE variant ("... in system") must NOT match: it is a different, host-level fault.
    assert not _FD_EXHAUSTION_RE.search("OSError: [Errno 23] Too many open files in system")
