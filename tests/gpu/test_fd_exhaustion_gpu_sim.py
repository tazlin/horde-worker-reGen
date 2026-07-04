"""Exercise file-descriptor exhaustion on a real GPU, through the sampling loop that triggers it.

Where ``tests/analysis/test_fd_exhaustion_repro.py`` exercises the mechanism on the CPU (a clamped
descriptor ceiling makes the free-RAM probe fail), this exercises the trigger path: a real GPU denoise
loop whose per-step free-RAM log line opens ``/proc/meminfo`` on each tqdm redraw. Under a full descriptor
table that open is refused, so the job faults inside sampling, and the captured error is fed to the
diagnosis detector to confirm the tooling recognises it end to end.

The GPU is genuinely exercised (a latent-sized tensor advanced by a denoise-style step loop), so this is
auto-skipped on a CUDA-less machine by ``tests/conftest`` via ``@pytest.mark.gpu``. It is also POSIX-gated:
``RLIMIT_NOFILE`` / ``EMFILE`` does not exist on Windows. The fidelity boundary is deliberate and small:
the free-RAM probe is driven once per step (as tqdm drives it in production) rather than through hordelib's
stdout-redirect shim, because the failing syscall (psutil opening ``/proc/meminfo`` with no free
descriptor) is identical either way.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.detectors import _FD_EXHAUSTION_RE, run_detectors
from horde_worker_regen.analysis.sessions import segment_sessions

resource = pytest.importorskip(
    "resource",
    reason="RLIMIT_NOFILE / EMFILE is POSIX-only; the fault cannot manifest on Windows (no resource module).",
)
torch = pytest.importorskip("torch", reason="the GPU simulation needs torch to occupy the device")

pytestmark = pytest.mark.gpu

_STEPS = 20  # A short SDXL-like step count, enough for the probe to be driven mid-sampling.


def _free_ram_probe() -> None:
    """Call the worker's free-RAM log line if importable, else its underlying syscall.

    ``hordelib.comfy_horde.log_free_ram`` is what the sampler calls to log free RAM; it ends in
    ``psutil.virtual_memory()``, which opens ``/proc/meminfo``. When hordelib is not importable in the
    test environment, calling ``psutil.virtual_memory()`` directly exercises the identical open.
    """
    try:
        from hordelib.comfy_horde import log_free_ram
    except Exception:  # noqa: BLE001 - fall back to the exact syscall the function makes.
        import psutil

        psutil.virtual_memory()
        return
    log_free_ram()


@contextmanager
def _descriptor_ceiling_reached() -> Iterator[None]:
    """Fill the descriptor table (see the CPU repro for the rationale), restoring it on exit."""
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
                break
        yield
    finally:
        for fd in opened:
            with suppress(OSError):
                os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def _run_gpu_denoise_with_periodic_ram_log() -> OSError:
    """Run a real GPU step loop that logs free RAM each step; return the EMFILE it faults with.

    Reproduces the production shape: genuine device work per step (a latent-sized tensor advanced by a
    trivial denoise-style op) with the free-RAM probe driven between steps, as tqdm drives it in the real
    sampler. Under the full descriptor table the probe raises ``EMFILE`` mid-sampling; that error is
    returned so the caller can assert on it. Fails the test if the loop somehow completes.
    """
    device = torch.device("cuda")
    latent = torch.randn(1, 4, 128, 128, device=device)  # SDXL 1024px latent shape.
    from tqdm import trange

    for _ in trange(_STEPS, desc="fd-exhaustion-sim", leave=False):
        latent = latent - 0.01 * torch.randn_like(latent)
        torch.cuda.synchronize()
        try:
            _free_ram_probe()
        except OSError as exc:
            if exc.errno == errno.EMFILE:
                return exc
            raise
    raise AssertionError("expected the free-RAM probe to fault with EMFILE during sampling, but it did not")


def test_gpu_sampling_faults_with_emfile_under_full_descriptor_table() -> None:
    """A real GPU denoise loop faults inside sampling when the descriptor table is full.

    The device is doing genuine work, and the free-RAM log line the worker emits each step is what fails,
    so the failure is exercised on hardware rather than in the abstract.
    """
    with _descriptor_ceiling_reached():
        emfile = _run_gpu_denoise_with_periodic_ram_log()
    assert emfile.errno == errno.EMFILE
    assert _FD_EXHAUSTION_RE.search(str(emfile)), str(emfile)


def test_detector_fires_on_the_simulated_fault(tmp_path: Path) -> None:
    """The simulated EMFILE, wrapped as the worker would report it, trips the file-descriptor detector.

    Closes the loop from real hardware fault to maintainer-facing finding: the error the GPU run produced
    is folded into the exact ``faulted on process`` line the orchestrator logs, and the detector must
    surface it as ``file_descriptor_exhaustion`` (not as an OOM).
    """
    with _descriptor_ceiling_reached():
        emfile = _run_gpu_denoise_with_periodic_ram_log()

    faulted_line = (
        "2026-06-24 20:09:24.000 | WARNING  | "
        "horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:892 - "
        "Job 597d4471 faulted on process 3 (RuntimeError: Pipeline failed to run - declared output node(s) "
        f"['output_image'] produced no results. Model: WAI-NSFW-illustrious-SDXL. Error: sampler (KSampler): {emfile}"
    )
    bridge = "2026-06-24 20:00:00.000 | DEBUG | x:y:1 - Setting up logger for main process\n" + faulted_line
    (tmp_path / "bridge.log").write_text(bridge, encoding="utf-8")

    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    findings = {f.id: f for f in run_detectors(build_session_context(session, bundle))}
    assert "file_descriptor_exhaustion" in findings
    assert "oom" not in findings
