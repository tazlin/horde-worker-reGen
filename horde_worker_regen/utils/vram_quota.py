"""Allocator-enforced per-process VRAM quotas.

The worker runs several GPU processes (inference pool, post-processing lane, safety), each with its own
caching allocator. The parent can schedule *when* work runs, but it cannot referee *how much* each
allocator keeps: freed tensors stay in a process's pool, and under Windows' WDDM an over-committed card
does not fail allocations, it silently demand-pages every process on the device. A quota moves that
arbitration into the allocator itself: a support process (lane, safety) is capped at the share of the
card its role justifies, so an overstep becomes a crisp out-of-memory inside the offender, on paths
that already degrade gracefully (a faulted chain falls back to a raw-image submit; a faulted safety
eval recycles the process), instead of an invisible device-wide slowdown.

CUDA-only by mechanism (``torch.cuda.set_per_process_memory_fraction`` is the caching-allocator cap);
on any other backend (XPU, DirectML, CPU) or on any failure this is a logged no-op, never an exception:
quotas are an optimization of a healthy worker, not a requirement to run one.
"""

from __future__ import annotations

from loguru import logger

POST_PROCESS_VRAM_QUOTA_MB = 4096.0
"""The dedicated post-processing lane's allocator cap (MB).

Sized for an upscaler/face-fixer chain's working set (a 4x ESRGAN pass on a 1MP image peaks around
3.1GB plus the resident module); a chain that genuinely needs more faults inside the lane and the job
is delivered with its raw images. Without the cap the lane's allocator pool was observed retaining
5GB+ of the card between chains."""

SAFETY_VRAM_QUOTA_MB = 4096.0
"""The safety process's allocator cap (MB).

Sized from its measured resident set: the safety checkers plus the CLIP interrogator's ViT model load
to roughly 3.3GB before any evaluation runs (a 2GB cap faults the model load itself), plus headroom
for per-eval activations. The cap's job is bounding, not shrinking: reducing safety's footprint
(CPU evaluation, post-eval unloads) is a separate lever."""


def apply_process_vram_quota_mb(quota_mb: float, *, device_index: int = 0) -> bool:
    """Cap this process's CUDA caching allocator at ``quota_mb`` on ``device_index``.

    Call from inside the child process after torch is importable. Returns True when the cap was
    applied; False (with a debug log) on any non-CUDA backend, absent torch, or error. Never raises.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("VRAM quota not applied: CUDA not available on this backend")
            return False
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        if total_bytes <= 0:
            return False
        fraction = min(1.0, (quota_mb * 1024.0 * 1024.0) / float(total_bytes))
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        logger.info(
            f"VRAM quota applied: this process's allocator is capped at {quota_mb:.0f}MB "
            f"({fraction:.0%} of device {device_index}).",
        )
        return True
    except Exception as quota_error:  # noqa: BLE001 - a quota is an optimization, never a startup failure
        logger.debug(f"VRAM quota not applied ({type(quota_error).__name__}: {quota_error})")
        return False
