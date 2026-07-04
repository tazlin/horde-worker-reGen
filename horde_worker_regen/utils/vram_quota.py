"""Allocator-enforced per-process VRAM quotas.

The worker runs several GPU processes (inference pool, post-processing lane, safety), each with its own
caching allocator. The parent can schedule *when* work runs, but it cannot referee *how much* each
allocator keeps: freed tensors stay in a process's pool, and under Windows' WDDM an over-committed card
does not fail allocations, it silently demand-pages every process on the device. A quota moves that
arbitration into the allocator itself: a support process (lane, safety) is capped at the share of the
card its role justifies, so an overstep becomes a crisp out-of-memory inside the offender, on paths
that already degrade deliberately (a faulted post-processing chain is submitted without images so the
horde reissues it; a faulted safety eval recycles the process), instead of an invisible device-wide slowdown.

CUDA-only by mechanism (``torch.cuda.set_per_process_memory_fraction`` is the caching-allocator cap);
on any other backend (XPU, DirectML, CPU) or on any failure this is a logged no-op, never an exception:
quotas are an optimization of a healthy worker, not a requirement to run one.
"""

from __future__ import annotations

from loguru import logger

POST_PROCESS_VRAM_QUOTA_FLOOR_MB = 4096.0
"""The smallest allocator cap the post-processing lane is ever given (MB).

A lane needs at least this much to load a post-processing module and run the smallest tile of a chain; a
card too small to spare it makes post-processing structurally unavailable rather than the cap shrinking
below a working set. On a card smaller than this the fraction clamps to the whole card anyway."""

POST_PROCESS_VRAM_QUOTA_CEILING_MB = 8192.0
"""The largest allocator cap the post-processing lane is ever given (MB).

The cap is a runaway/leak guard, not a per-job limiter: it must sit *above* the largest realistic PP
chain's working set (an upscale/face-fix peak observed up to ~6.4GB) so legitimate jobs run in VRAM,
while still bounding a pool that would otherwise squat the whole card between chains. Above this a bigger
card buys the inference pool headroom rather than letting the lane's cache grow without limit."""

POST_PROCESS_CARD_HEADROOM_MB = 4096.0
"""How much of the card the lane's guard always leaves for the inference pool and safety process (MB).

The guard never claims within this of the card total, so on a mid-size card the lane cannot starve the
inference pool even if its cache runs away. Deciding *when* a chain may co-reside with sampling remains
the orchestrator's admission gate; this only bounds the worst case."""


def effective_post_process_vram_quota_mb(total_vram_mb: float | None) -> float:
    """Return the lane's allocator-guard cap (MB) for a card of ``total_vram_mb`` total VRAM.

    The cap is a belt-and-suspenders runaway guard sized above the largest realistic PP chain where the
    card allows: the card total minus a reserved share for the inference pool, clamped between a floor
    (below which post-processing cannot run) and a ceiling (above which extra VRAM benefits inference, not
    the lane's cache). With no card reading (non-CUDA, cold start) it falls back to the floor. Pure so the
    parent's dispatch gate can derive the same number the lane enforces.
    """
    if total_vram_mb is None or total_vram_mb <= 0:
        return POST_PROCESS_VRAM_QUOTA_FLOOR_MB
    guard = total_vram_mb - POST_PROCESS_CARD_HEADROOM_MB
    return max(POST_PROCESS_VRAM_QUOTA_FLOOR_MB, min(guard, POST_PROCESS_VRAM_QUOTA_CEILING_MB))


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


def read_device_total_vram_mb(device_index: int = 0) -> float | None:
    """Return ``device_index``'s total VRAM (MB), or None on a non-CUDA backend, absent torch, or error.

    Call from inside a GPU child after torch is importable. Never raises: a missing reading degrades the
    lane's guard to its floor rather than failing startup.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        if total_bytes <= 0:
            return None
        return float(total_bytes) / (1024.0 * 1024.0)
    except Exception as read_error:  # noqa: BLE001 - a quota is an optimization, never a startup failure
        logger.debug(f"Device total VRAM unreadable ({type(read_error).__name__}: {read_error})")
        return None


def apply_post_process_vram_quota(*, device_index: int = 0) -> bool:
    """Cap the post-processing lane at its card-sized runaway guard on ``device_index``.

    Reads the pinned card's total VRAM and applies :func:`effective_post_process_vram_quota_mb`, so a card
    with headroom gives the lane enough to run a large upscale/face-fix chain in VRAM while a small card
    keeps the guard tight. A no-op (returns False) on any non-CUDA backend or error.
    """
    quota_mb = effective_post_process_vram_quota_mb(read_device_total_vram_mb(device_index))
    return apply_process_vram_quota_mb(quota_mb, device_index=device_index)
