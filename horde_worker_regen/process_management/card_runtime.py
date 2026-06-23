"""The per-card runtime plan: one :class:`CardRuntime` per GPU this worker drives.

A multi-GPU worker spawns a separate pool of inference processes per card, each pinned to that card and
gated by that card's own concurrency semaphores. :class:`CardRuntime` is the single typed record of
everything the orchestrator needs to know about one card at runtime: its stable index and backend kind,
its effective (per-card) config, its concurrency primitives, how many processes to run on it, and whether
its processes must be device-pinned. The process manager builds a ``dict[int, CardRuntime]`` keyed by
stable device index; the lifecycle manager spawns from it, and later scheduling/pop phases route against it.

On a single-GPU host the map has exactly one entry whose sizes equal the pre-multi-GPU globals, so the
single-card case is behaviourally identical to before this structure existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.synchronize import Semaphore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


@dataclass(frozen=True)
class CardRuntime:
    """Everything the orchestrator needs to run one GPU's pool of inference processes.

    Built once per driven card by the process manager and consumed by the lifecycle manager (to spawn and
    replace that card's processes with the right pinning and semaphores) and by later scheduling phases.
    """

    device_index: int
    """The stable (PCI-bus) index of this card, matching its key in the device map and process attribution."""
    kind: str
    """The accelerator backend (``cuda``/``rocm``/``xpu``/...), used to choose the device-pinning mechanism."""
    config: reGenBridgeData
    """The effective per-card config (the global config with this card's override applied), for routing/budget."""
    total_vram_mb: float | None
    """This card's total VRAM in MB (from the device map), for the heterogeneous weight-fit eligibility check.
    None when the device's capacity is unknown (CPU/dry-run/test paths), where the weight check abstains."""
    inference_semaphore: Semaphore
    """This card's concurrent-inference gate. Distinct per card so one card's sampling cannot block another's."""
    vae_decode_semaphore: Semaphore
    """This card's concurrent-VAE-decode gate."""
    gpu_sampling_lease: Semaphore
    """This card's GPU sampling lease (denoise-loop pipelining), independent of other cards' leases."""
    target_process_count: int
    """How many inference processes to run on this card (its ``queue_size`` + concurrency ceiling)."""
    max_concurrent_inference: int
    """This card's concurrent-sampling ceiling (its semaphore size before the lease opens the gate up)."""
    mask_kind: str | None
    """The accelerator kind to device-pin this card's processes with, or None to skip masking.

    None on a default single-GPU host (no env var written, behaviourally identical to before). Set to
    :attr:`kind` when the worker drives more than one card or the operator explicitly selected cards, so
    each process is masked to exactly its device."""
