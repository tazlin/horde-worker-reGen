"""Real-hardware capability probes: prove the whole catalog on the machine actually running them.

**This is the cold-start path, one isolated worker per probe.** Each parametrized case boots its own
worker (process spawn, torch import, engine init, checkpoint cold-load) and tears it down afterwards,
so it pays the full warm-up rampup *per probe*. A whole-catalog run is therefore slow and spends most
of its wall-clock booting rather than generating, with a correspondingly low GPU-core duty cycle: that
cost is inherent to the per-probe isolation, not a bug (the per-probe timing the run prints quantifies
it, and ``docs/explanation/duty-cycle.md`` explains why an isolated probe reads low).

Run this when that isolation is what you want:

- to reproduce or debug a single capability in a clean worker, selected by slug,
  ``pytest -m gpu -k controlnet`` (or ``-k sd15-baseline``) thanks to the parametrize id being the slug;
- to verify a capability free of any cross-probe state, where a fresh worker per case is the point.

Do *not* reach for a full run of this module as the fast or routine path. For a quick real-hardware
smoke use the canary (``pytest -m gpu -k canary`` in ``test_capability_canary``); for an amortized
multi-probe run that boots once and reuses the worker across checks, the warm path is the model (see
``test_capability_canary`` and the fake-mode ``tests/e2e/test_capability_warm_reuse``), and it is what
the benchmark executor itself uses so a real ``horde-benchmark run`` never pays this per-check rampup.

Each probe still self-skips via the same machine-fit gate the benchmark uses, so heavy tiers
(flux/qwen/zimage) and missing-model/no-key cases skip cleanly rather than fail. The whole module is
auto-skipped on a CUDA-less box by ``tests/conftest``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.benchmark.capabilities.capability import CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.executor import detect_machine_info
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from horde_worker_regen.benchmark.requirements import (
    civitai_token_available,
    compute_probe_requirements,
    requirement_skip_reason,
)
from tests._capability_probes import ALL_PROBES

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.gpu
@pytest.mark.parametrize("probe", ALL_PROBES, ids=lambda probe: probe.capability.slug)
async def test_capability_probe_real(probe: CapabilityProbe, record_probe_timing: Callable[[str, str], None]) -> None:
    """Every catalog probe the machine can host is PROVEN on real hardware; the rest self-skip."""
    machine = detect_machine_info(probe_devices=True)
    skip_reason = requirement_skip_reason(
        compute_probe_requirements(probe),
        machine=machine,
        process_mode="real",
        civitai_available=civitai_token_available(),
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    result = await run_capability_probe_async(
        probe,
        process_mode="real",
        total_vram_mb=machine.total_vram_mb,
    )
    if result.timing is not None:
        record_probe_timing(probe.capability.slug, result.timing.summary())
    assert result.verdict is CapabilityVerdict.PROVEN, "; ".join(result.reasons)
