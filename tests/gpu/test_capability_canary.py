"""GPU canary: the single briefest benchmark, run in isolation to measure and amortize the boot rampup.

This runs only the first part of the first benchmark, the SD1.5 baseline (one job: "can this machine
serve the simplest possible workload?"), so it is the fastest real-hardware signal that the whole probe
path works end to end. It also measures the thing that motivated it: a cold worker boot (process spawn,
torch import, engine init, checkpoint cold-load) is a large one-time cost, and the goal is to pay it
*once* and reuse the worker across checks rather than booting per check.

The test boots one :class:`WarmHarnessSession`, records how long that boot took (the rampup), then runs
the baseline check twice through it and asserts the inference processes were never respawned between the
checks. That identity is the proof that a warm worker behaves like the recovery supervisor: the rampup is
amortized, not repaid on every check. Auto-skipped on a CUDA-less box by ``tests/conftest``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions, build_capability_catalog
from horde_worker_regen.benchmark.capabilities.executor import detect_machine_info
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.requirements import (
    civitai_token_available,
    compute_probe_requirements,
    requirement_skip_reason,
)
from horde_worker_regen.harness import WarmHarnessSession

if TYPE_CHECKING:
    from collections.abc import Callable


def _canary_probe() -> CapabilityProbe:
    """The single briefest probe: the SD1.5 baseline with one job (the first part of the first benchmark)."""
    probes = build_capability_catalog(
        CatalogOptions(
            tiers=[BenchTier.SD15],
            jobs_per_level=1,
            include_features=False,
            include_alchemy=False,
        ),
    )
    return next(probe for probe in probes if probe.capability.kind is CapabilityKind.BASELINE)


@pytest.mark.gpu
async def test_capability_canary_warm_reuse(record_probe_timing: Callable[[str, str], None]) -> None:
    """The baseline check runs twice on one warm worker without respawning it; the boot is paid once."""
    probe = _canary_probe()
    machine = detect_machine_info(probe_devices=True)
    skip_reason = requirement_skip_reason(
        compute_probe_requirements(probe),
        machine=machine,
        process_mode="real",
        civitai_available=civitai_token_available(),
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    boot_started = time.monotonic()
    async with WarmHarnessSession(
        process_mode="real",
        model_names=sorted(probe.scenario.models_referenced()),
        max_threads_ceiling=1,
    ) as session:
        boot_seconds = time.monotonic() - boot_started
        launch_ids_before = {
            p.process_launch_identifier for p in session.manager._process_map.get_inference_processes()
        }

        first = await run_capability_probe_async(
            probe,
            process_mode="real",
            total_vram_mb=machine.total_vram_mb,
            warm_session=session,
        )
        second = await run_capability_probe_async(
            probe,
            process_mode="real",
            total_vram_mb=machine.total_vram_mb,
            warm_session=session,
        )

        launch_ids_after = {
            p.process_launch_identifier for p in session.manager._process_map.get_inference_processes()
        }

    record_probe_timing("canary-boot", f"warm worker boot {boot_seconds:.0f}s (one-time rampup, paid once)")
    if first.timing is not None:
        record_probe_timing("canary-check-1", first.timing.summary())
    if second.timing is not None:
        record_probe_timing("canary-check-2", second.timing.summary())

    assert first.verdict is CapabilityVerdict.PROVEN, "; ".join(first.reasons)
    assert second.verdict is CapabilityVerdict.PROVEN, "; ".join(second.reasons)
    assert launch_ids_before, "expected a warm inference process after boot"
    # The two checks ran on the same inference processes: no respawn, so the boot rampup was not repaid.
    assert launch_ids_before == launch_ids_after, "warm worker respawned between checks (rampup not amortized)"
