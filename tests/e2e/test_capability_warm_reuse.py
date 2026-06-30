"""The capability probe runner's warm path: many checks on one worker, no per-check rampup (fake mode).

The parametrized e2e/gpu probe tests each run a probe through the *cold* path (its own worker booted and
torn down per probe), which is what makes an isolated probe read as mostly startup. This test exercises
the *warm* path instead: one :class:`WarmHarnessSession` booted once, every probe run against it, proving
``run_capability_probe_async(warm_session=...)`` reuses the same inference processes the way the executor
(and a live worker) does. It is the fake-mode analog of the GPU canary, runnable in CI without hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.benchmark.capabilities.capability import CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from horde_worker_regen.harness import WarmHarnessSession
from tests._capability_probes import LIGHT_PROBES

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.e2e
async def test_warm_session_runs_probes_without_per_check_rampup(
    record_probe_timing: Callable[[str, str], None],
) -> None:
    """Every light probe is PROVEN on one warm worker, and no probe respawns the inference processes."""
    probes = LIGHT_PROBES
    model_names = sorted({model for probe in probes for model in probe.scenario.models_referenced()})

    def _threads(probe: CapabilityProbe) -> int:
        value = probe.bridge_data_overrides.get("max_threads", 1)
        return value if isinstance(value, int) else 1

    threads_ceiling = max((_threads(probe) for probe in probes), default=1)

    async with WarmHarnessSession(
        process_mode="fake",
        model_names=model_names,
        max_threads_ceiling=threads_ceiling,
    ) as session:
        launch_ids_before = {
            p.process_launch_identifier for p in session.manager._process_map.get_inference_processes()
        }
        assert launch_ids_before, "expected a warm inference process before running probes"

        for probe in probes:
            result = await run_capability_probe_async(probe, process_mode="fake", warm_session=session)
            if result.timing is not None:
                record_probe_timing(f"{probe.capability.slug} (warm)", result.timing.summary())
            assert result.verdict is CapabilityVerdict.PROVEN, "; ".join(result.reasons)

        launch_ids_after = {
            p.process_launch_identifier for p in session.manager._process_map.get_inference_processes()
        }

    # The same inference processes served every probe: the boot rampup was paid once, not per check.
    assert launch_ids_before == launch_ids_after, "warm worker respawned between probes (rampup not amortized)"
