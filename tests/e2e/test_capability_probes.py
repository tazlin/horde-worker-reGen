"""Fake-mode capability probes: a benchmark scenario run as a CI test, no GPU required.

Each cheap SD1.5 probe runs through the exact code path the benchmark uses (``run_capability_probe_async``
-> ``HarnessConfig.from_scenario`` -> the harness), so the benchmark and its test cannot drift. The
parametrize id is the capability slug, so ``-k sd15-threads`` selects exactly that probe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.benchmark.capabilities.capability import CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from tests._capability_probes import LIGHT_PROBES

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.e2e
@pytest.mark.parametrize("probe", LIGHT_PROBES, ids=lambda probe: probe.capability.slug)
async def test_capability_probe_fake(probe: CapabilityProbe, record_probe_timing: Callable[[str, str], None]) -> None:
    """Every light probe is PROVEN in fake mode (the synthetic worker completes its jobs cleanly)."""
    result = await run_capability_probe_async(probe, process_mode="fake")
    if result.timing is not None:
        record_probe_timing(probe.capability.slug, result.timing.summary())
    assert result.verdict is CapabilityVerdict.PROVEN, "; ".join(result.reasons)
