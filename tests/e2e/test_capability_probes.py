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


# Each probe boots a real worker and spawns real OS child processes through the harness, so the module is
# opt-in via -m slow.
pytestmark = pytest.mark.slow


_COLD_PROBE: CapabilityProbe = LIGHT_PROBES[0]
"""The one probe run on a cold worker here.

Every light probe is proven on one warm worker by ``test_capability_warm_reuse``; booting a cold worker
per probe as well proved the same verdicts at a fleet spawn each. What only a cold run covers is the
cold path itself (``HarnessConfig.from_scenario`` through a fresh boot), and one probe exercises it."""


@pytest.mark.e2e
@pytest.mark.parametrize("probe", [_COLD_PROBE], ids=lambda probe: probe.capability.slug)
async def test_capability_probe_fake(probe: CapabilityProbe, record_probe_timing: Callable[[str, str], None]) -> None:
    """A light probe is PROVEN on a cold fake-mode worker (the synthetic worker completes its jobs cleanly)."""
    result = await run_capability_probe_async(probe, process_mode="fake")
    if result.timing is not None:
        record_probe_timing(probe.capability.slug, result.timing.summary())
    assert result.verdict is CapabilityVerdict.PROVEN, "; ".join(result.reasons)
