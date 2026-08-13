"""Real-hardware capability probes: prove the whole catalog on the machine actually running them.

**The catalog runs warm, on one shared worker, exactly as the shipped executor runs it.** A single
:class:`WarmHarnessSession` boots once for the module and every catalog probe runs through it, so the
boot rampup (process spawn, torch import, engine init, checkpoint cold-load) is paid once instead of
once per probe. That is not a test-only shortcut: ``ProbeExecutor`` is built the same way, so a probe
here and the same probe in a real ``horde-benchmark run`` exercise the same path.

Two probes are deliberately exempt. **The tier baselines (``sd15-baseline``, ``sdxl-baseline``) keep
their cold boots**, because a cold boot is the thing they prove: "can this machine bring a worker up
from nothing and serve the simplest workload on this tier?" is a question a warm session has already
answered for itself. Everything the warm path could hide about startup, the baselines still catch.

Per-probe configuration is preserved, not flattened. Each probe's ``bridge_data_overrides`` (its
``allow_controlnet`` / ``allow_post_processing`` / ``alchemy_*`` settings) are applied to the running
worker as a live config change before it runs and replaced before the next one, so a warm result names
the same configuration a cold result would. Running the whole catalog under one everything-enabled
config would prove less than it claims.

Selection still works per probe: the parametrize id is the capability slug, so ``pytest -m gpu -k
controlnet`` (or ``-k sd15-baseline``) runs exactly that one. For the cheapest real-hardware smoke, use
the canary (``pytest -m gpu -k canary`` in ``test_capability_canary``); for the fake-mode equivalent of
the warm path see ``tests/e2e/test_capability_warm_reuse``.

Each probe still self-skips via the same machine-fit gate the benchmark uses, so heavy tiers
(flux/qwen/zimage) and missing-model/no-key cases skip cleanly rather than fail. The whole module is
auto-skipped on a CUDA-less box by ``tests/conftest``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS
from horde_worker_regen.benchmark.requirements import (
    civitai_token_available,
    compute_probe_requirements,
    requirement_skip_reason,
)
from horde_worker_regen.harness import WarmHarnessSession
from tests._capability_probes import ALL_PROBES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from horde_worker_regen.benchmark.report import MachineInfo

pytestmark = [pytest.mark.gpu, pytest.mark.asyncio(loop_scope="module")]

_WARM_MAX_THREADS_CEILING = 2
"""The warm worker's provisioned concurrency ceiling: the highest ``max_threads`` any catalog probe asks
for. Processes are launched to the ceiling once; each probe's live cap is set within it."""


def _keeps_cold_boot(probe: CapabilityProbe) -> bool:
    """Whether *probe* is one of the tier baselines, which prove the cold boot path and keep it."""
    return probe.capability.kind is CapabilityKind.BASELINE


def _warm_model_names() -> list[str]:
    """Every model the catalog might touch, so the one warm worker covers all of it.

    Mirrors ``ProbeExecutor._warm_model_names``: the union of the probes' referenced models plus each
    tier's base checkpoint. A probe installs only its own scenario, so naming a model it never uses
    costs nothing.
    """
    names: set[str] = set()
    for probe in ALL_PROBES:
        names.update(probe.scenario.models_referenced())
    for tier in {probe.capability.tier for probe in ALL_PROBES}:
        if tier in BENCH_TIER_MODELS:
            names.add(BENCH_TIER_MODELS[tier])
    return sorted(names)


class _LazyWarmSession:
    """Boots the module's shared warm worker on first use and hands the same one out thereafter.

    Booting inside the fixture body would make every selection pay the boot, including ``-k
    sd15-baseline``, which runs cold by design and would never touch the session. Deferring it to the
    first probe that actually wants a warm worker keeps a single-probe rerun as cheap as it was.
    """

    def __init__(self) -> None:
        self._session: WarmHarnessSession | None = None

    async def get(self) -> WarmHarnessSession:
        """Return the shared warm session, booting it if this is the first caller."""
        if self._session is None:
            self._session = await WarmHarnessSession(
                process_mode="real",
                model_names=_warm_model_names(),
                max_threads_ceiling=_WARM_MAX_THREADS_CEILING,
                # Provision the boot config to the whole catalog's ceiling, as ProbeExecutor does:
                # eligibility is judged against it, so a probe needing a feature or resolution the base
                # withholds is faulted at dispatch rather than run.
                scenarios=[probe.scenario for probe in ALL_PROBES],
            ).__aenter__()
        return self._session

    async def aclose(self) -> None:
        """Shut the warm worker down, if one was ever booted."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def warm_session() -> AsyncGenerator[_LazyWarmSession, None]:
    """One warm worker for the whole module's catalog probes (the boot is paid once, not 20 times)."""
    lazy = _LazyWarmSession()
    try:
        yield lazy
    finally:
        await lazy.aclose()


# The cold-boot baselines run before every warm probe, not in catalog order: each boots its own full
# worker, and once the first warm probe has booted the module's shared session, a cold boot would have
# to fit a second worker beside it on the same card. The stable sort keeps catalog order otherwise.
_ORDERED_PROBES = sorted(ALL_PROBES, key=lambda probe: 0 if _keeps_cold_boot(probe) else 1)


@pytest.mark.parametrize("probe", _ORDERED_PROBES, ids=lambda probe: probe.capability.slug)
async def test_capability_probe_real(
    probe: CapabilityProbe,
    gpu_machine_info: MachineInfo,
    warm_session: _LazyWarmSession,
    record_probe_timing: Callable[[str, str], None],
) -> None:
    """Every catalog probe the machine can host is PROVEN on real hardware; the rest self-skip."""
    skip_reason = requirement_skip_reason(
        compute_probe_requirements(probe),
        machine=gpu_machine_info,
        process_mode="real",
        civitai_available=civitai_token_available(),
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)

    # The baselines boot their own worker; every other probe rides the module's warm one.
    session = None if _keeps_cold_boot(probe) else await warm_session.get()

    result = await run_capability_probe_async(
        probe,
        process_mode="real",
        total_vram_mb=gpu_machine_info.total_vram_mb,
        warm_session=session,
    )
    if result.timing is not None:
        record_probe_timing(probe.capability.slug, result.timing.summary())
    assert result.verdict is CapabilityVerdict.PROVEN, "; ".join(result.reasons)
