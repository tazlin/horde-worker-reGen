"""A :class:`CapabilityProbe`: one runnable experiment that proves a single capability.

A probe binds a :class:`~horde_worker_regen.benchmark.capabilities.capability.Capability` to the
:class:`~horde_worker_regen.benchmark.scenarios.Scenario` that exercises it, the
:class:`~horde_worker_regen.benchmark.criteria.LevelCriteria` that decide pass/fail, and the explicit
set of capabilities that must already be proven before it is worth running (``requires``). It is the
capability-engine replacement for the old ``RampLevel``: the same workload and criteria, but with the
imperative stage/axis/rung lattice replaced by a declarative dependency edge.

Pure and torch-free: it carries only the (already torch-free) scenario and criteria models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from horde_worker_regen.benchmark.capabilities.capability import Capability
from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.scenarios import Scenario


class CapabilityProbe(BaseModel):
    """One experiment: run ``scenario`` under ``bridge_data_overrides`` and judge it by ``criteria``.

    The probe runs only when every capability in ``requires`` is proven and the machine can host it;
    this declarative edge replaces the old skip cascade (failed-tier-baseline / failed-axis-rung), so
    "a prerequisite did not hold" is the single reason a probe is skipped.
    """

    model_config = ConfigDict(frozen=True)

    capability: Capability
    scenario: Scenario
    criteria: LevelCriteria = Field(default_factory=LevelCriteria)
    requires: tuple[Capability, ...] = ()
    """Capabilities that must be proven first; an unmet prerequisite skips this probe."""
    bridge_data_overrides: dict[str, object] = Field(default_factory=dict)
    requires_network: bool = False
    timeout_seconds: float = 900.0
    baseline_hordelib: str = ""
    """The ``KNOWN_IMAGE_GENERATION_BASELINE`` value, for pre-flight burden estimates."""
    establishes_baseline: bool = False
    """The tier baseline probe records its observed it/s p50 as the tier reference."""

    @property
    def probe_id(self) -> str:
        """The probe's stable identifier, equal to its capability's slug (``sd15-controlnet``)."""
        return self.capability.slug


__all__ = ["CapabilityProbe"]
