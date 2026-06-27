"""Memory fail-fast preflight for the multi-model soak.

A multi-model soak wants one resident model per inference process so every process stays busy.
But N co-resident models can exceed VRAM (or RAM), and the worker's memory-pressure valves would
then silently evict and reload, defeating the very residency the soak is meant to exercise (and,
without a worker-owned budget, risking a cross-process OOM). So before committing to an N-model
topology the controller asks this module: *do N models fit?*, and if not, *what is the largest
pool that does?* That keeps the soak honest on machines that cannot hold the full pool, rather
than producing a thrash-dominated, misleading result.

The math is deliberately conservative: each resident model is charged its full single-job burden
(weights + a working set), which over-counts the working memory of an idle-resident model. Erring
high is the right direction for a fail-fast gate. Pure and table-testable; no torch/NVML here.
"""

from __future__ import annotations

import dataclasses

DEFAULT_VRAM_RESERVE_MB: float = 1500.0
"""Headroom kept free of model footprints, matching ``LevelCriteria.min_vram_headroom_mb``."""


@dataclasses.dataclass(frozen=True)
class SoakFitPlan:
    """The verdict on how many distinct models a soak topology can keep resident."""

    desired_models: int
    fitting_models: int
    per_model_vram_mb: float
    total_vram_mb: float
    reserve_vram_mb: float
    per_model_ram_mb: float | None = None
    total_ram_mb: float | None = None

    @property
    def fits(self) -> bool:
        """True when the full desired pool fits in memory."""
        return self.fitting_models >= self.desired_models

    @property
    def is_viable(self) -> bool:
        """True when at least one model fits (a soak can run, perhaps with a trimmed pool)."""
        return self.fitting_models >= 1

    @property
    def reason(self) -> str:
        """A human-readable summary of the fit decision."""
        if self.fits:
            return f"all {self.desired_models} soak models fit in {self.total_vram_mb:.0f} MB VRAM"
        if self.is_viable:
            return (
                f"only {self.fitting_models} of {self.desired_models} soak models fit "
                f"(~{self.per_model_vram_mb:.0f} MB each, {self.reserve_vram_mb:.0f} MB reserve, "
                f"{self.total_vram_mb:.0f} MB total VRAM); trimming the pool"
            )
        return (
            f"no soak model fits: ~{self.per_model_vram_mb:.0f} MB each leaves nothing under the "
            f"{self.reserve_vram_mb:.0f} MB reserve within {self.total_vram_mb:.0f} MB VRAM"
        )


def _fitting_count(*, per_unit_mb: float, total_mb: float, reserve_mb: float, desired: int) -> int:
    """How many ``per_unit_mb`` footprints fit under ``total_mb - reserve_mb`` (capped at desired).

    A non-positive per-unit estimate means "cannot estimate", so return ``desired`` so a missing
    burden number never blocks the soak (the run's own OOM guards remain the backstop).
    """
    if per_unit_mb <= 0:
        return desired
    budget = total_mb - reserve_mb
    if budget <= 0:
        return 0
    return max(0, min(desired, int(budget // per_unit_mb)))


def plan_soak_topology(
    *,
    desired_models: int,
    per_model_vram_mb: float,
    total_vram_mb: float,
    reserve_vram_mb: float = DEFAULT_VRAM_RESERVE_MB,
    per_model_ram_mb: float | None = None,
    total_ram_mb: float | None = None,
    reserve_ram_mb: float = 0.0,
) -> SoakFitPlan:
    """Decide how many of ``desired_models`` can stay co-resident within VRAM (and RAM, if given).

    The fitting count is the minimum of the VRAM-bound and (when both RAM figures are supplied)
    the RAM-bound counts, so the tighter resource governs.
    """
    fitting = _fitting_count(
        per_unit_mb=per_model_vram_mb,
        total_mb=total_vram_mb,
        reserve_mb=reserve_vram_mb,
        desired=desired_models,
    )
    if per_model_ram_mb is not None and total_ram_mb is not None:
        fitting = min(
            fitting,
            _fitting_count(
                per_unit_mb=per_model_ram_mb,
                total_mb=total_ram_mb,
                reserve_mb=reserve_ram_mb,
                desired=desired_models,
            ),
        )

    return SoakFitPlan(
        desired_models=desired_models,
        fitting_models=fitting,
        per_model_vram_mb=per_model_vram_mb,
        total_vram_mb=total_vram_mb,
        reserve_vram_mb=reserve_vram_mb,
        per_model_ram_mb=per_model_ram_mb,
        total_ram_mb=total_ram_mb,
    )
