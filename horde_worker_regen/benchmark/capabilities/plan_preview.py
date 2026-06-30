"""Project a capability catalog into per-probe plan rows: requirements plus the predicted run/skip verdict.

This is the no-boot preview behind ``horde-benchmark plan`` and the ``RampPlanned`` event the executor
emits before the first probe, so an operator (or the TUI) sees, up front, what each probe needs and
whether it will run on this machine. It reuses the same :func:`requirement_skip_reason` machine-fit gate
the executor applies, so the preview cannot drift from the run; the dynamic gates (an unmet prerequisite,
a catastrophe abort) are not predictable before the run and are left to the live supervisor.

Torch-free: it reads only the (torch-free) requirements projection and the plan-row model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from horde_worker_regen.benchmark.progress_channel import LevelPlanRow
from horde_worker_regen.benchmark.requirements import (
    civitai_token_available,
    compute_probe_requirements,
    requirement_skip_reason,
)

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.benchmark.capabilities.result import MachineInfo
    from horde_worker_regen.benchmark.requirements import LevelRequirements


def _download_summary(req: LevelRequirements) -> str:
    """Return a short phrase naming what a runnable-but-incomplete probe must fetch, or '' when nothing is."""
    parts: list[str] = []
    if req.models_missing:
        parts.append(f"{len(req.models_missing)} model{'s' if len(req.models_missing) != 1 else ''}")
    if req.controlnet_checkpoints_missing:
        parts.append("controlnet checkpoints")
    if req.controlnet_annotators_present is False:
        parts.append("controlnet annotators")
    return ", ".join(parts)


def _plan_row(req: LevelRequirements, verdict: str | None) -> LevelPlanRow:
    """Project a probe's requirements and pre-flight verdict into a compact plan row.

    A probe that fits this machine (no skip verdict) but lacks downloadable artifacts is "download first":
    runnable once fetched, so neither a green "ready" nor a grey "skip". A hard skip is never download-first.
    """
    download_summary = _download_summary(req) if verdict is None else ""
    return LevelPlanRow(
        level_id=req.level_id,
        stage=req.stage,
        tier=req.tier,
        estimated_vram_mb=req.estimated_vram_mb,
        min_disk_free_gb=req.min_disk_free_gb,
        free_disk_bytes=req.free_disk_bytes,
        download_bytes_needed=req.download_bytes_needed,
        num_models_missing=len(req.models_missing),
        requires_network=req.requires_network,
        requires_civitai_key=req.requires_civitai_key,
        requires_controlnet=req.requires_controlnet,
        controlnet_installed=req.controlnet_installed,
        controlnet_annotators_present=req.controlnet_annotators_present,
        controlnet_annotator_bytes=req.controlnet_annotator_bytes,
        features=req.features,
        will_run=verdict is None,
        verdict=verdict or "",
        needs_download=bool(download_summary),
        download_summary=download_summary,
    )


def build_capability_plan_rows(
    probes: list[CapabilityProbe],
    *,
    machine: MachineInfo,
    process_mode: str,
    force: bool = False,
    only_probe: str | None = None,
) -> list[LevelPlanRow]:
    """Build one plan row per probe: its requirements and the machine-fit verdict the run would reach."""
    civitai_available = civitai_token_available()
    rows: list[LevelPlanRow] = []
    for probe in probes:
        req = compute_probe_requirements(probe)
        if only_probe is not None and probe.probe_id != only_probe:
            verdict: str | None = "not selected (--only)"
        else:
            verdict = requirement_skip_reason(
                req,
                machine=machine,
                process_mode=process_mode,
                civitai_available=civitai_available,
                force=force,
            )
        rows.append(_plan_row(req, verdict))
    return rows


__all__ = ["build_capability_plan_rows"]
