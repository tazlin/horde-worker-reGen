"""Tests for per-level resource requirements and the machine-fit verdict."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark import requirements as requirements_mod
from horde_worker_regen.benchmark.ladder import LadderOptions, RampLevel, build_default_ladder
from horde_worker_regen.benchmark.report import MachineInfo
from horde_worker_regen.benchmark.requirements import (
    LevelRequirements,
    compute_level_requirements,
    requirement_skip_reason,
)
from horde_worker_regen.model_download_plan import DownloadPlan, ModelDiskInfo


def _present(_name: str) -> bool | None:
    """A presence resolver that reports every model as on disk (avoids touching the reference manager)."""
    return True


def _sd15_baseline_and_download_levels() -> tuple[RampLevel, RampLevel]:
    """Build a minimal sd15 ladder with the ad-hoc download level, returning (baseline, download)."""
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
            include_downloads=True,
        ),
    )
    baseline = next(level for level in ladder if level.establishes_tier_baseline)
    download = next(level for level in ladder if level.requires_network)
    return baseline, download


def test_compute_requirements_baseline_needs_no_network_or_key() -> None:
    """A plain sd15 baseline needs its checkpoint but neither network nor a CivitAI token."""
    baseline, _ = _sd15_baseline_and_download_levels()
    req = compute_level_requirements(baseline, present_resolver=_present)
    assert req.requires_network is False
    assert req.requires_civitai_key is False
    assert req.models_required
    assert req.models_missing == []


def test_compute_requirements_download_level_needs_network_and_key() -> None:
    """The ad-hoc lora level pulls from CivitAI, so it requires both network and a token."""
    _, download = _sd15_baseline_and_download_levels()
    req = compute_level_requirements(download, present_resolver=_present)
    assert req.requires_network is True
    assert req.requires_civitai_key is True
    assert "loras" in req.features


def _req(**overrides: object) -> LevelRequirements:
    """A minimal LevelRequirements for verdict tests; override only what the test exercises."""
    base: dict[str, object] = {
        "level_id": "A-sd15-baseline",
        "stage": "A",
        "tier": "sd15",
        "axis": "baseline",
        "baseline": "stable_diffusion_1",
    }
    base.update(overrides)
    return LevelRequirements(**base)  # type: ignore[arg-type]


def test_verdict_is_none_outside_real_mode() -> None:
    """fake/dry_run download and infer nothing, so resource gates never apply."""
    req = _req(estimated_vram_mb=999_999, requires_civitai_key=True)
    reason = requirement_skip_reason(
        req,
        machine=MachineInfo(total_vram_mb=1),
        process_mode="fake",
        civitai_available=False,
    )
    assert reason is None


def test_insufficient_vram_skips_unless_forced() -> None:
    """A level that needs more VRAM than the machine has is skipped, but --force attempts it anyway."""
    req = _req(estimated_vram_mb=24_000)
    machine = MachineInfo(total_vram_mb=8_000)
    skip = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        civitai_available=True,
    )
    assert skip is not None
    assert "insufficient VRAM" in skip

    forced = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        civitai_available=True,
        force=True,
    )
    assert forced is None


def test_missing_civitai_token_skips_lora_level_unless_forced() -> None:
    """A lora/TI level without a configured token is skipped with a clear reason; --force overrides."""
    req = _req(requires_civitai_key=True, estimated_vram_mb=2_000)
    machine = MachineInfo(total_vram_mb=8_000)
    skip = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        civitai_available=False,
    )
    assert skip is not None
    assert "CivitAI" in skip

    assert (
        requirement_skip_reason(
            req,
            machine=machine,
            process_mode="real",
            civitai_available=True,
        )
        is None
    )
    assert (
        requirement_skip_reason(
            req,
            machine=machine,
            process_mode="real",
            civitai_available=False,
            force=True,
        )
        is None
    )


def test_compute_requirements_uses_the_real_disk_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no injected resolver, the level's sized disk plan drives the missing models and download bytes."""
    fake_plan = DownloadPlan(
        models=[
            ModelDiskInfo(name="present", category=None, size_bytes=2 * 1024**3, on_disk=True, target_path="/a"),
            ModelDiskInfo(name="absent", category=None, size_bytes=5 * 1024**3, on_disk=False, target_path="/b"),
        ],
        present_bytes=2 * 1024**3,
        to_download_bytes=5 * 1024**3,
        total_bytes=7 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        fits=True,
        shortfall_bytes=0,
        unknown_size_models=[],
    )
    monkeypatch.setattr(requirements_mod, "models_disk_plan", lambda _names: fake_plan)
    baseline, _ = _sd15_baseline_and_download_levels()

    req = compute_level_requirements(baseline)

    assert req.models_missing == ["absent"]
    assert req.download_bytes_needed == 5 * 1024**3
    assert req.present_bytes == 2 * 1024**3
    assert req.free_disk_bytes == 100 * 1024**3
    assert [model.target_path for model in req.missing_models] == ["/b"]


def test_sizing_helpers_read_offline_never_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """``models_disk_plan`` and ``model_present_on_disk`` must read the reference offline, never online.

    These run inside short-lived benchmark subprocesses (the download modal's dry-run plan is one). An
    online manager would block on a per-process PRIMARY-API round-trip it cannot bound, which is what made
    the plan preview time out. This pins both helpers to ``ensure_offline_reference_manager`` so the
    network path cannot silently return.
    """
    import horde_worker_regen.reference_helper as reference_helper

    offline_calls = 0

    class _FakeOfflineManager:
        def get_all_model_references(self) -> dict:
            return {}

    def _fake_offline() -> _FakeOfflineManager:
        nonlocal offline_calls
        offline_calls += 1
        return _FakeOfflineManager()

    def _forbidden_online() -> object:
        raise AssertionError("benchmark sizing must not use the online (network) reference manager")

    monkeypatch.setattr(reference_helper, "ensure_offline_reference_manager", _fake_offline)
    monkeypatch.setattr(reference_helper, "ensure_model_reference_manager_initialized", _forbidden_online)

    requirements_mod.models_disk_plan(["anything"])
    requirements_mod.model_present_on_disk("anything")

    assert offline_calls == 2


def _sd15_controlnet_level() -> RampLevel:
    """Build an sd15 feature ladder and return the classic controlnet (preprocessor) level."""
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=True,
            include_alchemy=False,
        ),
    )
    return next(level for level in ladder if level.id.endswith("-controlnet"))


def test_controlnet_level_surfaces_installed_and_annotator_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A controlnet level reports requires_controlnet, the installed flag, and a non-zero annotator ROM."""
    monkeypatch.setattr(requirements_mod, "controlnet_installed", lambda: True)
    monkeypatch.setattr(requirements_mod, "controlnet_annotators_present", lambda control_types: True)
    monkeypatch.setattr(requirements_mod, "_controlnet_annotator_bytes", lambda control_types: 800 * 1024**2)

    req = compute_level_requirements(_sd15_controlnet_level(), present_resolver=_present)

    assert req.requires_controlnet is True
    assert req.controlnet_installed is True
    assert req.controlnet_annotators_present is True
    assert req.controlnet_annotator_bytes == 800 * 1024**2
    assert req.controlnet_install_hint == ""
    assert "controlnet" in req.features


def test_controlnet_annotator_presence_probed_only_when_extra_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Annotator presence is meaningful only with the extra: absent extra leaves ``present`` as None.

    Without onnxruntime the annotators are never fetched, so probing on-disk presence would just nag for
    something the worker cannot use anyway; the install gate is what a surface should show instead.
    """
    monkeypatch.setattr(requirements_mod, "controlnet_installed", lambda: False)

    def _should_not_be_called(control_types: list[str]) -> bool | None:
        raise AssertionError("annotator presence must not be probed when the controlnet extra is absent")

    monkeypatch.setattr(requirements_mod, "controlnet_annotators_present", _should_not_be_called)

    req = compute_level_requirements(_sd15_controlnet_level(), present_resolver=_present)

    assert req.controlnet_installed is False
    assert req.controlnet_annotators_present is None


def test_non_controlnet_level_has_no_controlnet_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain baseline never probes controlnet and reports it as not required, size zero."""
    monkeypatch.setattr(requirements_mod, "controlnet_installed", lambda: True)
    baseline, _ = _sd15_baseline_and_download_levels()

    req = compute_level_requirements(baseline, present_resolver=_present)

    assert req.requires_controlnet is False
    assert req.controlnet_installed is None
    assert req.controlnet_annotator_bytes == 0


def test_controlnet_not_installed_skips_with_install_hint() -> None:
    """A controlnet level on a worker without the extra is skipped with the install remedy; --force overrides."""
    machine = MachineInfo(total_vram_mb=8_000)
    req = _req(
        requires_controlnet=True,
        controlnet_installed=False,
        controlnet_install_hint="install `horde-worker-reGen[controlnet]`",
        estimated_vram_mb=2_000,
    )
    skip = requirement_skip_reason(req, machine=machine, process_mode="real", civitai_available=True)
    assert skip is not None
    assert "controlnet not installed" in skip
    assert "horde-worker-reGen[controlnet]" in skip

    forced = requirement_skip_reason(req, machine=machine, process_mode="real", civitai_available=True, force=True)
    assert forced is None

    installed = _req(requires_controlnet=True, controlnet_installed=True, estimated_vram_mb=2_000)
    assert requirement_skip_reason(installed, machine=machine, process_mode="real", civitai_available=True) is None


def test_absent_huge_checkpoint_is_a_hard_skip_even_when_forced() -> None:
    """A missing flux/qwen/zimage checkpoint cannot be forced: real-mode benchmarking never downloads weights."""
    for tier, missing_model in [
        ("flux", "Flux.1-Schnell fp8 (Compact)"),
        ("qwen", "Qwen-Image"),
        ("zimage", "Z-Image-Turbo"),
    ]:
        req = _req(tier=tier, models_missing=[missing_model])
        reason = requirement_skip_reason(
            req,
            machine=MachineInfo(total_vram_mb=80_000),
            process_mode="real",
            civitai_available=True,
            force=True,
        )
        assert reason is not None, f"expected hard skip for {tier}"
        assert "not present on disk" in reason
