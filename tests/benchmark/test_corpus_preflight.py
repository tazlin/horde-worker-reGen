"""Tests for the pricing-corpus preflight.

The preflight's job is to turn "this machine cannot produce admissible rows" into a named remedy before
hours are spent, so the subject here is the verdict and the ``fix`` text of each check. Every reading is
injected through :class:`Probes`, so none of this needs a GPU, weights, or a hordelib checkout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from horde_worker_regen.benchmark.corpus_preflight import (
    COMFY_PIN_PACKAGES,
    CORPUS_FREE_DISK_BYTES,
    CORPUS_TIER_MIN_VRAM_MB,
    HordelibFacts,
    PreflightCheck,
    PreflightReport,
    Probes,
    corpus_bench_tiers,
    format_report,
    requires_kudos_manifest,
    run_preflight,
    stamp_definition,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.pricing_corpus import CorpusMachineFacts, build_pricing_corpus_scenario

GOOD_PINS: dict[str, str] = {
    "comfy-kitchen": "0.2.31",
    "comfyui-embedded-docs": "0.5.10",
    "comfyui-frontend-package": "1.51.9",
    "comfyui-workflow-templates": "0.11.50",
}

MODELS = ["Deliberate", "AlbedoBase XL (SDXL)"]

HEALTHY_HORDELIB = HordelibFacts(
    importable=True,
    module_path="/checkout/hordelib/__init__.py",
    editable=True,
    version="7.6.1",
    source_path="/checkout",
    manifest_importable=True,
)


def healthy_probes(
    *,
    accelerators: Callable[[], list[str]] | None = None,
    vram_mb: Callable[[], int | None] | None = None,
    hordelib: Callable[[], HordelibFacts] | None = None,
    required_pins: Callable[[], dict[str, str]] | None = None,
    installed_versions: Callable[[Sequence[str]], dict[str, str | None]] | None = None,
    cache_home: Callable[[], str | None] | None = None,
    writable: Callable[[Path], bool] | None = None,
    free_bytes: Callable[[Path], int | None] | None = None,
    missing_models: Callable[[list[str]], list[str] | None] | None = None,
    civitai_token: Callable[[], str | None] | None = None,
    live_worker: Callable[[], str | None] | None = None,
) -> Probes:
    """Probes for a machine on which every check passes, with individual readings overridable."""
    return Probes(
        accelerators=accelerators or (lambda: ["NVIDIA L40S (46068 MB, cuda)"]),
        vram_mb=vram_mb or (lambda: 46068),
        hordelib=hordelib or (lambda: HEALTHY_HORDELIB),
        required_pins=required_pins or (lambda: dict(GOOD_PINS)),
        installed_versions=installed_versions
        or (lambda packages: {package: GOOD_PINS.get(package) for package in packages}),
        cache_home=cache_home or (lambda: "/models"),
        writable=writable or (lambda path: True),
        free_bytes=free_bytes or (lambda path: CORPUS_FREE_DISK_BYTES * 2),
        missing_models=missing_models or (lambda models: []),
        civitai_token=civitai_token or (lambda: "token"),
        live_worker=live_worker or (lambda: None),
    )


def check_named(report: PreflightReport, name: str) -> PreflightCheck:
    """The single check called ``name`` in a report."""
    matches = [check for check in report.checks if check.name == name]
    assert len(matches) == 1, f"expected one {name!r} check, got {len(matches)}"
    return matches[0]


def test_healthy_machine_passes_every_check() -> None:
    """A machine with every reading healthy passes, and the checks run in a stable order."""
    report = run_preflight("census", "alice-l40s", models=MODELS, require_manifest=True, probes=healthy_probes())

    assert report.passed, [check for check in report.checks if not check.passed]
    assert report.failures == []
    assert [check.name for check in report.checks] == [
        "machine id",
        "cuda",
        "vram",
        "hordelib",
        "kudos manifest",
        "comfy pins",
        "cache home",
        "models",
        "free disk",
        "civitai token",
        "no live worker",
    ]


def test_manifest_check_is_only_run_when_the_tier_needs_it() -> None:
    """A tier whose vocabularies are not manifest-derived is not held to the manifest."""
    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=healthy_probes())

    assert "kudos manifest" not in [check.name for check in report.checks]
    assert report.passed


def test_missing_model_names_it_and_gives_the_download_command() -> None:
    """A model that is not on disk is named, with the command that fetches it."""
    probes = healthy_probes(missing_models=lambda models: ["AlbedoBase XL (SDXL)"])

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "models")

    assert not report.passed
    assert "AlbedoBase XL (SDXL)" in check.detail
    assert check.fix == "horde-benchmark download --tiers sd15,sdxl"


def test_heavy_tier_download_command_covers_the_large_baselines() -> None:
    """The heavy tier's download fix names the large baselines it also needs."""
    probes = healthy_probes(missing_models=lambda models: ["Flux.1-Schnell fp8 (Compact)"])

    report = run_preflight("heavy", "alice-l40s", models=MODELS, require_manifest=True, probes=probes)

    assert check_named(report, "models").fix == "horde-benchmark download --tiers sd15,sdxl,flux,qwen,zimage"


def test_undeterminable_model_state_fails_rather_than_passing_silently() -> None:
    """Unknown on-disk state fails: an unchecked model is not a present one."""
    probes = healthy_probes(missing_models=lambda models: None)

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)

    assert not report.passed
    assert "could not determine" in check_named(report, "models").detail


def test_a_drifted_comfy_pin_fails_and_the_fix_installs_the_pinned_version() -> None:
    """A drifted ComfyUI pin fails, naming the found and wanted versions."""
    installed = dict(GOOD_PINS, **{"comfyui-frontend-package": "1.50.0"})
    probes = healthy_probes(installed_versions=lambda packages: {p: installed.get(p) for p in packages})

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "comfy pins")

    assert not report.passed
    assert "comfyui-frontend-package: 1.50.0 (need 1.51.9)" in check.detail
    assert check.fix == "uv pip install comfyui-frontend-package==1.51.9"


def test_an_absent_comfy_package_is_reported_as_absent() -> None:
    """A missing ComfyUI package fails with an install command for every pin."""
    probes = healthy_probes(installed_versions=lambda packages: dict.fromkeys(packages))

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "comfy pins")

    assert not report.passed
    assert "absent" in check.detail
    for package in COMFY_PIN_PACKAGES:
        assert f"{package}==" in check.fix


@pytest.mark.parametrize("machine_id", ["Alice-L40S", "ab", "-leading", "with_underscore", "x" * 42])
def test_a_malformed_machine_id_fails(machine_id: str) -> None:
    """A machine id that is not canonically shaped fails, since rows are keyed on it."""
    report = run_preflight("standard", machine_id, models=MODELS, require_manifest=False, probes=healthy_probes())

    assert not report.passed
    assert "--machine" in check_named(report, "machine id").fix


def test_no_machine_id_passes_but_says_the_rows_are_unattributable() -> None:
    """A run with no machine id is allowed, but the report says what it costs."""
    report = run_preflight("smoke", None, models=MODELS, require_manifest=False, probes=healthy_probes())

    assert report.passed
    assert "cannot be pooled" in check_named(report, "machine id").detail


def test_no_accelerator_fails_cuda() -> None:
    """A machine torch sees no accelerator on cannot price real inference."""
    report = run_preflight(
        "standard",
        "alice-l40s",
        models=MODELS,
        require_manifest=False,
        probes=healthy_probes(accelerators=list),
    )

    assert not report.passed
    assert not check_named(report, "cuda").passed


def test_a_non_editable_hordelib_fails_and_reports_its_path() -> None:
    """A non-editable hordelib fails and its path is reported, so it can be replaced."""
    facts = HordelibFacts(importable=True, module_path="/site-packages/hordelib/__init__.py", editable=False)
    report = run_preflight(
        "standard",
        "alice-l40s",
        models=MODELS,
        require_manifest=False,
        probes=healthy_probes(hordelib=lambda: facts),
    )
    check = check_named(report, "hordelib")

    assert not report.passed
    assert "/site-packages/hordelib/__init__.py" in check.detail
    assert check.fix.startswith("uv pip install -e")


def test_a_hordelib_without_the_manifest_fails_only_the_manifest_check() -> None:
    """A hordelib missing the manifest fails that check alone."""
    facts = HordelibFacts(importable=True, module_path="/checkout", editable=True, manifest_importable=False)
    report = run_preflight(
        "census",
        "alice-l40s",
        models=MODELS,
        require_manifest=True,
        probes=healthy_probes(hordelib=lambda: facts),
    )

    assert [check.name for check in report.failures] == ["kudos manifest"]


def test_an_unresolved_cache_home_fails_both_it_and_the_disk_check() -> None:
    """With no cache home there is nothing to measure free space on."""
    report = run_preflight(
        "standard",
        "alice-l40s",
        models=MODELS,
        require_manifest=False,
        probes=healthy_probes(cache_home=lambda: None),
    )

    assert {check.name for check in report.failures} == {"cache home", "free disk"}
    assert "AIWORKER_CACHE_HOME" in check_named(report, "cache home").fix


def test_a_full_cache_volume_fails_and_says_how_much_to_free() -> None:
    """A short cache volume fails with the shortfall spelled out."""
    probes = healthy_probes(free_bytes=lambda path: CORPUS_FREE_DISK_BYTES // 2)

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)

    assert "free up 10.0 GB" in check_named(report, "free disk").fix


def test_a_card_at_the_tier_threshold_passes() -> None:
    """A card holding exactly what the tier asks for is admitted."""
    probes = healthy_probes(vram_mb=lambda: CORPUS_TIER_MIN_VRAM_MB["heavy"])

    report = run_preflight("heavy", "alice-l40s", models=MODELS, require_manifest=True, probes=probes)
    check = check_named(report, "vram")

    assert check.passed, check.detail
    assert str(CORPUS_TIER_MIN_VRAM_MB["heavy"]) in check.detail


def test_a_card_below_the_tier_threshold_fails_and_names_a_tier_that_fits() -> None:
    """A card too small for the tier is turned away toward a tier it can actually run."""
    probes = healthy_probes(vram_mb=lambda: 12288)

    report = run_preflight("heavy", "alice-l40s", models=MODELS, require_manifest=True, probes=probes)
    check = check_named(report, "vram")

    assert not check.passed
    assert "12288 MB" in check.detail
    named = check.fix.rsplit(" ", 1)[-1]
    assert CORPUS_TIER_MIN_VRAM_MB[named] < CORPUS_TIER_MIN_VRAM_MB["heavy"]
    assert CORPUS_TIER_MIN_VRAM_MB[named] <= 12288


def test_a_card_below_every_tier_says_so_rather_than_naming_one() -> None:
    """When no tier fits, the fix asks for a bigger card instead of naming an impossible tier."""
    probes = healthy_probes(vram_mb=lambda: 4096)

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "vram")

    assert not check.passed
    assert str(min(CORPUS_TIER_MIN_VRAM_MB.values())) in check.fix
    assert "--tier" not in check.fix


def test_unreadable_vram_fails_and_says_the_probe_could_not_read_the_card() -> None:
    """An unanswerable card is a failure: a run admitted on an unknown card can overflow it."""
    probes = healthy_probes(vram_mb=lambda: None)

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "vram")

    assert not check.passed
    assert "could not read" in check.detail


def test_the_heavy_tier_does_not_need_a_civitai_token() -> None:
    """The heavy tier carries no LoRA cell, so a missing token is not a reason to refuse the run."""
    probes = healthy_probes(civitai_token=lambda: None)

    report = run_preflight("heavy", "alice-l40s", models=MODELS, require_manifest=True, probes=probes)
    check = check_named(report, "civitai token")

    assert check.passed
    assert check.detail == "not needed: the heavy tier has no LoRA cells"


def test_a_missing_civitai_token_fails() -> None:
    """Without a CivitAI token the LoRA cells would measure a failed fetch."""
    probes = healthy_probes(civitai_token=lambda: None)

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)

    assert "CIVIT_API_TOKEN" in check_named(report, "civitai token").fix


def test_a_live_worker_fails_with_its_reason() -> None:
    """A worker already holding the working directory blocks the run."""
    probes = healthy_probes(live_worker=lambda: "a .abort sentinel is present in the working directory")

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    check = check_named(report, "no live worker")

    assert ".abort" in check.detail
    assert "stop the running worker" in check.fix


def test_the_report_renders_every_check_with_its_fix() -> None:
    """The rendered table carries every check, its fix, and the failure count."""
    probes = healthy_probes(
        civitai_token=lambda: None, live_worker=lambda: "worker child process(es) still running: 1"
    )

    report = run_preflight("standard", "alice-l40s", models=MODELS, require_manifest=False, probes=probes)
    rendered = format_report(report)

    assert "tier standard, machine alice-l40s" in rendered
    assert "STATUS  CHECK" in rendered.replace("\n", " ")
    for check in report.checks:
        assert check.name in rendered
    assert "CIVIT_API_TOKEN" in rendered
    assert "2 of 10 checks failed" in rendered


def test_a_passing_report_says_so() -> None:
    """A clean report states that every check passed."""
    report = run_preflight("smoke", "alice-l40s", models=MODELS, require_manifest=False, probes=healthy_probes())

    assert "All 10 checks passed." in format_report(report)


def test_tier_mapping_matches_what_the_worker_env_needs() -> None:
    """Each corpus tier maps to the benchmark tiers its worker env must cover."""
    assert corpus_bench_tiers("smoke") == [BenchTier.SD15, BenchTier.SDXL]
    assert corpus_bench_tiers("standard") == [BenchTier.SD15, BenchTier.SDXL]
    assert corpus_bench_tiers("census") == [BenchTier.SD15, BenchTier.SDXL]
    assert corpus_bench_tiers("heavy") == [
        BenchTier.SD15,
        BenchTier.SDXL,
        BenchTier.FLUX,
        BenchTier.QWEN,
        BenchTier.ZIMAGE,
    ]


def test_only_the_manifest_backed_tiers_require_the_manifest() -> None:
    """Only the census and heavy tiers read the kudos feature manifest."""
    assert not requires_kudos_manifest("smoke")
    assert not requires_kudos_manifest("standard")
    assert requires_kudos_manifest("census")
    assert requires_kudos_manifest("heavy")


def test_stamping_records_the_run_provenance() -> None:
    """Stamping fills in the timestamp and machine facts without mutating the source."""
    _scenario, definition = build_pricing_corpus_scenario("smoke")
    facts = CorpusMachineFacts(machine_id="alice-l40s", gpu_model="NVIDIA L40S", vram_mb=46068)

    stamped = stamp_definition(definition, machine_id="alice-l40s", created_at=1234.5, facts=facts)

    assert stamped.created_at == 1234.5
    assert stamped.machine is not None
    assert stamped.machine.machine_id == "alice-l40s"
    assert stamped.machine.vram_mb == 46068
    # The built definition is not mutated, so a caller can still write the unstamped corpus description.
    assert definition.created_at is None
    assert definition.machine is None


def test_stamping_without_a_machine_id_leaves_the_machine_unset() -> None:
    """An unattributable run records its time but invents no machine."""
    _scenario, definition = build_pricing_corpus_scenario("smoke")

    stamped = stamp_definition(definition, machine_id=None, created_at=99.0)

    assert stamped.created_at == 99.0
    assert stamped.machine is None


def test_a_stamped_definition_survives_a_round_trip_through_its_artifact(tmp_path: Path) -> None:
    """The stamped provenance is preserved through the written artifact."""
    from horde_worker_regen.benchmark.pricing_corpus import PricingCorpusDefinition, write_definition_artifact

    _scenario, definition = build_pricing_corpus_scenario("smoke")
    facts = CorpusMachineFacts(machine_id="alice-l40s", torch_version="2.9.0+cu128")
    stamped = stamp_definition(definition, machine_id="alice-l40s", created_at=1234.5, facts=facts)

    path = write_definition_artifact(stamped, tmp_path / "corpus.json")
    reloaded = PricingCorpusDefinition.model_validate_json(path.read_text(encoding="utf-8"))

    assert reloaded.created_at == 1234.5
    assert reloaded.machine is not None
    assert reloaded.machine.torch_version == "2.9.0+cu128"
