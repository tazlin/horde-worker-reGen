"""Check that a machine can produce admissible pricing-corpus rows, before hours of it are spent.

A corpus run is long and its output is only useful if every row can be attributed to a machine and
compared against rows from other machines. The failures that make a run worthless are all knowable up
front: no CUDA, a card too small for the tier, a hordelib without the kudos feature manifest the census
vocabularies come from, a ComfyUI pin the inference children die on, a model that is not on disk, no
CivitAI token for the tiers that carry LoRA cells, or another worker already holding the card. Each
check therefore carries the exact command that clears it, so the operator fixes the machine rather than
reading a stack trace four hours in.

Every probe is injectable (:class:`Probes`), so the checks can be exercised on a box with no GPU and no
weights. The default probes stay torch-free in this process: the accelerator enumeration runs
out-of-process (the corpus CLI goes on to host the worker parent, which must never load torch) and
package versions are read from installed metadata rather than by importing anything.
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from loguru import logger

from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.pricing_corpus import (
    CorpusMachineFacts,
    PricingCorpusDefinition,
    tier_has_lora_cells,
)

HORDELIB_DISTRIBUTION = "horde_engine"
"""Distribution name behind the ``hordelib`` import name; its metadata carries the ComfyUI pins."""

KUDOS_MANIFEST_MODULE = "hordelib.kudos_training.manifest"
"""The feature manifest the census and heavy vocabularies are derived from."""

COMFY_PIN_PACKAGES: tuple[str, ...] = (
    "comfy-kitchen",
    "comfyui-embedded-docs",
    "comfyui-frontend-package",
    "comfyui-workflow-templates",
)
"""ComfyUI packages hordelib pins exactly; a drifted one kills the inference children at startup."""

_FALLBACK_COMFY_PINS: dict[str, str] = {
    "comfy-kitchen": "0.2.31",
    "comfyui-embedded-docs": "0.5.10",
    "comfyui-frontend-package": "1.51.9",
    "comfyui-workflow-templates": "0.11.50",
}
"""Pins to check against when hordelib's own metadata cannot be read.

Copied from the ``horde_engine`` distribution's requirements, which is where the authoritative values
live; :func:`required_comfy_pins` prefers those whenever the metadata is readable."""

MACHINE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
"""Machine ids are lowercase, dash-separated and stable across runs, since rows are keyed on them."""

CORPUS_FREE_DISK_BYTES = 20 * 1024**3
"""Free space the cache volume needs: the corpus fetches LoRAs and re-fetches evicted ones as it runs."""

CORPUS_TIER_MIN_VRAM_MB: dict[str, int] = {
    # The small tiers are SDXL-sized, which is the smallest card the corpus is worth running on at all.
    "smoke": 8192,
    "standard": 8192,
    "census": 8192,
    # The heavy families ship fp8 checkpoints of roughly 17-20 GB on disk that have to sit in VRAM beside
    # their text encoders, so a smaller card pages instead of sampling and prices the paging.
    "heavy": 24576,
}
"""VRAM each corpus tier needs on its largest card, in MB."""

_MANIFEST_TIERS: frozenset[str] = frozenset({"census", "heavy"})
"""Tiers whose vocabularies are read from the kudos feature manifest."""

_HEAVY_BENCH_TIERS: tuple[BenchTier, ...] = (
    BenchTier.SD15,
    BenchTier.SDXL,
    BenchTier.FLUX,
    BenchTier.QWEN,
    BenchTier.ZIMAGE,
)


def corpus_bench_tiers(tier: str) -> list[BenchTier]:
    """The benchmark tiers a corpus tier's worker environment must be prepared for.

    Only the heavy tier reaches past the two small baselines, and it needs the beta opt-in that
    ``ensure_worker_env`` derives from the presence of a beta tier in this list.
    """
    if tier == "heavy":
        return list(_HEAVY_BENCH_TIERS)
    return [BenchTier.SD15, BenchTier.SDXL]


def requires_kudos_manifest(tier: str) -> bool:
    """Whether a corpus tier's cells are built from the kudos feature manifest."""
    return tier in _MANIFEST_TIERS


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight condition, its verdict, and the exact remedy when it failed."""

    name: str
    passed: bool
    detail: str
    fix: str = ""
    """Command or edit that clears the check; empty when the check passed."""


@dataclass(frozen=True)
class PreflightReport:
    """The verdict for one machine and tier: every check that ran, in the order they ran."""

    tier: str
    machine_id: str | None
    checks: list[PreflightCheck]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[PreflightCheck]:
        """The failed checks, for a caller that wants to log only what needs fixing."""
        return [check for check in self.checks if not check.passed]


@dataclass(frozen=True)
class HordelibFacts:
    """How hordelib is installed in this interpreter, as the preflight needs to judge it."""

    importable: bool
    module_path: str | None = None
    editable: bool | None = None
    """True/False from the distribution's install record; None when the record is unreadable."""
    version: str | None = None
    source_path: str | None = None
    """The checkout an editable install points at, when the install record names one."""
    manifest_importable: bool = False


def probe_hordelib() -> HordelibFacts:
    """Read how hordelib is installed without importing it (importing it is expensive and torch-laden)."""
    spec = importlib.util.find_spec("hordelib")
    if spec is None:
        return HordelibFacts(importable=False)

    version: str | None = None
    editable: bool | None = None
    source_path: str | None = None
    try:
        distribution = importlib.metadata.distribution(HORDELIB_DISTRIBUTION)
        version = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            editable = bool(payload.get("dir_info", {}).get("editable"))
            url = payload.get("url")
            # The install record stores a file:// URL; an operator needs the path they would re-install from.
            source_path = url2pathname(urlparse(str(url)).path) if url else None
        else:
            editable = False
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
        logger.debug(f"Could not read {HORDELIB_DISTRIBUTION} install metadata: {type(error).__name__} {error}")

    manifest = False
    try:
        manifest = importlib.util.find_spec(KUDOS_MANIFEST_MODULE) is not None
    except (ImportError, AttributeError, ValueError):
        manifest = False

    return HordelibFacts(
        importable=True,
        module_path=spec.origin,
        editable=editable,
        version=version,
        source_path=source_path,
        manifest_importable=manifest,
    )


def required_comfy_pins() -> dict[str, str]:
    """The ComfyUI versions hordelib pins, read from its own requirements where they are readable."""
    pins = dict(_FALLBACK_COMFY_PINS)
    try:
        requirements = importlib.metadata.requires(HORDELIB_DISTRIBUTION) or []
    except importlib.metadata.PackageNotFoundError:
        return pins
    for requirement in requirements:
        name, separator, version = requirement.partition("==")
        if not separator:
            continue
        name = name.strip()
        if name in COMFY_PIN_PACKAGES:
            pins[name] = version.split(";")[0].strip()
    return pins


def installed_versions(packages: Sequence[str]) -> dict[str, str | None]:
    """The installed version of each named package, None where the package is absent."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@functools.cache
def _probed_accelerators() -> tuple[tuple[str, int, str], ...]:
    """The accelerators torch can see, as (name, total VRAM in MB, kind).

    Enumerated out of process so this one stays torch-free, and memoized because several checks read the
    same enumeration while the probe costs a subprocess launch and a torch import; the cards a preflight
    judges do not change under it.
    """
    try:
        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        return tuple((device.name, device.total_vram_mb, device.kind) for device in probe_accelerators())
    except Exception as error:  # noqa: BLE001 - an unavailable probe is a failed check, not a crash
        logger.debug(f"Accelerator probe failed: {type(error).__name__} {error}")
        return ()


def _probe_accelerator_names() -> list[str]:
    """The accelerators torch can see, rendered for the report."""
    return [f"{name} ({vram_mb} MB, {kind})" for name, vram_mb, kind in _probed_accelerators()]


def _probe_largest_vram_mb() -> int | None:
    """The largest visible accelerator's total VRAM in MB, or None when no card can be read."""
    return max((vram_mb for _name, vram_mb, _kind in _probed_accelerators()), default=None)


def _resolve_cache_home() -> str | None:
    """The model cache directory, resolved the way the worker resolves it."""
    from horde_worker_regen.analysis.system_info import resolve_cache_home

    return resolve_cache_home()


def _is_writable(path: Path) -> bool:
    """Whether a file can actually be created under ``path`` (permission bits alone are not proof)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".corpus-preflight-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _free_bytes(path: Path) -> int | None:
    """Free bytes on the volume holding ``path``, or None when it cannot be read."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _missing_models(models: list[str]) -> list[str] | None:
    """The named models that are not on disk, or None when on-disk state could not be determined."""
    from horde_worker_regen.benchmark.requirements import models_disk_plan

    plan = models_disk_plan(models)
    if plan is None:
        return None
    return [info.name for info in plan.models if not info.on_disk]


def _civitai_token() -> str | None:
    """The CivitAI token the LoRA cells will download with, from the env the worker config populates."""
    from horde_worker_regen.benchmark.requirements import _CIVITAI_TOKEN_ENV_VARS

    for name in _CIVITAI_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _live_worker_reason() -> str | None:
    """Why a worker looks live in this working directory, or None when nothing suggests one.

    reGen holds no exclusive process lock, so this reads the two durable signals it does leave: the
    ``.abort`` sentinel a stopping worker writes, and the owned-pid registry, whose records name a still
    running child when its pid and creation time both still match.
    """
    if Path(".abort").exists():
        return "a .abort sentinel is present in the working directory"

    try:
        import psutil

        from horde_worker_regen.app_state import default_app_state_dir
        from horde_worker_regen.process_management.lifecycle.owned_process_registry import (
            OWNED_PIDS_FILENAME,
            OwnedProcessRecord,
        )

        path = default_app_state_dir() / OWNED_PIDS_FILENAME
        if not path.is_file():
            return None
        records = [OwnedProcessRecord.model_validate(entry) for entry in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as error:  # noqa: BLE001 - an unreadable registry is not evidence of a live worker
        logger.debug(f"Could not read the owned-pid registry: {type(error).__name__} {error}")
        return None

    alive: list[int] = []
    for record in records:
        try:
            process = psutil.Process(record.os_pid)
            if abs(process.create_time() - record.create_time) <= 1.0:
                alive.append(record.os_pid)
        except (psutil.Error, OSError):
            continue
    if alive:
        return f"worker child process(es) still running: {', '.join(str(pid) for pid in alive)}"
    return None


@dataclass
class Probes:
    """The machine readings the checks are made from, injectable so the checks can be tested anywhere.

    Defaults are installed per instance (never as class attributes) so a plain function stays a plain
    function rather than binding as a method.
    """

    accelerators: Callable[[], list[str]] = field(default_factory=lambda: _probe_accelerator_names)
    vram_mb: Callable[[], int | None] = field(default_factory=lambda: _probe_largest_vram_mb)
    hordelib: Callable[[], HordelibFacts] = field(default_factory=lambda: probe_hordelib)
    required_pins: Callable[[], dict[str, str]] = field(default_factory=lambda: required_comfy_pins)
    installed_versions: Callable[[Sequence[str]], dict[str, str | None]] = field(
        default_factory=lambda: installed_versions,
    )
    cache_home: Callable[[], str | None] = field(default_factory=lambda: _resolve_cache_home)
    writable: Callable[[Path], bool] = field(default_factory=lambda: _is_writable)
    free_bytes: Callable[[Path], int | None] = field(default_factory=lambda: _free_bytes)
    missing_models: Callable[[list[str]], list[str] | None] = field(default_factory=lambda: _missing_models)
    civitai_token: Callable[[], str | None] = field(default_factory=lambda: _civitai_token)
    live_worker: Callable[[], str | None] = field(default_factory=lambda: _live_worker_reason)


def _check_machine_id(machine_id: str | None) -> PreflightCheck:
    """Judge the machine id: rows are keyed on it, so it must be given and canonically shaped."""
    if machine_id is None:
        return PreflightCheck(
            name="machine id",
            passed=True,
            detail="none given; this run's rows carry no machine facts and cannot be pooled with others",
        )
    if not MACHINE_ID_PATTERN.match(machine_id):
        return PreflightCheck(
            name="machine id",
            passed=False,
            detail=f"{machine_id!r} is not a valid machine id (lowercase letters, digits and dashes, 3-41 chars)",
            fix="re-run with --machine <owner>-<gpu>, for example --machine alice-l40s",
        )
    return PreflightCheck(name="machine id", passed=True, detail=machine_id)


def _check_cuda(probes: Probes) -> PreflightCheck:
    """Judge GPU availability; the corpus prices real inference, so a CPU box produces nothing."""
    devices = probes.accelerators()
    if not devices:
        return PreflightCheck(
            name="cuda",
            passed=False,
            detail="torch reported no accelerator",
            fix="install a GPU torch build in the worker venv (see docs/how-to/choose-a-pytorch-build.md)",
        )
    return PreflightCheck(name="cuda", passed=True, detail="; ".join(devices))


def _largest_fitting_tier(vram_mb: int) -> str | None:
    """The most demanding corpus tier this card holds, or None when it holds none of them."""
    fitting = [tier for tier, needed in CORPUS_TIER_MIN_VRAM_MB.items() if vram_mb >= needed]
    if not fitting:
        return None
    return sorted(fitting, key=lambda tier: (-CORPUS_TIER_MIN_VRAM_MB[tier], tier))[0]


def _check_vram(probes: Probes, tier: str) -> PreflightCheck:
    """Judge the card against the tier's working set; a tier that overflows prices paging, not inference."""
    # A tier the table does not name is held to the smallest requirement rather than crashing the report.
    needed = CORPUS_TIER_MIN_VRAM_MB.get(tier, min(CORPUS_TIER_MIN_VRAM_MB.values()))
    vram_mb = probes.vram_mb()
    if vram_mb is None:
        return PreflightCheck(
            name="vram",
            passed=False,
            detail="the accelerator probe could not read the card's VRAM",
            fix="install a GPU torch build in the worker venv (see docs/how-to/choose-a-pytorch-build.md)",
        )
    if vram_mb < needed:
        fitting = _largest_fitting_tier(vram_mb)
        smallest = min(CORPUS_TIER_MIN_VRAM_MB.values())
        return PreflightCheck(
            name="vram",
            passed=False,
            detail=f"{vram_mb} MB on the largest card; the {tier} tier needs {needed} MB",
            fix=(
                f"re-run with --tier {fitting}"
                if fitting is not None
                else f"run the corpus on a card with at least {smallest} MB"
            ),
        )
    return PreflightCheck(name="vram", passed=True, detail=f"{vram_mb} MB on the largest card, {needed} MB needed")


def _check_hordelib(facts: HordelibFacts) -> PreflightCheck:
    """Judge the hordelib install: it must be an editable checkout so its manifest can be matched to it."""
    if not facts.importable:
        return PreflightCheck(
            name="hordelib",
            passed=False,
            detail="hordelib is not importable",
            fix="uv pip install -e <path-to-hordelib-checkout>",
        )
    where = facts.source_path or facts.module_path or "(path unknown)"
    version = facts.version or "unknown version"
    if facts.editable is not True:
        state = "not installed editable" if facts.editable is False else "install record unreadable"
        return PreflightCheck(
            name="hordelib",
            passed=False,
            detail=f"{version} at {where} ({state})",
            fix="uv pip install -e <path-to-hordelib-checkout>",
        )
    return PreflightCheck(name="hordelib", passed=True, detail=f"{version} editable from {where}")


def _check_manifest(facts: HordelibFacts) -> PreflightCheck:
    """Judge manifest importability; the census and heavy vocabularies are read straight out of it."""
    if not facts.manifest_importable:
        return PreflightCheck(
            name="kudos manifest",
            passed=False,
            detail=f"{KUDOS_MANIFEST_MODULE} is not importable, so this tier's vocabularies cannot be built",
            fix="uv pip install -e <hordelib-checkout-that-ships-hordelib/kudos_training/manifest.py>",
        )
    return PreflightCheck(name="kudos manifest", passed=True, detail=f"{KUDOS_MANIFEST_MODULE} importable")


def _check_comfy_pins(probes: Probes) -> PreflightCheck:
    """Judge the ComfyUI pins; a drifted one makes every inference child exit during startup."""
    required = probes.required_pins()
    installed = probes.installed_versions(COMFY_PIN_PACKAGES)
    wrong = {
        package: (installed.get(package), version)
        for package, version in required.items()
        if installed.get(package) != version
    }
    if wrong:
        detail = "; ".join(
            f"{package}: {found or 'absent'} (need {want})" for package, (found, want) in sorted(wrong.items())
        )
        pins = " ".join(f"{package}=={want}" for package, (_found, want) in sorted(wrong.items()))
        return PreflightCheck(name="comfy pins", passed=False, detail=detail, fix=f"uv pip install {pins}")
    return PreflightCheck(
        name="comfy pins",
        passed=True,
        detail="; ".join(f"{package}=={version}" for package, version in sorted(required.items())),
    )


def _check_cache_home(probes: Probes) -> tuple[PreflightCheck, Path | None]:
    """Judge the model cache directory, and hand back the path the disk check measures."""
    cache_home = probes.cache_home()
    if not cache_home:
        return (
            PreflightCheck(
                name="cache home",
                passed=False,
                detail="neither AIWORKER_CACHE_HOME nor bridgeData.yaml `cache_home` resolves",
                fix="set `cache_home` in bridgeData.yaml, or export AIWORKER_CACHE_HOME=<models dir>",
            ),
            None,
        )
    path = Path(cache_home)
    if not probes.writable(path):
        return (
            PreflightCheck(
                name="cache home",
                passed=False,
                detail=f"{path} is not writable",
                fix=f"grant write access to {path}, or point `cache_home` at a writable directory",
            ),
            path,
        )
    return PreflightCheck(name="cache home", passed=True, detail=str(path)), path


def _check_models(probes: Probes, tier: str, models: list[str]) -> PreflightCheck:
    """Judge model presence; a model fetched mid-run prices the download, not the inference."""
    if not models:
        return PreflightCheck(name="models", passed=True, detail="no models to check")
    missing = probes.missing_models(models)
    tiers = ",".join(bench_tier.value for bench_tier in corpus_bench_tiers(tier))
    if missing is None:
        return PreflightCheck(
            name="models",
            passed=False,
            detail=f"could not determine on-disk state for {len(models)} model(s)",
            fix=f"horde-benchmark download --tiers {tiers}",
        )
    if missing:
        return PreflightCheck(
            name="models",
            passed=False,
            detail=f"not on disk: {', '.join(sorted(missing))}",
            fix=f"horde-benchmark download --tiers {tiers}",
        )
    return PreflightCheck(name="models", passed=True, detail=f"{len(models)} model(s) on disk")


def _check_free_disk(probes: Probes, cache_path: Path | None) -> PreflightCheck:
    """Judge free space on the cache volume; the LoRA cells fetch and re-fetch as the run proceeds."""
    if cache_path is None:
        return PreflightCheck(
            name="free disk",
            passed=False,
            detail="no cache home to measure",
            fix="resolve the cache home first (see the cache home check)",
        )
    free = probes.free_bytes(cache_path)
    if free is None:
        return PreflightCheck(
            name="free disk",
            passed=False,
            detail=f"could not read free space on the volume holding {cache_path}",
            fix=f"check that {cache_path} exists and is readable",
        )
    needed_gb = CORPUS_FREE_DISK_BYTES / 1024**3
    if free < CORPUS_FREE_DISK_BYTES:
        return PreflightCheck(
            name="free disk",
            passed=False,
            detail=f"{free / 1024**3:.1f} GB free on {cache_path}, {needed_gb:.0f} GB wanted",
            fix=f"free up {(CORPUS_FREE_DISK_BYTES - free) / 1024**3:.1f} GB on the volume holding {cache_path}",
        )
    return PreflightCheck(name="free disk", passed=True, detail=f"{free / 1024**3:.1f} GB free on {cache_path}")


def _check_civitai_token(probes: Probes, tier: str) -> PreflightCheck:
    """Judge the CivitAI token; without it the LoRA cells measure a failed fetch instead of a load."""
    if not tier_has_lora_cells(tier):
        return PreflightCheck(
            name="civitai token",
            passed=True,
            detail=f"not needed: the {tier} tier has no LoRA cells",
        )
    if not probes.civitai_token():
        return PreflightCheck(
            name="civitai token",
            passed=False,
            detail="no CivitAI token in the environment, so the LoRA cells cannot fetch their weights",
            fix="set `civitai_api_token` in bridgeData.yaml, or export CIVIT_API_TOKEN=<token>",
        )
    return PreflightCheck(name="civitai token", passed=True, detail="present")


def _check_live_worker(probes: Probes) -> PreflightCheck:
    """Judge whether the card is already taken; two workers on one card make both measurements junk."""
    reason = probes.live_worker()
    if reason:
        return PreflightCheck(
            name="no live worker",
            passed=False,
            detail=reason,
            fix="stop the running worker and remove any stale .abort file, then re-run",
        )
    return PreflightCheck(name="no live worker", passed=True, detail="nothing is holding this working directory")


def run_preflight(
    tier: str,
    machine_id: str | None,
    *,
    models: list[str],
    require_manifest: bool,
    probes: Probes | None = None,
) -> PreflightReport:
    """Check every condition a corpus run on this machine depends on, and report each with its remedy.

    Args:
        tier: The corpus tier about to run; it decides the download command a missing model is fixed by.
        machine_id: The operator-chosen machine id, or None when the tier does not need one (the run
            then produces rows that cannot be pooled with another machine's, which the report says).
        models: The image checkpoints the built scenario references.
        require_manifest: Whether this tier's cells come from the kudos feature manifest.
        probes: Machine readings to judge; the real ones by default.

    Returns:
        The report, whose ``passed`` is the run/do-not-run verdict.
    """
    probes = probes if probes is not None else Probes()
    hordelib_facts = probes.hordelib()

    checks = [
        _check_machine_id(machine_id),
        _check_cuda(probes),
        _check_vram(probes, tier),
        _check_hordelib(hordelib_facts),
    ]
    if require_manifest:
        checks.append(_check_manifest(hordelib_facts))
    checks.append(_check_comfy_pins(probes))

    cache_check, cache_path = _check_cache_home(probes)
    checks.append(cache_check)
    checks.append(_check_models(probes, tier, models))
    checks.append(_check_free_disk(probes, cache_path))
    checks.append(_check_civitai_token(probes, tier))
    checks.append(_check_live_worker(probes))

    return PreflightReport(tier=tier, machine_id=machine_id, checks=checks)


def format_report(report: PreflightReport) -> str:
    """Render a report as a fixed-width table: status, check, detail, and the fix for each failure."""
    rows = [
        ("OK" if check.passed else "FAIL", check.name, check.detail, check.fix if not check.passed else "")
        for check in report.checks
    ]
    headers = ("STATUS", "CHECK", "DETAIL", "FIX")
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

    def line(cells: tuple[str, str, str, str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    machine = report.machine_id or "(none)"
    out = [f"Pricing-corpus preflight: tier {report.tier}, machine {machine}", "", line(headers)]
    out.append("  ".join("-" * width for width in widths))
    out.extend(line(row) for row in rows)
    out.append("")
    if report.passed:
        out.append(f"All {len(report.checks)} checks passed.")
    else:
        out.append(f"{len(report.failures)} of {len(report.checks)} checks failed; apply the fixes above.")
    return "\n".join(out)


def collect_machine_facts(machine_id: str) -> CorpusMachineFacts:
    """Describe the measuring machine, so its rows stay attributable after they leave this box.

    Everything here is read without importing torch in this process: the device comes from the
    out-of-process accelerator probe (falling back to ``nvidia-smi``) and the versions come from
    installed package metadata.
    """
    gpu_model: str | None = None
    vram_mb: int | None = None
    driver_version: str | None = None

    try:
        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        devices = probe_accelerators()
        if devices:
            gpu_model = devices[0].name
            vram_mb = devices[0].total_vram_mb
    except Exception as error:  # noqa: BLE001 - machine facts are descriptive; a probe miss is not fatal
        logger.debug(f"Accelerator probe unavailable for machine facts: {type(error).__name__} {error}")

    try:
        from horde_worker_regen.analysis.system_info import nvidia_smi_summary

        summary = nvidia_smi_summary() or {}
        cards = summary.get("gpus") or []
        if isinstance(cards, list) and cards and isinstance(cards[0], dict):
            driver_version = cards[0].get("driver_version")
            gpu_model = gpu_model or cards[0].get("name")
    except Exception as error:  # noqa: BLE001 - nvidia-smi is absent on non-NVIDIA hosts
        logger.debug(f"nvidia-smi summary unavailable for machine facts: {type(error).__name__} {error}")

    try:
        vram_mb = vram_mb if vram_mb is not None else _nvidia_smi_vram_mb()
    except Exception as error:  # noqa: BLE001 - nvidia-smi is absent on non-NVIDIA hosts
        logger.debug(f"nvidia-smi summary unavailable for machine facts: {type(error).__name__} {error}")

    worker_version: str | None = None
    try:
        from horde_worker_regen.runtime_version import runtime_version

        worker_version = runtime_version()
    except Exception as error:  # noqa: BLE001 - a version read must not fail the run
        logger.debug(f"Could not read the worker version: {type(error).__name__} {error}")

    versions = installed_versions((HORDELIB_DISTRIBUTION, "torch"))
    return CorpusMachineFacts(
        machine_id=machine_id,
        hostname=platform.node() or None,
        gpu_model=gpu_model,
        vram_mb=vram_mb,
        driver_version=driver_version,
        os=platform.platform(),
        worker_version=worker_version,
        hordelib_version=versions.get(HORDELIB_DISTRIBUTION),
        torch_version=versions.get("torch"),
    )


def _nvidia_smi_vram_mb() -> int | None:
    """The first card's total VRAM in MB from ``nvidia-smi``, for hosts the torch probe cannot answer for."""
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        candidate = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(candidate) if candidate.exists() else None
    if not exe:
        return None
    try:
        output = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = next((line.strip() for line in (output or "").splitlines() if line.strip()), "")
    return int(first) if first.isdigit() else None


def stamp_definition(
    definition: PricingCorpusDefinition,
    *,
    machine_id: str | None,
    created_at: float,
    facts: CorpusMachineFacts | None = None,
) -> PricingCorpusDefinition:
    """Return the definition with the run's provenance filled in, ready to be written.

    ``created_at`` anchors the artifact to the stats stream that starts just after it, and the machine
    facts are what an assembler keys the run's rows on. A run given no machine id leaves ``machine``
    unset rather than inventing one, which keeps the artifact honest about being unattributable.

    Args:
        definition: The freshly built definition.
        machine_id: The operator-chosen machine id, or None when the run was not told which machine it is.
        created_at: Epoch seconds to record.
        facts: Pre-collected machine facts; collected here when omitted and a machine id was given.
    """
    if machine_id is None:
        return definition.model_copy(update={"created_at": created_at})
    machine = facts if facts is not None else collect_machine_facts(machine_id)
    return definition.model_copy(update={"created_at": created_at, "machine": machine})


__all__ = [
    "COMFY_PIN_PACKAGES",
    "CORPUS_FREE_DISK_BYTES",
    "CORPUS_TIER_MIN_VRAM_MB",
    "KUDOS_MANIFEST_MODULE",
    "MACHINE_ID_PATTERN",
    "HordelibFacts",
    "PreflightCheck",
    "PreflightReport",
    "Probes",
    "collect_machine_facts",
    "corpus_bench_tiers",
    "format_report",
    "installed_versions",
    "probe_hordelib",
    "required_comfy_pins",
    "requires_kudos_manifest",
    "run_preflight",
    "stamp_definition",
]
