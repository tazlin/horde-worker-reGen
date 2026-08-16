"""The `horde-benchmark` CLI: progressive worker benchmarking.

Subcommands:
- ``run``: prove the capability catalog on one warm worker via the canned-job harness (no API).
- ``plan``: preview each probe's resource needs and run/skip verdict (no worker is started).
- ``download``: fetch the checkpoints the selected tiers need, ahead of a timed run.
- ``pricing-corpus``: run the cost-attribution corpus whose stats records fit a pricing model.
- ``soak``: run one sustained-traffic mix for a fixed period, as a before/after performance vehicle.
- ``report``: re-render the markdown report from an existing output directory.
- ``monitor``: tail a run's progress.jsonl live (attach or replay).
- ``live``: open-loop load generation against a live AI-Horde API (separate phase).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.analysis.session_duty import (
    analyze_stats_files,
    discover_stats_sessions,
    render_session_duty_report,
)
from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.soak import SOAK_MIXES

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.analysis.session_duty import SessionDutyReport
    from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions
    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.benchmark.capabilities.result import CapabilityReport, MachineInfo
    from horde_worker_regen.benchmark.download_progress import DownloadEvent, DownloadModelRow
    from horde_worker_regen.model_download_core import DownloadControls
    from horde_worker_regen.model_download_plan import DownloadPlan


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``run`` subcommand: the capability-engine benchmark (one warm worker, no per-probe rampup)."""
    run = subparsers.add_parser(
        "run",
        help="Run the capability-probe benchmark: prove what this machine can do, on one warm worker.",
    )
    run.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers to attempt (sd15, sdxl, flux, qwen, zimage). flux/qwen/zimage "
        "are opt-in (very large) and auto-skip when the machine cannot host them.",
    )
    run.add_argument(
        "--process-mode",
        default="real",
        choices=("fake", "dry_run", "real"),
        help="real = GPU benchmark; fake/dry_run exercise the engine without inference.",
    )
    run.add_argument("--out", type=Path, default=None, help="Output directory (default: benchmark_results/<ts>).")
    run.add_argument("--jobs-per-level", type=int, default=4)
    run.add_argument("--probe-timeout", type=float, default=900.0, help="Per-probe timeout in seconds.")
    run.add_argument("--only", default=None, help="Run a single probe by its capability slug (e.g. sd15-controlnet).")
    run.add_argument("--include-downloads", action="store_true", help="Include the ad-hoc lora download probe.")
    run.add_argument("--no-alchemy", action="store_true", help="Skip the alchemy probes.")
    run.add_argument("--no-features", action="store_true", help="Skip the feature probes.")
    run.add_argument("--no-concurrency", action="store_true", help="Skip the concurrency probes.")
    run.add_argument(
        "--exclude-capability",
        action="append",
        default=[],
        choices=[kind.value for kind in CapabilityKind],
        metavar="CAPABILITY",
        help="Drop a single capability kind, independent of the coarse stage flags (repeatable).",
    )
    run.add_argument("--no-validate", action="store_true", help="Skip the post-run sustained-load soak.")
    run.add_argument("--soak-minutes", type=float, default=5.0, help="Duration of each per-tier soak (minutes).")
    run.add_argument(
        "--strict-duty",
        action="store_true",
        help="Fail a soak whose GPU duty cycle misses the 90%% target (off by default: advisory).",
    )
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-process state in the live view.",
    )


def _add_soak_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``soak`` subcommand: one sustained-traffic mix, run standalone as an A/B arm."""
    soak = subparsers.add_parser(
        "soak",
        help="Run one sustained-traffic soak mix for a fixed period and score it from its stats export.",
    )
    soak.add_argument(
        "--mix",
        default=SOAK_MIXES[0],
        choices=SOAK_MIXES,
        help="Which traffic mix to sustain (default: production_replay, the measured production cadence).",
    )
    soak.add_argument("--minutes", type=float, default=20.0, help="How long to sustain the mix (default: 20).")
    soak.add_argument(
        "--process-mode",
        default="real",
        choices=("fake", "dry_run", "real"),
        help="`real` measures the GPU; `fake`/`dry_run` exercise the plumbing without inference.",
    )
    soak.add_argument("--out", type=Path, default=None, help="Output directory (default: benchmark_results/<ts>).")
    soak.add_argument(
        "--label",
        default="",
        help="Free text naming this arm (e.g. 'baseline', 'unload-on'); recorded in the run manifest.",
    )
    soak.add_argument(
        "--bridge-data",
        type=Path,
        default=Path("bridgeData.yaml"),
        help="Worker config the throughput-relevant fields are carried over from (default: bridgeData.yaml).",
    )
    soak.add_argument(
        "--ignore-bridge-data",
        action="store_true",
        help="Do not carry any operator config over; run on harness defaults plus --override.",
    )
    soak.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Bridge-data override applied on top of the operator config (repeatable). This is the A/B knob.",
    )
    soak.add_argument(
        "--no-loras",
        action="store_true",
        help="Strip every LoRA and textual-inversion reference from the mix (see the real-mode caveat).",
    )
    soak.add_argument(
        "--shared-lora",
        action="append",
        default=[],
        metavar="REFERENCE",
        help="Resolvable LoRA reference for the mix's cache-hit pool (repeatable; at least 3 for real mode).",
    )
    soak.add_argument(
        "--unique-lora",
        action="append",
        default=[],
        metavar="REFERENCE",
        help="Resolvable LoRA reference for the mix's download-pressure pool (repeatable; at least 8).",
    )


def _add_stage_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the stage-inclusion flags shared by ``plan`` and ``download`` (so the two stay in lockstep)."""
    parser.add_argument("--include-downloads", action="store_true", help="Include the ad-hoc download probe.")
    parser.add_argument("--no-alchemy", action="store_true", help="Skip the alchemy probes.")
    parser.add_argument("--no-features", action="store_true", help="Skip the feature probes.")
    parser.add_argument("--no-concurrency", action="store_true", help="Skip the concurrency probes.")
    parser.add_argument(
        "--exclude-capability",
        action="append",
        default=[],
        choices=[kind.value for kind in CapabilityKind],
        metavar="CAPABILITY",
        help="Drop a single capability kind, independent of the coarse stage flags (repeatable).",
    )


def _add_plan_parser(subparsers: argparse._SubParsersAction) -> None:
    plan = subparsers.add_parser(
        "plan",
        help="Show each level's resource requirements and predicted run/skip verdict (no worker is started).",
    )
    plan.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers to plan (sd15, sdxl, flux, qwen).",
    )
    plan.add_argument(
        "--process-mode",
        default="real",
        choices=("fake", "dry_run", "real"),
        help="Resource gates (disk/VRAM/model presence) apply only in real mode; fake/dry_run always run.",
    )
    _add_stage_selection_args(plan)
    plan.add_argument(
        "--force",
        action="store_true",
        help="Reflect a forced ramp: show levels that do not fit the machine (or lack a CivitAI token) as RUN.",
    )
    plan.add_argument("--json", action="store_true", help="Emit the plan rows as JSON instead of a table.")


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    download = subparsers.add_parser(
        "download",
        help="Download the model checkpoints the selected tiers need, so the timed ramp is not slowed by "
        "downloading mid-run. Shows which models are needed, their size, and where they will be stored.",
    )
    download.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers whose checkpoints to download (sd15, sdxl, flux, qwen).",
    )
    # The download path always sets up the real worker env (so AIWORKER_CACHE_HOME resolves) and only ever
    # runs in real mode; only ever runs in real mode.
    download.set_defaults(process_mode="real")
    _add_stage_selection_args(download)
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the download plan (which models, their size, where they go) without downloading anything.",
    )
    download.add_argument(
        "--json-progress",
        action="store_true",
        help="Emit structured, line-delimited progress events for a parent process (used by the TUI).",
    )
    download.add_argument(
        "--control-stdin",
        action="store_true",
        help="Read pause/resume/rate control commands (one JSON object per line) from stdin (used by the TUI).",
    )
    download.add_argument("--directml", type=int, default=None, help="DirectML device index (for Windows AMD GPUs).")


def _add_pricing_corpus_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``pricing-corpus`` subcommand: the cost-attribution corpus used to fit a pricing model."""
    corpus = subparsers.add_parser(
        "pricing-corpus",
        help="Run the pricing corpus: a deterministic, axis-sweeping workload whose stats records are "
        "training data for a cost model. Emits a definition artifact that labels every job's cell.",
    )
    corpus.add_argument(
        "--tier",
        default="smoke",
        choices=("smoke", "standard", "census"),
        help="standard = the marginal-cost fit set (hours); census = every value of every categorical "
        "axis the kudos manifest encodes, plus a conflated sample (about four hours); smoke = a short "
        "subset that proves the corpus runs.",
    )
    corpus.add_argument(
        "--emit-definition",
        type=Path,
        default=None,
        help="Write the definition artifact here instead of next to the session stats.",
    )
    corpus.add_argument(
        "--dry-list",
        action="store_true",
        help="Print the ordered cell ids and exit; nothing is run and no worker is started.",
    )
    corpus.add_argument(
        "--lora-version-id",
        action="append",
        default=[],
        metavar="VERSION_ID",
        help="A real CivitAI LoRA version id for the LoRA cells (repeatable; the standard tier needs five).",
    )
    corpus.add_argument(
        "--ti-name",
        default=None,
        help="A real textual-inversion reference for the TI cell.",
    )
    corpus.add_argument(
        "--no-lora-eviction",
        action="store_true",
        help="Skip evicting the pinned LoRAs before the run; any still cached turns its miss cell into a "
        "hit measurement.",
    )
    corpus.add_argument("--out", type=Path, default=None, help="Output directory (default: benchmark_results/<ts>).")
    corpus.add_argument(
        "--timeout",
        type=float,
        default=6.0 * 60.0 * 60.0,
        help="Overall run timeout in seconds (the standard tier is a multi-hour workload).",
    )
    # The corpus measures real inference costs; a fabricated duration is not a price.
    corpus.set_defaults(process_mode="real")


def _parse_tiers(raw_tiers: str) -> list[BenchTier] | None:
    """Parse the comma-separated ``--tiers`` value into tiers, or None on an unknown tier (logged)."""
    tiers: list[BenchTier] = []
    for token in (token.strip() for token in raw_tiers.split(",")):
        if not token:
            continue
        try:
            tiers.append(BenchTier(token))
        except ValueError:
            logger.error(f"Unknown tier {token!r}; valid tiers: {', '.join(tier.value for tier in BenchTier)}")
            return None
    return tiers


def _prepare_catalog(
    args: argparse.Namespace,
    tiers: list[BenchTier],
    *,
    probe_devices: bool = True,
) -> tuple[list[CapabilityProbe], MachineInfo, CatalogOptions]:
    """Apply the worker env, detect the machine, and build the capability catalog for the selection.

    Shared by ``plan`` and ``download`` so the preview an operator sees is built from the exact same
    catalog (and the same detected VRAM) ``run`` would execute. ``probe_devices`` controls the GPU
    enumeration: callers that do not use the device info (the ``download`` path discards it) pass False
    to skip the out-of-process torch/CUDA probe, which is otherwise a cold, multi-minute startup cost.
    """
    from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions, build_capability_catalog
    from horde_worker_regen.benchmark.capabilities.executor import detect_machine_info
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env

    # The harness never reads bridgeData.yaml, so set AIWORKER_CACHE_HOME (and friends) here, before any
    # worker boots, so the real inference children resolve the worker's actual model directory instead of
    # hordelib's empty ./models fallback. Passing the tiers also opts into the beta reference when a beta
    # tier (qwen/zimage) is requested.
    ensure_worker_env(args.process_mode, tiers)
    machine = detect_machine_info(probe_devices=probe_devices and args.process_mode == "real")

    options = CatalogOptions(
        tiers=tiers,
        jobs_per_level=getattr(args, "jobs_per_level", 4),
        include_concurrency=not args.no_concurrency,
        include_features=not args.no_features,
        include_alchemy=not args.no_alchemy,
        include_downloads=args.include_downloads,
        excluded_kinds={CapabilityKind(value) for value in getattr(args, "exclude_capability", [])},
        probe_timeout_seconds=getattr(args, "probe_timeout", 900.0),
        total_vram_mb=machine.total_vram_mb,
    )
    return build_capability_catalog(options), machine, options


def _run_plan(args: argparse.Namespace) -> int:
    """Print each probe's resource requirements and predicted verdict against the detected machine."""
    from horde_worker_regen.benchmark.capabilities.plan_preview import build_capability_plan_rows
    from horde_worker_regen.benchmark.progress_console import format_plan_table

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    logger.info("Building benchmark plan (detecting hardware; no worker is started) ...")
    probes, machine, _options = _prepare_catalog(args, tiers)
    rows = build_capability_plan_rows(probes, machine=machine, process_mode=args.process_mode, force=args.force)

    if args.json:
        from horde_worker_regen.benchmark.progress_channel import encode_plan_rows

        # Sentinel-wrapped so a reader can isolate the payload from log lines/banners on this same stdout.
        print(encode_plan_rows(rows))  # noqa: T201
    else:
        print(format_plan_table(rows))  # noqa: T201
    return 0


def _format_download_plan(
    tiers: list[BenchTier],
    model_names: list[str],
    plan: DownloadPlan | None,
    annotator_row: DownloadModelRow | None = None,
) -> str:
    """Render the download plan as plain text: which models, their size, present-or-not, and where they go."""
    tier_label = ", ".join(tier.value for tier in tiers)
    lines = [f"Models needed for tiers {tier_label}: {len(model_names)}"]
    if plan is None:
        lines.append("  (could not size the on-disk picture; every model will be checked when downloading)")
        lines.extend(f"  [unknown ] {name}" for name in model_names)
        return "\n".join(lines)

    for info in plan.models:
        tag = "on disk " if info.on_disk else "download"
        size = f"{info.size_bytes / 1024**3:.1f} GB" if info.size_bytes else "size unknown"
        path = info.target_path or "(path undetermined)"
        lines.append(f"  [{tag}] {info.name} ({size})  ->  {path}")

    free = "unknown" if plan.free_disk_bytes is None else f"{plan.free_disk_bytes / 1024**3:.1f} GB free"
    lines.append(
        f"Already present: {plan.present_bytes / 1024**3:.1f} GB"
        f"  ·  To download: {plan.to_download_bytes / 1024**3:.1f} GB"
        f"  ·  Volume: {free}",
    )
    if not plan.fits:
        lines.append(f"  WARNING: not enough free space: about {plan.shortfall_bytes / 1024**3:.1f} GB short.")
    if annotator_row is not None:
        size = f"~{annotator_row.size_bytes / 1024**3:.1f} GB" if annotator_row.size_bytes else "size unknown"
        lines.append(f"Controlnet annotators (lazy, fetched on first use): {size}")
    return "\n".join(lines)


def _catalog_control_types(probes: list[CapabilityProbe]) -> list[str]:
    """Return the distinct controlnet ``control_type``s any probe in the catalog exercises (may be empty)."""
    return sorted({job.control_type for probe in probes for job in probe.scenario.image_jobs if job.control_type})


def _catalog_post_processors(probes: list[CapabilityProbe]) -> list[str]:
    """Return the distinct post-processor names any probe in the catalog exercises (may be empty)."""
    return sorted(
        {
            post_processor
            for probe in probes
            for job in probe.scenario.image_jobs
            for post_processor in (job.post_processing or ())
        },
    )


def _controlnet_annotator_row(
    control_types: list[str],
    *,
    on_disk: bool = False,
    size_bytes: int | None = None,
) -> DownloadModelRow | None:
    """Build the synthetic annotator plan row for *control_types*, or None when none apply.

    Pure and import-light on purpose: ``on_disk`` (resolved torch-free from the annotator catalog) and the
    optional ``size_bytes`` (a hordelib ROM figure the caller supplies only on the real-download path) are
    passed in, so the dry-run preview can build this row without the cold hordelib import that once timed the
    preview out.
    """
    if not control_types:
        return None
    from horde_worker_regen.benchmark.download_progress import DownloadModelRow

    return DownloadModelRow(
        name="ControlNet annotators",
        size_bytes=size_bytes or None,
        on_disk=on_disk,
        target_path="(annotator cache)",
        is_aux=True,  # a synthetic feature row; fetched via the annotator preload, never as an image model
    )


def _download_controlnet_annotators(*, directml: int | None) -> bool:
    """Download (and verify) the controlnet annotators via the worker's standard preload path.

    Mirrors ``download_process._download_controlnet_models``: idempotent and fast once the on-disk preload
    marker exists, so it is safe to call even when annotators are already present. Returns success.
    """
    import hordelib
    from hordelib.api import SharedModelManager

    extra_comfyui_args = [f"--directml={directml}"] if directml is not None else []
    hordelib.initialise(extra_comfyui_args=extra_comfyui_args)
    return bool(SharedModelManager.preload_annotators())


def _start_stdin_control_thread() -> DownloadControls:
    """Apply pause/resume/rate control lines from stdin to a DownloadControls, live during a download.

    The TUI's download modal writes one JSON control object per line to this subprocess's stdin; a daemon
    reader thread folds them into the controls the shared download core reads each chunk.
    """
    import threading

    from horde_worker_regen.benchmark.download_progress import decode_download_control
    from horde_worker_regen.model_download_core import DownloadControls

    controls = DownloadControls()

    def _reader() -> None:
        for line in sys.stdin:
            control = decode_download_control(line)
            if control is None:
                continue
            if control.cmd == "pause":
                controls.set_paused(True)
            elif control.cmd == "resume":
                controls.set_paused(False)
            elif control.cmd == "rate":
                controls.set_rate_limit(control.kbps)

    threading.Thread(target=_reader, name="benchmark-download-control", daemon=True).start()
    return controls


def _download_compvis_models(
    model_names: list[str],
    *,
    emit: Callable[[DownloadEvent], None],
    json_progress: bool,
    controls: DownloadControls | None = None,
) -> int:
    """Download each named image checkpoint; a thin alias for the compvis category of the shared core."""
    return _download_category_models(
        "compvis",
        model_names,
        emit=emit,
        json_progress=json_progress,
        controls=controls,
    )


def _download_category_models(
    category: str,
    model_names: list[str],
    *,
    emit: Callable[[DownloadEvent], None],
    json_progress: bool,
    controls: DownloadControls | None = None,
) -> int:
    """Download each named *category* model via the shared download core; return how many failed.

    The category name is also the ``SharedModelManager`` attribute that owns it (``compvis``, ``controlnet``,
    ``esrgan``, ...), so one path fetches image checkpoints and aux feature checkpoints alike with the same
    dedup + validate/retry the worker's download process uses
    (:func:`horde_worker_regen.model_download_core.ensure_models_present`). Checkpoints need only the model
    managers, not a full ``hordelib.initialise()`` (no torch/ComfyUI), which keeps this phase light and
    GPU-free.
    """
    from hordelib.api import SharedModelManager

    from horde_worker_regen.benchmark.download_progress import DownloadEvent
    from horde_worker_regen.model_download_core import ModelProgress, ensure_models_present

    SharedModelManager.load_model_managers()
    manager = getattr(SharedModelManager.manager, category, None)
    if manager is None:
        logger.error(f"Failed to load the {category!r} model manager; cannot download {model_names}.")
        return len(model_names)

    def on_start(name: str, index: int, total: int) -> None:
        emit(DownloadEvent(kind="model_started", name=name, index=index, total=total))
        if not json_progress:
            logger.info(f"[{index}/{total}] Downloading {name} ...")

    def on_progress(name: str, index: int, total: int, progress: ModelProgress) -> None:
        emit(
            DownloadEvent(
                kind="model_progress",
                name=name,
                index=index,
                total=total,
                downloaded_bytes=progress.downloaded_bytes,
                total_bytes=progress.total_bytes,
                speed_bps=progress.speed_bps,
                eta_seconds=progress.eta_seconds,
            ),
        )

    def on_finish(name: str, index: int, total: int, ok: bool) -> None:
        if not ok:
            logger.error(f"[{index}/{total}] Failed to download {name}.")
        elif not json_progress:
            logger.success(f"[{index}/{total}] {name}: done.")
        emit(DownloadEvent(kind="model_finished", name=name, index=index, total=total, ok=ok))

    outcome = ensure_models_present(
        manager,
        list(model_names),
        controls=controls,
        on_model_start=on_start,
        on_progress=on_progress,
        on_model_finish=on_finish,
    )
    return outcome.failed


def _run_download(args: argparse.Namespace) -> int:
    """Download the checkpoints the selected tiers need, after showing exactly what will be fetched and where."""
    from horde_worker_regen.benchmark.download_progress import (
        DownloadEvent,
        DownloadModelRow,
        encode_download_event,
    )
    from horde_worker_regen.benchmark.requirements import (
        annotators_present_offline,
        controlnet_checkpoint_files,
        models_disk_plan,
        post_processor_model_files,
    )

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    # The download path only needs the model set (it discards the machine info), so skip the GPU probe:
    # its cold out-of-process torch import is pointless here and, on a cold .exe install, was the dominant
    # cost behind the "Could not work out the download plan: timed out" preview failure. The probed VRAM
    # only sizes the post-processing sweep's *resolution*, which does not change which models are fetched.
    logger.info("Resolving the models the selected tiers need (no worker is started) ...")
    probes, _machine, _options = _prepare_catalog(args, tiers, probe_devices=False)
    model_names = sorted({name for probe in probes for name in probe.scenario.models_referenced()})
    if not model_names:
        logger.warning("The selected tiers reference no image checkpoints; nothing to download.")
        return 0

    plan = models_disk_plan(model_names)
    json_progress: bool = args.json_progress
    controls = _start_stdin_control_thread() if getattr(args, "control_stdin", False) else None

    # A feature exercises more than image checkpoints: controlnet needs its control-type checkpoints AND the
    # annotator ROMs, and post-processing needs its own models. Surface every such file as an explicit plan
    # row with real on-disk state, resolved torch-free from the model reference (no cold hordelib import), so
    # the preview never tells an operator with partially-present controlnets that nothing is missing.
    control_types = _catalog_control_types(probes)
    post_processors = _catalog_post_processors(probes)
    feature_files = controlnet_checkpoint_files(control_types) + post_processor_model_files(post_processors)

    # Annotator presence is now answered torch-free via the reference's annotator catalog (the dry-run no
    # longer has to hardcode them missing to dodge a cold hordelib import). The controlnet *extra*
    # (onnxruntime) is only checkable through hordelib, so that one probe stays on the real-download path.
    annotators_present = annotators_present_offline(control_types) if control_types else None
    annotator_size: int | None = None
    if args.dry_run:
        cn_installed = None
    else:
        from horde_worker_regen.benchmark.requirements import _controlnet_annotator_bytes, controlnet_installed

        cn_installed = controlnet_installed() if control_types else None
        # Size the ROMs only here (the hordelib import that would slow the dry-run preview is off that path).
        if control_types and cn_installed is not False:
            annotator_size = _controlnet_annotator_bytes(control_types) or None
    annotator_row = (
        _controlnet_annotator_row(control_types, on_disk=annotators_present is True, size_bytes=annotator_size)
        if control_types and cn_installed is not False
        else None
    )

    def emit(event: DownloadEvent) -> None:
        if json_progress:
            # Sentinel-wrapped so the TUI can isolate each event from interleaved log lines on this stdout.
            print(encode_download_event(event))  # noqa: T201

    # Feature checkpoints are real model files (unlike the lazy annotator ROMs): a confidently-absent one is
    # fetched alongside the image checkpoints. ``on_disk is True`` keeps an undeterminable file (no reference)
    # out of the present set without claiming it missing.
    feature_rows = [
        DownloadModelRow(
            name=feature.name,
            size_bytes=feature.size_bytes,
            on_disk=feature.on_disk is True,
            target_path=feature.target_path,
            is_aux=True,  # controlnet/post-proc checkpoints: fetched via the aux pass, never as image models
        )
        for feature in feature_files
    ]
    feature_missing = [feature for feature in feature_files if feature.on_disk is False]
    feature_missing_bytes = sum(feature.size_bytes or 0 for feature in feature_missing)

    if plan is not None:
        rows = [
            DownloadModelRow(
                name=info.name,
                size_bytes=info.size_bytes,
                on_disk=info.on_disk,
                target_path=info.target_path,
            )
            for info in plan.models
        ]
        # Only count annotator bytes as "to download" when they are not already on disk.
        annotator_bytes = (
            annotator_row.size_bytes or 0 if annotator_row is not None and not annotator_row.on_disk else 0
        )
        rows.extend(feature_rows)
        if annotator_row is not None:
            rows.append(annotator_row)
        emit(
            DownloadEvent(
                kind="planned",
                models=rows,
                present_bytes=plan.present_bytes,
                # Fold the feature checkpoints and the annotator ROM into the displayed "to download" so the
                # count and the byte figure agree; fits/shortfall stay checkpoint-based (the sized constraint).
                to_download_bytes=plan.to_download_bytes + feature_missing_bytes + annotator_bytes,
                free_disk_bytes=plan.free_disk_bytes,
                fits=plan.fits,
                shortfall_bytes=plan.shortfall_bytes,
            ),
        )
        missing = [info.name for info in plan.models if not info.on_disk]
    else:
        unsized_rows = [DownloadModelRow(name=name) for name in model_names]
        unsized_rows.extend(feature_rows)
        if annotator_row is not None:
            unsized_rows.append(annotator_row)
        emit(DownloadEvent(kind="planned", models=unsized_rows))
        missing = model_names

    if not json_progress:
        print(_format_download_plan(tiers, model_names, plan, annotator_row))  # noqa: T201

    # A controlnet level needs annotators, but the extra is not installed: nothing to fetch, so tell the
    # operator how to enable it rather than silently omitting the annotators from the plan.
    if control_types and cn_installed is False:
        from horde_worker_regen.capabilities import controlnet_install_hint

        with contextlib.suppress(Exception):
            logger.warning(
                f"A selected level uses controlnet, but the controlnet extra is not installed, so its "
                f"annotators were skipped: {controlnet_install_hint()}",
            )

    # Fetch annotators only when the extra is present and they are not already on disk. ``preload_annotators``
    # is idempotent, but skipping the (slow) hordelib.initialise + verify when they are confirmed present
    # is what lets "all required models already on disk" be reported truthfully.
    fetch_annotators = bool(control_types) and cn_installed is not False and annotators_present is not True

    if not missing and not feature_missing and not fetch_annotators:
        logger.success("All required models are already on disk; nothing to download.")
        emit(DownloadEvent(kind="complete", downloaded=0, failed=0))
        return 0

    if args.dry_run:
        if not json_progress:
            todo = [f"{len(missing)} model(s)"] if missing else []
            if feature_missing:
                todo.append(f"{len(feature_missing)} feature model(s)")
            if fetch_annotators:
                todo.append("controlnet annotators")
            print(f"\nDry run: would fetch {', '.join(todo)}. Re-run without --dry-run to fetch.")  # noqa: T201
        emit(DownloadEvent(kind="complete", downloaded=0, failed=0))
        return 0

    if missing and plan is not None and not plan.fits:
        logger.error(
            f"Not enough free disk to download: about {plan.shortfall_bytes / 1024**3:.1f} GB short. "
            "Free up space or select fewer tiers.",
        )
        emit(DownloadEvent(kind="complete", downloaded=0, failed=len(missing), detail="insufficient disk"))
        return 1

    failed = 0
    if missing:
        failed = _download_compvis_models(
            missing,
            emit=emit,
            json_progress=json_progress,
            controls=controls,
        )

    # Fetch the absent feature checkpoints through the same shared core, one model manager per category, so a
    # controlnet/post-processing level finds its weights present rather than cold-loading them mid-run.
    feature_failed = 0
    feature_missing_by_category: dict[str, list[str]] = {}
    for feature in feature_missing:
        feature_missing_by_category.setdefault(feature.category, []).append(feature.name)
    for category, names in feature_missing_by_category.items():
        feature_failed += _download_category_models(
            category,
            names,
            emit=emit,
            json_progress=json_progress,
            controls=controls,
        )

    annotators_failed = 0
    if fetch_annotators:
        emit(DownloadEvent(kind="model_started", name="ControlNet annotators", index=1, total=1))
        if not json_progress:
            logger.info("Downloading controlnet annotators ...")
        annotators_ok = False
        try:
            annotators_ok = _download_controlnet_annotators(directml=args.directml)
        except Exception as e:  # noqa: BLE001 - annotators are best-effort; report, do not crash the download
            logger.error(f"Failed to download controlnet annotators: {type(e).__name__} {e}")
        if not annotators_ok:
            annotators_failed = 1
            logger.error("Controlnet annotators failed to download.")
        elif not json_progress:
            logger.success("Controlnet annotators: done.")
        emit(DownloadEvent(kind="model_finished", name="ControlNet annotators", index=1, total=1, ok=annotators_ok))

    total_failed = failed + feature_failed + annotators_failed
    total_items = len(missing) + len(feature_missing) + (1 if fetch_annotators else 0)
    emit(DownloadEvent(kind="complete", downloaded=total_items - total_failed, failed=total_failed))
    if total_failed:
        logger.error(f"{total_failed} of {total_items} item(s) failed to download.")
        return 1
    logger.success(f"Downloaded {total_items} item(s); the benchmark can now run them without a mid-run fetch.")
    return 0


def _setup_benchmark_file_logging(out_dir: Path) -> None:
    """Give the benchmark controller process its own on-disk log in the run directory.

    The controller's loguru otherwise goes only to its stderr: under the TUI that is captured to the
    run's ``console.log``, but a CLI run leaves it on the terminal only, so nothing of the controller's
    own diagnostics (level lifecycle, abort reasons, "level died without a result", and (in warm mode,
    where the harness runs in-process here) the entire warm session) survives on disk. This writes
    those to ``controller.log`` for both paths. Writes are synchronous so a controller crash keeps its
    final lines. It also points the operator at the per-process child logs, which hordelib writes to
    ``logs/`` relative to the working directory (``bridge_*``/``stdout_*``/``stderr_*``/``trace_*``),
    not into the run directory, so a reader of the run dir knows where to look next.
    """
    with contextlib.suppress(Exception):
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            out_dir / "controller.log",
            level="DEBUG",
            rotation="25 MB",
            retention=10,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        logger.info(f"Benchmark controller log: {(out_dir / 'controller.log').resolve()}")
        logger.info(
            "Per-process worker (subprocess/grand-subprocess) logs are written by hordelib to "
            f"{Path('logs').resolve()} (bridge_*.log, stdout_*/stderr_*/trace_*, *.faulthandler).",
        )


def _run_run(args: argparse.Namespace) -> int:
    """Run the capability-probe benchmark on one warm worker and persist its report."""
    from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions
    from horde_worker_regen.benchmark.capabilities.executor import ProbeExecutor, detect_machine_info
    from horde_worker_regen.benchmark.progress_channel import (
        PROGRESS_FILENAME,
        JsonlProgressSink,
        MultiProgressSink,
        RampStarting,
    )
    from horde_worker_regen.benchmark.progress_console import ConsoleProgressSink
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / time.strftime("%Y%m%d-%H%M%S")
    _setup_benchmark_file_logging(out_dir)

    progress_sink = MultiProgressSink(
        [JsonlProgressSink(out_dir / PROGRESS_FILENAME), ConsoleProgressSink(verbose=args.verbose)],
    )
    progress_sink.emit(
        RampStarting(
            run_id=out_dir.name,
            process_mode=args.process_mode,
            phase="loading worker environment and detecting hardware",
        ),
    )

    # The harness never reads bridgeData.yaml, so set AIWORKER_CACHE_HOME (and friends) before booting the
    # worker, so the real inference children resolve the worker's model directory. Passing the tiers opts
    # into the beta reference when a beta tier (qwen/zimage) is requested.
    ensure_worker_env(args.process_mode, tiers)
    machine = detect_machine_info(probe_devices=args.process_mode == "real")

    options = CatalogOptions(
        tiers=tiers,
        jobs_per_level=args.jobs_per_level,
        include_concurrency=not args.no_concurrency,
        include_features=not args.no_features,
        include_alchemy=not args.no_alchemy,
        include_downloads=args.include_downloads,
        excluded_kinds={CapabilityKind(value) for value in args.exclude_capability},
        probe_timeout_seconds=args.probe_timeout,
        total_vram_mb=machine.total_vram_mb,
    )
    executor = ProbeExecutor(
        catalog_options=options,
        process_mode=args.process_mode,
        machine=machine,
        out_dir=out_dir,
        run_soak=not args.no_validate,
        soak_seconds=args.soak_minutes * 60.0,
        strict_duty_cycle=args.strict_duty,
        only_probe=args.only,
        progress_sink=progress_sink,
    )
    try:
        report = executor.run()
    except Exception:
        logger.exception("The benchmark executor crashed.")
        raise
    finally:
        progress_sink.close()
    _record_capability_benchmark_in_app_state(report, out_dir)

    proven = sum(1 for probe in report.probes if probe.verdict == "proven")
    print(f"\nBenchmark complete: {proven}/{len(report.probes)} probes proven.")  # noqa: T201
    print(f"Report: {out_dir / 'report.md'}")  # noqa: T201
    if report.findings:
        print(f"Robustness findings: {len(report.findings)} (see the remediation queue in the report)")  # noqa: T201
    print("\nSuggested bridgeData:")  # noqa: T201
    print(report.suggested_bridge_data.as_yaml_block())  # noqa: T201
    return 0


def _evict_pinned_loras(version_ids: tuple[str, ...]) -> bool:
    """Evict the corpus's LoRA versions from the cache so the first use of each is a genuine miss.

    Deletion goes through the LoRA model manager's ``delete_lora`` (removing the file and its
    reference-db entry together), in a subprocess: the model-manager import chain must stay out of this
    process, which goes on to host the worker parent. An id that is already absent only draws the
    manager's not-found warning, so eviction is idempotent across reruns.

    Returns:
        True when the eviction subprocess completed; False when it failed, since running on without it
        would silently turn the miss cells into hit measurements.
    """
    import subprocess

    script = (
        "import sys\n"
        "from hordelib.model_manager.lora import LoraModelManager\n"
        "manager = LoraModelManager()\n"
        "for version_id in sys.argv[1:]:\n"
        "    manager.delete_lora(version_id)\n"
    )
    logger.info(f"Evicting {len(version_ids)} pinned LoRA version(s) so the miss cells measure real fetches.")
    result = subprocess.run([sys.executable, "-c", script, *version_ids], check=False)
    if result.returncode != 0:
        logger.error(f"LoRA eviction failed (exit {result.returncode}); refusing to run with a warm cache.")
        return False
    return True


def _pricing_corpus_bridge_overrides(tier: str) -> dict[str, object]:
    """Return the bridge capabilities a corpus tier's workload needs advertised.

    The harness derives its envelope from the workload, while the census also pins the complete capability
    surface explicitly: the corpus is a declared cross-capability measurement, so a future template or harness
    derivation change must not silently narrow what the worker offers.
    """
    if tier != "census":
        return {}
    return {
        "allow_controlnet": True,
        "extended_controlnet": True,
        "allow_img2img": True,
        "allow_inpainting": True,
        "allow_lora": True,
        "allow_post_processing": True,
    }


def _run_pricing_corpus(args: argparse.Namespace) -> int:
    """Build (and, unless listing, run) the pricing corpus, persisting its definition artifact."""
    from horde_worker_regen.benchmark.pricing_corpus import (
        PRICING_CORPUS_LORA_VERSION_IDS,
        PRICING_CORPUS_TI_NAME,
        PricingCorpusError,
        build_pricing_corpus_scenario,
        write_definition_artifact,
    )
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env
    from horde_worker_regen.harness import HarnessConfig, run_harness
    from horde_worker_regen.reference_helper import ensure_model_reference_manager_initialized
    from horde_worker_regen.stats_operations import default_stats_dir

    lora_version_ids = tuple(args.lora_version_id) if args.lora_version_id else PRICING_CORPUS_LORA_VERSION_IDS
    try:
        scenario, definition = build_pricing_corpus_scenario(
            args.tier,
            lora_version_ids=lora_version_ids,
            ti_name=args.ti_name if args.ti_name else PRICING_CORPUS_TI_NAME,
        )
    except PricingCorpusError as error:
        logger.error(str(error))
        return 2

    if args.emit_definition is not None:
        write_definition_artifact(definition, args.emit_definition)
        logger.info(f"Wrote the corpus definition to {args.emit_definition.resolve()}.")

    if args.dry_list:
        for job in definition.jobs:
            print(f"{job.position:4d}  {job.permutation:8s}  {job.cell_id}")  # noqa: T201
        print(  # noqa: T201
            f"\n{len(definition.jobs)} jobs ({definition.warmup_job_count} warmup) over "
            f"{len(definition.cells)} cells; shuffle seeds {', '.join(definition.shuffle_seeds)}.",
        )
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / f"pricing-corpus-{stamp}"
    _setup_benchmark_file_logging(out_dir)
    if args.emit_definition is None:
        # The artifact is the only key from a stats record back to the cell (and so to the axis values)
        # that produced it, so it lands with the stats stream rather than in a run directory that a later
        # reader of the stats has no reason to look in.
        definition_path = default_stats_dir() / f"pricing-corpus-{definition.tier}-{stamp}.json"
        write_definition_artifact(definition, definition_path)
        logger.info(f"Wrote the corpus definition to {definition_path.resolve()}.")

    if args.no_lora_eviction:
        logger.warning(
            "LoRA eviction skipped: any pinned LoRA already cached makes its miss cell measure a hit.",
        )
    elif not _evict_pinned_loras(lora_version_ids):
        return 2

    ensure_worker_env(args.process_mode, [BenchTier.SD15, BenchTier.SDXL])
    # The corpus exists to price jobs, and price varies by model class, so every job's stats record must
    # carry the model's real baseline. Initializing the reference here (a plain sync context, before the
    # harness event loop starts) is what lets the harness resolve real records instead of stubbing them.
    try:
        ensure_model_reference_manager_initialized()
    except Exception as reference_error:  # noqa: BLE001 - a reference miss degrades records, not the run
        logger.warning(
            f"Could not initialize the model reference ({type(reference_error).__name__}); corpus records "
            "will carry stubbed baselines.",
        )
    logger.info(
        f"Running the {definition.tier} pricing corpus: {len(definition.jobs)} jobs over "
        f"{len(definition.cells)} cells, models {', '.join(scenario.models_referenced())}.",
    )
    result = run_harness(
        HarnessConfig.from_scenario(
            scenario,
            process_mode=args.process_mode,
            timeout_seconds=args.timeout,
            bridge_data_overrides=_pricing_corpus_bridge_overrides(definition.tier),
        ),
    )
    logger.info(
        f"Corpus finished: {result.num_jobs_completed}/{result.num_jobs_expected} jobs completed, "
        f"{result.num_jobs_faulted} faulted, in {result.elapsed_seconds:.0f}s ({result.exit_reason}).",
    )
    return 0 if result.succeeded else 1


_SOAK_START_MARGIN_SECONDS = 180.0
"""Headroom over the soak period for the cold boot and model load before the mix starts flowing."""

_SOAK_DRAIN_MARGIN_SECONDS = 60.0
"""Headroom after the soak period for in-flight jobs to finish and the worker to shut down cleanly."""

_SOAK_TOP_SLOT_DUTY_BUCKETS = 4
"""How many non-sampling slot-duty buckets the closing summary names."""


def _soak_stats_session(stats_dir: Path, known_session_ids: set[str]) -> tuple[str, list[Path]] | None:
    """The stats session this soak wrote, or None when the export produced nothing.

    Identified as a session id that was not present before the run, so a stats directory holding earlier
    sessions cannot be mistaken for this one. Falls back to the only session present when the directory
    held nothing beforehand, which is the first-run case.
    """
    sessions = discover_stats_sessions(stats_dir)
    fresh = [(session_id, paths) for session_id, paths in sessions if session_id not in known_session_ids]
    if fresh:
        return fresh[-1]
    if len(sessions) == 1 and not known_session_ids:
        return sessions[0]
    return None


def _print_soak_summary(report: SessionDutyReport, *, label: str, stats_paths: list[Path]) -> None:
    """Print the stats paths and the throughput headline a soak arm is compared on.

    Kudos is unavailable offline (no horde priced this work), so the throughput readings are the ones the
    worker measured itself: completed jobs, and the sampling slot-duty seconds that are the denoise time
    those jobs spent on the card. The remaining slot-duty buckets name what the empty slot-seconds were
    waiting on, which is where an arm's difference shows up.
    """
    print(f"\nStats: {', '.join(str(path.resolve()) for path in stats_paths)}")  # noqa: T201
    print(render_session_duty_report([report]))  # noqa: T201
    sampling_seconds = report.slot_duty_seconds.get("sampling", 0.0)
    arm = f" [{label}]" if label else ""
    print(  # noqa: T201
        f"soak{arm}: {report.completed_jobs} jobs completed, {sampling_seconds:.0f}s denoise "
        f"(sampling slot-duty); kudos is not available offline.",
    )
    other = sorted(
        ((name, seconds) for name, seconds in report.slot_duty_seconds.items() if name != "sampling"),
        key=lambda item: item[1],
        reverse=True,
    )[:_SOAK_TOP_SLOT_DUTY_BUCKETS]
    if other:
        buckets = "  ".join(f"{name} {seconds:.0f}s" for name, seconds in other)
        print(f"top non-sampling slot duty: {buckets}")  # noqa: T201


def _run_soak(args: argparse.Namespace) -> int:
    """Run one sustained-traffic soak mix and score it from the stats export it wrote."""
    from horde_worker_regen.benchmark.gate_driver import _parse_overrides
    from horde_worker_regen.benchmark.soak import build_soak_mix_scenario, operator_throughput_overrides
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env
    from horde_worker_regen.harness import HarnessConfig, run_harness
    from horde_worker_regen.reference_helper import ensure_model_reference_manager_initialized
    from horde_worker_regen.stats_operations import default_stats_dir

    try:
        cli_overrides = _parse_overrides(args.override)
        scenario = build_soak_mix_scenario(
            args.mix,
            soak_seconds=args.minutes * 60.0,
            shared_lora_references=args.shared_lora or None,
            unique_lora_references=args.unique_lora or None,
            include_auxiliary_references=not args.no_loras,
        )
    except (argparse.ArgumentTypeError, ValueError) as error:
        logger.error(str(error))
        return 2

    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / time.strftime("%Y%m%d-%H%M%S")
    _setup_benchmark_file_logging(out_dir)

    bridge_overrides: dict[str, object] = {}
    if not args.ignore_bridge_data:
        bridge_overrides.update(operator_throughput_overrides(args.bridge_data))
        if not bridge_overrides:
            logger.warning(
                f"No throughput fields were read from {args.bridge_data}; the soak runs on harness defaults, "
                "which are not the configuration this worker serves under.",
            )
    bridge_overrides.update(cli_overrides)

    if args.process_mode == "real" and not args.no_loras and not (args.shared_lora and args.unique_lora):
        logger.warning(
            f"The {args.mix} mix carries LoRA references and none were supplied, so its LoRA-bearing "
            "templates use synthetic names that cannot resolve: those jobs would measure the prefetch "
            "failure path. Pass --shared-lora/--unique-lora with resolvable references, or --no-loras.",
        )

    ensure_worker_env(args.process_mode)
    if args.process_mode == "real":
        # Comparing two arms requires every stats record to carry the model's real baseline. Initializing
        # the reference here (a plain sync context, before the harness event loop starts) is what lets the
        # harness resolve real records instead of stubbing them.
        try:
            ensure_model_reference_manager_initialized()
        except Exception as reference_error:  # noqa: BLE001 - a reference miss degrades records, not the run
            logger.warning(
                f"Could not initialize the model reference ({type(reference_error).__name__}); soak records "
                "will carry stubbed baselines.",
            )

    stats_dir = default_stats_dir()
    known_session_ids = {session_id for session_id, _ in discover_stats_sessions(stats_dir)}

    logger.info(
        f"Soaking the {scenario.name} mix for {args.minutes:.1f} min in {args.process_mode} mode "
        f"({len(scenario.image_jobs)} templates over {len(scenario.models_referenced())} models); "
        f"bridge overrides: {bridge_overrides or 'none'}.",
    )
    result = run_harness(
        HarnessConfig.from_scenario(
            scenario,
            process_mode=args.process_mode,
            timeout_seconds=args.minutes * 60.0 + _SOAK_START_MARGIN_SECONDS + _SOAK_DRAIN_MARGIN_SECONDS,
            bridge_data_overrides=bridge_overrides,
        ),
    )
    # The harness reconfigures loguru's sinks for the run, so the closing summary goes to stdout to
    # reach the operator, and to the log for the run directory's own record.
    finished = (
        f"Soak finished: {result.num_jobs_completed} jobs completed, {result.num_jobs_faulted} faulted, "
        f"in {result.elapsed_seconds:.0f}s ({result.exit_reason})."
    )
    logger.info(finished)
    print(f"\n{finished}")  # noqa: T201

    manifest: dict[str, object] = {
        "label": args.label,
        "mix": args.mix,
        "scenario_id": scenario.name,
        "minutes": args.minutes,
        "process_mode": args.process_mode,
        "loras_included": not args.no_loras,
        "bridge_overrides": bridge_overrides,
        "jobs_completed": result.num_jobs_completed,
        "jobs_faulted": result.num_jobs_faulted,
        "elapsed_seconds": result.elapsed_seconds,
        "exit_reason": result.exit_reason,
    }

    session = _soak_stats_session(stats_dir, known_session_ids)
    if session is None:
        no_stats = (
            f"No stats session was written to {stats_dir} (the export is real-mode only), so this soak "
            "cannot be scored offline."
        )
        logger.warning(no_stats)
        print(no_stats)  # noqa: T201
    else:
        session_id, stats_paths = session
        manifest["stats_session_id"] = session_id
        manifest["stats_files"] = [str(path.resolve()) for path in stats_paths]
        _print_soak_summary(
            analyze_stats_files(session_id=session_id, paths=stats_paths),
            label=args.label,
            stats_paths=stats_paths,
        )

    manifest_path = out_dir / "soak.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Run manifest: {manifest_path.resolve()}")  # noqa: T201
    return 0 if result.succeeded else 1


def _record_capability_benchmark_in_app_state(report: CapabilityReport, out_dir: Path) -> None:
    """Record a finished capability run in app state, best-effort (bookkeeping must not fail the run)."""
    try:
        from horde_worker_regen.app_state import AppStateStore, build_capability_benchmark_record

        record = build_capability_benchmark_record(report, results_dir=out_dir)
        AppStateStore().record_benchmark(record)
        logger.info(f"Recorded benchmark {record.run_id} in app state ({AppStateStore().path}).")
    except Exception as app_state_error:  # noqa: BLE001 - app-state bookkeeping must not fail the run
        logger.debug(f"Could not record benchmark in app state: {app_state_error}")


_MONITOR_POLL_SECONDS = 0.5
_MONITOR_IDLE_POLLS_BEFORE_EXIT = 3
"""Consecutive empty polls (with a report present) after which `monitor` concludes a finished run."""


def _run_monitor(args: argparse.Namespace) -> int:
    """Tail a run's ``progress.jsonl`` and render it live, for attaching to or replaying a ramp."""
    from horde_worker_regen.benchmark.progress_channel import PROGRESS_FILENAME, ProgressTailer, RampFinished
    from horde_worker_regen.benchmark.progress_console import format_progress_event

    out_dir: Path = args.out_dir
    progress_path = out_dir / PROGRESS_FILENAME
    if not progress_path.exists():
        logger.error(f"No {PROGRESS_FILENAME} found in {out_dir}")
        return 1

    tailer = ProgressTailer(progress_path)
    saw_ramp_finished = False
    idle_polls = 0
    while not saw_ramp_finished:
        events = tailer.poll()
        if not events:
            idle_polls += 1
            if (out_dir / "report.json").exists() and idle_polls >= _MONITOR_IDLE_POLLS_BEFORE_EXIT:
                break
            time.sleep(_MONITOR_POLL_SECONDS)
            continue
        idle_polls = 0
        for event in events:
            line = format_progress_event(event, verbose=args.verbose)
            if line is not None:
                print(line)  # noqa: T201
            if isinstance(event, RampFinished):
                saw_ramp_finished = True
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Re-render a run's markdown report from its persisted ``report.json``."""
    from horde_worker_regen.benchmark.capabilities.report_render import render_markdown
    from horde_worker_regen.benchmark.capabilities.result import CapabilityReport

    report_path = args.out_dir / "report.json"
    if not report_path.exists():
        logger.error(f"No report.json found in {args.out_dir}")
        return 1
    try:
        report = CapabilityReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.error(f"Could not parse a capability report from {report_path}: {error}")
        return 1

    markdown = render_markdown(report)
    (args.out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="horde-benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_run_parser(subparsers)
    _add_plan_parser(subparsers)
    _add_download_parser(subparsers)
    _add_pricing_corpus_parser(subparsers)
    _add_soak_parser(subparsers)

    report = subparsers.add_parser("report", help="Re-render the markdown report from an output directory.")
    report.add_argument("out_dir", type=Path)

    monitor = subparsers.add_parser("monitor", help="Tail a run's progress.jsonl live (attach or replay).")
    monitor.add_argument("out_dir", type=Path)
    monitor.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the per-process state summary on each progress line.",
    )

    subparsers.add_parser("live", help="Open-loop load generation against a live API (not yet implemented).")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_run(args)
    if args.command == "plan":
        return _run_plan(args)
    if args.command == "download":
        return _run_download(args)
    if args.command == "pricing-corpus":
        return _run_pricing_corpus(args)
    if args.command == "soak":
        return _run_soak(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "monitor":
        return _run_monitor(args)
    if args.command == "live":
        logger.error("The live load-generation path is a separate phase and not implemented yet.")
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
