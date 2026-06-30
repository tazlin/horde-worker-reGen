"""The `horde-benchmark` CLI: progressive worker benchmarking.

Subcommands:
- ``ramp``: run the ramp ladder via the canned-job harness (reproducible, no API).
- ``plan``: preview each level's resource needs and run/skip verdict (no worker is started).
- ``download``: fetch the checkpoints the selected tiers need, ahead of a timed ramp.
- ``report``: re-render the markdown report from an existing output directory.
- ``monitor``: tail a run's progress.jsonl live (attach or replay).
- ``live``: open-loop load generation against a live AI-Horde API (separate phase).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
from horde_worker_regen.benchmark.enums import SELECTABLE_AXES, BenchAxis, BenchTier

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.benchmark.capabilities.result import CapabilityReport
    from horde_worker_regen.benchmark.download_progress import DownloadEvent, DownloadModelRow
    from horde_worker_regen.benchmark.ladder import LadderOptions, RampLevel
    from horde_worker_regen.benchmark.report import BenchmarkReport, MachineInfo
    from horde_worker_regen.model_download_core import DownloadControls
    from horde_worker_regen.model_download_plan import DownloadPlan


def _add_ramp_parser(subparsers: argparse._SubParsersAction) -> None:
    ramp = subparsers.add_parser("ramp", help="Run the progressive ramp benchmark via the harness.")
    ramp.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers to attempt (sd15, sdxl, flux, qwen, zimage). flux/qwen/zimage "
        "are opt-in (very large: 17-20 GB download, 13-16 GB VRAM) and auto-skip when the machine cannot "
        "hold them; qwen and zimage are beta models sourced from the pending reference.",
    )
    ramp.add_argument(
        "--process-mode",
        default="real",
        choices=("fake", "dry_run", "real"),
        help="real = GPU benchmark; fake/dry_run exercise the ramp machinery without inference.",
    )
    ramp.add_argument("--out", type=Path, default=None, help="Output directory (default: benchmark_results/<ts>).")
    ramp.add_argument("--jobs-per-level", type=int, default=4)
    ramp.add_argument("--level-timeout", type=float, default=900.0, help="Per-level timeout in seconds.")
    ramp.add_argument("--resume", action="store_true", help="Reuse existing level results in --out.")
    ramp.add_argument("--only-level", default=None, help="Run a single level id (remediation loop).")
    ramp.add_argument("--skip-downloads", action="store_true", help="Skip levels that need network access.")
    _add_stage_selection_args(ramp)
    ramp.add_argument(
        "--force",
        action="store_true",
        help="Attempt levels that would otherwise be skipped for not fitting this machine (insufficient "
        "VRAM/disk) or lacking a CivitAI token. An absent checkpoint is still skipped.",
    )
    ramp.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-ramp sustained-load validation soak.",
    )
    ramp.add_argument(
        "--soak-minutes",
        type=float,
        default=5.0,
        help="Duration of each per-tier validation soak (minutes).",
    )
    ramp.add_argument(
        "--strict-duty",
        action="store_true",
        help="Fail a validation soak whose GPU duty cycle misses the 90%% target (a hard gate). Off by "
        "default: the soak reports the shortfall with attribution but passes on stability and throughput. "
        "Use when enforcing the duty-cycle target on a reference machine.",
    )
    ramp.add_argument(
        "--warm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one warm worker across fixed-scenario levels instead of cold-starting a fresh worker "
        "(and respawning every inference process) per level. On by default; pass --no-warm to run each "
        "level in its own isolated subprocess (full crash isolation at the cost of per-level startup).",
    )
    ramp.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-process state in the live view and raise spawned worker logging to TRACE.",
    )
    ramp.add_argument(
        "--no-abort-on-catastrophe",
        action="store_true",
        help="Keep running after a level hangs/crashes instead of aborting the whole ramp "
        "(by default the first catastrophic failure stops the ramp, since the worker stack is shared).",
    )


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


def _add_stage_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the stage-inclusion flags shared by ``ramp`` and ``plan`` (so the two stay in lockstep)."""
    parser.add_argument("--include-downloads", action="store_true", help="Include the ad-hoc download levels.")
    parser.add_argument("--no-alchemy", action="store_true", help="Skip the alchemy levels.")
    parser.add_argument("--no-features", action="store_true", help="Skip the feature levels (stage C).")
    parser.add_argument("--no-concurrency", action="store_true", help="Skip the concurrency levels (stage B).")
    parser.add_argument(
        "--exclude-axis",
        action="append",
        default=[],
        choices=[axis.value for axis in SELECTABLE_AXES],
        metavar="AXIS",
        help="Drop a single ramp axis, independent of the coarse stage flags (repeatable). One of: "
        + ", ".join(axis.value for axis in SELECTABLE_AXES)
        + ".",
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
    # runs in real mode; kept as a fixed attribute so the shared _prepare_ladder helper has what it needs.
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


def _prepare_ladder(
    args: argparse.Namespace,
    tiers: list[BenchTier],
    *,
    probe_devices: bool = True,
) -> tuple[list[RampLevel], MachineInfo, LadderOptions]:
    """Apply the worker env, detect the machine, and build the ladder for the given selection.

    Shared by ``ramp`` and ``plan`` so the plan an operator previews is built from the exact same
    ladder (and the same detected VRAM) the ramp would run. ``probe_devices`` controls the GPU
    enumeration: callers that do not use the device info (the ``download`` path discards it) pass False
    to skip the out-of-process torch/CUDA probe, which is otherwise a cold, multi-minute startup cost.
    """
    from horde_worker_regen.benchmark.controller import detect_machine_info
    from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env

    # The harness never reads bridgeData.yaml, so set AIWORKER_CACHE_HOME (and friends) here, before
    # spawning level subprocesses, so the real inference children resolve the worker's actual model
    # directory instead of hordelib's empty ./models fallback. Children inherit this process's env.
    # Passing the tiers also opts into the beta reference when a beta tier (qwen) is requested.
    ensure_worker_env(args.process_mode, tiers)

    # Detect the machine once: the ladder uses the VRAM to size the post-processing sweep, and the
    # controller reuses the same info instead of re-detecting. Skipped (probe_devices=False) where the
    # caller ignores the device info, since the probe runs a cold out-of-process torch import.
    machine = detect_machine_info(probe_devices=probe_devices)

    options = LadderOptions(
        tiers=tiers,
        jobs_per_level=getattr(args, "jobs_per_level", 4),
        include_concurrency=not args.no_concurrency,
        include_features=not args.no_features,
        include_alchemy=not args.no_alchemy,
        include_downloads=args.include_downloads,
        excluded_axes={BenchAxis(value) for value in getattr(args, "exclude_axis", [])},
        level_timeout_seconds=getattr(args, "level_timeout", 900.0),
        total_vram_mb=machine.total_vram_mb,
    )
    return build_default_ladder(options), machine, options


def _run_plan(args: argparse.Namespace) -> int:
    """Print each level's resource requirements and predicted verdict against the detected machine."""
    from horde_worker_regen.benchmark.controller import BenchmarkController
    from horde_worker_regen.benchmark.progress_console import format_plan_table

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    logger.info("Building benchmark plan (detecting hardware; no worker is started) ...")
    ladder, machine, _options = _prepare_ladder(args, tiers)

    # The controller owns the pre-flight verdict, so build one (without running it) to reuse that logic.
    controller = BenchmarkController(
        ladder,
        Path("benchmark_plan"),
        process_mode=args.process_mode,
        force=args.force,
        machine=machine,
    )
    rows = controller.build_plan_rows(machine)

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


def _ladder_control_types(ladder: list[RampLevel]) -> list[str]:
    """Return the distinct controlnet ``control_type``s any level in the ladder exercises (may be empty)."""
    return sorted({job.control_type for level in ladder for job in level.scenario.image_jobs if job.control_type})


def _ladder_post_processors(ladder: list[RampLevel]) -> list[str]:
    """Return the distinct post-processor names any level in the ladder exercises (may be empty)."""
    return sorted(
        {
            post_processor
            for level in ladder
            for job in level.scenario.image_jobs
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
    ladder, _machine, _options = _prepare_ladder(args, tiers, probe_devices=False)
    model_names = sorted({name for level in ladder for name in level.scenario.models_referenced()})
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
    control_types = _ladder_control_types(ladder)
    post_processors = _ladder_post_processors(ladder)
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


def _setup_controller_file_logging(out_dir: Path) -> None:
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


def _run_ramp(args: argparse.Namespace) -> int:
    from horde_worker_regen.benchmark.progress_channel import (
        PROGRESS_FILENAME,
        JsonlProgressSink,
        MultiProgressSink,
        RampStarting,
    )
    from horde_worker_regen.benchmark.progress_console import ConsoleProgressSink

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / time.strftime("%Y%m%d-%H%M%S")
    _setup_controller_file_logging(out_dir)

    # Tee progress to a durable JSONL log (for the TUI / `monitor` to tail) and a live console view.
    # The sink is created up front, before the slow hardware-detection and import phase below, so the
    # first heartbeat creates progress.jsonl immediately and the TUI has something to render. Without it
    # the entire startup window (torch/hordelib import + GPU probe) is dark on both the progress file and
    # the buffered console, so a slow or wedged startup is indistinguishable from a hang.
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

    from horde_worker_regen.benchmark.controller import BenchmarkController

    ladder, machine, _options = _prepare_ladder(args, tiers)
    logger.info(f"Ramp ladder has {len(ladder)} level(s); output in {out_dir}")

    controller = BenchmarkController(
        ladder,
        out_dir,
        process_mode=args.process_mode,
        resume=args.resume,
        only_level=args.only_level,
        skip_downloads=args.skip_downloads,
        validate=not args.no_validate,
        soak_seconds=args.soak_minutes * 60.0,
        strict_duty_cycle=args.strict_duty,
        progress_sink=progress_sink,
        warm=args.warm,
        verbose=args.verbose,
        abort_on_catastrophe=not args.no_abort_on_catastrophe,
        force=args.force,
        machine=machine,
    )
    try:
        report = controller.run()
    except Exception:
        # With controller.log now installed, surface a controller-process crash to disk (not just stderr).
        logger.exception("The benchmark controller crashed.")
        raise
    finally:
        progress_sink.close()
    _record_benchmark_in_app_state(report, out_dir)

    passed = sum(1 for level in report.levels if level.outcome == "passed")
    print(f"\nBenchmark complete: {passed}/{len(report.levels)} levels passed.")  # noqa: T201
    print(f"Report: {out_dir / 'report.md'}")  # noqa: T201
    if report.findings:
        print(f"Robustness findings: {len(report.findings)} (see the remediation queue in the report)")  # noqa: T201
    print("\nSuggested bridgeData:")  # noqa: T201
    print(report.suggested_bridge_data.as_yaml_block())  # noqa: T201
    return 0


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
    _setup_controller_file_logging(out_dir)

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


def _record_capability_benchmark_in_app_state(report: CapabilityReport, out_dir: Path) -> None:
    """Record a finished capability run in app state, best-effort (bookkeeping must not fail the run)."""
    try:
        from horde_worker_regen.app_state import AppStateStore, build_capability_benchmark_record

        record = build_capability_benchmark_record(report, results_dir=out_dir)
        AppStateStore().record_benchmark(record)
        logger.info(f"Recorded benchmark {record.run_id} in app state ({AppStateStore().path}).")
    except Exception as app_state_error:  # noqa: BLE001 - app-state bookkeeping must not fail the run
        logger.debug(f"Could not record benchmark in app state: {app_state_error}")


def _record_benchmark_in_app_state(report: BenchmarkReport, out_dir: Path) -> None:
    """Record a finished CLI ramp as the canonical run-to-run benchmark, best-effort.

    Both the CLI and the TUI funnel through the same ``build_benchmark_record`` adapter so the durable
    pointer is identical regardless of how the ramp was launched. Failure here only loses bookkeeping,
    so it is logged at debug rather than raised.
    """
    try:
        from horde_worker_regen.app_state import AppStateStore, build_benchmark_record

        record = build_benchmark_record(report, results_dir=out_dir)
        AppStateStore().record_benchmark(record)
        logger.info(f"Recorded benchmark {record.run_id} in app state ({AppStateStore().path}).")
    except Exception as app_state_error:  # noqa: BLE001 - app-state bookkeeping must not fail the ramp
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
    """Re-render a run's markdown report, auto-detecting the capability (v5) or legacy report format."""
    import json

    report_path = args.out_dir / "report.json"
    if not report_path.exists():
        logger.error(f"No report.json found in {args.out_dir}")
        return 1

    # A capability report carries ``probes``; a legacy report carries ``levels``. Render with whichever
    # engine produced it so ``report`` works for both ``run`` and the older ``ramp`` output.
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error(f"Could not read {report_path}: {error}")
        return 1

    if isinstance(data, dict) and "probes" in data:
        from horde_worker_regen.benchmark.capabilities.report_render import render_markdown as render_capability
        from horde_worker_regen.benchmark.capabilities.result import CapabilityReport

        markdown = render_capability(CapabilityReport.model_validate(data))
    else:
        from horde_worker_regen.benchmark.controller import load_existing_report
        from horde_worker_regen.benchmark.report import render_markdown

        report = load_existing_report(args.out_dir)
        if report is None:
            logger.error(f"Could not parse a benchmark report from {report_path}")
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
    _add_ramp_parser(subparsers)
    _add_plan_parser(subparsers)
    _add_download_parser(subparsers)

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
    if args.command == "ramp":
        return _run_ramp(args)
    if args.command == "plan":
        return _run_plan(args)
    if args.command == "download":
        return _run_download(args)
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
