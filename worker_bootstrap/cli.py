"""Command-line entry for the worker bootstrap brain (detect / sync / launch / preload / install)."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import backend as backend_mod
from worker_bootstrap import config_seed, consent, detect, gitbin, paths, runner, sync_plan, updater, uvbin

_BACKEND_ENV = "HORDE_WORKER_BACKEND"
_FEATURES_ENV = "HORDE_WORKER_FEATURES"

_DEFAULT_CONFIRM_MB = 1500

# launch mode -> the uv-run command (console scripts from pyproject [project.scripts]). "bridge" is handled
# separately because it downloads models before starting the worker, matching the old horde-bridge.cmd.
_LAUNCH_COMMANDS: dict[str, list[str]] = {
    "web": ["horde-worker-web"],
    "terminal": ["horde-worker"],
    "host": ["horde-worker-host"],
    "benchmark": ["horde-benchmark"],
}


def _print_amd_unsupported() -> None:
    """Explain that no usable AMD backend was found and how to force a choice."""
    print(
        "An AMD GPU was detected, but no installable GPU backend was matched. The installer supports "
        "ComfyUI's experimental AMD Windows ROCm profile for supported Radeon/Ryzen AI GPUs, plus Linux "
        "ROCm when the ROCm runtime is present. "
        "Re-run with HORDE_WORKER_BACKEND=cpu for the CPU build (~100x slower), or force a known profile "
        "with HORDE_WORKER_BACKEND=rocm-windows if your card is supported but was not recognized.",
        file=sys.stderr,
    )


def _print_cpu_notice() -> None:
    """Explain that the CPU build runs in alchemist-only mode (image generation disabled)."""
    print(
        "Using the CPU build: image generation is disabled (CPU inference is ~100x slower), so the worker "
        "runs in alchemist-only mode (upscaling, face-fixing, interrogation). Set alchemist: true in "
        "bridgeData.yaml (done automatically for a fresh CPU install). Reinstall a GPU build later with "
        "update-runtime --cu132 (or the matching build) to enable image generation.",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class _SyncOptions:
    """Resolved knobs for the preview/hold/prune behaviour of a sync (CLI flag > env var > default)."""

    preview: bool
    hold: bool
    confirm_threshold_bytes: int
    headless_policy: str
    prune: bool


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var; unset/empty falls back to *default*; ``0/false/no/off`` mean False."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _sync_options(args: argparse.Namespace) -> _SyncOptions:
    """Resolve sync behaviour from CLI flags (when present), then env vars, then safe defaults.

    ``args`` may lack the sync flags (e.g. a ``launch`` first-run sync), in which case every flag reads
    ``None`` and the env/default takes over.
    """
    preview = not getattr(args, "no_sync_preview", False) and _env_bool("HORDE_WORKER_SYNC_PREVIEW", True)

    hold_flag = getattr(args, "hold_torch", None)
    hold = hold_flag if hold_flag is not None else _env_bool("HORDE_WORKER_SYNC_HOLD", False)

    confirm_mb = getattr(args, "confirm_above_mb", None)
    if confirm_mb is None:
        raw_mb = os.environ.get("HORDE_WORKER_SYNC_CONFIRM_MB", "")
        try:
            confirm_mb = int(raw_mb) if raw_mb.strip() else _DEFAULT_CONFIRM_MB
        except ValueError:
            confirm_mb = _DEFAULT_CONFIRM_MB

    policy = getattr(args, "headless_policy", None) or os.environ.get("HORDE_WORKER_SYNC_HEADLESS_POLICY") or "proceed"
    policy = policy.strip().lower()
    if policy not in ("proceed", "hold"):
        policy = "proceed"

    prune = not getattr(args, "no_prune", False) and _env_bool("HORDE_WORKER_SYNC_PRUNE", True)

    return _SyncOptions(
        preview=preview,
        hold=hold,
        confirm_threshold_bytes=max(confirm_mb, 0) * 1024 * 1024,
        headless_policy=policy,
        prune=prune,
    )


def _apply_cache_mode_flag(args: argparse.Namespace) -> None:
    """Honour ``--cache-mode`` by setting the env var the shims/runner read (CLI flag > env var).

    Shared mode also clears any ``UV_CACHE_DIR`` a shim pre-set, so uv falls back to its own default
    (system) cache rather than the isolated one for this run.
    """
    mode = getattr(args, "cache_mode", None)
    if mode is None:
        return
    os.environ["HORDE_WORKER_UV_CACHE_MODE"] = mode
    if mode == "shared":
        os.environ.pop("UV_CACHE_DIR", None)


def _cache_is_owned(root: Path) -> bool:
    """Return whether uv's cache for this run is the isolated cache we created (so safe to auto-prune)."""
    if paths.uv_cache_mode() == "shared":
        return False
    preset = os.environ.get("UV_CACHE_DIR")
    return preset is None or Path(preset) == paths.uv_cache_dir(root)


def _effective_cache_dir(root: Path) -> str:
    """Return a human label for the uv cache uv will use this run (for the preview footer)."""
    if paths.uv_cache_mode() == "shared":
        return "uv default (shared) cache"
    return os.environ.get("UV_CACHE_DIR") or str(paths.uv_cache_dir(root))


def _maybe_prune(uv: str, root: Path, options: _SyncOptions) -> None:
    """Auto-prune the owned uv cache after a successful sync (never a shared/redirected cache).

    The install is already complete and on disk by the time this runs, so every non-success outcome
    below is reported as a skipped cleanup, never a failed install. The progress line is printed (and
    flushed) before the blocking call so a slow prune cannot look like a hang.
    """
    if not options.prune or not _cache_is_owned(root):
        return
    print("Tidying the worker's uv cache (removing superseded wheels; this can take a moment)...", flush=True)
    rc, reclaimed = runner.uv_cache_prune(uv, root=root)
    if rc == 0:
        if reclaimed:
            reclaimed_human = sync_plan.human_bytes(reclaimed)
            print(f"Reclaimed {reclaimed_human} from the worker's uv cache ({_effective_cache_dir(root)}).")
        else:
            print("uv cache already tidy; nothing to reclaim.")
        return
    if rc == runner.PRUNE_TIMED_OUT:
        print(
            "Cache cleanup timed out and was skipped; the install is complete. "
            "Raise HORDE_WORKER_PRUNE_TIMEOUT or pass --no-prune to silence this.",
            file=sys.stderr,
        )
    elif rc == runner.PRUNE_INTERRUPTED:
        print("Cache cleanup was interrupted and skipped; the install is complete.", file=sys.stderr)
    else:
        print("Cache cleanup did not complete (non-fatal); the install is complete.", file=sys.stderr)


def _is_headless() -> bool:
    """Return whether this run is non-interactive (no terminal) or consent was captured upstream."""
    return consent.consent_env_var() is not None or not consent.is_interactive()


def _run_sync(uv: str, root: Path, token: str, feature_extras: tuple[str, ...], options: _SyncOptions) -> int:
    """Run the sync, first showing a download preview and honouring a hold/cancel when one applies.

    Falls back to the plain locked sync whenever the preview is disabled, there is no venv yet (nothing
    to limp along on), or the dry-run cannot be produced/parsed: the preview must never block an update.
    """
    if not options.preview or not paths.venv_dir(root).exists():
        return runner.uv_sync(uv, token, extras=feature_extras, root=root)

    rc_dry, output = runner.uv_sync_dry_run(uv, token, extras=feature_extras, root=root)
    changes = sync_plan.parse_dry_run(output) if rc_dry == 0 else []
    if rc_dry != 0 or not changes:
        if rc_dry != 0:
            print("Could not preview the sync; proceeding with the normal locked sync.", file=sys.stderr)
        return runner.uv_sync(uv, token, extras=feature_extras, root=root)

    overrides_path = paths.sync_overrides_file(root)
    installed = sync_plan.installed_versions(paths.venv_dir(root))
    overrides_text = sync_plan.held_overrides_text(changes, installed)
    holdable = False
    if overrides_text is not None and _write_overrides(overrides_path, overrides_text):
        holdable = (
            runner.uv_sync_held(
                uv,
                token,
                overrides_path=overrides_path,
                extras=feature_extras,
                root=root,
                dry_run=True,
            )
            == 0
        )

    plan = sync_plan.build_plan(
        changes,
        holdable=holdable,
        cache_dir=_effective_cache_dir(root),
        cache_is_owned=_cache_is_owned(root),
        free_disk_bytes=sync_plan.free_bytes(root),
    )
    print(sync_plan.format_sync_plan(plan))

    action = sync_plan.decide(
        plan,
        hold_requested=options.hold,
        headless=_is_headless(),
        headless_policy=options.headless_policy,
        confirm_threshold_bytes=options.confirm_threshold_bytes,
        interactive=consent.is_interactive(),
    )
    if action == "abort":
        print("Sync cancelled; keeping the current environment.")
        return 1
    if action == "hold":
        print("Limping along: holding torch/torchvision at the installed version; updating everything else.")
        return runner.uv_sync_held(uv, token, overrides_path=overrides_path, extras=feature_extras, root=root)
    return runner.uv_sync(uv, token, extras=feature_extras, root=root)


def _write_overrides(path: Path, text: str) -> bool:
    """Write the uv override file used to hold packages; return False (skip the hold) if it cannot be written."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        return False
    return True


def _reconcile_backend(root: Path, token: str) -> str:
    """Clamp the resolved token to the live GPU's arch window so an unrunnable build is never installed.

    A persisted ``bin/backend`` (or the cu126 default) can name a build with no kernel image for the
    card (a cu126 token on a Blackwell GPU, say), which torch only rejects at the first kernel launch
    ("no CUDA kernels for this GPU"). When the live capability says the token cannot run, swap to the
    build that can and re-persist it so the correction sticks across future syncs instead of reasserting
    the broken token on every update. A non-CUDA token, or an unreadable capability, is left untouched.
    """
    reconciled = detect.reconcile_backend_for_gpu(token)
    if reconciled == token:
        return token
    print(
        f"Adjusting the torch build from {token} to {reconciled}: this GPU's compute capability has no "
        f"kernel image in the {token} wheel, so {token} would install but fail at the first kernel launch. "
        f"Installing {reconciled} instead (and recording it for future updates).",
        file=sys.stderr,
    )
    backend_mod.write_backend_file(paths.backend_file(root), reconciled)
    return reconciled


def _verify_installed_torch_arch(uv: str, root: Path, token: str) -> None:
    """Compare the freshly-installed torch wheel's kernels against the live GPU, warning on a true mismatch.

    Closes the loop on the *prediction* the build selection makes: the build was chosen on the belief it
    has kernels for this GPU, so if the installed wheel's own arch list proves otherwise, the table in
    :mod:`worker_bootstrap.detect` is out of date for this card. That is a worker bug, not something the
    user can fix by reinstalling the same build, so it is surfaced here as a maintainer-actionable report
    (with a user stopgap) rather than left to the worker's later generic "unsupported GPU" runtime fault.

    Best-effort and never fatal: the install already succeeded on disk. A genuinely too-old driver is a
    different case and is not flagged here -- the wheel would still list the card's architecture, so this
    fires only on a real kernel-image gap.
    """
    arch_list = runner.query_torch_arch_list(uv, root=root)
    if not arch_list or not any(arch.startswith("sm_") for arch in arch_list):
        return  # torch absent, or a CPU/ROCm build whose arch tags do not apply
    capability = detect.live_compute_capability()
    if capability == (0, 0) or detect.gpu_arch_supported(arch_list, capability):
        return
    cap_tag = f"sm_{capability[0]}{capability[1]}"
    print(
        f"WARNING: the torch build just installed ({token}) has no CUDA kernels for this GPU "
        f"(compute capability {capability[0]}.{capability[1]}, {cap_tag}); the wheel was built for "
        f"{' '.join(arch_list)}. This build was selected believing it would run this GPU, so this is "
        f"most likely a worker bug: the build-selection table in worker_bootstrap.detect is out of date "
        f"for {cap_tag}. Please report it (quoting this message) at "
        f"https://github.com/Haidra-Org/horde-worker-reGen/issues . As a stopgap, forcing a newer build "
        f"with HORDE_WORKER_BACKEND (for example cu132) may work if one carries kernels for your card.",
        file=sys.stderr,
    )


def _sync(uv: str, root: Path, *, cli_flag: str | None, options: _SyncOptions) -> int:
    """Disclose, gain consent, ensure git, seed config, then run the sync (with preview) or ROCm path."""
    token = backend_mod.resolve_backend(
        cli_flag=cli_flag,
        env_value=os.environ.get(_BACKEND_ENV),
        file_value=backend_mod.read_backend_file(paths.backend_file(root)),
        # Detect here too (not only at install/detect time): an absent bin/backend must pick the build
        # this machine can actually run rather than blindly defaulting to cu126.
        detected=detect.detect_backend(),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2
    # Belt-and-suspenders: detection can be bypassed by a stale persisted token or a forced override, so
    # cross-check whatever was resolved against the live GPU and clamp an unrunnable build before installing.
    token = _reconcile_backend(root, token)

    # Disclose what is about to be installed (and from where) and gain consent before any heavy download.
    # The git line tells the user up front whether their existing git is used or a portable one is fetched.
    system_git = gitbin.find_system_git()
    if not consent.ensure_consent(
        notice_path=paths.install_notice(root),
        marker_path=paths.consent_marker(root),
        detail_lines=[f"  - GPU backend to install: {token}", gitbin.notice_line(system_git)],
    ):
        return 1

    # Resolve git now (during the long install), not mid-job: hordelib clones ComfyUI with a bare `git`.
    git_resolution = gitbin.ensure_git(root)
    if not git_resolution.ok:
        print(git_resolution.message, file=sys.stderr)
        return 1

    config_seed.seed_config(
        template=paths.template_config(root),
        target=paths.bridge_config(root),
        backend_token=token,
    )
    from worker_bootstrap import rocm

    if rocm.is_rocm_token(token):
        rc = rocm.sync_rocm(uv, root=root, hold=options.hold, token=token)
        if rc == 0:
            _maybe_prune(uv, root, options)
            _write_sync_stamp(root)
        return rc
    try:
        backend_mod.validate_locked_extra(token, paths.pyproject_path(root))
        feature_extras = backend_mod.desired_feature_extras(token, env_value=os.environ.get(_FEATURES_ENV))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    features_note = ", ".join(feature_extras) if feature_extras else "none (lean base)"
    print(f"Installing dependencies for GPU backend: {token} (features: {features_note})")
    rc = _run_sync(uv, root, token, feature_extras, options)
    if rc == 0:
        # Verify the prediction against ground truth now that torch is on disk: an arch mismatch here
        # means the build map is stale, which the user cannot fix by reinstalling the same build.
        _verify_installed_torch_arch(uv, root, token)
        _maybe_prune(uv, root, options)
        _write_sync_stamp(root)
    return rc


def _lock_fingerprint(root: Path) -> str:
    """Return a content fingerprint of ``uv.lock``, or ``""`` when it cannot be read.

    The lockfile pins every resolved version a sync installs, so its hash is what distinguishes an
    install whose dependencies have moved (an in-place update overlaid a new lock) from one that has not.
    """
    try:
        return hashlib.sha256(paths.lock_path(root).read_bytes()).hexdigest()
    except OSError:
        return ""


def _venv_matches_lock(root: Path) -> bool:
    """Whether the venv's recorded sync stamp matches the current lockfile.

    A missing lock cannot prove staleness, so it is treated as current (never forces a re-sync loop). A
    missing stamp (an install predating this check, or one never synced through here) is treated as stale
    so the venv is reconciled once.
    """
    fingerprint = _lock_fingerprint(root)
    if fingerprint == "":
        return True
    try:
        recorded = paths.sync_stamp_file(root).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == fingerprint


def _write_sync_stamp(root: Path) -> None:
    """Record the current lock fingerprint after a successful sync (best-effort).

    A missing stamp only costs one extra reconcile sync on the next launch, so any write failure is
    swallowed rather than failing an otherwise-successful install.
    """
    fingerprint = _lock_fingerprint(root)
    if fingerprint == "":
        return
    stamp = paths.sync_stamp_file(root)
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(fingerprint, encoding="utf-8")
    except OSError:
        pass


def _ensure_synced(uv: str, root: Path, *, cli_flag: str | None, options: _SyncOptions) -> int:
    """Sync the venv when it is missing or stale relative to ``uv.lock``.

    An in-place update overlays a new lockfile but preserves the existing venv, so a plain launch must
    notice the venv no longer matches the lock and re-sync; otherwise the worker code runs ahead of its
    installed dependencies (the cause of the post-update crash-loop). An unchanged install matches its
    stamp and starts immediately, with no re-resolve.
    """
    if paths.venv_dir(root).exists() and _venv_matches_lock(root):
        return 0
    return _sync(uv, root, cli_flag=cli_flag, options=options)


def _offer_cpu_mode(args: argparse.Namespace, token: str) -> str:
    """Let an interactive user opt into CPU/alchemist-only mode when a GPU build was auto-detected.

    A backend chosen explicitly (a ``--backend`` flag or ``HORDE_WORKER_BACKEND``) is respected as-is; a
    non-interactive run never prompts. See :func:`worker_bootstrap.backend.choose_backend_interactively`.
    """
    explicitly_chosen = bool(args.backend) or bool(os.environ.get(_BACKEND_ENV))
    return backend_mod.choose_backend_interactively(
        token,
        explicitly_chosen=explicitly_chosen,
        interactive=consent.is_interactive(),
    )


def _cmd_detect(args: argparse.Namespace, root: Path, uv: str) -> int:  # noqa: ARG001  (uv unused here)
    """Detect (and optionally persist) the backend token, honouring a flag/env override."""
    token = backend_mod.resolve_backend(
        cli_flag=args.backend,
        env_value=os.environ.get(_BACKEND_ENV),
        detected=detect.detect_backend(),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2
    token = _offer_cpu_mode(args, token)
    if token == detect.CPU:
        _print_cpu_notice()
    if args.write:
        backend_mod.write_backend_file(paths.backend_file(root), token)
    print(token)
    return 0


def _cmd_sync(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Install/update dependencies for the resolved backend."""
    _apply_cache_mode_flag(args)
    return _sync(uv, root, cli_flag=args.backend, options=_sync_options(args))


def _maybe_offer_update(root: Path) -> None:
    """On launch, check for a newer release and (per policy) apply it before syncing/starting.

    Best-effort and never fatal: a failed check or a declined prompt simply proceeds with the current
    install. ``off`` skips entirely; ``auto`` applies without asking; ``prompt`` (default) asks on an
    interactive run and only notifies (never blocks) when headless, so an unattended/service start is
    never wedged waiting on input. The applied overlay changes ``uv.lock`` when dependencies moved, which
    the subsequent lock-aware sync then installs before the worker starts.

    The self-applier bows out for installs whose updates are owned elsewhere (winget, a git checkout): the
    in-worker notifier still tells the user how to update those. To avoid a launch being gated on the GitHub
    API every time, the check is throttled, and a version the user explicitly skipped is not re-offered
    until a newer one is released.
    """
    policy = updater.auto_update_policy()
    if policy == "off":
        return
    allowed, _reason = updater.self_update_allowed(root)
    if not allowed:
        return
    if not updater.should_check_now(root):
        return
    try:
        info = updater.check_for_update(root)
    except Exception as error:  # noqa: BLE001 - a launch must never fail because the update check did
        print(f"Update check skipped: {error}", file=sys.stderr)
        return
    updater.record_check(root)
    if not info.available or info.latest is None:
        return
    if updater.is_version_skipped(root, info.latest):
        return

    kind = "beta update" if info.is_prerelease else "update"
    if policy == "prompt":
        if not consent.is_interactive():
            print(
                f"A worker {kind} is available ({info.current} -> {info.latest}). Run `update` to apply, "
                "or set HORDE_WORKER_AUTO_UPDATE=auto.",
                file=sys.stderr,
            )
            return
        answer = (
            input(f"A worker {kind} is available ({info.current} -> {info.latest}). Update now? [Y/n/skip] ")
            .strip()
            .lower()
        )
        if answer in ("s", "skip"):
            updater.mark_version_skipped(root, info.latest)
            print(f"Skipping {info.latest}; you won't be asked again until a newer version is released.")
            return
        if answer in ("n", "no"):
            print("Skipping the update for this launch.")
            return

    print(f"Updating to {info.latest} ...")
    result = updater.perform_update(root, info)
    if result.ok:
        print(result.message)
        updater.clear_skip(root)
        updater.sync_arp_version(root, info.latest)
    else:
        print(f"Update skipped: {result.message}", file=sys.stderr)


def _cmd_launch(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Start the worker in the requested mode, syncing first if the venv is missing or stale."""
    _maybe_offer_update(root)
    rc = _ensure_synced(uv, root, cli_flag=args.backend, options=_sync_options(args))
    if rc != 0:
        return rc
    if args.mode == "bridge":
        rc = runner.uv_run(uv, ["python", "-s", "download_models.py"], root=root)
        if rc != 0:
            return rc
        return runner.uv_run(uv, ["python", "-s", "run_worker.py", *args.rest], root=root)
    return runner.uv_run(uv, [*_LAUNCH_COMMANDS[args.mode], *args.rest], root=root)


def _cmd_preload(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Download/verify models, then exit."""
    rc = _ensure_synced(uv, root, cli_flag=None, options=_sync_options(args))
    if rc != 0:
        return rc
    return runner.uv_run(uv, ["python", "-s", "download_models.py"], root=root)


def _cmd_run(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Run an arbitrary command in the worker venv (uv run --no-sync), for back-compat passthrough."""
    return runner.uv_run(uv, list(args.rest), root=root)


def _cmd_update(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Check for, and (unless ``--check``) apply, the latest release in place, then re-sync dependencies.

    Unlike the launch-time offer, this ignores the skip/throttle state (running ``update`` is itself the
    intent to update now). It refuses to overlay an install whose updates are owned elsewhere (winget, a
    git checkout), but ``--check`` still reports availability for those.
    """
    repo_override: str | None = getattr(args, "repo", None) or None

    if args.check:
        info = updater.check_for_update(root, repo=repo_override)
        channel_note = " (beta channel)" if info.channel == "beta" else ""
        if info.latest is None:
            print("Could not determine the latest version (the update check failed).", file=sys.stderr)
            return 1
        if info.available:
            print(f"Update available{channel_note}: {info.current} -> {info.latest}")
        else:
            print(f"Up to date ({info.current}){channel_note}.")
        return 0

    # Gate before the network call on the apply path: an install whose updates are owned elsewhere is
    # refused regardless of what is available.
    allowed, reason = updater.self_update_allowed(root)
    if not allowed:
        print(reason, file=sys.stderr)
        return 1

    info = updater.check_for_update(root, repo=repo_override)
    channel_note = " (beta channel)" if info.channel == "beta" else ""
    if info.latest is None:
        print("Could not check for updates; leaving the current install unchanged.", file=sys.stderr)
        return 1

    # Persist a --repo override as soon as the repo proves reachable (info.latest is not None), whether or
    # not an update is actually available. This covers the "switch to official channel" case where both repos
    # are at the same version: the user still wants future plain `update` runs to use the new origin.
    if repo_override:
        updater.write_repo_to_install_info(root, repo_override)

    if not info.available:
        msg = f"Already up to date ({info.current}){channel_note}."
        if repo_override:
            msg += f" Update origin set to {repo_override}."
        print(msg)
        return 0

    if not args.yes and consent.is_interactive():
        answer = input(f"Update {info.current} -> {info.latest}{channel_note}? [Y/n] ")
        if answer.strip().lower() in ("n", "no"):
            print("Update cancelled.")
            return 1

    result = updater.perform_update(root, info)
    if not result.ok:
        print(f"Update failed: {result.message}", file=sys.stderr)
        return 1
    updater.clear_skip(root)
    updater.sync_arp_version(root, info.latest)

    # The overlay may have moved uv.lock; reconcile the venv now so the next launch starts on the new deps.
    # Hold the success line until the reconcile lands so a failed sync is not reported as a clean update
    # (the overlay invalidated the sync stamp, so the next launch retries the sync regardless).
    rc = _sync(uv, root, cli_flag=None, options=_sync_options(args))
    if rc != 0:
        print(
            f"{result.message} The dependency sync did not complete; the next launch will retry it.",
            file=sys.stderr,
        )
        return rc
    print(result.message)
    return 0


def _cmd_apply_bundle(args: argparse.Namespace, root: Path, uv: str) -> int:  # noqa: ARG001  (uv unused here)
    """Overlay an already-extracted release bundle onto this install, mirror-pruning the import roots.

    The one-line installers extract the downloaded release to a temp dir, copy it into place (so the shims
    and this bootstrap exist to run), then call this so a reinstall over an older version applies through
    the SAME pruning overlay the self-updater uses, rather than a plain unzip that would leave a renamed or
    removed module behind to shadow the new code. Stdlib-only, so it runs in the bare bootstrap environment
    before the venv exists.
    """
    bundle = Path(args.bundle).expanduser()
    if not bundle.is_dir():
        print(f"ERROR: bundle directory not found: {bundle}", file=sys.stderr)
        return 1
    try:
        updater.apply_bundle(bundle, root)
    except OSError as error:
        print(f"ERROR: could not apply the release bundle: {error}", file=sys.stderr)
        return 1
    print(f"Applied the release bundle to {root} (stale modules pruned from the import roots).")
    return 0


def _cmd_install(args: argparse.Namespace, root: Path, uv: str) -> int:
    """One-shot first run: detect + persist backend, sync, then launch the web dashboard."""
    token = backend_mod.resolve_backend(
        cli_flag=args.backend,
        env_value=os.environ.get(_BACKEND_ENV),
        detected=detect.detect_backend(),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2
    token = _offer_cpu_mode(args, token)
    if token == detect.CPU:
        _print_cpu_notice()
    backend_mod.write_backend_file(paths.backend_file(root), token)
    _apply_cache_mode_flag(args)
    rc = _sync(uv, root, cli_flag=token, options=_sync_options(args))
    if rc != 0 or args.no_launch:
        return rc
    return runner.uv_run(uv, _LAUNCH_COMMANDS["web"], root=root)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with one subcommand per bootstrap action."""
    parser = argparse.ArgumentParser(prog="bootstrap.py", description="AI Horde Worker bootstrap.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend_flag(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--backend",
            default=None,
            help=(
                "Force a torch build (cu126/cu130/cu132/cpu/rocm/rocm-windows) instead of "
                "detecting/reading bin/backend."
            ),
        )
        # Convenience shortcuts kept for back-compat with the old update-runtime.cmd/sh flag interface
        # (e.g. `update-runtime.cmd --cu126`); each is just `--backend <build>`.
        for build in ("cu126", "cu130", "cu132", "cpu", "rocm", "rocm-windows"):
            target.add_argument(
                f"--{build}",
                dest="backend",
                action="store_const",
                const=build,
                help=f"Shortcut for --backend {build}.",
            )

    def add_sync_flags(target: argparse.ArgumentParser) -> None:
        """Add the preview/hold/prune/cache-mode knobs (CLI flag > env var > default)."""
        target.add_argument(
            "--no-sync-preview",
            action="store_true",
            help="Skip the pre-sync download preview (env: HORDE_WORKER_SYNC_PREVIEW=0).",
        )
        target.add_argument(
            "--hold-torch",
            dest="hold_torch",
            action="store_const",
            const=True,
            default=None,
            help="Limp along: keep the installed torch/torchvision when no dependency requires the upgrade.",
        )
        target.add_argument(
            "--no-hold-torch",
            dest="hold_torch",
            action="store_const",
            const=False,
            help="Always take torch/torchvision upgrades (opposite of --hold-torch).",
        )
        target.add_argument(
            "--confirm-above-mb",
            type=int,
            default=None,
            help=f"Confirm before downloads larger than N MB on an interactive run (default {_DEFAULT_CONFIRM_MB}).",
        )
        target.add_argument(
            "--headless-policy",
            choices=["proceed", "hold"],
            default=None,
            help="Non-interactive behaviour for big optional upgrades: take them (proceed) or hold (default proceed).",
        )
        target.add_argument(
            "--no-prune",
            action="store_true",
            help="Do not auto-prune the worker's owned uv cache after a successful sync.",
        )
        target.add_argument(
            "--cache-mode",
            choices=["isolated", "shared"],
            default=None,
            help="Use the isolated worker cache (default) or uv's shared cache (env HORDE_WORKER_UV_CACHE_MODE).",
        )

    p_detect = sub.add_parser("detect", help="Detect the GPU/torch build for this machine.")
    add_backend_flag(p_detect)
    p_detect.add_argument("--write", action="store_true", help="Persist the result to bin/backend.")

    p_sync = sub.add_parser("sync", help="Install/update dependencies for the selected build.")
    add_backend_flag(p_sync)
    add_sync_flags(p_sync)

    p_launch = sub.add_parser("launch", help="Start the worker (syncing first if needed).")
    p_launch.add_argument("mode", choices=["web", "terminal", "bridge", "host", "benchmark"])
    add_backend_flag(p_launch)
    # No sync flags here: launch only syncs on first run (no venv), where the preview is skipped anyway,
    # and argparse.REMAINDER below would otherwise swallow them as worker passthrough.
    p_launch.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments passed through to the worker.")

    sub.add_parser("preload", help="Download/verify models, then exit.")

    p_install = sub.add_parser("install", help="Detect, sync, and launch (one-shot first run).")
    add_backend_flag(p_install)
    add_sync_flags(p_install)
    p_install.add_argument("--no-launch", action="store_true", help="Install only; do not start the worker.")

    p_run = sub.add_parser("run", help="Run an arbitrary command in the worker venv (uv run --no-sync).")
    p_run.add_argument("rest", nargs=argparse.REMAINDER, help="The command and its arguments.")

    p_apply = sub.add_parser(
        "apply-bundle",
        help="Overlay an extracted release bundle onto this install, pruning stale modules (installer use).",
    )
    p_apply.add_argument("bundle", help="Path to the extracted release bundle directory.")

    p_update = sub.add_parser("update", help="Update the worker to the latest release in place, then re-sync.")
    p_update.add_argument("--check", action="store_true", help="Only report whether an update is available.")
    p_update.add_argument("--yes", action="store_true", help="Apply without prompting (for non-interactive use).")
    p_update.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/REPO",
        help=(
            "Pull releases from this repo instead of the recorded origin (env: HORDE_WORKER_UPDATE_REPO). "
            "Persisted to bin/install-info on a successful check so future updates use the same origin "
            "without the flag. Use this to switch from a beta fork to the official release channel or "
            "vice versa (e.g. --repo Haidra-Org/horde-worker-reGen)."
        ),
    )

    return parser


_HANDLERS = {
    "detect": _cmd_detect,
    "sync": _cmd_sync,
    "launch": _cmd_launch,
    "preload": _cmd_preload,
    "install": _cmd_install,
    "run": _cmd_run,
    "update": _cmd_update,
    "apply-bundle": _cmd_apply_bundle,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the matching bootstrap action; return its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Back-compat: the old runtime.cmd/.sh were a generic `uv run` wrapper, so things like
    # `runtime.cmd python -s download_models.py` (and the Dockerfiles README) pass a bare command. If the
    # first token is not a known subcommand (and not a flag), treat the whole line as `run <command...>`.
    if argv and not argv[0].startswith("-") and argv[0] not in _HANDLERS:
        argv = ["run", *argv]
    args = _build_parser().parse_args(argv)
    root = paths.install_root()
    uv = uvbin.uv_executable(root)
    return _HANDLERS[args.command](args, root, uv)
