"""The main entry point for the reGen worker."""

import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import contextlib
import dataclasses
import io
import multiprocessing
import os
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from typing import override

import regex as re
from loguru import logger


def main(
    ctx: BaseContext,
    load_from_env_vars: bool = False,
    *,
    amd_gpu: bool = False,
    directml: int | None = None,
    supervisor_connection: Connection | None = None,
) -> None:
    """Check for a valid config and start the driver ('main') process for the reGen worker."""
    from pydantic import ValidationError

    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, reGenBridgeData
    from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME
    from horde_worker_regen.process_management.main_entry_point import start_working
    from horde_worker_regen.reference_helper import ensure_model_reference_manager_initialized

    horde_model_reference_manager = ensure_model_reference_manager_initialized()

    bridge_data: reGenBridgeData | None = None
    try:
        if load_from_env_vars:
            bridge_data = BridgeDataLoader.load_from_env_vars(
                horde_model_reference_manager=horde_model_reference_manager,
            )
            if len(bridge_data.api_key) == 10:
                logger.error(
                    "API key is the default. This is almost certainly not what you want. "
                    "Please check your environment variables are being set correctly and try again.",
                )

                logger.error("Exiting...")
                return
        else:
            bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=horde_model_reference_manager,
            )
    except ConnectionRefusedError:
        logger.error("Could not connect to the the horde. Is it down?")
        input("Press any key to exit...")
        return
    except Exception as e:
        logger.exception(e)

        if "No such file or directory" in str(e):
            logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

        if isinstance(e, ValidationError):
            # Print a list of fields that failed validation
            logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
            for error in e.errors():
                logger.error(f"{error['loc'][0]}: {error['msg']}")

        input("Press any key to exit...")
        return

    if not bridge_data:
        logger.error("Failed to load bridge data. Exiting...")
        return

    bridge_data.load_env_vars()

    start_working(
        ctx=ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=horde_model_reference_manager,
        amd_gpu=amd_gpu,
        directml=directml,
        supervisor_connection=supervisor_connection,
    )

    logger.info("Worker has finished working.")
    logger.info("Exiting...")


class LogConsoleRewriter(io.StringIO):
    """Makes the console output more readable by shortening certain strings."""

    def __init__(self, original_iostream: io.TextIOBase) -> None:
        """Initialise the rewriter."""
        self.original_iostream = original_iostream

        pattern = r"\[36m(\d+)"

        self.line_number_pattern = re.compile(pattern)

    @override
    def write(self, message: str) -> int:
        """Rewrite the message to make it more readable where possible."""
        replacements = [
            ("horde_worker_regen.process_management.process_manager", "*"),
            ("horde_worker_regen.", "[HWR]"),
            ("print_status_method", ""),
            ("receive_and_handle_process_messages", "[ % ]"),
            ("print_status_method", "[ i ]"),
            ("start_inference_processes", "[SIP]"),
            ("_start_inference_process", "[SIP]"),
            ("start_inference_process", "[SIP]"),
            ("start_safety_process", "[SSP]"),
            ("start_inference", "[ % ]"),
            ("log_kudos_info", "[ i ]"),
            ("submit_single_generation", "[ - ]"),
            ("preload_models", "[ % ]"),
            ("api_job_pop", "[ + ]"),
            ("_process_control_loop", "[ # ]"),
            ("_bridge_data_loop", "[ C ]"),
            ("enable_performance_mode", "[ C ]"),
        ]

        for old, new in replacements:
            message = message.replace(old, new)

        replacement = ""

        message = self.line_number_pattern.sub(replacement, message)

        if self.original_iostream is None:
            raise ValueError("self.original_iostream. is None!")

        return self.original_iostream.write(message)

    @override
    def flush(self) -> None:
        """Flush the buffer to the original stdout."""
        self.original_iostream.flush()


@dataclasses.dataclass
class WorkerLaunchOptions:
    """Explicit worker launch options, decoupled from argparse so a supervisor can pass them directly."""

    verbosity: int = 0
    no_logging: bool = False
    load_config_from_env_vars: bool = False
    amd: bool = False
    worker_name: str | None = None
    directml: int | None = None


def _redirect_streams_to_file(path: str) -> None:
    """Point this process's stdout/stderr (Python and OS-fd level) at a file.

    The supervised worker is a spawned child that inherits the TUI's terminal, so any console
    output would corrupt the Textual UI. The TUI surfaces worker output by tailing logs/bridge*.log
    instead, so here we send the raw streams to a file. ``os.dup2`` also captures C-level writes
    and loguru's ``sys.__stdout__`` console sink; if it is unavailable we fall back to the
    Python-level reassignment alone.
    """
    from pathlib import Path

    Path("logs").mkdir(exist_ok=True)
    # Intentionally kept open for the process lifetime: it backs fds 1/2 via dup2 below.
    stream = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    with contextlib.suppress(Exception):
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    sys.stdout = stream
    sys.stderr = stream


def _record_worker_start_in_app_state() -> None:
    """Record that a worker session is starting on this version, best-effort.

    Stamps ``worker_version_last_ran`` in the durable app state so a later version bump can mark a
    stale benchmark for re-running. Any failure is swallowed (logged at debug): app-state bookkeeping
    must never block worker startup.
    """
    try:
        from horde_worker_regen import __version__
        from horde_worker_regen.app_state import AppStateStore

        AppStateStore().record_worker_started(worker_version=__version__)
    except Exception as app_state_error:  # noqa: BLE001 - app-state must never block worker startup
        logger.debug(f"Could not record worker start in app state: {app_state_error}")


def _run_release_update_check() -> None:
    """Look up the latest release and log whether the worker is current (the blocking body of the check).

    When a newer release is found it is recorded in ``AIWORKER_NEWER_RELEASE_AVAILABLE`` so the periodic
    status report can re-nag. Any failure is swallowed: an update check must never affect the worker.
    """
    from horde_worker_regen.update_check import NEWER_RELEASE_ENV_VAR, check_for_update, current_version

    try:
        info = check_for_update()
    except Exception as update_error:  # noqa: BLE001 - an update check must never affect the worker
        logger.debug(f"Release update check failed: {update_error}")
        return
    if info is None:
        logger.info(f"Worker v{current_version()} is up to date.")
        return
    os.environ[NEWER_RELEASE_ENV_VAR] = info.latest_version
    logger.warning(
        f"Update available: v{current_version()} -> v{info.latest_version}. Update with "
        "'winget upgrade Haidra.HordeWorker', or re-run the installer (the same install command).",
    )


def _start_release_update_check() -> None:
    """Log, off the startup path, whether a newer worker release exists (best-effort, non-blocking).

    The GitHub release lookup is a network call, so it runs on a daemon thread to keep worker startup
    instant and offline-safe. This is the headless/console counterpart to the dashboard's update
    notification; both share :mod:`horde_worker_regen.update_check`, and neither touches
    ``_version_meta.json`` (which exists only for the operator-controlled hard minimum-version gate).
    """
    import threading

    from horde_worker_regen.update_check import update_check_disabled

    if update_check_disabled():
        return

    threading.Thread(target=_run_release_update_check, name="release-update-check", daemon=True).start()


def _log_benchmark_hint() -> None:
    """Log a one-time, non-blocking hint when no current benchmark exists for this worker version.

    The interactive onboarding lives in the TUI; for a headless/container/service worker this keeps startup
    unattended while still pointing the operator at the benchmark. Any failure is swallowed (logged at debug).
    """
    try:
        from horde_worker_regen import __version__
        from horde_worker_regen.app_state import AppStateStore, BenchmarkAvailability, benchmark_status_summary

        availability = benchmark_status_summary(AppStateStore().load(), current_version=__version__)
        if availability is BenchmarkAvailability.CURRENT:
            return
        preface = "No benchmark on record" if availability is BenchmarkAvailability.NONE else "Benchmark is stale"
        logger.info(
            f"{preface} for worker v{__version__}. Run 'horde-benchmark ramp' (or launch the TUI with "
            "'horde-worker') to benchmark and auto-tune this worker.",
        )
    except Exception as hint_error:  # noqa: BLE001 - the hint must never block worker startup
        logger.debug(f"Could not emit benchmark hint: {hint_error}")


def _prepare_runtime(options: WorkerLaunchOptions, *, supervised: bool = False) -> None:
    """Shared worker pre-flight: spawn method, env vars, version check, telemetry, and logging.

    Args:
        options: The launch options.
        supervised: When True, the worker was launched by the TUI over a pipe. Console output is
            redirected to a file (so it cannot corrupt the TUI) and console verbosity is minimised;
            the loguru file sinks (logs/bridge*.log) are unaffected and remain the TUI's log source.
    """
    with contextlib.suppress(Exception):
        multiprocessing.set_start_method("spawn", force=True)

    if supervised:
        # Redirect before anything prints, so not even the start-method banner leaks to the TUI.
        _redirect_streams_to_file("logs/bridge_main_console.log")

    if os.path.exists(".abort"):
        with logger.catch(reraise=True):
            os.remove(".abort")
            logger.debug("Removed .abort file")

    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

    os.environ["HORDE_SDK_DISABLE_CUSTOM_SINKS"] = "1"

    # Spawned workers can't see the CLI -v count, so pass the operator's verbosity intent down
    # via env (inherited across the spawn). Workers floor it at DEBUG; setdefault lets an
    # explicitly-exported value win. See worker_entry_points.resolve_worker_log_verbosity.
    from horde_worker_regen.process_management.worker_entry_points import WORKER_LOG_VERBOSITY_ENV

    os.environ.setdefault(WORKER_LOG_VERBOSITY_ENV, str(options.verbosity))

    if options.worker_name:
        os.environ["AIWORKER_DREAMER_WORKER_NAME"] = options.worker_name

    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    if not options.load_config_from_env_vars:
        # Note: 'load_env_vars_from_config' means to translate the config file to environment variables
        # if 'load_config_from_env_vars' is True, then we are ignoring the config file
        load_env_vars_from_config()

    from horde_worker_regen.version_meta import do_version_check

    do_version_check()

    _start_release_update_check()

    _record_worker_start_in_app_state()

    if not supervised:
        rewriter_stdout = LogConsoleRewriter(sys.stdout)  # type: ignore
        sys.stdout = rewriter_stdout

        rewriter_stderr = LogConsoleRewriter(sys.stderr)  # type: ignore
        sys.stderr = rewriter_stderr

    # OpenTelemetry tracing is opt-in only (AIWORKER_REGEN_ENABLE_TELEMETRY). Force it off here —
    # before any hordelib import and before worker processes are spawned — so the kill switch is
    # inherited by every child. Left on, hordelib's per-ComfyUI-op spans starve the inference loop
    # and depress GPU duty cycle even with no collector running. See telemetry.py.
    from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

    claim_logfire_ownership()
    enforce_telemetry_default_off()
    # configure_telemetry()  # opt-in: only call when AIWORKER_REGEN_ENABLE_TELEMETRY is set

    AIWORKER_LIMITED_CONSOLE_MESSAGES = os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES")

    logger.remove()
    # From the torch-free ``utils.logger`` submodule, not the ``hordelib.api`` facade: this runs in the
    # long-lived main/orchestrator process, and the facade would drag torch (~500MB) into it at startup.
    from hordelib.utils.logger import HordeLog

    target_verbosity = options.verbosity

    if supervised:
        target_verbosity = 0  # Console is redirected to a file; keep it terse (file sinks unaffected).
    elif AIWORKER_LIMITED_CONSOLE_MESSAGES:
        if target_verbosity > 2:
            print(
                "Warning: AIWORKER_LIMITED_CONSOLE_MESSAGES is set"
                " but verbosity is set to 3 or higher. Setting verbosity to 2.",
            )

        target_verbosity = 2
    elif options.no_logging:
        target_verbosity = 0  # Disable logging to the console
    elif options.verbosity == 0:
        target_verbosity = 3  # Default to INFO or higher (Warning, Error, Critical)

    # Initialise logging with loguru
    HordeLog.initialise(
        setup_logging=True,
        process_id=None,
        verbosity_count=target_verbosity,
    )

    if not supervised:
        # The TUI shows an interactive onboarding modal instead; a headless worker just gets a hint.
        _log_benchmark_hint()


def init() -> None:
    """Initialise the worker from CLI args and run it (the headless entry point)."""
    # Create args for -v, allowing -vvv
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity of output")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging to the console")
    parser.add_argument(
        "-e",
        "--load-config-from-env-vars",
        action="store_true",
        default=False,
        help="Load the config only from environment variables. This is useful for running the worker in a container.",
    )
    parser.add_argument(
        "--amd",
        "--amd-gpu",
        action="store_true",
        default=False,
        help="Enable AMD GPU-specific optimisations",
    )
    parser.add_argument(
        "-n",
        "--worker-name",
        type=str,
        default=None,
        help="Override the worker name from the config file, for running multiple workers on one machine",
    )
    parser.add_argument(
        "--directml",
        type=int,
        default=None,
        help="Enable directml and specify device to use.",
    )

    args = parser.parse_args()

    options = WorkerLaunchOptions(
        verbosity=args.v,
        no_logging=args.no_logging,
        load_config_from_env_vars=args.load_config_from_env_vars,
        amd=args.amd,
        worker_name=args.worker_name,
        directml=args.directml,
    )

    from horde_worker_regen.process_management.child_crash_capture import (
        enable_child_faulthandler,
        write_startup_crash,
    )

    # Headless prints to the terminal, but mirror the supervised path so a startup crash also leaves a
    # discoverable bridge_main_startup.log (and faulthandler trace) instead of only terminal scrollback.
    enable_child_faulthandler("main")
    try:
        _prepare_runtime(options)
        # We only need to download the legacy DBs once, so we do it here instead of in the worker processes
        main(
            multiprocessing.get_context("spawn"),
            options.load_config_from_env_vars,
            amd_gpu=options.amd,
            directml=options.directml,
        )
    except Exception as worker_error:
        with contextlib.suppress(Exception):
            logger.exception("The worker crashed.")
        write_startup_crash("main", worker_error)
        raise


def run_supervised(supervisor_connection: Connection, options: WorkerLaunchOptions) -> None:
    """Worker entry point used by the TUI supervisor (a spawned child of the TUI process).

    Identical to :func:`init` but driven by an explicit options object instead of argv, with the
    console redirected to a file and a supervisor pipe wired in for state snapshots and control.
    This must be a top-level function so it is picklable as a ``multiprocessing`` spawn target.
    """
    from horde_worker_regen.process_management.child_crash_capture import (
        enable_child_faulthandler,
        write_startup_crash,
    )

    # The whole preflight runs before hordelib opens bridge.log, and this child's stderr is the TUI's
    # discarded one, so without this an early crash leaves no log at all. Capture it before anything risky.
    enable_child_faulthandler("main")
    try:
        _prepare_runtime(options, supervised=True)
        main(
            multiprocessing.get_context("spawn"),
            options.load_config_from_env_vars,
            amd_gpu=options.amd,
            directml=options.directml,
            supervisor_connection=supervisor_connection,
        )
    except Exception as worker_error:
        # Route the crash through loguru (so it reaches bridge.log once a sink exists) and, regardless,
        # to the loguru-independent startup file (so the no-sink preflight window is never silent).
        with contextlib.suppress(Exception):
            logger.exception("The supervised worker crashed.")
        write_startup_crash("main", worker_error)
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    init()
