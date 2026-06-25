import sys
import time
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext

from horde_model_reference.model_reference_manager import ModelReferenceManager
from loguru import logger

from horde_worker_regen.app_state import (
    AppStateStore,
    KnownGoodSettings,
    KnownGoodSource,
    WorkerRunRecord,
    compute_config_digest,
)
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.capabilities import coerce_bridge_data_to_capabilities
from horde_worker_regen.process_management.config.worker_identity import WorkerNameConfigError, verify_worker_identity
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

_KNOWN_GOOD_MIN_SESSION_SECONDS = 600.0
"""A session must run at least this long (cleanly, with jobs done) to be trusted as known-good."""

_KNOWN_GOOD_CONFIG_FIELDS = (
    "max_threads",
    "queue_size",
    "max_batch",
    "allow_lora",
    "allow_controlnet",
    "allow_post_processing",
    "models_to_load",
    "alchemist",
)
"""The performance-relevant bridgeData fields captured in a known-good snapshot (mirrors the
benchmark's suggested-bridge-data set, so the two sources of known-good stay comparable)."""


def start_working(
    ctx: BaseContext,
    bridge_data: reGenBridgeData,
    horde_model_reference_manager: ModelReferenceManager,
    *,
    amd_gpu: bool = False,
    directml: int | None = None,
    supervisor_connection: Connection | None = None,
) -> None:
    """Create and start process manager.

    Args:
        ctx: The multiprocessing context to use.
        bridge_data: The validated bridge configuration.
        horde_model_reference_manager: The model reference manager.
        amd_gpu: Whether the GPU is an AMD GPU.
        directml: The directml device id, when applicable.
        supervisor_connection: The worker end of a supervisor pipe when launched by the TUI; None headless.
    """
    # Fail fast on a worker-name misconfiguration before spawning any child processes.
    try:
        verify_worker_identity(bridge_data)
    except WorkerNameConfigError as name_error:
        logger.error(str(name_error))
        sys.exit(1)

    # Disable any advertised feature whose backend packages are not installed (e.g. controlnet /
    # post-processing on a lean non-NVIDIA install) so the worker never pops a job it cannot serve.
    # Covers env-var configs too (which never hot-reload); file reloads are re-coerced in the manager.
    coerce_bridge_data_to_capabilities(bridge_data)

    process_manager = HordeWorkerProcessManager(
        ctx=ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=horde_model_reference_manager,
        amd_gpu=amd_gpu,
        directml=directml,
        supervisor_connection=supervisor_connection,
        enable_background_downloads=True,
    )

    try:
        process_manager.start()
    finally:
        _persist_session_state(process_manager, bridge_data)


def _persist_session_state(process_manager: HordeWorkerProcessManager, bridge_data: reGenBridgeData) -> None:
    """Record this session's outcome (and known-good settings) in durable app state, best-effort.

    Runs in the shutdown ``finally`` so even an exception-driven exit still records the run. Any failure
    here is swallowed (logged at debug): persistence must never turn worker shutdown into a crash.
    """
    try:
        store = AppStateStore()
        run_record = process_manager.build_run_record()
        store.record_worker_finished(run_record)
        if _session_qualifies_as_known_good(run_record):
            store.record_known_good(_known_good_from_bridge_data(bridge_data, run_record.worker_version))
    except Exception as persist_error:  # noqa: BLE001 - persistence must never break worker shutdown
        logger.debug(f"Could not persist session app state: {persist_error}")


def _session_qualifies_as_known_good(run_record: WorkerRunRecord) -> bool:
    """Return whether a session ran cleanly and long enough to trust its configuration."""
    long_enough = (run_record.duration_seconds or 0.0) >= _KNOWN_GOOD_MIN_SESSION_SECONDS
    return run_record.clean_exit and run_record.jobs_submitted > 0 and long_enough


def _known_good_from_bridge_data(bridge_data: reGenBridgeData, worker_version: str) -> KnownGoodSettings:
    """Build a known-good record from the performance-relevant fields of a clean session's config."""
    snapshot = _relevant_config_snapshot(bridge_data)
    return KnownGoodSettings(
        config_digest=compute_config_digest(snapshot),
        config_snapshot=snapshot,
        validated_at=time.time(),
        worker_version=worker_version,
        source=KnownGoodSource.CLEAN_RUN,
    )


def _relevant_config_snapshot(bridge_data: reGenBridgeData) -> dict[str, object]:
    """Return the subset of bridgeData fields that define a configuration's performance profile."""
    snapshot: dict[str, object] = {}
    for field_name in _KNOWN_GOOD_CONFIG_FIELDS:
        if hasattr(bridge_data, field_name):
            snapshot[field_name] = getattr(bridge_data, field_name)
    return snapshot
