import asyncio
from multiprocessing.context import BaseContext

from horde_model_reference.model_reference_manager import ModelReferenceManager
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


def start_working(
    ctx: BaseContext,
    bridge_data: reGenBridgeData,
    horde_model_reference_manager: ModelReferenceManager,
    *,
    amd_gpu: bool = False,
    directml: int | None = None,
    use_tui: bool = False,
) -> None:
    """Create and start process manager."""
    process_manager = HordeWorkerProcessManager(
        ctx=ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=horde_model_reference_manager,
        amd_gpu=amd_gpu,
        directml=directml,
    )

    if use_tui:
        try:
            from horde_worker_regen.tui import HordeWorkerTUI

            logger.info("Starting worker with Textual UI...")

            # Create and run the TUI app
            app = HordeWorkerTUI(process_manager)

            # Integrate with loguru to capture logs
            app.integrate_with_loguru()

            # Run the TUI alongside the worker
            asyncio.run(app.run_async_with_worker())

        except ImportError:
            logger.error(
                "Textual UI requested but textual package is not installed. "
                "Install it with: pip install textual"
            )
            logger.info("Falling back to standard console mode...")
            process_manager.start()
        except Exception as e:
            logger.error(f"Failed to start TUI: {e}")
            logger.exception(e)
            logger.info("Falling back to standard console mode...")
            process_manager.start()
    else:
        process_manager.start()
