"""Textual UI package for Horde Worker reGen.

This package provides an optional terminal user interface for monitoring
and managing the Horde Worker. The TUI displays real-time information about:
- Worker status and uptime
- Process states and resource usage
- Job queues and processing
- Kudos generation and statistics
- Configuration settings
- Live logs
"""

from horde_worker_regen.tui.app import HordeWorkerTUI

__all__ = ["HordeWorkerTUI"]
