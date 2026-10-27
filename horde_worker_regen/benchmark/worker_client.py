"""Connect a normal supervised worker to a local benchmark controller."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import urllib.error
import urllib.request
from multiprocessing.connection import Connection
from typing import cast

from loguru import logger

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorControlMessage,
    WorkerLivenessFrame,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions, run_supervised


def _relay(connection: Connection, url: str, token: str, stopped: threading.Event) -> None:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    while not stopped.wait(0.1):
        try:
            snapshot: WorkerStateSnapshot | None = None
            liveness: WorkerLivenessFrame | None = None
            while connection.poll():
                frame = connection.recv()
                if isinstance(frame, WorkerStateSnapshot):
                    snapshot = frame
                elif isinstance(frame, WorkerLivenessFrame):
                    liveness = frame
            if liveness is not None:
                request = urllib.request.Request(
                    f"{url}/worker-liveness", data=liveness.model_dump_json().encode(), headers=headers, method="POST"
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
            if snapshot is not None:
                request = urllib.request.Request(
                    f"{url}/worker-state", data=snapshot.model_dump_json().encode(), headers=headers, method="POST"
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
            request = urllib.request.Request(f"{url}/worker-commands", headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                commands = json.load(response)
            for raw in commands:
                connection.send(SupervisorControlMessage.model_validate(raw))
        except (OSError, EOFError, urllib.error.URLError) as error:
            logger.warning(f"Benchmark supervisor connection ended: {error}")
            stopped.set()
            # EOF follows the ordinary supervisor-loss path in the worker.
            connection.close()
            return


def main() -> None:
    """Run normal worker startup with the existing supervisor pipe protocol."""
    url = os.environ["HORDE_BENCHMARK_SERVICE_URL"]
    token = os.environ["HORDE_BENCHMARK_SERVICE_TOKEN"]
    ctx = multiprocessing.get_context("spawn")
    worker_connection, controller_connection = ctx.Pipe()
    stopped = threading.Event()
    relay = threading.Thread(target=_relay, args=(controller_connection, url, token, stopped), daemon=True)
    relay.start()
    try:
        run_supervised(cast("Connection", worker_connection), WorkerLaunchOptions())
    finally:
        stopped.set()
        worker_connection.close()
        relay.join(timeout=6)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
