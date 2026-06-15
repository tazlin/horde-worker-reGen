"""Length-prefixed JSON framing for the worker-host socket.

The TUI normally owns the worker over a ``multiprocessing`` pipe. In browser/served mode that does not
work: ``textual-serve`` runs a fresh TUI subprocess per browser session, so the worker must outlive any
one session. The [`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost] owns the single worker and
serves its state over a localhost socket; each TUI session attaches as a client. This module is the wire
format they share.

Frames are a 4-byte big-endian length followed by a UTF-8 JSON object with a ``type`` field. The payloads
reuse the already-JSON-round-trippable channel models
([`WorkerStateSnapshot`][horde_worker_regen.process_management.supervisor_channel.WorkerStateSnapshot],
[`SupervisorControlMessage`][horde_worker_regen.process_management.supervisor_channel.SupervisorControlMessage]),
so the snapshot/command schema stays defined in one place.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

from horde_worker_regen.process_management.supervisor_channel import (
    SUPERVISOR_PROTOCOL_VERSION,
    SupervisorControlMessage,
    WorkerStateSnapshot,
)

DEFAULT_HOST_ADDRESS = "127.0.0.1"
DEFAULT_HOST_PORT = 7717
ATTACH_ENV_VAR = "HORDE_WORKER_ATTACH"
"""``host:port`` of a running worker host; set by the web launcher so the served TUI attaches."""

_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 32 * 1024 * 1024
"""Reject absurd frame lengths so a desync or hostile peer cannot force a huge allocation."""

# Frame ``type`` discriminators.
MSG_HELLO = "hello"
MSG_SNAPSHOT = "snapshot"
MSG_STATUS = "status"
MSG_COMMAND = "command"
MSG_LIFECYCLE = "lifecycle"

# Lifecycle actions a client may request of the host (process-level, distinct from worker commands).
LIFECYCLE_START = "start"
LIFECYCLE_STOP = "stop"
LIFECYCLE_RESTART = "restart"
LIFECYCLE_SHUTDOWN = "shutdown"
"""Stop the worker and the host itself. Sent by the launcher on exit, never by a browser session."""


def encode_frame(message: dict[str, Any]) -> bytes:
    """Serialize a message dict to a length-prefixed JSON frame."""
    payload = json.dumps(message).encode("utf-8")
    return _HEADER.pack(len(payload)) + payload


def _recv_exactly(sock: socket.socket, count: int) -> bytes | None:
    """Read exactly ``count`` bytes, or None if the peer closes the connection first."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict[str, Any] | None:
    """Read one frame from ``sock``; returns the message dict, or None on a clean close.

    Raises ValueError on a malformed/oversized frame so the caller can drop the connection.
    """
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME_BYTES:
        raise ValueError(f"Frame length {length} exceeds the maximum of {_MAX_FRAME_BYTES} bytes")
    payload = _recv_exactly(sock, length)
    if payload is None:
        return None
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Frame payload was not a JSON object")
    return decoded


def send_frame(sock: socket.socket, message: dict[str, Any]) -> None:
    """Encode and send a single frame on ``sock``."""
    sock.sendall(encode_frame(message))


# region message builders


def hello_message() -> dict[str, Any]:
    """The host's greeting, carrying the protocol version the client checks on connect."""
    return {"type": MSG_HELLO, "protocol_version": SUPERVISOR_PROTOCOL_VERSION}


def snapshot_message(snapshot: WorkerStateSnapshot) -> dict[str, Any]:
    """Wrap a worker-state snapshot for the wire."""
    return {"type": MSG_SNAPSHOT, "snapshot": snapshot.model_dump(mode="json")}


def status_message(*, status: str, restart_attempts: int, mode: str, worker_running: bool) -> dict[str, Any]:
    """Wrap the host's supervisor status for the wire."""
    return {
        "type": MSG_STATUS,
        "status": status,
        "restart_attempts": restart_attempts,
        "mode": mode,
        "worker_running": worker_running,
    }


def command_message(command: SupervisorControlMessage) -> dict[str, Any]:
    """Wrap a worker control command (forwarded by the host to the worker)."""
    return {"type": MSG_COMMAND, "command": command.model_dump(mode="json")}


def lifecycle_message(action: str) -> dict[str, Any]:
    """Wrap a process-level lifecycle request (start/stop/restart the worker)."""
    return {"type": MSG_LIFECYCLE, "action": action}


# endregion

# region message parsers


def parse_snapshot(message: dict[str, Any]) -> WorkerStateSnapshot:
    """Reconstruct a snapshot from a received ``snapshot`` frame."""
    return WorkerStateSnapshot.model_validate(message["snapshot"])


def parse_command(message: dict[str, Any]) -> SupervisorControlMessage:
    """Reconstruct a control command from a received ``command`` frame."""
    return SupervisorControlMessage.model_validate(message["command"])


# endregion


def resolve_attach_address(value: str) -> tuple[str, int]:
    """Parse a ``host:port`` string into a ``(host, port)`` tuple (IPv6 literals use ``[addr]:port``)."""
    host, separator, port = value.rpartition(":")
    if not separator:
        raise ValueError(f"Expected host:port, got {value!r}")
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host or DEFAULT_HOST_ADDRESS, int(port)
