"""Tests for the image-utilities lane's parent-side adapter.

These exercise the real :class:`UtilitiesProcessAdapter` against a stdlib ``http.server`` standing in for
the capability service and a fake launcher seam, so no cross-venv subprocess is spawned. They assert the
control-message translation, the state/heartbeat/memory emission, and the liveness behaviour.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from horde_image_utilities.client import HordeImageUtilitiesClient
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeAnnotationResultMessage,
    HordeAnnotatorAvailabilityMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeStartAnnotationControlMessage,
)
from horde_worker_regen.process_management.lifecycle.utilities_adapter import UtilitiesProcessAdapter


def _make_small_png() -> bytes:
    """Return a tiny valid PNG so the rembg client can decode a real image in tests."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


_SMALL_PNG = _make_small_png()
_CANNED_CONTROL_MAP = _SMALL_PNG
_MEMORY_JSON: dict[str, object] = {
    "process_rss_bytes": 123_456_789,
    "torch_allocated_bytes": 64 * 1024 * 1024,
    "torch_reserved_bytes": 128 * 1024 * 1024,
    "torch_device_total_bytes": 24576 * 1024 * 1024,
    "onnxruntime_session_count": 1,
    "loaded_model_names": ["canny"],
}


class _ServiceState:
    """Shared, thread-safe record of what the fake capability HTTP service was asked to do."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.health_ok = True
        self.annotate_status = 200
        self.paths_hit: list[str] = []
        # canny is servable; seg is unavailable (its backend is not portable, 501 today); depth's weights
        # are still missing, so it is not yet servable either.
        self.annotators_payload: list[dict[str, object]] = [
            {"name": "canny", "runtime": "none", "available": True, "weights_present": "present", "loaded": False},
            {"name": "seg", "runtime": "torch", "available": False, "weights_present": "missing", "loaded": False},
            {"name": "depth", "runtime": "torch", "available": True, "weights_present": "missing", "loaded": False},
        ]

    def note(self, path: str) -> None:
        with self.lock:
            self.paths_hit.append(path)

    def hits(self, path: str) -> int:
        with self.lock:
            return self.paths_hit.count(path)


def _make_handler(state: _ServiceState) -> type[BaseHTTPRequestHandler]:
    """Return a request handler bound to the given service state."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - matches parent signature
            return

        def _json(self, status: int, body: Mapping[str, object]) -> None:
            import json

            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - http.server dispatch name
            state.note(self.path)
            if self.path == "/health":
                self.send_response(200 if state.health_ok else 503)
                self.end_headers()
                return
            if self.path == "/ops/memory":
                self._json(200, _MEMORY_JSON)
                return
            if self.path == "/annotators":
                import json

                body = json.dumps(state.annotators_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - http.server dispatch name
            state.note(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            if self.path == "/ops/release-cache":
                self._json(200, _MEMORY_JSON)
                return
            if self.path == "/ops/shutdown":
                self._json(200, {"shutdown_scheduled": True, "detail": "bye"})
                return
            if self.path.startswith("/rembg/remove-background"):
                if state.annotate_status != 200:
                    self.send_response(state.annotate_status)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(_SMALL_PNG)))
                self.end_headers()
                self.wfile.write(_SMALL_PNG)
                return
            if self.path.startswith("/annotators/"):
                if state.annotate_status != 200:
                    self.send_response(state.annotate_status)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(_CANNED_CONTROL_MAP)))
                self.end_headers()
                self.wfile.write(_CANNED_CONTROL_MAP)
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


class _FakeServer:
    """A capability-service launcher seam pointed at an already-running fake HTTP service."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self.running = True
        self.exit: int | None = None
        self.start_called = False
        self.stop_called = False
        self.fake_pid: int | None = 4242

    @property
    def is_running(self) -> bool:
        return self.running

    @property
    def exit_code(self) -> int | None:
        return self.exit

    @property
    def pid(self) -> int | None:
        return self.fake_pid

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> HordeImageUtilitiesClient:
        return HordeImageUtilitiesClient(self._base_url, timeout=5.0)

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True
        self.running = False


@pytest.fixture()
def http_service() -> Iterator[tuple[_ServiceState, str]]:
    """Start a fake capability HTTP service on a loopback port for the duration of a test."""
    state = _ServiceState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    base_url = f"http://{host}:{port}"
    try:
        yield state, base_url
    finally:
        server.shutdown()
        server.server_close()


def _drain(q: queue.Queue[object]) -> list[object]:
    """Return every message currently on the queue without blocking."""
    out: list[object] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def _wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _build_adapter(
    base_url: str,
    q: queue.Queue[object],
    child_conn: Connection,
    *,
    server: _FakeServer | None = None,
    health_failure_grace_seconds: float = 0.3,
) -> tuple[UtilitiesProcessAdapter, _FakeServer]:
    fake_server = server if server is not None else _FakeServer(base_url)
    adapter = UtilitiesProcessAdapter(
        process_id=5,
        process_message_queue=q,  # type: ignore[arg-type]
        control_connection=child_conn,
        process_launch_identifier=0,
        server=fake_server,
        heartbeat_interval_seconds=0.05,
        memory_interval_seconds=0.05,
        health_failure_grace_seconds=health_failure_grace_seconds,
    )
    return adapter, fake_server


def _states(messages: list[object]) -> list[HordeProcessState]:
    return [m.process_state for m in messages if isinstance(m, HordeProcessStateChangeMessage)]


def test_healthy_bringup_emits_idle_heartbeats_and_memory(http_service: tuple[_ServiceState, str]) -> None:
    """A healthy service brings up to WAITING_FOR_JOB and emits heartbeats and memory on cadence."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, fake_server = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(
            lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)),
        ), "expected the lane to reach WAITING_FOR_JOB"
        assert _wait_for(lambda: any(isinstance(m, HordeProcessHeartbeatMessage) for m in list(q.queue)))
        assert _wait_for(lambda: any(isinstance(m, HordeProcessMemoryMessage) for m in list(q.queue)))

        memory = next(m for m in _drain(q) if isinstance(m, HordeProcessMemoryMessage))
        assert memory.ram_usage_bytes == _MEMORY_JSON["process_rss_bytes"]
        assert memory.process_reserved_mb == 128
        assert fake_server.start_called is True
    finally:
        adapter.stop()
        parent_conn.close()


def test_start_annotation_round_trips_bytes_into_a_result(http_service: tuple[_ServiceState, str]) -> None:
    """START_ANNOTATION posts the source bytes and returns the control map in a result message."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        job_id = GenerationID(root="00000000-0000-0000-0000-000000000001")
        parent_conn.send(
            HordeStartAnnotationControlMessage(
                job_id=job_id,
                control_type="canny",
                source_image_bytes=b"source-image",
                resolution=512,
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAnnotationResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAnnotationResultMessage))
        assert result.job_id == job_id
        assert result.control_map_bytes == _CANNED_CONTROL_MAP
        assert result.state == GENERATION_STATE.ok
        assert result.fault_reason is None
        assert state.hits("/annotators/canny") == 1
    finally:
        adapter.stop()
        parent_conn.close()


def test_annotation_missing_weights_faults_with_reason(http_service: tuple[_ServiceState, str]) -> None:
    """A 409 from the annotators endpoint faults the result and carries the error reason."""
    state, base_url = http_service
    state.annotate_status = 409
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(
            HordeStartAnnotationControlMessage(
                job_id=GenerationID(root="00000000-0000-0000-0000-000000000002"),
                control_type="depth",
                source_image_bytes=b"source-image",
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAnnotationResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAnnotationResultMessage))
        assert result.control_map_bytes is None
        assert result.state == GENERATION_STATE.faulted
        assert result.fault_reason is not None and "409" in result.fault_reason
    finally:
        adapter.stop()
        parent_conn.close()


def test_release_allocator_cache_hits_the_endpoint(http_service: tuple[_ServiceState, str]) -> None:
    """RELEASE_ALLOCATOR_CACHE translates to the service's release-cache op."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE))
        assert _wait_for(lambda: state.hits("/ops/release-cache") == 1)
    finally:
        adapter.stop()
        parent_conn.close()


def test_end_process_shuts_down_and_stops_the_service(http_service: tuple[_ServiceState, str]) -> None:
    """END_PROCESS asks the service to shut down, stops the launcher, and emits the ending states."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, fake_server = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
        assert _wait_for(lambda: fake_server.stop_called)
        assert _wait_for(lambda: state.hits("/ops/shutdown") == 1)
        assert _wait_for(lambda: HordeProcessState.PROCESS_ENDED in _states(list(q.queue)))
        assert adapter.handle.is_alive() is False
    finally:
        parent_conn.close()


def test_subprocess_exit_surfaces_not_alive(http_service: tuple[_ServiceState, str]) -> None:
    """When the service subprocess exits, the handle reports not-alive so the reaper can recover it."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, fake_server = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: adapter.handle.is_alive() is True)
        fake_server.running = False
        fake_server.exit = 1
        assert adapter.handle.is_alive() is False
        assert adapter.handle.exitcode == 1
    finally:
        adapter.stop()
        parent_conn.close()


def test_servable_control_types_probe_excludes_unavailable_and_missing_weights(
    http_service: tuple[_ServiceState, str],
) -> None:
    """The lane emits and caches only the annotators that are available with present weights.

    canny is servable; seg (backend not portable) and depth (weights still missing) are excluded, so a
    controlnet job for seg/depth falls through to the in-graph preprocessor rather than being pre-annotated.
    """
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: any(isinstance(m, HordeAnnotatorAvailabilityMessage) for m in list(q.queue)))
        message = next(m for m in _drain(q) if isinstance(m, HordeAnnotatorAvailabilityMessage))
        assert message.servable_control_types == ["canny"]
        assert adapter.servable_control_types == frozenset({"canny"})
    finally:
        adapter.stop()
        parent_conn.close()


def test_strip_background_form_routes_to_utilities_and_matches_pp_shape(
    http_service: tuple[_ServiceState, str],
) -> None:
    """A strip_background alchemy form runs remove_background and emits a PP-lane-shaped alchemy result."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(
            HordeAlchemyControlMessage(
                control_flag=HordeControlFlag.START_ALCHEMY,
                form=AlchemyFormSpec(
                    form_id="00000000-0000-0000-0000-0000000000aa",
                    form="strip_background",
                    source_image_bytes=_SMALL_PNG,
                    r2_upload="https://example.invalid/upload",
                ),
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAlchemyResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAlchemyResultMessage))
        assert result.form == "strip_background"
        assert result.form_id == "00000000-0000-0000-0000-0000000000aa"
        assert result.state == GENERATION_STATE.ok
        # Mirrors the post-processing lane's result shape: WebP image bytes, no inline payload.
        assert result.image_bytes is not None and result.image_bytes[:4] == b"RIFF"
        assert result.result_payload is None
        assert state.hits("/rembg/remove-background") == 0  # path carries a query string; counted below
        assert any(p.startswith("/rembg/remove-background") for p in state.paths_hit)
    finally:
        adapter.stop()
        parent_conn.close()


def test_annotation_form_routes_to_utilities_and_matches_image_result_shape(
    http_service: tuple[_ServiceState, str],
) -> None:
    """An annotation alchemy form runs the requested detector and emits WebP bytes for R2."""
    state, base_url = http_service
    result_queue: queue.Queue[object] = queue.Queue()
    parent_connection, child_connection = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, result_queue, child_connection)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(result_queue.queue)))
        parent_connection.send(
            HordeAlchemyControlMessage(
                control_flag=HordeControlFlag.START_ALCHEMY,
                form=AlchemyFormSpec(
                    form_id="00000000-0000-0000-0000-0000000000ac",
                    form="annotation",
                    source_image_bytes=_SMALL_PNG,
                    control_type="canny",
                    r2_upload="https://example.invalid/upload",
                ),
            ),
        )
        assert _wait_for(
            lambda: any(isinstance(message, HordeAlchemyResultMessage) for message in list(result_queue.queue)),
        )
        result = next(message for message in _drain(result_queue) if isinstance(message, HordeAlchemyResultMessage))
        assert result.form == "annotation"
        assert result.state == GENERATION_STATE.ok
        assert result.image_bytes is not None and result.image_bytes[:4] == b"RIFF"
        assert any(path.startswith("/annotators/canny") for path in state.paths_hit)
    finally:
        adapter.stop()
        parent_connection.close()


def test_strip_background_service_failure_faults_the_form(http_service: tuple[_ServiceState, str]) -> None:
    """A service failure during background removal faults the alchemy form (never wedges the lane)."""
    state, base_url = http_service
    state.annotate_status = 500
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(
            HordeAlchemyControlMessage(
                control_flag=HordeControlFlag.START_ALCHEMY,
                form=AlchemyFormSpec(
                    form_id="00000000-0000-0000-0000-0000000000bb",
                    form="strip_background",
                    source_image_bytes=_SMALL_PNG,
                ),
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAlchemyResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAlchemyResultMessage))
        assert result.state == GENERATION_STATE.faulted
        assert result.image_bytes is None
        # The lane returns to idle after the fault, proving it keeps serving (no wedge).
        assert _wait_for(lambda: adapter.handle.is_alive() is True)
    finally:
        adapter.stop()
        parent_conn.close()


def test_annotation_missing_weights_faults_the_form(http_service: tuple[_ServiceState, str]) -> None:
    """A 409 (missing weights) from the annotator faults the form rather than wedging the lane."""
    state, base_url = http_service
    state.annotate_status = 409
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(
            HordeAlchemyControlMessage(
                control_flag=HordeControlFlag.START_ALCHEMY,
                form=AlchemyFormSpec(
                    form_id="00000000-0000-0000-0000-0000000000ad",
                    form="annotation",
                    source_image_bytes=_SMALL_PNG,
                    control_type="canny",
                ),
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAlchemyResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAlchemyResultMessage))
        assert result.state == GENERATION_STATE.faulted
        assert result.image_bytes is None
        # The lane returns to idle after the fault, proving it keeps serving (no wedge).
        assert _wait_for(lambda: adapter.handle.is_alive() is True)
    finally:
        adapter.stop()
        parent_conn.close()


def test_annotation_without_control_type_faults_without_calling_the_service(
    http_service: tuple[_ServiceState, str],
) -> None:
    """A hostile annotation form with no control_type faults cleanly and never reaches the annotator."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, _ = _build_adapter(base_url, q, child_conn)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        parent_conn.send(
            HordeAlchemyControlMessage(
                control_flag=HordeControlFlag.START_ALCHEMY,
                form=AlchemyFormSpec(
                    form_id="00000000-0000-0000-0000-0000000000ae",
                    form="annotation",
                    source_image_bytes=_SMALL_PNG,
                    control_type=None,
                ),
            ),
        )
        assert _wait_for(lambda: any(isinstance(m, HordeAlchemyResultMessage) for m in list(q.queue)))
        result = next(m for m in _drain(q) if isinstance(m, HordeAlchemyResultMessage))
        assert result.state == GENERATION_STATE.faulted
        assert result.image_bytes is None
        assert not any(path.startswith("/annotators/") for path in state.paths_hit)
        assert _wait_for(lambda: adapter.handle.is_alive() is True)
    finally:
        adapter.stop()
        parent_conn.close()


def test_unresponsive_but_alive_service_is_recycled(http_service: tuple[_ServiceState, str]) -> None:
    """A service that fails health while its subprocess is alive is stopped so the lane can be recovered."""
    state, base_url = http_service
    q: queue.Queue[object] = queue.Queue()
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    adapter, fake_server = _build_adapter(base_url, q, child_conn, health_failure_grace_seconds=0.2)

    adapter.start()
    try:
        assert _wait_for(lambda: HordeProcessState.WAITING_FOR_JOB in _states(list(q.queue)))
        # Break health while the subprocess stays "alive": the adapter must convert the hang into a
        # recoverable death by stopping the service after the grace window.
        state.health_ok = False
        assert _wait_for(lambda: fake_server.stop_called, timeout=3.0)
    finally:
        adapter.stop()
        parent_conn.close()
