"""Parent-side bridge that makes the out-of-venv image-utilities service a first-class worker child.

Every other worker child is a multiprocessing spawn that speaks the IPC message vocabulary from inside
its own ``HordeProcess`` loop. The image-utilities capability service is different: it runs from a
separate virtual environment as ``python -m horde_image_utilities`` (a uvicorn server on loopback HTTP),
so its native, accelerator-gated stack never enters the worker's main environment. This module bridges
that subprocess into the ordinary child contract:

- :class:`UtilitiesProcessHandle` presents the OS-process control surface (:class:`ChildProcessHandle`)
  over the capability-service subprocess, so the crash reaper and teardown drive it like any spawn.
- :class:`UtilitiesProcessAdapter` owns the capability service and runs the parent-side threads that
  translate control messages into HTTP calls and emit the state / heartbeat / memory messages the child
  would otherwise send itself, so the process map and the TUI see a normal child.

Liveness integration: the adapter polls the service's health endpoint on the child's heartbeat cadence
and emits a heartbeat only while it answers. A service that stops answering while its subprocess is still
alive (an unresponsive-but-not-dead hang) is not silently tolerated: after a grace window the adapter
stops the subprocess outright, which makes :meth:`UtilitiesProcessHandle.is_alive` report False so the
lifecycle's existing crash reaper recovers it. There is no bespoke silence watchdog for this lane; the
choice to convert an unresponsive service into a dead one lets the one existing recovery path handle both.

The capability service launcher exposes its subprocess pid, so this lane surfaces an OS pid the same way a
spawned child does: :meth:`UtilitiesProcessHandle.pid` reports it, and every emitted state / heartbeat /
memory message carries it as ``reported_os_pid``. The parent overwrites its handle-derived ``os_pid`` from
that field, so per-PID telemetry (WDDM paging attribution, the owned-PID registry) attributes the utilities
subprocess correctly. Teardown is still driven through the control pipe (``END_PROCESS``) and the handle's
terminate/kill, both of which stop the subprocess.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from horde_sdk.ai_horde_api import GENERATION_STATE
from loguru import logger

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeAnnotationResultMessage,
    HordeAnnotatorAvailabilityMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeStartAnnotationControlMessage,
    HordeStartStripControlMessage,
    HordeStripResultMessage,
)
from horde_worker_regen.process_management.lifecycle.process_info import ChildProcessHandle

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore

if TYPE_CHECKING:
    from horde_image_utilities.client import HordeImageUtilitiesClient


_ANNOTATE_FILE_FIELD = "file"
"""The multipart form field the annotators endpoint reads the source PNG from (mirrors the rembg field)."""

_MEGABYTE = 1024 * 1024


class CapabilityServer(Protocol):
    """The subset of a capability-service launcher the adapter drives (satisfied by ``CapabilityServerProcess``)."""

    @property
    def is_running(self) -> bool:
        """Return whether the service subprocess is currently alive."""
        ...

    @property
    def exit_code(self) -> int | None:
        """Return the subprocess exit code, or None if it has not started or is still running."""
        ...

    @property
    def base_url(self) -> str:
        """Return the loopback base URL the service is reachable at."""
        ...

    @property
    def pid(self) -> int | None:
        """Return the OS process id of the service subprocess, or None when it is not running."""
        ...

    @property
    def client(self) -> HordeImageUtilitiesClient:
        """Return a client bound to the service's base URL."""
        ...

    def start(self) -> None:
        """Start the subprocess and block until it reports healthy."""
        ...

    def stop(self) -> None:
        """Terminate the subprocess, escalating to kill if it does not exit."""
        ...


def encode_annotation_multipart(image_png_bytes: bytes, resolution: int) -> tuple[bytes, str]:
    """Encode a source PNG plus a ``resolution`` form field as a ``multipart/form-data`` body.

    Args:
        image_png_bytes: PNG-encoded source image content.
        resolution: The resolution the annotator should produce the control map at.

    Returns:
        A tuple of ``(body, content_type)`` suitable for an HTTP POST.
    """
    boundary = uuid.uuid4().hex
    image_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{_ANNOTATE_FILE_FIELD}"; filename="image.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    resolution_part = (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="resolution"\r\n\r\n{resolution}'
    ).encode()
    closing = f"\r\n--{boundary}--\r\n".encode()
    body = image_part + image_png_bytes + resolution_part + closing
    return body, f"multipart/form-data; boundary={boundary}"


class UtilitiesProcessHandle:
    """OS-process control surface over a capability-service subprocess (a :class:`ChildProcessHandle`).

    While the service is still coming up (its subprocess not yet launched) the handle reports alive so the
    crash reaper does not race the bring-up. Once bring-up has completed it tracks the subprocess directly;
    once it has failed it reports dead so the reaper recovers the lane.
    """

    def __init__(self, server: CapabilityServer, on_stop: Callable[[], None]) -> None:
        """Initialise the handle over the given capability service.

        Args:
            server: The capability-service launcher this handle controls.
            on_stop: A zero-argument callable invoked to stop the whole adapter on terminate/kill.
        """
        self._server = server
        self._on_stop = on_stop
        self.bringup_complete = False
        """Set True once the service has been brought up healthy."""
        self.bringup_failed = False
        """Set True once bring-up has definitively failed (the reaper then recovers the lane)."""

    @property
    def pid(self) -> int | None:
        """Return the service subprocess's OS pid, or None before it has launched (or after it exits)."""
        return self._server.pid

    @property
    def exitcode(self) -> int | None:
        """Return the subprocess exit code, or None while running or not started."""
        return self._server.exit_code

    def is_alive(self) -> bool:
        """Return whether the lane should be treated as alive."""
        if self.bringup_failed:
            return False
        if not self.bringup_complete:
            return True
        return self._server.is_running

    def terminate(self) -> None:
        """Stop the adapter and its subprocess."""
        self._on_stop()

    def kill(self) -> None:
        """Stop the adapter and its subprocess (the capability launcher escalates to kill itself)."""
        self._on_stop()

    def join(self, timeout: float | None = None) -> None:
        """No-op: :meth:`terminate` already waits on the subprocess through the launcher's stop."""
        return


class UtilitiesLaneAdapter(Protocol):
    """The construction/lifecycle seam the process lifecycle uses for the utilities lane (injectable in tests)."""

    @property
    def handle(self) -> ChildProcessHandle:
        """Return the OS-process control surface for the lane."""
        ...

    def start(self) -> None:
        """Start the capability service and the parent-side bridge threads."""
        ...


class UtilitiesProcessAdapter:
    """Bridges a capability-service subprocess into the ordinary worker-child contract.

    Owns the capability service and three daemon threads: a bring-up thread (starts the service and marks
    it idle when healthy), a control thread (translates control-pipe messages into HTTP calls and emits
    annotation results), and a cadence thread (polls health/memory and emits heartbeat / state / memory
    messages on the child heartbeat cadence).
    """

    def __init__(
        self,
        *,
        process_id: int,
        process_message_queue: ProcessQueue,
        control_connection: Connection,
        process_launch_identifier: int,
        server: CapabilityServer,
        device_index: int = 0,
        heartbeat_interval_seconds: float = 1.0,
        memory_interval_seconds: float = 5.0,
        annotate_timeout_seconds: float = 120.0,
        health_failure_grace_seconds: float = 15.0,
    ) -> None:
        """Initialise the adapter.

        Args:
            process_id: The logical slot id (not an OS pid) this lane occupies.
            process_message_queue: The queue the parent receives child messages on.
            control_connection: The parent-held pipe end the lifecycle sends control messages on.
            process_launch_identifier: The unique identifier for this launch.
            server: The capability service to own.
            device_index: The stable index of the GPU this lane is pinned to.
            heartbeat_interval_seconds: The cadence at which health is polled and heartbeats are emitted.
            memory_interval_seconds: The cadence at which a memory report is emitted.
            annotate_timeout_seconds: Per-request timeout for an annotation HTTP call.
            health_failure_grace_seconds: How long the service may fail health while its subprocess is \
                alive before the adapter stops it (converting an unresponsive service into a recoverable death).
        """
        self._process_id = process_id
        self._process_message_queue = process_message_queue
        self._control_connection = control_connection
        self._process_launch_identifier = process_launch_identifier
        self._server = server
        self._device_index = device_index
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._memory_interval_seconds = memory_interval_seconds
        self._annotate_timeout_seconds = annotate_timeout_seconds
        self._health_failure_grace_seconds = health_failure_grace_seconds

        self._handle = UtilitiesProcessHandle(server, self.stop)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self._state_lock = threading.Lock()
        self._annotating = False
        self._current_state = HordeProcessState.PROCESS_STARTING
        self._last_memory_report_at = 0.0
        self._first_health_failure_at: float | None = None
        self._servable_control_types: frozenset[str] = frozenset()

    @property
    def handle(self) -> ChildProcessHandle:
        """Return the OS-process control surface for the lane."""
        return self._handle

    def start(self) -> None:
        """Emit the starting state and launch the bring-up, control, and cadence threads."""
        self._send_state(HordeProcessState.PROCESS_STARTING, "Image utilities process starting")
        self._threads = [
            threading.Thread(
                target=self._bringup_loop, name=f"horde-utilities-bringup-{self._process_id}", daemon=True
            ),
            threading.Thread(
                target=self._control_loop, name=f"horde-utilities-control-{self._process_id}", daemon=True
            ),
            threading.Thread(
                target=self._cadence_loop, name=f"horde-utilities-cadence-{self._process_id}", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        """Signal the threads to stop and tear down the capability service (idempotent)."""
        self._stop.set()
        try:
            self._server.stop()
        except Exception as e:
            logger.warning(f"Image utilities service stop raised: {type(e).__name__} {e}")

    def _bringup_loop(self) -> None:
        """Start the capability service, then mark the lane idle (or dead on failure)."""
        try:
            self._server.start()
        except Exception as e:
            logger.error(f"Image utilities service failed to become healthy: {type(e).__name__} {e}")
            self._handle.bringup_failed = True
            return
        if self._stop.is_set():
            return
        self._handle.bringup_complete = True
        self._send_state(HordeProcessState.WAITING_FOR_JOB, "Image utilities service healthy")
        self._send_memory_report()
        self._refresh_servable_control_types()

    def _control_loop(self) -> None:
        """Drain the control pipe and dispatch each message until told to stop."""
        while not self._stop.is_set():
            try:
                if not self._control_connection.poll(0.1):
                    continue
                message = self._control_connection.recv()
            except (EOFError, OSError):
                return
            self._handle_control_message(message)

    def _handle_control_message(self, message: object) -> None:
        """Translate one control message into the corresponding capability-service action."""
        if not isinstance(message, HordeControlMessage):
            logger.critical(f"Image utilities lane received unexpected message type: {type(message).__name__}")
            return

        if message.control_flag == HordeControlFlag.END_PROCESS:
            self._send_state(HordeProcessState.PROCESS_ENDING, "Image utilities process ending")
            try:
                self._server.client.shutdown()
            except Exception as e:
                logger.debug(f"Image utilities graceful shutdown request failed: {type(e).__name__} {e}")
            self.stop()
            self._send_state(HordeProcessState.PROCESS_ENDED, "Image utilities process ended")
            return

        if message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
            try:
                self._server.client.release_cache()
            except Exception as e:
                logger.warning(f"Image utilities release-cache failed: {type(e).__name__} {e}")
            return

        if isinstance(message, HordeStartAnnotationControlMessage):
            self._handle_annotation(message)
            return

        if isinstance(message, HordeStartStripControlMessage):
            self._handle_strip(message)
            return

        if isinstance(message, HordeAlchemyControlMessage):
            self._handle_alchemy(message)
            return

        logger.error(
            f"Dropped a control message the image utilities lane does not support: {message.control_flag.name}",
        )

    def _handle_annotation(self, message: HordeStartAnnotationControlMessage) -> None:
        """Run one annotation over HTTP and put the result on the parent's message queue."""
        self._set_annotating(True)
        self._send_state(HordeProcessState.ALCHEMY_STARTING, f"Annotating {message.control_type}")
        started_at = time.monotonic()
        control_map_bytes: bytes | None = None
        state = GENERATION_STATE.ok
        fault_reason: str | None = None
        try:
            control_map_bytes = self.annotate(
                message.control_type,
                message.source_image_bytes,
                message.resolution,
            )
        except Exception as e:
            state = GENERATION_STATE.faulted
            fault_reason = f"{type(e).__name__}: {e}"
            logger.error(f"Image utilities annotation ({message.control_type}) failed: {fault_reason}")

        time_elapsed = time.monotonic() - started_at
        self._process_message_queue.put(
            HordeAnnotationResultMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info="Annotation complete" if fault_reason is None else fault_reason,
                time_elapsed=time_elapsed,
                job_id=message.job_id,
                control_map_bytes=control_map_bytes,
                state=state,
                fault_reason=fault_reason,
            ),
        )
        self._send_state(HordeProcessState.ALCHEMY_COMPLETE, f"Annotation {message.control_type} complete")
        self._set_annotating(False)

    def _handle_strip(self, message: HordeStartStripControlMessage) -> None:
        """Strip the background from each of a generation job's images and put the result on the queue.

        Reuses the same background-removal translation the standalone ``strip_background`` alchemy form uses
        (:meth:`remove_background`), but emits a :class:`HordeStripResultMessage` so the parent routes the
        stripped images back into the generation job's safety/submit tail rather than the alchemy submit path.
        A failure on any image faults the whole stage (empty result), which the parent turns into a no-image
        fault, matching how the post-processing lane treats a failed pass.
        """
        self._set_annotating(True)
        self._send_state(HordeProcessState.ALCHEMY_STARTING, "Background strip")
        started_at = time.monotonic()
        stripped: list[bytes] = []
        state = GENERATION_STATE.ok
        fault_reason: str | None = None
        try:
            for image_bytes in message.images_bytes:
                stripped.append(self.remove_background(image_bytes))
        except Exception as e:
            state = GENERATION_STATE.faulted
            fault_reason = f"{type(e).__name__}: {e}"
            stripped = []
            logger.error(f"Image utilities background strip failed: {fault_reason}")

        time_elapsed = time.monotonic() - started_at
        self._process_message_queue.put(
            HordeStripResultMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info="Background strip complete" if fault_reason is None else fault_reason,
                time_elapsed=time_elapsed,
                job_id=message.job_id,
                images_bytes=stripped,
                state=state,
                fault_reason=fault_reason,
            ),
        )
        completed_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self._send_state(completed_state, "Background strip complete")
        self._set_annotating(False)

    def _handle_alchemy(self, message: HordeAlchemyControlMessage) -> None:
        """Run one image-utilities alchemy form and emit the standard alchemy result.

        The lane serves ``strip_background`` and ``annotation``. Any other form arriving here is a routing
        error. Both successful paths return WebP bytes ready for the existing R2 submit flow.
        """
        from horde_sdk.generation_parameters.alchemy.consts import is_annotation_form, is_strip_background_form

        form = message.form
        self._set_annotating(True)
        self._send_state(HordeProcessState.ALCHEMY_STARTING, f"Alchemy {form.form} ({form.form_id})")
        started_at = time.monotonic()
        image_bytes: bytes | None = None
        state = GENERATION_STATE.ok
        fault_reason: str | None = None

        if is_annotation_form(form.form):
            if form.control_type is None:
                state = GENERATION_STATE.faulted
                fault_reason = "annotation alchemy form is missing control_type"
                logger.error(fault_reason)
            else:
                try:
                    image_bytes = self.annotate_for_alchemy(form.control_type, form.source_image_bytes)
                except Exception as e:
                    state = GENERATION_STATE.faulted
                    fault_reason = f"{type(e).__name__}: {e}"
                    logger.error(f"Image utilities annotation ({form.form_id}) failed: {fault_reason}")
        elif is_strip_background_form(form.form):
            try:
                image_bytes = self.remove_background(form.source_image_bytes)
            except Exception as e:
                state = GENERATION_STATE.faulted
                fault_reason = f"{type(e).__name__}: {e}"
                logger.error(f"Image utilities strip_background ({form.form_id}) failed: {fault_reason}")
        else:
            state = GENERATION_STATE.faulted
            fault_reason = f"image utilities lane does not serve alchemy form '{form.form}'"
            logger.error(fault_reason)

        time_elapsed = time.monotonic() - started_at
        self._process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info=f"Alchemy form {form.form} ({form.form_id})" if fault_reason is None else fault_reason,
                time_elapsed=time_elapsed,
                form_id=form.form_id,
                form=form.form,
                state=state,
                image_bytes=image_bytes,
            ),
        )
        completed_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self._send_state(completed_state, f"Alchemy {form.form} ({form.form_id}) complete")
        self._set_annotating(False)

    def annotate_for_alchemy(self, control_type: str, image_bytes: bytes) -> bytes:
        """Return an annotation result encoded as WebP for the alchemy R2 result path.

        Args:
            control_type: The control-map detector identifier.
            image_bytes: Encoded source image bytes.

        Returns:
            WebP-encoded control-map bytes.
        """
        import io

        from PIL import Image

        control_map_bytes = self.annotate(control_type, image_bytes)
        control_map = Image.open(io.BytesIO(control_map_bytes))
        buffer = io.BytesIO()
        control_map.save(buffer, format="WebP", quality=95, method=6)
        return buffer.getvalue()

    def remove_background(self, image_bytes: bytes) -> bytes:
        """Strip an image's background via the service and return the result WebP-encoded for R2.

        The service's client decodes the PNG, runs rembg, and returns a ``PIL.Image``; this encodes that
        to WebP with the same settings the post-processing lane uses (``quality=95, method=6``) so the
        submit path is byte-shape-identical regardless of which lane produced the result.

        Raises:
            Exception: Any client-side or service-side failure (surfaced as a faulted result by the caller).
        """
        import io

        from PIL import Image

        source_image = Image.open(io.BytesIO(image_bytes))
        result_image = self._server.client.remove_background(source_image)
        buffer = io.BytesIO()
        result_image.save(buffer, format="WebP", quality=95, method=6)
        return buffer.getvalue()

    def annotate(self, control_type: str, image_bytes: bytes, resolution: int = 512) -> bytes:
        """POST a source image to the annotators endpoint and return the control map PNG bytes.

        Raises:
            urllib.error.HTTPError: If the service returns a non-2xx status (for example 409 when the \
                annotator's weights are missing).
            urllib.error.URLError: If the service is unreachable.
        """
        body, content_type = encode_annotation_multipart(image_bytes, resolution)
        url = f"{self._server.base_url}/annotators/{urllib.parse.quote(control_type)}"
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", content_type)
        with urllib.request.urlopen(request, timeout=self._annotate_timeout_seconds) as response:
            return response.read()

    @property
    def servable_control_types(self) -> frozenset[str]:
        """The control types the lane could annotate at the last probe (thread-safe read)."""
        with self._state_lock:
            return self._servable_control_types

    def list_annotators(self) -> list[dict[str, object]]:
        """GET the service's per-detector annotator availability, returning the parsed JSON list.

        Raises:
            urllib.error.HTTPError / urllib.error.URLError: On a non-2xx status or an unreachable service.
        """
        import json

        url = f"{self._server.base_url}/annotators"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=self._annotate_timeout_seconds) as response:
            payload = json.loads(response.read())
        return payload if isinstance(payload, list) else []

    @staticmethod
    def _servable_from_annotators(annotators: list[dict[str, object]]) -> frozenset[str]:
        """Reduce the ``GET /annotators`` payload to the set of control types the lane can serve now.

        A control type is servable when its heavy backend is importable (``available``) and its weights are
        not reported missing (a weightless detector reports ``present``/``unknown``, so only a genuinely
        absent checkpoint excludes it). This is the availability-driven predicate the job flow keys
        pre-annotation on: anything not in this set falls through to hordelib's in-graph preprocessor.
        """
        servable: set[str] = set()
        for entry in annotators:
            name = entry.get("name")
            available = entry.get("available")
            weights_present = entry.get("weights_present")
            if isinstance(name, str) and available is True and weights_present != "missing":
                servable.add(name)
        return frozenset(servable)

    def _refresh_servable_control_types(self) -> None:
        """Re-probe servable annotators and emit the availability snapshot (best-effort).

        A probe failure leaves the previously-known set in place rather than clearing it, so a single
        transient error does not momentarily strip the lane's controlnet offers.
        """
        try:
            annotators = self.list_annotators()
        except Exception as e:
            logger.debug(f"Image utilities annotator availability probe failed: {type(e).__name__} {e}")
            return
        servable = self._servable_from_annotators(annotators)
        with self._state_lock:
            self._servable_control_types = servable
        self._process_message_queue.put(
            HordeAnnotatorAvailabilityMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info="Annotator availability",
                time_elapsed=None,
                servable_control_types=sorted(servable),
            ),
        )

    def _cadence_loop(self) -> None:
        """Poll health and memory on the child cadence, emitting heartbeat / state / memory messages."""
        while not self._stop.wait(self._heartbeat_interval_seconds):
            healthy = self._poll_health()
            if healthy:
                self._first_health_failure_at = None
                self._mark_idle_if_ready()
                self._send_heartbeat()
            elif self._server.is_running and self._handle.bringup_complete:
                self._note_health_failure()

            now = time.monotonic()
            if now - self._last_memory_report_at >= self._memory_interval_seconds:
                self._last_memory_report_at = now
                self._send_memory_report()
                # Re-probe servable annotators on the same cadence so a control type whose weights land
                # after bring-up (a still-running download-process pre-place) becomes servable without a
                # restart; a no-change refresh re-emits the same snapshot cheaply.
                if healthy:
                    self._refresh_servable_control_types()

    def _poll_health(self) -> bool:
        """Return whether the service answers its health endpoint (never raises)."""
        try:
            return self._server.client.health()
        except Exception:
            return False

    def _note_health_failure(self) -> None:
        """Track a health failure while the subprocess is alive, stopping it once past the grace window."""
        now = time.monotonic()
        if self._first_health_failure_at is None:
            self._first_health_failure_at = now
        elapsed = now - self._first_health_failure_at
        if elapsed < self._health_failure_grace_seconds:
            logger.warning(f"Image utilities service unresponsive for {elapsed:.0f}s (subprocess still alive)")
            return
        logger.error(
            f"Image utilities service unresponsive for {elapsed:.0f}s; stopping it so the lane can be recovered",
        )
        self.stop()

    def _mark_idle_if_ready(self) -> None:
        """Emit ``WAITING_FOR_JOB`` when the lane is healthy, past bring-up, and not mid-annotation."""
        with self._state_lock:
            if not self._handle.bringup_complete or self._annotating:
                return
            if self._current_state == HordeProcessState.WAITING_FOR_JOB:
                return
        self._send_state(HordeProcessState.WAITING_FOR_JOB, "Image utilities service idle")

    def _set_annotating(self, annotating: bool) -> None:
        """Record whether an annotation is in flight (guards the cadence thread from overwriting its state)."""
        with self._state_lock:
            self._annotating = annotating

    def _send_state(self, state: HordeProcessState, info: str) -> None:
        """Emit a process state-change message and record the current state."""
        with self._state_lock:
            self._current_state = state
        self._process_message_queue.put(
            HordeProcessStateChangeMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info=info,
                time_elapsed=None,
                process_state=state,
            ),
        )

    def _send_heartbeat(self) -> None:
        """Emit a liveness heartbeat for the lane."""
        self._process_message_queue.put(
            HordeProcessHeartbeatMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info="Heartbeat",
                time_elapsed=None,
                heartbeat_type=HordeHeartbeatType.OTHER,
            ),
        )

    def _send_memory_report(self) -> None:
        """Poll the service's memory snapshot and emit a memory-report message (best-effort)."""
        ram_usage_bytes = 0
        process_reserved_mb: int | None = None
        process_allocated_mb: int | None = None
        try:
            report = self._server.client.get_memory_report()
            ram_usage_bytes = report.process_rss_bytes or 0
            if report.torch_reserved_bytes is not None:
                process_reserved_mb = report.torch_reserved_bytes // _MEGABYTE
            if report.torch_allocated_bytes is not None:
                process_allocated_mb = report.torch_allocated_bytes // _MEGABYTE
        except Exception as e:
            logger.debug(f"Image utilities memory report unavailable: {type(e).__name__} {e}")

        # ``vram_usage_mb`` / ``vram_total_mb`` are deliberately left unset: the capability service reports
        # only its own torch figures, not a device-wide used reading, so populating a total-without-usage
        # entry would misstate device-wide free VRAM. ``process_reserved_mb`` is the honest per-process
        # charge and feeds only the committed-VRAM ledger.
        self._process_message_queue.put(
            HordeProcessMemoryMessage(
                process_id=self._process_id,
                process_launch_identifier=self._process_launch_identifier,
                reported_os_pid=self._server.pid,
                info="Memory report",
                time_elapsed=None,
                ram_usage_bytes=ram_usage_bytes,
                process_reserved_mb=process_reserved_mb,
                process_allocated_mb=process_allocated_mb,
                device_index=self._device_index,
                sampled_at=time.time(),
            ),
        )
