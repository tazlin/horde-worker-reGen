"""Tests for the intentionally small browser-native supervisor-protocol client."""

from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    TEXT_BACKEND_PROCESS_ID,
    ProcessSnapshot,
    TextBackendDetail,
    WorkerConfigSummary,
    WorkerStateSnapshot,
    WorkLedgerEntry,
    WorkLedgerProgressUnit,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.tui.native_dashboard import (
    NATIVE_ACTION_PATH,
    NATIVE_DASHBOARD_PATH,
    NATIVE_STATE_PATH,
    NativeDashboardAction,
    NativeDashboardBridge,
    NativeDashboardWeb,
    build_native_dashboard_state,
    load_native_dashboard_html,
)
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


class _NativeSupervisorDouble:
    """Minimal attached-supervisor double recording the native adapter's control requests."""

    def __init__(self, snapshot: WorkerStateSnapshot | None = None) -> None:
        self.latest_snapshot = snapshot
        self._status = SupervisorStatus.RUNNING if snapshot is not None else SupervisorStatus.STOPPED
        self._alive = snapshot is not None
        self._connected = True
        self._lifecycle_resolved = True
        self.actions: list[str] = []
        self.maintenance: list[bool] = []

    @property
    def status(self) -> SupervisorStatus:
        return self._status

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def lifecycle_resolved(self) -> bool:
        return self._lifecycle_resolved

    def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self.actions.append("start")

    def request_graceful_stop(self, *, timeout: float = 0.0) -> None:
        self.actions.append("stop")

    def request_pause(self) -> bool:
        self.actions.append("pause")
        return self.connected

    def request_resume(self) -> bool:
        self.actions.append("resume")
        return self.connected

    def request_set_server_maintenance(self, enabled: bool) -> bool:
        self.maintenance.append(enabled)
        return self.connected


def _snapshot() -> WorkerStateSnapshot:
    """A populated snapshot proving each selected native headline survives projection."""
    return WorkerStateSnapshot(
        timestamp=200.0,
        session_start_time=100.0,
        config=WorkerConfigSummary(
            dreamer_name="Native Worker",
            worker_version="12.0.0",
            horde_username="horde-user",
        ),
        num_jobs_popped=20,
        num_jobs_submitted=17,
        num_jobs_faulted=2,
        num_process_recoveries=3,
        num_job_slowdowns=4,
        time_spent_no_jobs_available=55.0,
        kudos_per_hour=900.0,
        kudos_this_session=1234.0,
        gpu_utilization_mean_percent=72.0,
        jobs_pending_inference=5,
        jobs_in_progress=2,
        jobs_pending_post_processing=1,
        jobs_being_post_processed=1,
        jobs_pending_safety_check=2,
        jobs_being_safety_checked=1,
        jobs_pending_submit=3,
        supervisor_paused=True,
        maintenance_mode=True,
        worker_details_maintenance=False,
        active_models=["AlbedoBase XL"],
        api_messages=["Planned maintenance later", "Second", "Third", "Not projected"],
        processes=[
            ProcessSnapshot(
                process_id=1,
                process_type="INFERENCE",
                device_index=0,
                last_process_state="INFERENCE_STARTING",
                is_alive=True,
                is_busy=True,
                loaded_horde_model_name="AlbedoBase XL",
                current_job_id="active-job-id",
                last_heartbeat_timestamp=198.0,
                last_current_step=7,
                last_total_steps=20,
                vram_usage_mb=8192,
                total_vram_mb=12288,
            ),
            ProcessSnapshot(
                process_id=2,
                process_type="SAFETY",
                last_process_state="WAITING_FOR_JOB",
                is_alive=False,
                is_busy=False,
            ),
        ],
        work_ledger=[
            WorkLedgerEntry(
                job_id="active-job-id",
                stage=WorkLedgerStage.INFERENCE,
                model="AlbedoBase XL",
                process_id=1,
                device_index=0,
                age_seconds=12.0,
            ),
            WorkLedgerEntry(
                job_id="finished-job-id",
                stage=WorkLedgerStage.COMPLETED,
                model="AlbedoBase XL",
            ),
        ],
    )


def test_state_projection_is_small_and_preserves_high_level_facts() -> None:
    """The native API selects operational facts instead of leaking the full snapshot/config schema."""
    state = build_native_dashboard_state(_NativeSupervisorDouble(_snapshot()))
    payload = state.model_dump(mode="json")

    assert state.worker_name == "Native Worker"
    assert state.jobs_submitted == 17
    assert state.jobs_post_processing == 2
    assert state.jobs_safety == 3
    assert state.local_paused is True
    assert state.horde_maintenance is False
    assert state.processes_alive == 1
    assert state.processes_total == 2
    assert state.processes_busy == 1
    assert len(state.active_jobs) == 1
    assert state.active_jobs[0].progress_percent == 35.0
    assert state.process_states[0].heartbeat_age_seconds == 2.0
    assert state.api_messages == ["Planned maintenance later", "Second", "Third"]
    assert "config" not in payload
    assert "processes" not in payload
    assert "recent_jobs" not in payload
    assert "work_ledger" not in payload


def test_state_projection_handles_an_attached_host_without_a_snapshot() -> None:
    """A newly attached page receives coherent lifecycle state before the worker's first snapshot."""
    supervisor = _NativeSupervisorDouble()
    supervisor._connected = False
    supervisor._lifecycle_resolved = False

    state = build_native_dashboard_state(supervisor)

    assert state.connected is False
    assert state.lifecycle_resolved is False
    assert state.lifecycle == "stopped"
    assert state.worker_name is None


def test_action_bridge_uses_explicit_local_and_horde_maintenance_commands() -> None:
    """The adapter never toggles from aggregate maintenance; it dispatches the requested state verbatim."""
    supervisor = _NativeSupervisorDouble(_snapshot())
    bridge = NativeDashboardBridge(supervisor)

    for action in (
        NativeDashboardAction.START,
        NativeDashboardAction.STOP,
        NativeDashboardAction.PAUSE,
        NativeDashboardAction.RESUME,
        NativeDashboardAction.MAINTENANCE_ON,
        NativeDashboardAction.MAINTENANCE_OFF,
    ):
        assert bridge.perform(action).accepted

    assert supervisor.actions == ["start", "stop", "pause", "resume"]
    assert supervisor.maintenance == [True, False]


def test_native_page_is_dependency_free_and_links_to_the_full_dashboard() -> None:
    """The companion is one native page with no framework/build pipeline and an escape hatch to Textual."""
    html = load_native_dashboard_html()

    assert '<meta name="viewport"' in html
    assert 'fetch("/native/api/state"' in html
    assert 'fetch("/native/api/action"' in html
    assert 'href="/"' in html
    assert 'id="view-toggle"' in html
    assert 'id="active-jobs"' in html
    assert 'id="processes"' in html
    assert "horde-native-glance" in html
    assert "height: 100dvh" in html
    assert 'style.setProperty("--entry-columns"' in html
    assert "style.setProperty(shareProperty" in html
    assert "scroll-snap-type" not in html
    assert 'get("view") === "glance"' in html
    assert "react" not in html.lower()
    assert "htmx" not in html.lower()


async def test_native_http_handlers_serve_state_and_validate_actions() -> None:
    """The HTTP boundary returns projected JSON and rejects non-JSON or unknown controls."""
    supervisor = _NativeSupervisorDouble(_snapshot())
    native = NativeDashboardWeb(("127.0.0.1", 1))
    native._bridge = NativeDashboardBridge(supervisor)
    app = web.Application()
    app.router.add_get(NATIVE_DASHBOARD_PATH, native.handle_page)
    app.router.add_get(NATIVE_STATE_PATH, native.handle_state)
    app.router.add_post(NATIVE_ACTION_PATH, native.handle_action)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        page = await client.get(NATIVE_DASHBOARD_PATH)
        assert page.status == 200
        assert "Worker overview" in await page.text()

        state = await client.get(NATIVE_STATE_PATH)
        payload = await state.json()
        assert state.status == 200
        assert payload["jobs_submitted"] == 17
        assert payload["active_jobs"][0]["progress_percent"] == 35.0
        assert payload["processes_alive"] == 1
        assert payload["process_states"][0]["state"] == "INFERENCE_STARTING"
        assert "config" not in payload

        action = await client.post(NATIVE_ACTION_PATH, json={"action": "pause"})
        assert action.status == 202
        assert supervisor.actions[-1] == "pause"

        wrong_type = await client.post(NATIVE_ACTION_PATH, data="action=pause")
        assert wrong_type.status == 415
        unknown = await client.post(NATIVE_ACTION_PATH, json={"action": "erase_everything"})
        assert unknown.status == 400
    finally:
        await client.close()


def test_native_process_rows_carry_the_supervised_text_backend() -> None:
    """The native page's process list includes the backend, with its pid, state and measured footprint."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        timestamp=2000.0,
        processes=[
            ProcessSnapshot(
                process_id=0,
                process_type="INFERENCE",
                last_process_state="INFERENCE_STARTING",
                is_alive=True,
                is_busy=True,
            ),
            ProcessSnapshot(
                process_id=TEXT_BACKEND_PROCESS_ID,
                process_type="TEXT_BACKEND",
                device_index=None,
                last_process_state="SERVING",
                is_alive=True,
                is_busy=False,
                is_external=True,
                os_pid=48213,
                display_state="ready",
                loaded_horde_model_name="Llama-3.2-3B",
                text_backend=TextBackendDetail(
                    kind="koboldcpp",
                    model_name="Llama-3.2-3B",
                    port=5001,
                    footprint_mb=2700,
                    last_health_ok_at=1958.0,
                ),
            ),
        ],
    )

    rows = build_native_dashboard_state(_NativeSupervisorDouble(snapshot)).process_states

    assert [row.is_external for row in rows] == [False, True]
    backend_row = rows[1]
    assert backend_row.process_id == 48213
    assert backend_row.process_type == "TEXT_BACKEND"
    assert backend_row.device_index is None
    assert backend_row.state == "ready"
    assert backend_row.model == "Llama-3.2-3B"
    assert backend_row.footprint_mb == 2700
    assert backend_row.health_age_seconds == 42.0


def _scribe_snapshot() -> WorkerStateSnapshot:
    """A scribe worker serving one generation through a supervised backend."""
    return WorkerStateSnapshot(
        timestamp=200.0,
        config=WorkerConfigSummary(
            dreamer_name="Native Worker",
            scribe_name="Native Scribe",
            worker_version="12.0.0",
            scribe=True,
        ),
        enabled_workloads=[WorkloadKind.TEXT_GENERATION.value],
        text_backend_ready=True,
        text_model_name="koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M",
        text_total_submitted=9,
        text_total_faulted=1,
        text_jobs_in_flight=1,
        processes=[
            ProcessSnapshot(
                process_id=TEXT_BACKEND_PROCESS_ID,
                process_type="TEXT_BACKEND",
                device_index=None,
                last_process_state="SERVING",
                is_alive=True,
                is_busy=True,
                is_external=True,
                os_pid=48213,
                display_state="ready",
                loaded_horde_model_name="koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M",
                text_backend=TextBackendDetail(
                    kind="koboldcpp",
                    port=5001,
                    footprint_mb=2700,
                    active_requests=1,
                    queued_requests=2,
                    tokens_per_second=16.0,
                ),
            ),
        ],
        work_ledger=[
            WorkLedgerEntry(
                job_id="text-job-id",
                stage=WorkLedgerStage.INFERENCE,
                workload=WorkloadKind.TEXT_GENERATION,
                model="koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M",
                progress_current=12,
                progress_unit=WorkLedgerProgressUnit.CHUNKS,
                iterations_per_second=16.0,
                age_seconds=3.0,
            ),
        ],
    )


def test_the_projection_carries_a_text_generation_as_active_work() -> None:
    """A text generation is work in hand, so it reaches the page through the ledger every job uses.

    It runs in a separate program, so it has no process and no percentage: the row says what it counted
    and in which unit instead of being given a fraction nothing supports.
    """
    state = build_native_dashboard_state(_NativeSupervisorDouble(_scribe_snapshot()))

    assert len(state.active_jobs) == 1
    job = state.active_jobs[0]
    assert job.workload == WorkloadKind.TEXT_GENERATION.value
    assert job.progress_unit == WorkLedgerProgressUnit.CHUNKS.value
    assert job.progress_current == 12
    assert job.progress_percent is None
    assert job.process_id is None


def test_the_projection_carries_the_backend_row_activity_and_the_scribe_identity() -> None:
    """The backend's row states what the flow is asking of it, which the supervisor cannot see itself."""
    state = build_native_dashboard_state(_NativeSupervisorDouble(_scribe_snapshot()))

    assert state.worker_name == "Native Scribe"
    assert state.text_backend_ready is True
    assert state.text_in_flight == 1
    assert state.text_submitted == 9
    assert state.text_faulted == 1
    assert state.text_tokens_per_second == 16.0
    backend_row = next(row for row in state.process_states if row.is_external)
    assert backend_row.active_requests == 1
    assert backend_row.queued_requests == 2
    assert backend_row.tokens_per_second == 16.0
    assert backend_row.busy is True


def test_the_native_page_renders_the_text_section_and_the_row_units() -> None:
    """The page's own markup has to carry the fields the projection now sends, or they render nowhere."""
    page = load_native_dashboard_html()

    assert 'id="text-section"' in page
    assert 'id="text-backend"' in page
    assert 'id="text-rate"' in page
    assert "state.text_in_flight" in page
    assert "job.progress_unit" in page
    assert "text_generation" in page
