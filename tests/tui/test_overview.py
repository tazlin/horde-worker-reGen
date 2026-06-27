"""Tests for the overview dashboard rendering helpers."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    FeatureReadinessSummary,
    JobQueueEntry,
    OrchestrationIntentSnapshot,
    PopGovernorsSnapshot,
    PopGovernorStatus,
    ProcessSnapshot,
    SystemMemorySnapshot,
    WholeCardResidencyStatus,
    WorkerConfigSummary,
    WorkerStateSnapshot,
    WorkLedgerEntry,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.models.feature_readiness import (
    FeatureReadiness,
    FeatureReadinessState,
    GatedFeature,
)
from horde_worker_regen.tui.health import derive
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


def _render(renderable: object, width: int = 160) -> str:
    """Render a Rich renderable to plain text at the given console width."""
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _gated(feature: GatedFeature, state: FeatureReadinessState) -> FeatureReadiness:
    return FeatureReadiness(feature=feature, label=feature.value.replace("_", " ").title(), state=state)


def test_compact_readiness_line_names_engaged_features() -> None:
    """The overview's compact line summarizes engaged gated features with their offer verbs."""
    summary = FeatureReadinessSummary(
        gated=[
            _gated(GatedFeature.CONTROLNET, FeatureReadinessState.WAITING),
            _gated(GatedFeature.POST_PROCESSING, FeatureReadinessState.OFFERED),
        ],
    )
    line = OverviewView()._feature_readiness_line(summary)
    assert line is not None
    text = _render(line)
    assert "downloading" in text
    assert "offered" in text


def test_compact_readiness_line_omits_fully_disabled_features() -> None:
    """An operator using none of the gated features sees no readiness line (no noise)."""
    summary = FeatureReadinessSummary(
        gated=[
            _gated(GatedFeature.CONTROLNET, FeatureReadinessState.DISABLED),
            _gated(GatedFeature.POST_PROCESSING, FeatureReadinessState.DISABLED),
        ],
    )
    assert OverviewView()._feature_readiness_line(summary) is None
    assert OverviewView()._feature_readiness_line(None) is None


def _busy_process() -> ProcessSnapshot:
    """A sampling inference process carrying resolution/batch/step detail."""
    return ProcessSnapshot(
        process_id=1,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        loaded_horde_model_name="AlbedoBase XL",
        loaded_horde_model_baseline="stable_diffusion_xl",
        current_job_id="7f3a1c9e-4b2c-4d6e-8a1f-0c2b07d49abc",
        batch_amount=2,
        current_job_width=832,
        current_job_height=1216,
        current_job_steps=28,
        last_current_step=14,
        last_total_steps=28,
        last_iterations_per_second=8.0,
        vram_usage_mb=8000,
        total_vram_mb=24000,
    )


def _downloading_snapshot(*, paused: bool = False) -> WorkerStateSnapshot:
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        worker_registered=True,
    )
    snapshot.download_plan = DownloadPlanSummary(num_present=2, num_to_download=3)
    snapshot.downloads = DownloadStatusSnapshot(
        phase=DownloadPhase.PAUSED if paused else DownloadPhase.DOWNLOADING,
        current=CurrentDownloadStatus(
            model_name="HugeCheckpoint",
            feature="image model",
            target_dir="/models",
            downloaded_bytes=3 * 1024**2,
            total_bytes=12 * 1024**2,
            speed_bps=1.5 * 1024**2,
        ),
        paused=paused,
    )
    return snapshot


def test_overview_hero_shows_download_line_when_active() -> None:
    """The hero renders a slim download line (model, current-file %, speed) while a fetch is in flight."""
    snapshot = _downloading_snapshot()
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "downloading" in hero
    assert "HugeCheckpoint" in hero
    assert "25%" in hero  # 3 of 12 MB
    assert "MB/s" in hero


def test_overview_hero_omits_download_line_when_idle() -> None:
    """With no download in flight the hero shows no download line, keeping an idle worker uncluttered."""
    snapshot = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"))
    snapshot.downloads = DownloadStatusSnapshot(phase=DownloadPhase.IDLE)
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "downloading" not in hero


def test_overview_compact_bar_includes_download_progress() -> None:
    """The thin status bar appends the current-download percent and speed when a fetch is active."""
    snapshot = _downloading_snapshot()
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    bar = _render(OverviewView()._render_compact_bar(report, snapshot, frame=0))

    assert "25%" in bar
    assert "MB/s" in bar


def test_overview_hero_shows_system_memory_line() -> None:
    """The hero renders a RAM line with the in-use/total figure and the worker's per-role breakdown."""
    _GB = 1024**3
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        worker_registered=True,
        system_memory=SystemMemorySnapshot(
            total_bytes=64 * _GB,
            available_bytes=20 * _GB,
            worker_rss_by_role={"orchestrator": 1 * _GB, "inference": 18 * _GB, "safety": 2 * _GB},
        ),
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "RAM 44.0 GB / 64.0 GB" in hero
    assert "worker 21.0 GB" in hero
    assert "inference 18.0 GB" in hero


def test_overview_hero_omits_memory_line_without_sample() -> None:
    """With no memory sample (older worker) the hero simply omits the RAM line."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        worker_registered=True,
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "RAM " not in hero


def test_overview_shows_lora_pause_when_background_download_blocks_pops() -> None:
    """The overview explains temporary LoRA pop suppression."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(
            dreamer_name="Tester",
            worker_version="12.0.0",
            allow_lora=True,
            effective_allow_lora=False,
            allow_controlnet=True,
        ),
        worker_registered=True,
        lora_pops_blocked_by_downloads=True,
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "LoRA pops paused while background downloads are active." in hero
    assert OverviewView._allow_summary(snapshot) == "img2img, lora paused, controlnet, post"


def test_process_table_keeps_process_owned_state() -> None:
    """The process table focuses on slot/process attributes, not active job attributes."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot), width=200)
    assert "Resident model" in text
    assert "AlbedoBase XL" in text
    assert "SDXL" in text
    assert "832×1216" not in text
    assert "7f3a1c9e" not in text
    assert "HB type" not in text


def test_process_table_detailed_adds_process_diagnostics() -> None:
    """The F6 detail view reveals process heartbeat columns when the terminal is wide enough."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    table = OverviewView()._render_process_table(snapshot, detailed=True, available_width=200)
    text = _render(table, width=200)
    assert "HB type" in text
    assert "Heartbeat" in text
    assert "Steps" not in text


def test_process_table_sheds_to_essentials_on_narrow_terminal() -> None:
    """At 80 columns the table keeps process identity/state and sheds richer columns."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot, available_width=80))
    assert "State" in text
    assert "Progress" not in text
    assert "Resident model" not in text
    assert "GPU VRAM" not in text
    assert "832×1216" not in text


def test_process_table_clamps_details_intent_to_width() -> None:
    """Requesting details on a too-narrow terminal sheds the diagnostic columns (width clamps intent)."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot, detailed=True, available_width=120))
    assert "HB type" not in text


def test_process_table_caption_hints_at_hidden_columns() -> None:
    """When width clamps below the wanted density, the caption names the hidden count and the width to reveal."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot, available_width=100))
    assert "more columns" in text
    assert "cols wide" in text


def test_process_table_residency_caption_wins_over_shed_hint() -> None:
    """An active whole-card residency caption takes the caption slot even when columns are also shed."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_flux_holder_process()],
        whole_card_residency=_active_residency(),
    )
    text = _render(OverviewView()._render_process_table(snapshot, available_width=100))
    assert "Whole-card residency" in text
    assert "more columns" not in text


def test_intent_hides_why_when_it_duplicates_detailed_gate() -> None:
    """The detailed intent panel should not repeat identical Why and Gate text."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        orchestration_intent=OrchestrationIntentSnapshot(
            summary="Waiting for dispatch.",
            why="blocked by full queue",
            raw_gate="blocked by full queue",
        ),
    )

    text = _render(OverviewView._render_intent(snapshot, detailed=True))

    assert "Gate" in text
    assert "blocked by full queue" in text
    assert " Why  " not in text


def test_intent_keeps_duplicate_why_in_normal_mode_where_gate_is_hidden() -> None:
    """Normal mode keeps Why when Gate would not be rendered anyway."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        orchestration_intent=OrchestrationIntentSnapshot(
            summary="Waiting for dispatch.",
            why="blocked by full queue",
            raw_gate="blocked by full queue",
        ),
    )

    text = _render(OverviewView._render_intent(snapshot, detailed=False))

    assert "Why" in text
    assert "blocked by full queue" in text
    assert "Gate" not in text


def test_intent_keeps_why_when_gate_differs() -> None:
    """Different Why and Gate values both remain visible in detailed mode."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        orchestration_intent=OrchestrationIntentSnapshot(
            summary="Waiting for dispatch.",
            why="queue is full",
            raw_gate="no free process",
        ),
    )

    text = _render(OverviewView._render_intent(snapshot, detailed=True))

    assert "Why" in text
    assert "queue is full" in text
    assert "Gate" in text
    assert "no free process" in text


def test_alchemy_panel_visibility_tracks_enabled_or_active_alchemy() -> None:
    """Normal mode can show alchemy when it is configured or currently carrying forms."""
    disabled = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0", alchemist=False),
    )
    enabled = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0", alchemist=True),
    )
    active = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0", alchemist=False),
        alchemy_forms_in_flight=1,
    )

    assert OverviewView._show_alchemy_panel(disabled) is False
    assert OverviewView._show_alchemy_panel(enabled) is True
    assert OverviewView._show_alchemy_panel(active) is True


def test_work_ledger_shows_job_id_progress_and_size() -> None:
    """The work ledger owns active job details: id, progress, size, and intent."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        work_ledger=[
            WorkLedgerEntry(
                job_id="7f3a1c9e-4b2c-4d6e-8a1f-0c2b07d49abc",
                stage=WorkLedgerStage.INFERENCE,
                model="AlbedoBase XL",
                baseline="stable_diffusion_xl",
                process_id=1,
                device_index=0,
                progress_current=14,
                progress_total=28,
                width=832,
                height=1216,
                steps=28,
                intent="sampling",
            ),
        ],
    )
    text = _render(OverviewView()._render_work_ledger(snapshot, detailed=False, available_width=200), width=200)
    assert "7f3a1c9e" in text
    assert "14/28" in text
    assert "832×1216" in text
    assert "sampling" in text


def test_work_ledger_can_summarize_recent_jobs_without_hiding_active_work() -> None:
    """The recent-job toggle hides terminal rows but keeps in-progress rows and a compact count."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        work_ledger=[
            WorkLedgerEntry(
                job_id="active-job-1234",
                stage=WorkLedgerStage.INFERENCE,
                model="AlbedoBase XL",
                progress_current=7,
                progress_total=28,
                intent="sampling",
            ),
            WorkLedgerEntry(
                job_id="done-job-1234",
                stage=WorkLedgerStage.COMPLETED,
                model="Deliberate",
                e2e_seconds=12.0,
            ),
            WorkLedgerEntry(
                job_id="fault-job-1234",
                stage=WorkLedgerStage.FAULTED,
                model="BrokenModel",
                faulted=True,
            ),
        ],
    )

    text = _render(
        OverviewView()._render_work_ledger(
            snapshot,
            detailed=False,
            available_width=200,
            show_recent_jobs=False,
        ),
        width=200,
    )

    assert "AlbedoBase XL" in text
    assert "7/28" in text
    assert "Deliberate" not in text
    assert "BrokenModel" not in text
    assert "1 job completed recently; 1 faulted" in text


def test_queue_table_shows_job_id_and_baseline() -> None:
    """The queue table carries each pending job's id prefix and resolved baseline."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        pending_jobs=[
            JobQueueEntry(
                job_id="9c2b07d4-aaaa-bbbb-cccc-ddddeeeeffff",
                model="Deliberate",
                baseline="stable_diffusion_1",
                steps=30,
                width=1024,
                height=1024,
            ),
        ],
    )
    text = _render(OverviewView()._render_queue_table(snapshot))
    assert "9c2b07d4" in text
    assert "SD1.5" in text


def test_queue_table_sheds_wide_columns_when_cramped() -> None:
    """A cramped queue keeps job id and model but sheds the wide columns, hinting at the clamp."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        pending_jobs=[
            JobQueueEntry(
                job_id="9c2b07d4-aaaa-bbbb-cccc-ddddeeeeffff",
                model="Deliberate",
                baseline="stable_diffusion_1",
                steps=30,
                width=1024,
                height=1024,
            ),
        ],
    )
    text = _render(OverviewView()._render_queue_table(snapshot, available_width=48), width=48)
    assert "9c2b07d4" in text
    assert "Steps" not in text
    assert "more column" in text


def test_recent_jobs_table_shows_baseline_size_and_timings() -> None:
    """Recent jobs surface baseline, size, and the queue/safety/E2E timings (favouring more data)."""
    from horde_worker_regen.process_management.ipc.supervisor_channel import RecentJobRecord

    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        recent_jobs=[
            RecentJobRecord(
                job_id="5d11aa22-1234-5678-9abc-def012345678",
                faulted=False,
                queue_wait_seconds=1.2,
                safety_seconds=0.3,
                e2e_seconds=2.4,
                model_name="Deliberate",
                baseline="stable_diffusion_1",
                steps=30,
                width=768,
                height=1024,
            ),
        ],
    )
    text = _render(OverviewView()._render_recent_jobs(snapshot))
    assert "5d11aa22" in text
    assert "SD1.5" in text
    assert "768×1024" in text


def test_pipeline_strip_shows_lifecycle_stages() -> None:
    """The job-pipeline strip labels each lifecycle stage with its live count."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        jobs_pending_inference=3,
        jobs_in_progress=1,
        jobs_pending_safety_check=0,
        jobs_pending_submit=2,
        num_jobs_submitted=42,
    )
    text = _render(OverviewView()._render_pipeline_strip(snapshot))
    assert "Queue" in text and "Inference" in text and "Safety" in text and "Submit" in text
    assert "42 submitted" in text


def test_queue_lane_renders_upcoming_blocks() -> None:
    """The queue lane renders an 'Up next' line of blocks for pending jobs."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        pending_jobs=[
            JobQueueEntry(job_id="a", model="Deliberate", steps=30, width=1024, height=1024),
            JobQueueEntry(job_id="b", model="SDXL", steps=25, width=512, height=768),
        ],
    )
    text = _render(OverviewView()._render_queue_table(snapshot))
    assert "Up next" in text
    assert "1024²" in text


def test_trends_panel_shows_value_direction_and_window() -> None:
    """Recorded GPU-duty/kudos/job history renders the kudos, jobs, and GPU-duty trend rows."""
    view = OverviewView()
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        gpu_utilization_mean_percent=70.0,
        gpu_utilization_busy_fraction=0.7,
        kudos_per_hour=12000.0,
        num_jobs_submitted=42,
    )
    for percent, kudos in ((40.0, 8000.0), (55.0, 10000.0), (80.0, 12000.0)):
        snapshot.gpu_utilization_mean_percent = percent
        snapshot.kudos_per_hour = kudos
        view._record_trends(snapshot)
    text = _render(view._render_trends(snapshot))
    assert "Kudos/hr" in text and "Jobs/hr" in text and "GPU duty" in text
    assert "42 done" in text


def test_compact_bar_summarizes_worker_in_one_line() -> None:
    """The thin compact bar carries the phase, kudos, GPU duty, and pipeline counts."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
        gpu_utilization_mean_percent=87.0,
        gpu_utilization_busy_fraction=0.87,
        kudos_per_hour=1240.0,
        num_jobs_submitted=1284,
        jobs_pending_inference=6,
        jobs_in_progress=2,
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)
    text = _render(OverviewView()._render_compact_bar(report, snapshot, frame=0))
    assert "1,284 done" in text
    assert "1,240" in text
    assert "gpu 87%" in text


def test_compact_bar_handles_missing_snapshot() -> None:
    """With no snapshot yet, the compact bar still states the phase and headline without raising."""
    report = derive(None, SupervisorStatus.STOPPED, None)
    text = _render(OverviewView()._render_compact_bar(report, None, frame=0))
    assert report.phase.value.upper() in text


def _active_residency(**overrides: object) -> WholeCardResidencyStatus:
    """An active whole-card residency for Flux on a 16GB card (2 siblings paused, safety off-GPU)."""
    base: dict[str, object] = {
        "possible": True,
        "enabled": True,
        "safety_off_gpu_enabled": True,
        "cooldown_seconds": 45,
        "per_process_overhead_mb": 1288,
        "total_vram_mb": 16375,
        "active": True,
        "model": "Flux.1-dev",
        "phase": "establishing",
        "safety_paused": True,
        "processes_now": 1,
        "processes_target": 1,
        "processes_max": 3,
        "cooldown_remaining_seconds": 40.0,
        "weights_mb": 11500,
        "reserve_mb": 3700,
        "free_now_mb": 57,
        "free_if_alone_mb": 15087,
        "max_resident_processes": 1,
    }
    base.update(overrides)
    return WholeCardResidencyStatus(**base)  # type: ignore[arg-type]


def _flux_holder_process() -> ProcessSnapshot:
    """The single inference process holding the whole-card Flux model."""
    return ProcessSnapshot(
        process_id=0,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        loaded_horde_model_name="Flux.1-dev",
        vram_usage_mb=14000,
        total_vram_mb=16375,
    )


def test_hero_shows_whole_card_residency_banner_when_active() -> None:
    """An active residency adds a hero line naming the model, the reduced processes, and why."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_flux_holder_process()],
        worker_registered=True,
        whole_card_residency=_active_residency(),
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "whole-card residency" in hero
    assert "Flux.1-dev" in hero
    assert "sole use of the GPU" in hero
    assert "safety off-GPU" in hero


def test_process_table_marks_holder_and_explains_paused_siblings() -> None:
    """The Processes table tags the holder row and captions why the sibling rows are gone."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_flux_holder_process()],
        whole_card_residency=_active_residency(),
    )

    text = _render(OverviewView()._render_process_table(snapshot))

    assert "★" in text
    assert "Whole-card residency" in text
    assert "2 idle inference processes paused" in text
    assert "safety off-GPU" in text


def test_residency_panel_shows_forecast_numbers_when_active() -> None:
    """The details-only residency panel surfaces the hard forecast numbers behind the decision."""
    text = _render(OverviewView._render_residency_panel(_active_residency()))

    assert "Whole-card residency" in text
    assert "Weights" in text
    assert "establishing" in text
    assert "Restores in" in text


def test_residency_panel_armed_when_only_possible() -> None:
    """When the feature can engage but is not active, the panel shows the armed posture, not live rows."""
    text = _render(
        OverviewView._render_residency_panel(_active_residency(active=False, possible=True, model=None)),
    )

    assert "armed" in text
    assert "Weights" not in text


def test_governors_panel_shows_active_with_countdown() -> None:
    """An active timed governor renders with its label, reason, and a remaining-time countdown."""
    governors = PopGovernorsSnapshot(
        governors=[
            PopGovernorStatus(
                name="large_model_reentry",
                label="Large-model re-entry cooldown",
                active=True,
                reason="cooling down before serving any very-large model",
                current_spell_seconds=12.0,
                expected_remaining_seconds=33.0,
                triggers=1,
                total_active_seconds=12.0,
                fraction_of_session=0.1,
            ),
        ],
        any_active=True,
    )
    text = _render(OverviewView._render_governors_panel(governors, detailed=False))

    assert "Pop governors" in text
    assert "Large-model re-entry cooldown" in text
    assert "left" in text  # the countdown


def test_governors_panel_details_lists_released_history() -> None:
    """In details mode a governor that has released is shown dim with its trigger count and total time."""
    governors = PopGovernorsSnapshot(
        governors=[
            PopGovernorStatus(
                name="whole_card_residency",
                label="Whole-card residency",
                active=False,
                triggers=3,
                total_active_seconds=90.0,
                fraction_of_session=0.3,
            ),
        ],
        any_active=False,
    )
    text = _render(OverviewView._render_governors_panel(governors, detailed=True))

    assert "Whole-card residency" in text
    assert "3x" in text
    assert "30% of session" in text
