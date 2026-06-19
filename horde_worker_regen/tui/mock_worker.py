"""A synthetic worker that speaks the supervisor protocol without a GPU or hordelib.

Used for ``--process-mode fake``: it emits believable :class:`WorkerStateSnapshot` frames, including a
realistic warm-up lifecycle and a periodic simulated network blip, and honours control commands, so
the TUI (and ``textual serve``) can be developed, demoed, and tested end-to-end with no models, no
torch, and no API key. It intentionally imports nothing heavy.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque

from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadItem,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    JobFeatureSummary,
    JobQueueEntry,
    ProcessSnapshot,
    RecentJobRecord,
    SupervisorChannel,
    SupervisorCommand,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions

_TICK_SECONDS = 0.5

_MOCK_MODELS = ["AlbedoBase XL (SDXL)", "Deliberate", "Flux.1-Schnell fp8 (Compact)"]

# A pool of believable job shapes (width, height, steps, batch) so the resolution/batch columns and the
# queue/pipeline visualizations have varied, realistic content under --process-mode fake.
_MOCK_JOB_SHAPES: list[tuple[int, int, int, int]] = [
    (1024, 1024, 30, 1),
    (832, 1216, 28, 1),
    (512, 768, 25, 4),
    (1152, 896, 32, 2),
    (768, 768, 20, 1),
]

# One-time warm-up, then the repeating steady-state job cycle (name, duration seconds).
_WARMUP_PHASES: list[tuple[str, float]] = [
    ("PROCESS_STARTING", 1.5),
    ("DOWNLOADING_MODEL", 4.0),
    ("PRELOADING_MODEL", 2.5),
]
_STEADY_PHASES: list[tuple[str, float]] = [
    ("WAITING_FOR_JOB", 1.2),
    ("PRELOADING_MODEL", 1.0),
    ("INFERENCE_STARTING", 4.5),
    ("INFERENCE_COMPLETE", 0.6),
]
_PHASES = _WARMUP_PHASES + _STEADY_PHASES
_STEADY_START = len(_WARMUP_PHASES)

# Simulated network outage: every period, for the first `_BLIP_LENGTH` seconds (after warm-up).
_BLIP_PERIOD = 55
_BLIP_LENGTH = 6
_WARMUP_GRACE = 14

# Simulated "no jobs available" stretch (with skip reasons) and a horde-set maintenance window, so the
# condition-surfacing UI is demonstrable under --process-mode fake.
_NO_WORK_PERIOD = 30
_NO_WORK_LENGTH = 8
_MOCK_SKIP_REASONS = {"models": 3, "max_pixels": 1, "nsfw": 2}
_MAINTENANCE_PERIOD = 80
_MAINTENANCE_LENGTH = 10


class _MockProcess:
    """A single synthetic inference process: one warm-up, then a repeating job cycle."""

    def __init__(self, process_id: int) -> None:
        self.process_id = process_id
        self._index = 0
        self._phase_started = time.monotonic()
        self._fraction = 0.0
        self.model = random.choice(_MOCK_MODELS)
        self.width, self.height, self.total_steps, self.batch = random.choice(_MOCK_JOB_SHAPES)
        self.step = 0
        self.its = 0.0
        self.vram_mb = 1800
        self.vram_high_water_mb = 1800
        self.ram_bytes = 6 * 1024**3
        self.last_inference_started = 0.0
        self.num_jobs_completed = 0

    @property
    def state(self) -> str:
        """The current ``HordeProcessState`` name."""
        return _PHASES[self._index][0]

    def restart(self) -> None:
        """Simulate a process slot being replaced (re-runs warm-up)."""
        self._index = 0
        self._phase_started = time.monotonic()
        self.step = 0
        self.its = 0.0
        self.vram_mb = 0
        self.model = random.choice(_MOCK_MODELS)

    def tick(self, *, paused: bool) -> bool:
        """Advance the lifecycle. Returns True when a steady-state job just completed."""
        now = time.monotonic()
        phase_name, duration = _PHASES[self._index]
        elapsed = now - self._phase_started
        self._fraction = min(elapsed / duration, 1.0) if duration else 1.0

        if phase_name == "INFERENCE_STARTING":
            self.step = int(self._fraction * self.total_steps)
            self.its = round(random.uniform(6.5, 11.5), 2)
            self.vram_mb = 1800 + int(self._fraction * 7000)
            self.vram_high_water_mb = max(self.vram_high_water_mb, self.vram_mb)
            if elapsed < _TICK_SECONDS:
                self.last_inference_started = time.time()
        elif phase_name in ("DOWNLOADING_MODEL", "PRELOADING_MODEL"):
            self.vram_mb = 200 + int(self._fraction * 1600)
        elif phase_name == "WAITING_FOR_JOB":
            self.step = 0
            self.its = 0.0
            self.vram_mb = 1800
            if paused:
                self._phase_started = now  # Stay parked while paused.
                return False

        if elapsed < duration:
            return False

        completed = phase_name == "INFERENCE_COMPLETE"
        self._index += 1
        if self._index >= len(_PHASES):
            self._index = _STEADY_START  # Warm-up runs once; then loop the steady cycle.
        self._phase_started = now
        if completed:
            self.model = random.choice(_MOCK_MODELS)
            self.width, self.height, self.total_steps, self.batch = random.choice(_MOCK_JOB_SHAPES)
            self.num_jobs_completed += 1
        return completed

    @property
    def is_busy(self) -> bool:
        """Whether this process is doing work (not idly waiting)."""
        return self.state != "WAITING_FOR_JOB"

    def to_snapshot(self) -> ProcessSnapshot:
        """Project this mock process into a wire snapshot."""
        sampling = self.state == "INFERENCE_STARTING"
        loading = self.state in ("DOWNLOADING_MODEL", "PRELOADING_MODEL")
        percent = (
            int(self.step / self.total_steps * 100) if sampling else int(self._fraction * 100) if loading else None
        )
        return ProcessSnapshot(
            process_id=self.process_id,
            process_type="INFERENCE",
            last_process_state=self.state,
            is_alive=True,
            is_busy=self.is_busy,
            loaded_horde_model_name=self.model if self.is_busy else None,
            loaded_horde_model_baseline="stable_diffusion_xl",
            current_job_id=f"mock-{self.process_id}-{int(self._phase_started)}" if sampling else None,
            last_heartbeat_timestamp=time.time(),
            last_heartbeat_type="INFERENCE_STEP" if sampling else "OTHER",
            last_heartbeat_percent_complete=percent,
            ram_usage_bytes=self.ram_bytes,
            vram_usage_mb=self.vram_mb,
            total_vram_mb=24000,
            batch_amount=self.batch if self.is_busy else 1,
            current_job_width=self.width if self.is_busy else None,
            current_job_height=self.height if self.is_busy else None,
            current_job_steps=self.total_steps if self.is_busy else None,
            last_iterations_per_second=self.its if sampling else None,
            last_current_step=self.step if sampling else None,
            last_total_steps=self.total_steps if sampling else None,
            vram_used_high_water_mb=self.vram_high_water_mb,
            ram_used_high_water_mb=self.ram_bytes // 1024**2,
            num_jobs_completed=self.num_jobs_completed,
        )


class _MockSafetyProcess:
    """A synthetic safety process: idle between checks, but with a fresh heartbeat and a climbing tally.

    Exercises the two safety-specific fixes at once: the idle heartbeat (so it never looks dead) and the
    per-process "Checked" counter (so its otherwise-invisible work is visible).
    """

    def __init__(self, process_id: int) -> None:
        self.process_id = process_id
        self.num_jobs_completed = 0
        self._busy_until = 0.0

    def on_job_checked(self) -> None:
        """Record a completed safety check and briefly flash the evaluating state."""
        self.num_jobs_completed += 1
        self._busy_until = time.monotonic() + 0.2

    @property
    def state(self) -> str:
        """``EVALUATING_SAFETY`` during the brief check flash, otherwise ``WAITING_FOR_JOB``."""
        return "EVALUATING_SAFETY" if time.monotonic() < self._busy_until else "WAITING_FOR_JOB"

    def to_snapshot(self) -> ProcessSnapshot:
        """Project the safety process into a wire snapshot (always alive, heartbeat fresh)."""
        return ProcessSnapshot(
            process_id=self.process_id,
            process_type="SAFETY",
            last_process_state=self.state,
            is_alive=True,
            is_busy=self.state != "WAITING_FOR_JOB",
            last_heartbeat_timestamp=time.time(),
            last_heartbeat_type="OTHER",
            ram_usage_bytes=2 * 1024**3,
            vram_usage_mb=0,
            total_vram_mb=24000,
            ram_used_high_water_mb=2 * 1024,
            num_jobs_completed=self.num_jobs_completed,
        )


_GB = 1024**3

# A synthetic background-download lifecycle so the Downloads tab is demonstrable under --process-mode fake.
_MOCK_DOWNLOAD_QUEUE: list[tuple[str, str, int]] = [
    ("Flux.1-Schnell fp8 (Compact)", "image model", 12 * _GB),
    ("AlbedoBase XL (SDXL)", "image model", 6 * _GB),
]
_MOCK_DOWNLOAD_SPEED_BPS = 90 * 1024 * 1024  # ~90 MB/s, a believable fast connection.


class _MockDownloads:
    """Drives a believable phase timeline (initializing -> scanning -> downloading -> idle)."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._plan = DownloadPlanSummary(
            present_bytes=6 * _GB,
            to_download_bytes=sum(size for _, _, size in _MOCK_DOWNLOAD_QUEUE),
            total_bytes=6 * _GB + sum(size for _, _, size in _MOCK_DOWNLOAD_QUEUE),
            free_disk_bytes=412 * _GB,
            fits=True,
            shortfall_bytes=0,
            num_present=1,
            num_to_download=len(_MOCK_DOWNLOAD_QUEUE),
            sizes_complete=True,
        )

    @property
    def plan(self) -> DownloadPlanSummary:
        """The static disk-plan summary for the mock config."""
        return self._plan

    def snapshot(self) -> DownloadStatusSnapshot:
        """Project the elapsed-time timeline into a download-status snapshot."""
        elapsed = time.monotonic() - self._started
        if elapsed < 2.0:
            return DownloadStatusSnapshot(phase=DownloadPhase.INITIALIZING)
        if elapsed < 5.0:
            return DownloadStatusSnapshot(phase=DownloadPhase.SCANNING, present_model_names=["Deliberate"])

        download_elapsed = elapsed - 5.0
        present = ["Deliberate"]
        for index, (model_name, feature, size) in enumerate(_MOCK_DOWNLOAD_QUEUE):
            done_before = sum(size for _, _, size in _MOCK_DOWNLOAD_QUEUE[:index])
            downloaded_total = _MOCK_DOWNLOAD_SPEED_BPS * download_elapsed
            into_this = downloaded_total - done_before
            if into_this < 0:
                continue
            if into_this < size:
                remaining = (size - into_this) / _MOCK_DOWNLOAD_SPEED_BPS
                pending = [
                    DownloadItem(model_name=name, feature=feat, size_bytes=sz)
                    for name, feat, sz in _MOCK_DOWNLOAD_QUEUE[index + 1 :]
                ]
                return DownloadStatusSnapshot(
                    phase=DownloadPhase.DOWNLOADING,
                    current=CurrentDownloadStatus(
                        model_name=model_name,
                        feature=feature,
                        target_dir="models/compvis",
                        downloaded_bytes=int(into_this),
                        total_bytes=size,
                        speed_bps=float(_MOCK_DOWNLOAD_SPEED_BPS),
                        eta_seconds=remaining,
                    ),
                    pending=pending,
                    present_model_names=present,
                )
            present.append(model_name)

        return DownloadStatusSnapshot(phase=DownloadPhase.IDLE, present_model_names=present)


def run_mock_worker(connection: object, options: WorkerLaunchOptions) -> None:
    """Synthetic worker entry point (a spawn target): emit snapshots and honour control commands.

    Args:
        connection: The worker end of the supervisor pipe.
        options: The launch options (only ``worker_name`` is used, for display parity).
    """
    import horde_worker_regen

    channel = SupervisorChannel(connection)  # type: ignore[arg-type]
    safety = _MockSafetyProcess(0)
    processes = [_MockProcess(1), _MockProcess(2)]
    downloads = _MockDownloads()

    paused = False
    jobs_submitted = 0
    jobs_faulted = 0
    jobs_popped = 0
    kudos_session = 0.0
    session_start = time.time()
    recent_jobs: deque[RecentJobRecord] = deque(maxlen=25)

    config = WorkerConfigSummary(
        dreamer_name=os.getenv("AIWORKER_DREAMER_WORKER_NAME") or options.worker_name or "Mock Dreamer",
        worker_version=horde_worker_regen.__version__,
        horde_username="mock_user",
        num_models=len(_MOCK_MODELS),
        max_power=32,
        max_threads=len(processes),
        queue_size=1,
        safety_on_gpu=True,
        allow_lora=True,
        allow_post_processing=True,
        high_memory_mode=True,
    )

    while True:
        channel.note_alive()
        for command in channel.drain_commands():
            if command.command in (SupervisorCommand.PAUSE, SupervisorCommand.DRAIN):
                paused = True
            elif command.command is SupervisorCommand.RESUME:
                paused = False
            elif command.command is SupervisorCommand.RESTART_PROCESS:
                for process in processes:
                    if process.process_id == command.process_id:
                        process.restart()
            elif command.command is SupervisorCommand.SHUTDOWN:
                return

        for process in processes:
            if process.tick(paused=paused):
                jobs_submitted += 1
                jobs_popped += 1
                kudos_session += random.uniform(8.0, 28.0)
                safety.on_job_checked()
                faulted = random.random() < 0.04
                if faulted:
                    jobs_faulted += 1
                recent_jobs.append(
                    RecentJobRecord(
                        job_id=f"mock-{jobs_submitted}",
                        faulted=faulted,
                        queue_wait_seconds=random.uniform(0.1, 3.0),
                        e2e_seconds=random.uniform(3.5, 9.0),
                        model_name=process.model,
                        steps=process.total_steps,
                        width=process.width,
                        height=process.height,
                    ),
                )

        elapsed = time.time() - session_start
        blip = elapsed > _WARMUP_GRACE and (int(elapsed) % _BLIP_PERIOD) < _BLIP_LENGTH
        no_work = not paused and elapsed > _WARMUP_GRACE and (int(elapsed) % _NO_WORK_PERIOD) < _NO_WORK_LENGTH
        horde_maintenance = elapsed > _WARMUP_GRACE and (int(elapsed) % _MAINTENANCE_PERIOD) < _MAINTENANCE_LENGTH
        last_pop = max((p.last_inference_started for p in processes), default=0.0)
        session_hours = max(elapsed / 3600.0, 1e-6)

        # A small, slowly-rotating synthetic queue so the pipeline strip and "Up next" lane have content.
        queue_depth = 0 if no_work else (int(elapsed) // 3) % 5
        pending_jobs = [
            JobQueueEntry(
                job_id=f"mock-queued-{index}",
                model=_MOCK_MODELS[index % len(_MOCK_MODELS)],
                steps=shape[2],
                width=shape[0],
                height=shape[1],
                features=JobFeatureSummary(loras=1) if index % 2 else None,
            )
            for index, shape in enumerate(_MOCK_JOB_SHAPES[:queue_depth])
        ]
        sampling_count = sum(1 for p in processes if p.state == "INFERENCE_STARTING")

        snapshot = WorkerStateSnapshot(
            session_start_time=session_start,
            maintenance_mode=paused,
            worker_details_maintenance=horde_maintenance,
            config=config,
            processes=[safety.to_snapshot(), *(process.to_snapshot() for process in processes)],
            num_jobs_popped=jobs_popped,
            num_jobs_submitted=jobs_submitted,
            num_jobs_faulted=jobs_faulted,
            worker_registered=elapsed > 3,
            user_info_failed=blip,
            user_info_failed_reason="HTTP error ((ConnectionError) simulated outage)" if blip else None,
            in_error_backoff=blip,
            seconds_since_last_pop=(time.time() - last_pop) if last_pop else None,
            last_pop_no_jobs_available=no_work,
            last_pop_skipped_reasons=dict(_MOCK_SKIP_REASONS) if no_work else {},
            api_messages=["Simulated network issue; the worker is retrying."] if blip else [],
            pending_megapixelsteps=random.randint(0, 12),
            jobs_pending_inference=len(pending_jobs),
            jobs_in_progress=sampling_count,
            jobs_pending_safety_check=sum(1 for p in processes if p.state == "INFERENCE_COMPLETE"),
            jobs_pending_submit=(int(elapsed) // 2) % 3,
            pending_jobs=pending_jobs,
            recent_jobs=list(recent_jobs),
            kudos_per_hour=kudos_session / session_hours if kudos_session else None,
            kudos_this_session=kudos_session,
            active_models=sorted({p.model for p in processes if p.is_busy}),
            gpu_utilization_mean_percent=round(random.uniform(55.0, 92.0), 1),
            gpu_utilization_busy_fraction=round(random.uniform(0.55, 0.95), 2),
            vram_high_water_mb_per_process={p.process_id: p.vram_high_water_mb for p in processes},
            disk_free_bytes={"G:\\": 412 * 1024**3},
            downloads=downloads.snapshot(),
            download_plan=downloads.plan,
        )
        if not channel.send_snapshot(snapshot):
            return
        time.sleep(_TICK_SECONDS)
