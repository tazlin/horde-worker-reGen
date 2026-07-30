"""Behavioural tests for the disaggregated stage progress emitters (sample stage and VAE decode).

The pinned sampler and the VAE lane translate hordelib ``ProgressReport``s into heartbeats so the
dashboard can show mid-pipeline progress. The sample-stage emitter must forward per-step progress
without touching the monolithic path's VAE-decode / inference-slot handoff (over-releasing the inference
semaphore or acquiring the VAE-decode semaphore in a sampler wedges the pipeline). The decode emitter
forwards a tiled decode's per-tile steps and stays silent for a single-shot decode.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from hordelib.api import ComfyUIProgress, ComfyUIProgressUnit, ProgressReport, ProgressState

from horde_worker_regen.process_management.ipc.messages import (
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
)
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess


def _progress(
    *,
    current_step: int,
    total_steps: int,
    percent: int = 0,
    rate: float = 5.0,
    rate_unit: ComfyUIProgressUnit = ComfyUIProgressUnit.ITERATIONS_PER_SECOND,
) -> ProgressReport:
    """Build a ProgressReport carrying one ComfyUI step report."""
    return ProgressReport(
        hordelib_progress_state=ProgressState.progress,
        comfyui_progress=ComfyUIProgress(
            percent=percent,
            current_step=current_step,
            total_steps=total_steps,
            rate=rate,
            rate_unit=rate_unit,
        ),
    )


class _SamplerHeartbeatFake:
    """A minimal stand-in exposing only the state the sample-stage emitter is permitted to touch.

    It deliberately omits the VAE-decode semaphore, the inference-slot release, and the
    ``_current_job_inference_steps_complete`` completion latch. If the emitter reached for any of that
    machinery the attribute access would raise, so a green test proves the emitter never touches it.
    """

    def __init__(self) -> None:
        self._last_progress_step_seen: int | None = None
        self._nonadvancing_progress_repeats = 0
        self._last_job_inference_rate: str | None = None
        self.heartbeats: list[SimpleNamespace] = []

    def send_heartbeat_message(
        self,
        heartbeat_type: HordeHeartbeatType,
        *,
        process_warning: str | None = None,
        percent_complete: int | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        iterations_per_second: float | None = None,
        nonadvancing_step_repeats: int = 0,
    ) -> None:
        self.heartbeats.append(
            SimpleNamespace(
                heartbeat_type=heartbeat_type,
                percent_complete=percent_complete,
                current_step=current_step,
                total_steps=total_steps,
                iterations_per_second=iterations_per_second,
                nonadvancing_step_repeats=nonadvancing_step_repeats,
            ),
        )


def _emit_sample(fake: _SamplerHeartbeatFake, report: ProgressReport) -> None:
    """Invoke the (unbound) sample-stage emitter against the fake recorder."""
    HordeInferenceProcess._emit_sample_stage_progress(fake, report)  # type: ignore[arg-type]


class TestSampleStageEmitter:
    """The sample-stage emitter forwards step progress and never touches the VAE/slot handoff."""

    def test_advancing_step_emits_inference_step_with_populated_fields(self) -> None:
        """A mid-sampling step becomes an INFERENCE_STEP heartbeat carrying the step counts."""
        fake = _SamplerHeartbeatFake()

        _emit_sample(fake, _progress(current_step=3, total_steps=30, percent=10))

        assert len(fake.heartbeats) == 1
        beat = fake.heartbeats[0]
        assert beat.heartbeat_type == HordeHeartbeatType.INFERENCE_STEP
        assert beat.current_step == 3
        assert beat.total_steps == 30
        assert beat.percent_complete == 10
        assert beat.iterations_per_second == 5.0

    def test_terminal_step_emits_pipeline_state_change_not_inference_step(self) -> None:
        """The final step (current == total) reports a state change, not another INFERENCE_STEP."""
        fake = _SamplerHeartbeatFake()

        _emit_sample(fake, _progress(current_step=30, total_steps=30))

        assert len(fake.heartbeats) == 1
        assert fake.heartbeats[0].heartbeat_type == HordeHeartbeatType.PIPELINE_STATE_CHANGE

    def test_seconds_per_iteration_rate_is_normalized_to_iterations_per_second(self) -> None:
        """A seconds/iteration rate is inverted to iterations/second for the heartbeat."""
        fake = _SamplerHeartbeatFake()

        _emit_sample(
            fake,
            _progress(current_step=2, total_steps=30, rate=2.0, rate_unit=ComfyUIProgressUnit.SECONDS_PER_ITERATION),
        )

        assert fake.heartbeats[0].iterations_per_second == 0.5

    def test_repeated_step_increments_the_nonadvancing_counter(self) -> None:
        """Re-reporting the same step raises the non-advancing repeat count the hang watchdog reads."""
        fake = _SamplerHeartbeatFake()

        _emit_sample(fake, _progress(current_step=5, total_steps=30))
        _emit_sample(fake, _progress(current_step=5, total_steps=30))

        assert fake.heartbeats[0].nonadvancing_step_repeats == 0
        assert fake.heartbeats[1].nonadvancing_step_repeats == 1

    def test_emitter_never_reaches_for_the_vae_or_slot_machinery(self) -> None:
        """The fake omits the VAE/slot attributes; a clean run proves the emitter never touches them."""
        fake = _SamplerHeartbeatFake()

        _emit_sample(fake, _progress(current_step=1, total_steps=30))

        assert not hasattr(fake, "_vae_decode_semaphore")
        assert not hasattr(fake, "_current_job_inference_steps_complete")
        assert not hasattr(fake, "_inference_slot_released")


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what a lane sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        self.messages.append(message)


def _make_vae_lane(queue: _FakeQueue) -> HordeVaeLaneProcess:
    return HordeVaeLaneProcess(
        process_id=1,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )


def _inference_step_beats(queue: _FakeQueue) -> list[HordeProcessHeartbeatMessage]:
    return [
        message
        for message in queue.messages
        if isinstance(message, HordeProcessHeartbeatMessage)
        and message.heartbeat_type == HordeHeartbeatType.INFERENCE_STEP
    ]


class TestVaeDecodeEmitter:
    """The VAE decode emitter forwards tiled-decode steps and stays silent for a single-shot decode."""

    def test_tiled_decode_step_emits_inference_step_heartbeat(self) -> None:
        """A per-tile step report becomes an INFERENCE_STEP heartbeat carrying the tile counts."""
        queue = _FakeQueue()
        lane = _make_vae_lane(queue)
        queue.messages.clear()

        lane._emit_decode_progress(_progress(current_step=2, total_steps=8, percent=25))

        beats = _inference_step_beats(queue)
        assert len(beats) == 1
        assert beats[0].current_step == 2
        assert beats[0].total_steps == 8
        assert beats[0].percent_complete == 25

    def test_single_shot_decode_emits_no_step_heartbeat(self) -> None:
        """A decode with no intermediate step (current_step 0) emits nothing; POST_PROCESSING stands."""
        queue = _FakeQueue()
        lane = _make_vae_lane(queue)
        queue.messages.clear()

        lane._emit_decode_progress(_progress(current_step=0, total_steps=0))

        assert _inference_step_beats(queue) == []
