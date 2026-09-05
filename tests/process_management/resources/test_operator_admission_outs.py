"""The operator's outs over the measured admission: the margin, the probe delay, the lane rungs, the pin.

Each is a field on the worker config with a per-card twin under ``gpu_overrides``; these tests pin the
platform-aware margin default and its override, the probe delay's two regimes, the whole-card pin by name or
baseline id, and the re-armed probe after a process cycle.
"""

from __future__ import annotations

from dataclasses import replace

from horde_worker_regen.bridge_data.data_model import GpuOverride, reGenBridgeData
from horde_worker_regen.bridge_data.gpu_config import resolve_effective_gpu_config
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.admission_identity import (
    _ADMISSION_NOISE_BUFFER_MB,
    admission_margin_mb,
    admission_noise_buffer_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    _MEASURED_ATTEMPT_BAND_MB,
    _STARVATION_DIAGNOSTIC_SECONDS,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramRequest,
    VramRequestKind,
)
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async


class TestAdmissionMargin:
    """The admission margin: platform default, operator override, and the governor buffer left alone."""

    def test_platform_default_is_the_physics_buffer_on_wddm_and_half_elsewhere(self) -> None:
        """WDDM keeps the 5 % buffer; a device-wide reading scales to 2.5 %; both floor at 512 MB."""
        assert admission_margin_mb(24074.0, platform="win32") == admission_noise_buffer_mb(24074.0)
        assert admission_margin_mb(24074.0, platform="linux") == 0.025 * 24074.0
        assert admission_margin_mb(8192.0, platform="linux") == _ADMISSION_NOISE_BUFFER_MB
        assert admission_margin_mb(None, platform="linux") == _ADMISSION_NOISE_BUFFER_MB

    def test_the_operator_override_wins_whole(self) -> None:
        """An explicit margin replaces the derived one, down to zero, on either platform."""
        assert admission_margin_mb(24074.0, override_mb=300.0, platform="win32") == 300.0
        assert admission_margin_mb(24074.0, override_mb=0.0, platform="linux") == 0.0
        assert admission_margin_mb(24074.0, override_mb=-5.0) == 0.0

    def test_the_governor_buffer_does_not_follow_the_margin(self) -> None:
        """The device-free governor's floors stay on the physics buffer whatever the margin does."""
        assert admission_noise_buffer_mb(24074.0) == 0.05 * 24074.0


class TestPerCardFields:
    """The outs exist on the worker config and resolve per card."""

    def test_defaults_and_per_card_override_resolution(self) -> None:
        """Every out has a worker default and resolves per card through gpu_overrides."""
        base = reGenBridgeData(api_key="0000000000")
        assert base.vram_admission_noise_mb is None
        assert base.measured_load_probe_seconds == 10
        assert base.starved_head_lane_reclaim is True
        assert base.starved_head_utilities_pause is True
        assert base.whole_card_models == []

        card = resolve_effective_gpu_config(
            base,
            GpuOverride(
                vram_admission_noise_mb=256,
                measured_load_probe_seconds=0,
                starved_head_lane_reclaim=False,
                starved_head_utilities_pause=False,
                whole_card_models=["qwen_image"],
            ),
        )
        assert card.vram_admission_noise_mb == 256
        assert card.measured_load_probe_seconds == 0
        assert card.starved_head_lane_reclaim is False
        assert card.starved_head_utilities_pause is False
        assert card.whole_card_models == ["qwen_image"]
        assert base.vram_admission_noise_mb is None, "the base config is untouched"


class TestWholeCardPin:
    """A pinned model keeps its whole-card claim whatever the measurement says."""

    def test_a_pinned_model_keeps_its_claim_against_measurement(self) -> None:
        """The measured retirement never applies to a pinned model; an unpinned twin retires as before."""
        forecast = resource_budget.StreamForecast(
            weights_mb=11500.0,
            footprint_mb=16400.0,
            reserve_mb=2048.0,
            base_reserve_mb=1519.0,
            free_now_mb=20000.0,
            free_if_alone_mb=23586.0,
            free_after_model_evict_mb=20000.0,
            total_vram_mb=24074.0,
            per_process_overhead_mb=488.0,
            marginal_process_overhead_mb=487.0,
            wants_whole_card=True,
            measured_footprint_mb=11500.0,
            measured_observation_count=5,
        )
        assert forecast.measured_retires_whole_card_intent is True
        pinned = replace(forecast, whole_card_pinned=True)
        assert pinned.measured_retires_whole_card_intent is False
        assert pinned.needs_exclusive_residency is True


class TestProbeDelay:
    """The probe delay: the configured wait inside the band, the long horizon past it."""

    @staticmethod
    def _state(device_free_mb: float) -> DeviceVramState:
        return DeviceVramState(
            total_vram_mb=24074.0,
            baseline_mb=1000.0,
            committed_vram_mb=2000.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            device_free_mb=device_free_mb,
            noise_buffer_mb=1203.7,
        )

    @staticmethod
    def _head(starved_seconds: float, probe_after_seconds: float | None = None) -> VramRequest:
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="Z-Image-Turbo",
            baseline="z_image_turbo",
            device_index=0,
            target_process_id=2,
            candidate_delta_mb=19424.0,
            is_head_of_queue=True,
            head_job_id="head",
            starved_seconds=starved_seconds,
        )
        if probe_after_seconds is not None:
            request = replace(request, probe_after_seconds=probe_after_seconds)
        return request

    def _probes_at(self, device_free_mb: float, starved_seconds: float, probe_after_seconds: float | None) -> bool:
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._state(device_free_mb)}))
        return arbiter.evaluate(self._head(starved_seconds, probe_after_seconds)).measured_attempt

    def test_a_small_shortfall_probes_after_the_configured_delay(self) -> None:
        """Inside the band the card's probe delay applies (the teardown grace by default), and 0 means at once."""
        assert self._probes_at(20157.0, _FIRST_PARTY_TEARDOWN_GRACE_SECONDS - 1.0, None) is False
        assert self._probes_at(20157.0, _FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0, None) is True
        assert self._probes_at(20157.0, 0.0, 0.0) is True
        assert self._probes_at(20157.0, 30.0, 45.0) is False

    def test_a_large_shortfall_keeps_the_long_horizon(self) -> None:
        """Past the band the ledger is the likelier liar, so the diagnostic horizon still gates the probe."""
        short_by_gigabytes = 20157.0 - 2 * _MEASURED_ATTEMPT_BAND_MB
        assert self._probes_at(short_by_gigabytes, _FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0, None) is False
        assert self._probes_at(short_by_gigabytes, _STARVATION_DIAGNOSTIC_SECONDS + 1.0, None) is True


async def test_a_process_cycle_rearms_a_spent_probe() -> None:
    """A spent probe is forgotten once the process that carried it is gone, so the head can earn one more."""
    tracker = JobTracker()
    job = make_job_pop_response("Z-Image-Turbo")
    await track_popped_job_async(tracker, job)
    tracker.mark_measured_attempt(job, candidate_mb=19424.0, device_index=0)
    assert tracker.has_spent_measured_attempt_on_device(job, 0) is True
    assert tracker.is_measured_attempt_on_device(job, 0) is True

    tracker.rearm_measured_attempt(job)

    assert tracker.has_spent_measured_attempt_on_device(job, 0) is False
    assert tracker.is_measured_attempt_on_device(job, 0) is False
