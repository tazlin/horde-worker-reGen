"""The post-processing fault breaker and the arbiter-derived post-processing offer.

A post-processing peak that cannot be hosted faults the job and the horde reissues it, but a worker that
keeps faulting trips the horde's forced-maintenance (the spiral this guards against). These pin the worker's
self-protective breaker: a rolling-window counter fed by both fault sources (the planner's unhostable-peak
faults and watchdog-reaped post-processing stalls), a trip when the count *exceeds* the threshold, and
recovery once the window is quiet and the VRAM arbiter does not rule the peak out on every driven card. The
pop-time offer is the same arbiter verdict: it is withheld only while every driven card returns DENY for the
representative chain, never on a transient shortage. A structural whole-card disable never auto-recovers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import HordeImageResult
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.vram_arbiter import DeviceVramState, MeasuredVramSnapshot
from tests.process_management.conftest import make_job_pop_response, make_testable_process_manager

_LARGE_CARD_TOTAL_MB = 24000.0
"""A card whose achievable ceiling seats the post-processing peak, so the arbiter can only FITS or DEFER."""

_TINY_CARD_TOTAL_MB = 3000.0
"""A card whose achievable ceiling (total less the 512 MB noise buffer) is below the post-processing peak."""


def _card(*, total_vram_mb: float, device_free_mb: float) -> DeviceVramState:
    """Build one card's frozen measurement with no reservations, so the identity reads device-free directly."""
    return DeviceVramState(
        total_vram_mb=total_vram_mb,
        baseline_mb=0.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=device_free_mb,
    )


def _roomy_card() -> DeviceVramState:
    """A large card with ample free VRAM: the post-processing peak FITS."""
    return _card(total_vram_mb=_LARGE_CARD_TOTAL_MB, device_free_mb=21000.0)


def _full_card(device_free_mb: float = 2000.0) -> DeviceVramState:
    """A large card whose free VRAM is below the post-processing peak: the arbiter DEFERS to its ladder."""
    return _card(total_vram_mb=_LARGE_CARD_TOTAL_MB, device_free_mb=device_free_mb)


def _tiny_card() -> DeviceVramState:
    """A card that could never seat the post-processing peak even emptied: the arbiter DENIES."""
    return _card(total_vram_mb=_TINY_CARD_TOTAL_MB, device_free_mb=2500.0)


def _install_cycle(manager: HordeWorkerProcessManager, devices: dict[int, DeviceVramState]) -> None:
    """Freeze ``devices`` into the manager's real arbiter and mirror their readings as the driven cards."""
    manager._vram_arbiter.begin_cycle(MeasuredVramSnapshot(devices=devices))
    manager._last_device_free_mb_by_device.clear()
    for device_index, state in devices.items():
        assert state.device_free_mb is not None
        manager._last_device_free_mb_by_device[device_index] = state.device_free_mb


@contextmanager
def _captured_messages() -> Iterator[list[str]]:
    """Collect every loguru message emitted inside the block."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="TRACE")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _make_pp_job_info() -> HordeJobInfo:
    """Build a job waiting for the post-processing lane, carrying one raw image result."""
    return HordeJobInfo(
        sdk_api_job_info=make_job_pop_response(post_processing=["RealESRGAN_x4plus"]),
        job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
        state=GENERATION_STATE.ok,
        censored=False,
        time_popped=time.time(),
    )


class TestPostProcessingFaultCounter:
    """The rolling-window counter the breaker reads, fed by both fault sources."""

    def test_counts_within_window_and_excludes_older(self) -> None:
        """Each recorded fault is counted within the window; a zero-width window counts none."""
        job_tracker = JobTracker()
        assert job_tracker.count_recent_post_processing_faults(600) == 0

        job_tracker.note_post_processing_overcommit_fault()
        job_tracker.note_post_processing_overcommit_fault()

        assert job_tracker.count_recent_post_processing_faults(600) == 2
        # A zero-width window excludes faults recorded a moment ago (the boundary is strict-enough to prune).
        assert job_tracker.count_recent_post_processing_faults(-1) == 0


class TestPostProcessingFaultBreaker:
    """The trip/latch behaviour the control loop drives via ``_apply_post_processing_fault_breaker``."""

    def test_trips_only_after_exceeding_threshold_and_latches(self) -> None:
        """The breaker tolerates exactly the threshold and trips on the next fault, then latches."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=True,
            post_processing_fault_threshold=4,
            post_processing_fault_window_seconds=1800,
        )

        # Exactly the threshold is tolerated (the trip is strictly greater-than).
        for _ in range(4):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False

        # One more crosses it: the breaker trips and stamps the time.
        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_tripped_at > 0

    def test_latch_holds_while_faults_remain_in_window(self) -> None:
        """Once tripped the latch persists across later checks while the faults are still in the window."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=600,
        )
        for _ in range(2):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_disabled_flag_prevents_trip(self) -> None:
        """With the breaker disabled, no number of faults latches it off."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=False,
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=1800,
        )
        for _ in range(10):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False


class TestPostProcessingBreakerAutoRecovery:
    """The fault-count breaker re-enables once the window is quiet and the offer probe finds a card."""

    @staticmethod
    def _trip_breaker(manager: HordeWorkerProcessManager) -> None:
        """Drive the breaker to its latched state with two faults at the currently-mocked wall-clock time."""
        for _ in range(2):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()

    def test_stays_disabled_while_every_card_denies_then_re_enables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A quiet window on a card the arbiter denies stays latched; a card that can seat the peak re-enables."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=60,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_auto_recoverable is True

        monkeypatch.setattr(time, "time", lambda: base_time + 61.0)
        _install_cycle(manager, {0: _tiny_card()})
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

        # A full card is a DEFER, which the dispatch ladder resolves: that is enough to re-enable.
        _install_cycle(manager, {0: _full_card()})
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False
        assert manager._state.post_processing_breaker_auto_recoverable is False

    def test_offerable_card_does_not_re_enable_while_faults_remain_in_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A card that fits the peak cannot re-enable while an over-commit fault is still in the window."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=600,
        )
        _install_cycle(manager, {0: _roomy_card()})
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True

        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_no_device_reading_re_enables_on_the_window_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host with no driven-card reading has no card to deny on, so a quiet window re-enables."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=60,
            device_free_mb=None,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True

        monkeypatch.setattr(time, "time", lambda: base_time + 61.0)
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False

    async def test_latch_drain_faults_do_not_rearm_the_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pending work the latch faults mid-window does not count, so the window still elapses on time."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=60,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True

        # Mid-window, a job reaches the post-processing queue and the latch faults it without images.
        monkeypatch.setattr(time, "time", lambda: base_time + 30.0)
        job_info = _make_pp_job_info()
        await manager._job_tracker.queue_for_post_processing(job_info)
        await manager.start_post_processing()
        assert job_info.state == GENERATION_STATE.faulted
        assert manager._job_tracker.count_recent_post_processing_faults(60) == 2

        # Past the window measured from the trip: only the two genuine faults existed, so the latch clears.
        monkeypatch.setattr(time, "time", lambda: base_time + 61.0)
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False

    def test_structural_whole_card_latch_never_auto_recovers(self) -> None:
        """A latch set without the auto-recoverable marker (the whole-card conflict) stays disabled."""
        manager = make_testable_process_manager()
        _install_cycle(manager, {0: _roomy_card()})
        # The scheduler's whole-card disable sets the latch but leaves ``auto_recoverable`` false (default).
        manager._state.post_processing_disabled_by_breaker = True
        manager._state.post_processing_breaker_auto_recoverable = False

        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True


class TestPostProcessingHeadroomGate:
    """The offer is withheld only while the arbiter DENIES the post-processing peak on every driven card."""

    def test_deny_on_the_only_card_withholds_and_logs_once(self) -> None:
        """A structural DENY closes the offer with one warning; a later fit reopens it with one line."""
        manager = make_testable_process_manager()
        _install_cycle(manager, {0: _tiny_card()})

        with _captured_messages() as messages:
            manager._apply_post_processing_headroom_gate()
            manager._apply_post_processing_headroom_gate()
            assert manager._state.post_processing_withheld_for_headroom is True

            _install_cycle(manager, {0: _roomy_card()})
            manager._apply_post_processing_headroom_gate()
            manager._apply_post_processing_headroom_gate()
            assert manager._state.post_processing_withheld_for_headroom is False

        withholding = [message for message in messages if message.startswith("Withholding post-processing")]
        readvertising = [message for message in messages if message.startswith("Re-advertising post-processing")]
        assert len(withholding) == 1
        assert "achievable ceiling" in withholding[0]
        assert len(readvertising) == 1

    def test_defer_keeps_the_offer_open_below_the_peak(self) -> None:
        """Free VRAM below the post-processing peak is a DEFER for the ladder, never a closed offer."""
        manager = make_testable_process_manager()
        _install_cycle(manager, {0: _full_card(device_free_mb=1500.0)})

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is False

    def test_one_fitting_card_of_two_keeps_the_offer_open(self) -> None:
        """A DENY on one driven card does not withhold while another card can seat the peak."""
        manager = make_testable_process_manager()
        _install_cycle(manager, {0: _tiny_card(), 1: _roomy_card()})

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is False

    def test_every_card_denying_withholds(self) -> None:
        """Every driven card denying the peak withholds the offer."""
        manager = make_testable_process_manager()
        _install_cycle(manager, {0: _tiny_card(), 1: _tiny_card()})

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is True

    def test_no_cycle_state_keeps_the_offer_open(self) -> None:
        """A card with no cycle snapshot yields a relaxed FITS, so the offer stays open."""
        manager = make_testable_process_manager(device_free_mb=2000.0)
        assert manager._vram_arbiter.has_cycle is False

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is False

    def test_no_device_reading_keeps_the_offer_open(self) -> None:
        """A host with no driven-card reading has no card to deny on."""
        manager = make_testable_process_manager(device_free_mb=None)

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is False

    def test_disabled_master_switch_forces_the_offer_open(self) -> None:
        """With the self-protection switch off, even a structural DENY does not withhold advertising."""
        manager = make_testable_process_manager(post_processing_fault_breaker_enabled=False)
        _install_cycle(manager, {0: _tiny_card()})
        manager._state.post_processing_withheld_for_headroom = True

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is False

    def test_free_vram_swings_never_close_an_open_offer(self) -> None:
        """Readings that used to close the offer (inside the old band, then above it, then a dip) leave it open."""
        manager = make_testable_process_manager()
        readings_mb = (21000.0, 6000.0, 20000.0, 1400.0, 6000.0, 23000.0)

        with _captured_messages() as messages:
            for reading_mb in readings_mb:
                _install_cycle(manager, {0: _full_card(device_free_mb=reading_mb)})
                manager._apply_post_processing_headroom_gate()
                assert manager._state.post_processing_withheld_for_headroom is False

        assert not [message for message in messages if "post-processing advertising" in message]
