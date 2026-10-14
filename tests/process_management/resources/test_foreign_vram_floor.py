"""Unit tests for the sustained foreign-VRAM floor tracker and the managed text backend as a known tenant.

The tracker smooths the per-tick foreign reading (``total - device_free - worker_footprint``) into the
minimum over a trailing window, on a hand-advanced monotonic clock. Taking the minimum makes a transient
foreign spike unable to raise the floor (and so unable to wrongly deny a servable model), while the warm-up
gate withholds any floor until a full window has been observed so a cold-start transient cannot set it early.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

from loguru import logger

from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.device_memory_source import (
    DeviceMemoryReading,
    DeviceMemorySourceKind,
    ReadingProvenance,
    nvml_reading,
)
from horde_worker_regen.process_management.resources.foreign_vram_floor import (
    FOREIGN_FLOOR_WINDOW_SECONDS,
    ForeignVramFloorTracker,
    ManagedVramTenant,
    floor_with_known_tenant_mb,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.retention import RetentionFit
from tests.process_management.conftest import (
    device_wide_reading,
    device_wide_reading_provider,
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def test_no_floor_until_a_full_window_is_observed() -> None:
    """Within the warm-up window the floor is withheld (None), preserving the pre-foreign boundary."""
    tracker = ForeignVramFloorTracker()
    assert tracker.update(0, 1900.0, now=0.0) is None
    assert tracker.update(0, 1900.0, now=FOREIGN_FLOOR_WINDOW_SECONDS - 1.0) is None
    # The first sample at or beyond a full window of observation yields the sustained minimum.
    assert tracker.update(0, 1900.0, now=FOREIGN_FLOOR_WINDOW_SECONDS) == 1900.0


def test_sustained_minimum_over_the_window() -> None:
    """The reported floor is the minimum foreign reading across the trailing window's samples."""
    tracker = ForeignVramFloorTracker()
    readings = [2200.0, 1900.0, 2100.0, 2000.0]
    floor = None
    for index, foreign in enumerate(readings):
        floor = tracker.update(0, foreign, now=float(index))
    # Still inside warm-up (span 3s << window), so no floor is reported yet.
    assert floor is None
    assert tracker.update(0, 2050.0, now=FOREIGN_FLOOR_WINDOW_SECONDS) == 1900.0


def test_transient_spike_does_not_raise_the_floor() -> None:
    """A brief foreign spike raises the instantaneous reading but not the window minimum."""
    tracker = ForeignVramFloorTracker()
    # A steady 1900 floor established over a full window.
    step = 10.0
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += step
    # A game opens: one sample spikes to 9000. The sustained floor stays at the steady minimum.
    spiked = tracker.update(0, 9000.0, now=now)
    assert spiked == 1900.0


def test_missing_worker_report_contributes_no_sample() -> None:
    """A None reading (children not yet reporting VRAM) adds no sample and holds the current floor."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += 10.0
    established = tracker.update(0, 1900.0, now=now)
    assert established == 1900.0
    # A tick with no measurable foreign reading must not disturb the established floor.
    assert tracker.update(0, None, now=now + 1.0) == 1900.0


def test_stale_samples_age_out_and_the_floor_is_forgotten() -> None:
    """When readings stop for longer than the window, the aged-out history forgets the floor (None)."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += 10.0
    assert tracker.update(0, 1900.0, now=now) == 1900.0
    # A long gap with no reports: every sample falls outside the trailing window and is pruned.
    assert tracker.update(0, None, now=now + 2 * FOREIGN_FLOOR_WINDOW_SECONDS) is None


def test_non_positive_readings_are_not_samples() -> None:
    """Zeros and negatives interleaved with positive readings leave the floor at the positive minimum.

    A reading at or below zero is the worker's own footprint figure meeting the card's use, never evidence that
    nothing foreign is on the card.
    """
    tracker = ForeignVramFloorTracker()
    readings = [3300.0, 0.0, 3250.0, -4345.0, 3400.0, 0.0]
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, readings[int(now / 10.0) % len(readings)], now=now)
        now += 10.0
    assert tracker.update(0, 0.0, now=now) == 3250.0


def test_only_non_positive_readings_learn_no_floor() -> None:
    """A card whose readings never rise above zero keeps the floor unknown, the pre-foreign behaviour."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= 3 * FOREIGN_FLOOR_WINDOW_SECONDS:
        assert tracker.update(0, 0.0 if int(now) % 20 else -500.0, now=now) is None
        now += 10.0


def test_non_positive_readings_do_not_hold_a_floor_past_its_window() -> None:
    """Positive samples still age out while only non-positive readings arrive, so the floor is forgotten."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += 10.0
    assert tracker.update(0, 0.0, now=now) == 1900.0
    last_positive_at = now - 10.0
    while now <= last_positive_at + FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 0.0, now=now)
        now += 10.0
    assert tracker.update(0, 0.0, now=now) is None


def test_per_card_isolation() -> None:
    """Each card keeps its own sample history and floor."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        tracker.update(1, 3500.0, now=now)
        now += 10.0
    assert tracker.update(0, 1900.0, now=now) == 1900.0
    assert tracker.update(1, 3500.0, now=now) == 3500.0


# ---- the managed text backend as a known tenant ---------------------------------------------------------------

_SMALL_CARD_MB = 10240.0
"""A 10 GB card, the size of one a managed text backend can fill on its own."""

_LARGE_CARD_MB = 32768.0

_TENANT_MB = 9500.0
"""The managed backend's measured footprint on the small card."""


def _two_card_scheduler(
    *,
    free_mb_by_device: dict[int, float] | None = None,
) -> tuple[InferenceScheduler, list[ManagedVramTenant | None]]:
    """Return a scheduler over a small card 0 and a large card 1, plus the mutable tenant it reads.

    Each card holds one reporting inference process, so both totals are known. ``free_mb_by_device`` is the
    parent's measured device-free reading per card, or None for a host that has read neither card yet.
    """
    bridge_data = make_mock_bridge_data(safety_on_gpu=True)
    small = make_mock_process_info(1, model_name=None, device_index=0)
    small.total_vram_mb = int(_SMALL_CARD_MB)
    large = make_mock_process_info(2, model_name=None, device_index=1)
    large.total_vram_mb = int(_LARGE_CARD_MB)
    card_runtimes = make_test_card_runtimes(device_indices=(0, 1), config=bridge_data)
    card_runtimes[0] = replace(card_runtimes[0], total_vram_mb=_SMALL_CARD_MB)
    card_runtimes[1] = replace(card_runtimes[1], total_vram_mb=_LARGE_CARD_MB)
    scheduler = _make_inference_scheduler(
        bridge_data=bridge_data,
        process_map=ProcessMap({1: small, 2: large}),
        card_runtimes=card_runtimes,
        device_free_mb=None,
    )
    if free_mb_by_device is not None:
        scheduler.set_device_reading_provider(device_wide_reading_provider(free_mb_by_device.get))
    tenant_box: list[ManagedVramTenant | None] = [None]
    scheduler.set_managed_tenant_provider(lambda: tenant_box[0])
    return scheduler, tenant_box


class TestFloorWithKnownTenant:
    """The standing floor is the larger of the measured floor and the tenant, never their sum."""

    def test_no_figures_keeps_the_floor_unknown(self) -> None:
        """With neither a learned floor nor a tenant the floor stays None, the pre-foreign behaviour."""
        assert floor_with_known_tenant_mb(None, None) is None

    def test_tenant_alone_is_the_floor(self) -> None:
        """A tenant measured before the learned window has covered it is the floor on its own."""
        assert floor_with_known_tenant_mb(None, _TENANT_MB) == _TENANT_MB

    def test_learned_floor_above_the_tenant_is_not_double_counted(self) -> None:
        """A learned floor that already contains the tenant stands as it is."""
        assert floor_with_known_tenant_mb(9800.0, _TENANT_MB) == 9800.0

    def test_tenant_above_a_small_learned_floor_raises_it(self) -> None:
        """A learned floor that has not yet covered the tenant is raised to the tenant's footprint."""
        assert floor_with_known_tenant_mb(600.0, _TENANT_MB) == _TENANT_MB


class TestFloorSamplesOnlyADeviceWideReading:
    """The floor is total less free less the worker's share, which only a device-wide figure can answer."""

    @staticmethod
    def _sampled_foreign_mb(provenance: ReadingProvenance) -> float | None:
        """Return the foreign sample the tracker is handed for card 0 under a reading of ``provenance``."""
        scheduler, _tenant_box = _two_card_scheduler()
        scheduler._process_map[1].process_reserved_mb = 500  # type: ignore[index]
        samples: list[float | None] = []
        track = scheduler._foreign_vram_floor.update

        def record(device_key: int, foreign_now_mb: float | None, *, now: float) -> float | None:
            samples.append(foreign_now_mb)
            return track(device_key, foreign_now_mb, now=now)

        scheduler._foreign_vram_floor.update = record  # type: ignore[method-assign]
        scheduler.build_vram_arbiter_device_state(
            0,
            device_reading=DeviceMemoryReading(
                free_mb=600.0,
                total_mb=_SMALL_CARD_MB,
                source=DeviceMemorySourceKind.CHILD_REPORT,
                provenance=provenance,
            ),
        )
        # The build's reading is sampled first. Later reads in the same build go through the provider, which
        # this scheduler does not have, and hand the tracker None whatever the provenance.
        return samples[0]

    def test_a_device_wide_reading_contributes_a_sample(self) -> None:
        """Other programs are inside a device-wide figure, so what the worker does not hold is theirs."""
        assert self._sampled_foreign_mb(ReadingProvenance.DEVICE_WIDE) is not None

    def test_a_process_local_reading_contributes_none(self) -> None:
        """A process-local figure does not include other programs, so it says nothing about what they hold."""
        assert self._sampled_foreign_mb(ReadingProvenance.PROCESS_LOCAL) is None

    def test_a_host_memory_reading_contributes_none(self) -> None:
        """System RAM in use is no measure of what another program holds on the card."""
        assert self._sampled_foreign_mb(ReadingProvenance.HOST_MEMORY) is None


class TestFloorMeasuresTheWorkerWhereTheDriverReportsPerProcess:
    """Per-process device usage replaces the ledger estimate in the foreign reading wherever the driver gives it."""

    _WORKER_PID = 4242
    _WORKER_USES_MB = 2500.0
    """What the driver lists the worker's process as using on card 0."""
    _EXTERNAL_TENANT_MB = 2000.0
    """Device memory another program holds on card 0, which the driver's compute-process list does not show."""

    def _scheduler_over_card_zero(self, *, ledger_reserved_mb: int) -> tuple[InferenceScheduler, list[float]]:
        """Return a two-card scheduler whose card 0 process reports ``ledger_reserved_mb``, and its clock."""
        scheduler, _tenant_box = _two_card_scheduler()
        worker = scheduler._process_map[1]  # type: ignore[index]
        worker.os_pid = self._WORKER_PID
        worker.process_reserved_mb = ledger_reserved_mb
        scheduler._process_map[2].os_pid = self._WORKER_PID + 1  # type: ignore[index]
        now = [0.0]
        scheduler._foreign_floor_clock = lambda: now[0]
        return scheduler, now

    def _card_zero_reading(self, *, per_process: bool) -> DeviceMemoryReading:
        """Return card 0's NVML reading with the worker and the external tenant on it."""
        free_mb = _SMALL_CARD_MB - self._WORKER_USES_MB - self._EXTERNAL_TENANT_MB
        return nvml_reading(
            free_mb=free_mb,
            total_mb=_SMALL_CARD_MB,
            process_used_mb={self._WORKER_PID: self._WORKER_USES_MB} if per_process else None,
        )

    def _floor_after_one_window(
        self, scheduler: InferenceScheduler, now: list[float], reading: DeviceMemoryReading
    ) -> float | None:
        """Build card 0's arbiter state every 10 s over one full window and return the floor it carries."""
        floor_mb: float | None = None
        while now[0] <= FOREIGN_FLOOR_WINDOW_SECONDS:
            floor_mb = scheduler.build_vram_arbiter_device_state(0, device_reading=reading).foreign_floor_mb
            now[0] += 10.0
        return floor_mb

    def test_the_floor_learns_the_tenant_when_the_ledger_overcharges(self) -> None:
        """A ledger reservation above what the driver lists would read at or below zero; the measurement does not."""
        scheduler, now = self._scheduler_over_card_zero(ledger_reserved_mb=6000)
        reading = self._card_zero_reading(per_process=True)
        ledger_mb = scheduler._process_map.committed_vram_mb(
            context_constant_mb=scheduler.resolved_context_constant_mb(),
            device_index=0,
        )
        assert _SMALL_CARD_MB - reading.free_mb - ledger_mb <= 0

        assert self._floor_after_one_window(scheduler, now, reading) == self._EXTERNAL_TENANT_MB

    def test_without_per_process_figures_the_ledger_estimate_is_used(self) -> None:
        """Under WDDM the driver gives no per-process figure, so the reading subtracts the ledger estimate."""
        scheduler, now = self._scheduler_over_card_zero(ledger_reserved_mb=1000)
        reading = self._card_zero_reading(per_process=False)
        ledger_mb = scheduler._process_map.committed_vram_mb(
            context_constant_mb=scheduler.resolved_context_constant_mb(),
            device_index=0,
        )

        assert self._floor_after_one_window(scheduler, now, reading) == _SMALL_CARD_MB - reading.free_mb - ledger_mb

    def test_a_list_without_any_worker_pid_keeps_the_ledger_estimate(self) -> None:
        """A process list that names none of the worker's pids (another pid namespace) says nothing about it."""
        scheduler, now = self._scheduler_over_card_zero(ledger_reserved_mb=1000)
        reading = nvml_reading(
            free_mb=_SMALL_CARD_MB - self._WORKER_USES_MB - self._EXTERNAL_TENANT_MB,
            total_mb=_SMALL_CARD_MB,
            process_used_mb={1: self._WORKER_USES_MB},
        )
        ledger_mb = scheduler._process_map.committed_vram_mb(
            context_constant_mb=scheduler.resolved_context_constant_mb(),
            device_index=0,
        )

        assert self._floor_after_one_window(scheduler, now, reading) == _SMALL_CARD_MB - reading.free_mb - ledger_mb


class TestManagedTenantIsChargedOnItsCard:
    """A serving managed text backend is a standing floor on its card, seen by every reader of the floor."""

    def test_the_floor_holds_the_footprint_before_any_window_elapses(self) -> None:
        """The snapshot, the arbiter state and the ceiling all see the tenant with no learned floor yet."""
        scheduler, tenant_box = _two_card_scheduler(free_mb_by_device={0: 600.0, 1: 30000.0})
        assert scheduler.current_foreign_floor_mb(0) is None

        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)

        assert scheduler.current_foreign_floor_mb(0) == _TENANT_MB
        assert scheduler.current_foreign_floor_mb(1) is None
        card_zero = scheduler.snapshot().card(0)
        assert card_zero.foreign_floor_mb is not None
        assert card_zero.foreign_floor_mb >= _TENANT_MB
        device_state = scheduler.build_vram_arbiter_device_state(
            0,
            device_reading=device_wide_reading(600.0, _SMALL_CARD_MB),
        )
        assert device_state.foreign_floor_mb is not None
        assert device_state.foreign_floor_mb >= _TENANT_MB
        ceiling_mb = scheduler.achievable_ceiling_mb(0)
        assert ceiling_mb is not None
        assert ceiling_mb == device_state.achievable_ceiling_mb()
        assert ceiling_mb < _SMALL_CARD_MB - _TENANT_MB

    def test_a_stopped_backend_releases_the_floor(self) -> None:
        """A backend that stops serving takes its standing floor with it."""
        scheduler, tenant_box = _two_card_scheduler()
        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)
        assert scheduler.current_foreign_floor_mb(0) == _TENANT_MB

        tenant_box[0] = None

        assert scheduler.current_foreign_floor_mb(0) is None
        ceiling_mb = scheduler.achievable_ceiling_mb(0)
        assert ceiling_mb is not None
        assert ceiling_mb > _SMALL_CARD_MB - _TENANT_MB

    def test_a_learned_floor_above_the_footprint_is_not_double_counted(self) -> None:
        """Once the learned window has covered the tenant, the floor is the learned figure alone."""
        scheduler, tenant_box = _two_card_scheduler()
        now = [0.0]
        scheduler._foreign_floor_clock = lambda: now[0]
        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)
        while now[0] <= FOREIGN_FLOOR_WINDOW_SECONDS:
            scheduler._foreign_vram_floor.update(0, 9800.0, now=now[0])
            now[0] += 10.0

        assert scheduler.current_foreign_floor_mb(0) == 9800.0

    def test_the_charge_is_logged_once_with_card_footprint_and_ceiling(self) -> None:
        """One INFO line when the tenant is charged; repeated reads say nothing more."""
        scheduler, tenant_box = _two_card_scheduler()
        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
        try:
            for _ in range(5):
                scheduler.current_foreign_floor_mb(0)
                scheduler.achievable_ceiling_mb(0)
        finally:
            logger.remove(sink_id)

        charged = [message for message in messages if "managed text backend holds" in message]
        assert len(charged) == 1
        ceiling_mb = scheduler.achievable_ceiling_mb(0)
        assert ceiling_mb is not None
        assert "device 0" in charged[0]
        assert f"{_TENANT_MB:.0f} MB" in charged[0]
        assert f"{ceiling_mb:.0f} MB" in charged[0]

    def test_the_safety_chooser_prefers_the_card_whose_room_holds_safety(self) -> None:
        """Card 0 reads more free now, but its tenant leaves less room than safety costs, so card 1 wins."""
        scheduler, tenant_box = _two_card_scheduler(free_mb_by_device={0: 700.0, 1: 500.0})
        assert scheduler._choose_safety_gpu_card() == 0

        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)

        room_mb = scheduler.achievable_ceiling_mb(0)
        assert room_mb is not None
        assert room_mb < scheduler._safety_footprint_mb()
        assert scheduler._choose_safety_gpu_card() == 1

    def test_the_safety_chooser_without_readings_charges_the_tenant(self) -> None:
        """With no measured free VRAM the larger card wins on its total, until a tenant fills it."""
        scheduler, tenant_box = _two_card_scheduler()
        assert scheduler._choose_safety_gpu_card() == 1

        tenant_box[0] = ManagedVramTenant(device_index=1, footprint_mb=_LARGE_CARD_MB - 2000.0)

        assert scheduler._choose_safety_gpu_card() == 0


class TestManagedTenantServiceability:
    """Model serviceability judges a card against the tenant, so the offer drops what cannot fit beside it."""

    def test_the_serviceability_baseline_is_the_larger_figure(self) -> None:
        """The reconciler's baseline and the tenant are never added."""
        scheduler, tenant_box = _two_card_scheduler()
        scheduler.set_admission_baseline_provider(lambda _device_index: 1024.0)
        assert scheduler.serviceability_baseline_mb(0) == 1024.0

        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)

        assert scheduler.serviceability_baseline_mb(0) == _TENANT_MB
        assert scheduler.serviceability_baseline_mb(1) == 1024.0
        scheduler.set_admission_baseline_provider(lambda _device_index: 9900.0)
        assert scheduler.serviceability_baseline_mb(0) == 9900.0

    def test_a_model_that_cannot_fit_beside_the_tenant_is_unserviceable(self) -> None:
        """A model the card serves alone becomes unserviceable there once the tenant is charged."""
        model = "sd15_model"
        bridge_data = make_mock_bridge_data(image_models_to_load=[model])
        reference = {model: make_mock_model_reference_record(model)}
        process_info = make_mock_process_info(0, model_name=None)
        process_info.total_vram_mb = int(_SMALL_CARD_MB)
        scheduler = _make_inference_scheduler(
            bridge_data=bridge_data,
            model_metadata=make_test_model_metadata(reference),
            process_map=ProcessMap({0: process_info}),
            card_runtimes=make_test_card_runtimes(config=bridge_data, total_vram_mb=_SMALL_CARD_MB),
        )
        tenant_box: list[ManagedVramTenant | None] = [None]
        scheduler.set_managed_tenant_provider(lambda: tenant_box[0])
        job = make_job_pop_response(model)
        assert scheduler._unserviceable_job_reason(job) is None

        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=_TENANT_MB)

        reason = scheduler._unserviceable_job_reason(job)
        assert reason is not None
        assert "cannot fit any serving card" in reason


class TestManagedTenantIsChargedByRetention:
    """The retention static fit prices from the card total, so the tenant must be charged there too."""

    def test_a_tenant_of_n_mb_reduces_the_retention_fit_by_n(self) -> None:
        """A job whose peak fits beside the retained residents stops fitting once a tenant holds the card."""
        tenant_mb = 4000.0
        scheduler, tenant_box = _two_card_scheduler()
        target = scheduler._process_map[1]
        predicted_box = [0.0]

        def fixed_peak(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
            return Mock(fits=True, predicted_mb=predicted_box[0], reserve_mb=0.0)

        scheduler._vram_budget.check_job = fixed_peak  # type: ignore[method-assign]
        job = make_job_pop_response("sd15_model")

        without_tenant = scheduler._retention_fit(job, target=target, device_index=0, include_target_retained=True)
        assert isinstance(without_tenant, RetentionFit)
        assert without_tenant.foreign_floor_mb == 0.0
        predicted_box[0] = without_tenant.effective_available_mb - without_tenant.noise_mb - tenant_mb / 2
        fits_alone = scheduler._retention_fit(job, target=target, device_index=0, include_target_retained=True)
        assert isinstance(fits_alone, RetentionFit)
        assert fits_alone.granted is True

        tenant_box[0] = ManagedVramTenant(device_index=0, footprint_mb=tenant_mb)

        with_tenant = scheduler._retention_fit(job, target=target, device_index=0, include_target_retained=True)
        assert isinstance(with_tenant, RetentionFit)
        assert with_tenant.foreign_floor_mb == tenant_mb
        assert with_tenant.effective_available_mb == fits_alone.effective_available_mb - tenant_mb
        assert with_tenant.granted is False
        assert f"foreign floor {tenant_mb:.0f}MB" in with_tenant.describe()

    def test_a_tenant_on_another_card_leaves_the_retention_fit_alone(self) -> None:
        """The tenant is charged on its own card only."""
        scheduler, tenant_box = _two_card_scheduler()
        target = scheduler._process_map[1]
        job = make_job_pop_response("sd15_model")
        tenant_box[0] = ManagedVramTenant(device_index=1, footprint_mb=_TENANT_MB)

        fit = scheduler._retention_fit(job, target=target, device_index=0, include_target_retained=True)

        assert isinstance(fit, RetentionFit)
        assert fit.foreign_floor_mb == 0.0
