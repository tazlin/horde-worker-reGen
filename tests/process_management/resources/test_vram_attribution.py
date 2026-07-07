"""Tests for the observational per-process VRAM attribution layer.

Covers the three pure pieces of the measurement + reconciliation plumbing:

* the platform-aware per-process context-constant resolution,
* the committed-VRAM ledger (``ProcessMap.committed_vram_mb``), and
* the drift reconciler (baseline capture, drift arithmetic, persistent-vs-transient warning).
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessMemoryMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.resource_budget import (
    _LINUX_CONTEXT_CONSTANT_MB,
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    _WIN32_CONTEXT_CONSTANT_MB,
    platform_context_constant_mb,
)
from horde_worker_regen.process_management.resources.vram_attribution import (
    _LEDGER_PHANTOM_TOLERANCE_MB,
    _RECALIBRATE_INTERVAL_SECONDS,
    _REPORT_STALENESS_SECONDS,
    DriftObservation,
    VramAttributionReconciler,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


class TestPlatformContextConstant:
    """The per-process context charge resolves measured-first, then platform seed, then generic fallback."""

    def test_windows_seed(self) -> None:
        """With no measured marginal, Windows uses the 243 MB probed seed."""
        assert platform_context_constant_mb(None, platform="win32") == _WIN32_CONTEXT_CONSTANT_MB

    def test_linux_seed(self) -> None:
        """With no measured marginal, Linux uses the 144 MB probed seed."""
        assert platform_context_constant_mb(None, platform="linux") == _LINUX_CONTEXT_CONSTANT_MB

    def test_unknown_platform_uses_generic_seed(self) -> None:
        """An unknown platform falls back to the generic marginal-context seed."""
        assert platform_context_constant_mb(None, platform="darwin") == _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB

    def test_measured_marginal_overrides_platform_seed(self) -> None:
        """A measured marginal (> 0) wins over the platform seed on every platform."""
        assert platform_context_constant_mb(310.0, platform="win32") == 310.0
        assert platform_context_constant_mb(310.0, platform="linux") == 310.0

    def test_non_positive_marginal_ignored(self) -> None:
        """A zero or negative measured marginal is ignored in favour of the seed."""
        assert platform_context_constant_mb(0.0, platform="linux") == _LINUX_CONTEXT_CONSTANT_MB
        assert platform_context_constant_mb(-5.0, platform="win32") == _WIN32_CONTEXT_CONSTANT_MB


class TestCommittedVramLedger:
    """``committed_vram_mb`` sums ``context_constant + process_reserved_mb + process_aimdo_mb`` over live procs."""

    def _reporting_process(
        self,
        process_id: int,
        reserved_mb: int | None,
        *,
        aimdo_mb: int | None = None,
        device_index: int = 0,
        state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
        process_type: HordeProcessType = HordeProcessType.INFERENCE,
    ) -> object:
        info = make_mock_process_info(
            process_id,
            state=state,
            process_type=process_type,
            device_index=device_index,
        )
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.process_aimdo_mb = aimdo_mb  # type: ignore[attr-defined]
        return info

    def test_sums_context_plus_reserved(self) -> None:
        """Two GPU processes each contribute context_constant + their own reserved figure."""
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 6000),
                1: self._reporting_process(1, 2000),
            },
        )
        # (200 + 6000) + (200 + 2000)
        assert process_map.committed_vram_mb(context_constant_mb=200.0) == 8400.0

    def test_includes_aimdo_pool_in_footprint(self) -> None:
        """An INFERENCE child's direct-IO weight pool is charged on top of context + reserved (disjoint terms).

        Regression for the attribution hole: a child measured at ~24MB torch-reserved while owning ~10GB of
        weights in the native pool must contribute those weights, or the ledger silently under-counts a nearly
        full card.
        """
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 24, aimdo_mb=10000),
            },
        )
        # 200 (ctx) + 24 (reserved) + 10000 (aimdo)
        assert process_map.committed_vram_mb(context_constant_mb=200.0) == 10224.0

    def test_missing_aimdo_report_is_treated_as_zero(self) -> None:
        """A process that reports reserved but no aimdo figure (the torch-only lane) charges reserved only."""
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 6000, aimdo_mb=None),
            },
        )
        assert process_map.committed_vram_mb(context_constant_mb=200.0) == 6200.0

    def test_excludes_processes_without_reserved_report(self) -> None:
        """A process that has not reported an allocator reservation (CPU / cold start) is not charged."""
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 6000),
                1: self._reporting_process(1, None),
            },
        )
        assert process_map.committed_vram_mb(context_constant_mb=200.0) == 6200.0

    def test_excludes_terminal_processes(self) -> None:
        """A process in a terminal shutdown state no longer commits VRAM and is excluded."""
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 6000),
                1: self._reporting_process(1, 4000, state=HordeProcessState.PROCESS_ENDED),
            },
        )
        assert process_map.committed_vram_mb(context_constant_mb=200.0) == 6200.0

    def test_device_index_filter(self) -> None:
        """A device filter charges only the processes pinned to that card."""
        process_map = ProcessMap(
            {
                0: self._reporting_process(0, 6000, device_index=0),
                1: self._reporting_process(1, 3000, device_index=1),
            },
        )
        assert process_map.committed_vram_mb(context_constant_mb=100.0, device_index=1) == 3100.0

    def test_empty_map_is_zero(self) -> None:
        """No GPU-reporting processes means nothing is committed."""
        assert ProcessMap().committed_vram_mb(context_constant_mb=200.0) == 0.0


class TestDriftReconciler:
    """Baseline capture, drift arithmetic, and the persistent-vs-transient warning gate."""

    def test_baseline_is_min_device_used_while_no_model_resident(self) -> None:
        """The baseline captures the minimum quiet reading and ignores readings with a model resident."""
        reconciler = VramAttributionReconciler()
        reconciler.note_baseline(1200.0, any_model_resident=False)
        reconciler.note_baseline(1000.0, any_model_resident=False)
        reconciler.note_baseline(500.0, any_model_resident=True)  # ignored: a model is resident
        assert reconciler.baseline_estimate_mb == 1000.0

    def test_drift_is_used_minus_baseline_plus_committed(self) -> None:
        """Drift is device_used - (baseline + committed)."""
        reconciler = VramAttributionReconciler()
        reconciler.note_baseline(1000.0, any_model_resident=False)
        observation = reconciler.observe(device_used_mb=8000.0, committed_vram_mb=6000.0, now=0.0)
        assert observation.drift_mb == 1000.0

    def test_no_warn_without_baseline(self) -> None:
        """Without a captured baseline the drift is uncomputable and never warns."""
        reconciler = VramAttributionReconciler()
        observation = reconciler.observe(device_used_mb=8000.0, committed_vram_mb=1000.0, now=0.0)
        assert observation.drift_mb is None
        assert observation.should_warn is False

    def test_no_warn_without_device_used(self) -> None:
        """Without a device-used reading the reconciliation degrades and never warns."""
        reconciler = VramAttributionReconciler()
        reconciler.note_baseline(1000.0, any_model_resident=False)
        observation = reconciler.observe(device_used_mb=None, committed_vram_mb=1000.0, now=0.0)
        assert observation.drift_mb is None
        assert observation.should_warn is False

    def test_warns_only_on_persistent_positive_drift(self) -> None:
        """A single over-threshold observation does not warn; two consecutive ones do."""
        reconciler = VramAttributionReconciler(
            drift_warn_threshold_mb=1024.0,
            consecutive_observations=2,
            warn_interval_seconds=60.0,
        )
        reconciler.note_baseline(1000.0, any_model_resident=False)
        first = reconciler.observe(device_used_mb=1000.0 + 2000.0, committed_vram_mb=0.0, now=0.0)
        assert first.drift_mb == 2000.0
        assert first.should_warn is False  # first over-threshold reading, streak == 1
        second = reconciler.observe(device_used_mb=1000.0 + 2000.0, committed_vram_mb=0.0, now=1.0)
        assert second.should_warn is True  # streak == 2

    def test_transient_spike_does_not_warn(self) -> None:
        """An over-threshold reading followed by an in-threshold reading resets the streak (no warning)."""
        reconciler = VramAttributionReconciler(consecutive_observations=2, warn_interval_seconds=60.0)
        reconciler.note_baseline(1000.0, any_model_resident=False)
        reconciler.observe(device_used_mb=1000.0 + 2000.0, committed_vram_mb=0.0, now=0.0)
        # Committed now accounts for the VRAM, so drift falls back under threshold.
        settled = reconciler.observe(device_used_mb=1000.0 + 2000.0, committed_vram_mb=2000.0, now=1.0)
        assert settled.drift_mb == 0.0
        assert settled.consecutive_over_threshold == 0
        assert settled.should_warn is False

    def test_warning_is_rate_limited(self) -> None:
        """Sustained drift warns at most once per interval."""
        reconciler = VramAttributionReconciler(consecutive_observations=2, warn_interval_seconds=60.0)
        reconciler.note_baseline(1000.0, any_model_resident=False)
        reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=0.0)
        warned = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=1.0)
        assert warned.should_warn is True
        within_interval = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=30.0)
        assert within_interval.should_warn is False
        after_interval = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=61.0)
        assert after_interval.should_warn is True


class TestMemoryMessageRoundTrip:
    """The new per-process attribution fields survive the message default/explicit round-trip."""

    def test_defaults_none(self) -> None:
        """A memory message without the new fields defaults them to None (older children)."""
        message = HordeProcessMemoryMessage(
            process_id=0,
            process_launch_identifier=0,
            info="m",
            ram_usage_bytes=1,
        )
        assert message.process_allocated_mb is None
        assert message.process_reserved_mb is None
        assert message.process_peak_reserved_mb is None
        assert message.process_aimdo_mb is None
        assert message.sampled_at is None

    def test_carries_explicit_values(self) -> None:
        """Explicit per-process figures round-trip through the message model."""
        message = HordeProcessMemoryMessage(
            process_id=0,
            process_launch_identifier=0,
            info="m",
            ram_usage_bytes=1,
            process_allocated_mb=5000,
            process_reserved_mb=6000,
            process_peak_reserved_mb=6500,
            process_aimdo_mb=10000,
            sampled_at=1234.5,
        )
        assert message.process_allocated_mb == 5000
        assert message.process_reserved_mb == 6000
        assert message.process_peak_reserved_mb == 6500
        assert message.process_aimdo_mb == 10000
        assert message.sampled_at == 1234.5

    def test_on_memory_report_stores_fields(self) -> None:
        """``ProcessMap.on_memory_report`` persists the per-process figures onto the process info."""
        info = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: info})
        process_map.on_memory_report(
            process_id=0,
            ram_usage_bytes=1,
            vram_usage_mb=100,
            total_vram_mb=16000,
            process_reserved_mb=6000,
            process_allocated_mb=5000,
            process_peak_reserved_mb=6500,
            process_aimdo_mb=10000,
            report_sampled_at=1234.5,
        )
        assert info.process_reserved_mb == 6000
        assert info.process_allocated_mb == 5000
        assert info.process_peak_reserved_mb == 6500
        assert info.process_aimdo_mb == 10000
        assert info.report_sampled_at == 1234.5


class TestCommittedReportAge:
    """``ProcessMap.oldest_committed_report_age_seconds`` ages the least-fresh committed-ledger contributor."""

    def _contributor(
        self,
        process_id: int,
        *,
        reserved_mb: int | None,
        sampled_at: float | None,
        state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    ) -> object:
        info = make_mock_process_info(process_id, state=state)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.report_sampled_at = sampled_at  # type: ignore[attr-defined]
        return info

    def test_none_when_no_contributors(self) -> None:
        """With nobody charged to the ledger there is nothing to age."""
        process_map = ProcessMap({0: self._contributor(0, reserved_mb=None, sampled_at=100.0)})
        assert process_map.oldest_committed_report_age_seconds(now=200.0) is None

    def test_returns_max_age_over_contributors(self) -> None:
        """The oldest (least-fresh) contributor's age is the ledger's age."""
        process_map = ProcessMap(
            {
                0: self._contributor(0, reserved_mb=6000, sampled_at=190.0),
                1: self._contributor(1, reserved_mb=2000, sampled_at=170.0),
            },
        )
        assert process_map.oldest_committed_report_age_seconds(now=200.0) == 30.0

    def test_missing_sampled_at_is_infinite(self) -> None:
        """A contributor that never carried a sample timestamp cannot be dated, so it is maximally stale."""
        process_map = ProcessMap(
            {
                0: self._contributor(0, reserved_mb=6000, sampled_at=None),
                1: self._contributor(1, reserved_mb=2000, sampled_at=195.0),
            },
        )
        assert process_map.oldest_committed_report_age_seconds(now=200.0) == float("inf")


class TestStalenessAwareReconciliation:
    """A stale committed ledger is an UNKNOWN tenant: the reconciler skips drift computation and never warns."""

    def test_stale_ledger_suppresses_warning_and_resets_streak(self) -> None:
        """A stale observation returns no drift, resets the streak, and cannot warn even mid-streak."""
        reconciler = VramAttributionReconciler(consecutive_observations=2, warn_interval_seconds=60.0)
        reconciler.note_baseline(1000.0, any_model_resident=False)
        first = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=0.0)
        assert first.consecutive_over_threshold == 1
        stale = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, committed_is_stale=True, now=1.0)
        assert stale.drift_mb is None
        assert stale.should_warn is False
        assert stale.consecutive_over_threshold == 0

    def test_fresh_ledger_after_stale_window_reconciles_from_scratch(self) -> None:
        """Once reports are fresh again the streak rebuilds; a lone post-staleness reading does not warn."""
        reconciler = VramAttributionReconciler(consecutive_observations=2, warn_interval_seconds=60.0)
        reconciler.note_baseline(1000.0, any_model_resident=False)
        reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, committed_is_stale=True, now=0.0)
        resumed = reconciler.observe(device_used_mb=4000.0, committed_vram_mb=0.0, now=1.0)
        assert resumed.drift_mb == 3000.0
        assert resumed.consecutive_over_threshold == 1
        assert resumed.should_warn is False


class TestPhysicalPressureTrigger:
    """The no-candidate pressure trigger fires on physical overcommit (committed + baseline > total)."""

    def _reconciler(self) -> VramAttributionReconciler:
        reconciler = VramAttributionReconciler()
        reconciler.note_baseline(1700.0, any_model_resident=False)
        return reconciler

    def test_streak_confirms_before_firing(self) -> None:
        """A single physical-overcommit observation does not fire; the confirming streak does, once."""
        reconciler = self._reconciler()
        # committed 15000 + baseline 1700 = 16700 > total 16375: physically over-committed.
        first = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        assert first.over_physical_ceiling is True
        assert first.should_unload is False  # one transient observation is not enough
        second = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        assert second.should_unload is True
        assert second.consecutive_over_ceiling == 2

    def test_hysteresis_suppresses_reissue_until_it_clears(self) -> None:
        """After firing, a sustained over-commit does not re-fire until it drops below the ceiling."""
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        fired = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        assert fired.should_unload is True
        # Still over-committed: suppressed, no repeat unload.
        held = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        assert held.over_physical_ceiling is True
        assert held.should_unload is False
        # Clears below the ceiling, then over-commits again: eligible to fire once more after the streak.
        reconciler.observe_physical_pressure(committed_vram_mb=5000.0, total_vram_mb=16375.0)
        reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        refired = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        assert refired.should_unload is True

    def test_admission_ceiling_exceedance_without_physical_overcommit_does_not_fire(self) -> None:
        """A committed sum above the admission ceiling but within the physical total never fires a pressure unload."""
        reconciler = self._reconciler()
        # Admission ceiling = (16375 - 1700) - 512 = 14163. Committed 14500 exceeds it, but
        # committed + baseline = 16200 < 16375, so there is no physical overcommit: must not fire.
        first = reconciler.observe_physical_pressure(committed_vram_mb=14500.0, total_vram_mb=16375.0)
        second = reconciler.observe_physical_pressure(committed_vram_mb=14500.0, total_vram_mb=16375.0)
        assert first.over_physical_ceiling is False
        assert second.over_physical_ceiling is False
        assert second.should_unload is False

    def test_stale_or_uncomputable_resets_and_never_fires(self) -> None:
        """A stale ledger, missing baseline, or unknown total resets the streak and signals no unload."""
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=16375.0)
        stale = reconciler.observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
            committed_is_stale=True,
        )
        assert stale.should_unload is False
        assert stale.consecutive_over_ceiling == 0
        no_total = reconciler.observe_physical_pressure(committed_vram_mb=15000.0, total_vram_mb=None)
        assert no_total.should_unload is False
        no_baseline = VramAttributionReconciler().observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
        )
        assert no_baseline.should_unload is False


class TestPhantomOvercommitRecalibration:
    """A committed over-commit the device-used truth contradicts recalibrates the ledger, never wedges on it.

    Reconstructs the structural wedge: the committed ledger read ~21926MB while the card physically held ~1288MB
    (device nearly empty, model map empty, lanes idle). The over-commit is a phantom (stale allocator-cache
    reservation no eviction can cure); the correct response is to recalibrate the ledger to device truth and
    never latch the pressure path suppressed.
    """

    def _reconciler(self) -> VramAttributionReconciler:
        reconciler = VramAttributionReconciler()
        reconciler.note_baseline(1499.0, any_model_resident=False)
        return reconciler

    def test_phantom_signals_recalibrate_not_unload(self) -> None:
        """Committed 21926 while device holds only 1288 is a phantom: recalibrate (after the streak), never evict."""
        reconciler = self._reconciler()
        first = reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=0.0,
        )
        assert first.over_physical_ceiling is True
        assert first.ledger_phantom is True
        assert first.should_unload is False
        assert first.should_recalibrate is False  # one transient reading is not enough
        second = reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=1.0,
        )
        assert second.should_recalibrate is True
        assert second.should_unload is False

    def test_phantom_never_latches_suppressed(self) -> None:
        """A persistent phantom recalibrates once per interval forever; it never suppresses (the wedge fix).

        The old pressure path latched ``_pressure_suppressed`` on a figure committed could not drop below,
        wedging admission permanently. A phantom must instead stay eligible to recalibrate.
        """
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=0.0,
        )
        fired = reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=1.0,
        )
        assert fired.should_recalibrate is True
        within_interval = reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=5.0,
        )
        assert within_interval.ledger_phantom is True
        assert within_interval.should_recalibrate is False  # rate-limited, not suppressed
        after_interval = reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=_RECALIBRATE_INTERVAL_SECONDS + 2.0,
        )
        assert after_interval.should_recalibrate is True  # still eligible: never wedged

    def test_recalibration_clears_once_committed_tracks_truth(self) -> None:
        """Once the recalibrated lanes re-report and committed falls below the ceiling, the phantom clears."""
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=0.0,
        )
        reconciler.observe_physical_pressure(
            committed_vram_mb=21926.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=1.0,
        )
        # Lanes released their cache and re-reported: committed now matches device truth, under the ceiling.
        cleared = reconciler.observe_physical_pressure(
            committed_vram_mb=1500.0,
            total_vram_mb=16375.0,
            device_used_mb=1288.0,
            now=2.0,
        )
        assert cleared.over_physical_ceiling is False
        assert cleared.ledger_phantom is False
        assert cleared.should_recalibrate is False
        assert cleared.should_unload is False

    def test_corroborated_overcommit_still_unloads(self) -> None:
        """A real over-commit the device-used truth corroborates evicts as before, never recalibrates.

        committed 15000 + baseline 1499 = 16499 > 16375, and device_used 16200 corroborates the card is
        genuinely near full, so this is not a phantom: the idle-model reclaim path fires.
        """
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
            device_used_mb=16200.0,
            now=0.0,
        )
        second = reconciler.observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
            device_used_mb=16200.0,
            now=1.0,
        )
        assert second.ledger_phantom is False
        assert second.should_unload is True
        assert second.should_recalibrate is False

    def test_missing_device_used_trusts_ledger_and_unloads(self) -> None:
        """With no device-used reading the ledger is trusted (legacy behaviour): the unload path, not recalibrate."""
        reconciler = self._reconciler()
        reconciler.observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
            device_used_mb=None,
            now=0.0,
        )
        second = reconciler.observe_physical_pressure(
            committed_vram_mb=15000.0,
            total_vram_mb=16375.0,
            device_used_mb=None,
            now=1.0,
        )
        assert second.should_unload is True
        assert second.should_recalibrate is False
        assert second.ledger_phantom is False

    def test_phantom_tolerance_boundary(self) -> None:
        """A committed sum within the phantom tolerance of device-used is corroborated, not a phantom."""
        reconciler = self._reconciler()
        # committed just over the physical ceiling but within tolerance of device_used: corroborated.
        device_used = 16000.0
        committed = device_used + _LEDGER_PHANTOM_TOLERANCE_MB - 1.0
        reconciler.observe_physical_pressure(
            committed_vram_mb=committed,
            total_vram_mb=16375.0,
            device_used_mb=device_used,
            now=0.0,
        )
        second = reconciler.observe_physical_pressure(
            committed_vram_mb=committed,
            total_vram_mb=16375.0,
            device_used_mb=device_used,
            now=1.0,
        )
        assert second.ledger_phantom is False
        assert second.should_unload is True


class TestCommittedLedgerHygiene:
    """The committed ledger charges each reporting process once and drops a replaced incarnation's charge."""

    def test_reporting_process_charged_once_no_context_double_count(self) -> None:
        """``memory_reserved`` excludes the CUDA context, so a reporting lane is charged ctx + reserved exactly once.

        Pins the no-double-charge invariant the audit established: the context constant is additive to reserved
        (which does not already contain it), never folded in twice.
        """
        info = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        info.process_reserved_mb = 6000
        info.process_aimdo_mb = None
        process_map = ProcessMap({0: info})
        assert process_map.committed_vram_mb(context_constant_mb=243.0) == 6243.0

    def test_process_replacement_drops_old_incarnation_charge(self) -> None:
        """``on_process_ending`` (retire/replace) clears the reserved figure so the dead charge leaves at once.

        Rebuild hygiene: a soft reset that replaces a lane must not carry the old incarnation's footprint into the
        new one's charge. The fresh incarnation re-enters the ledger only when it re-reports a reservation.
        """
        info = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        info.process_reserved_mb = 6000
        info.report_sampled_at = time.time()
        process_map = ProcessMap({0: info})
        assert process_map.committed_vram_mb(context_constant_mb=243.0) == 6243.0
        process_map.on_process_ending(0)
        assert process_map.committed_vram_mb(context_constant_mb=243.0) == 0.0
        assert process_map.oldest_committed_report_age_seconds(now=time.time()) is None


class TestSchedulerLedgerRecalibration:
    """``recalibrate_committed_ledger`` asks only idle, reporting, live lanes to release their allocator cache."""

    def _reporting_lane(
        self,
        process_id: int,
        *,
        reserved_mb: int | None,
        state: HordeProcessState,
    ) -> HordeProcessInfo:
        info = make_mock_process_info(process_id, model_name=None, state=state)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        return info

    def test_releases_cache_on_idle_reporting_lanes_only(self) -> None:
        """Busy, terminal, and never-reported lanes are skipped; each idle reporting lane is asked exactly once."""
        pm = make_testable_process_manager()
        scheduler = pm._inference_scheduler
        asked: list[int] = []
        scheduler.release_allocator_cache = Mock(  # type: ignore[attr-defined]
            side_effect=lambda pid: asked.append(pid) or True,
        )
        pm._process_map.clear()
        pm._process_map[1] = self._reporting_lane(1, reserved_mb=5000, state=HordeProcessState.WAITING_FOR_JOB)  # type: ignore[index]
        pm._process_map[2] = self._reporting_lane(2, reserved_mb=6000, state=HordeProcessState.INFERENCE_STARTING)  # type: ignore[index]
        pm._process_map[3] = self._reporting_lane(3, reserved_mb=None, state=HordeProcessState.WAITING_FOR_JOB)  # type: ignore[index]
        pm._process_map[4] = self._reporting_lane(4, reserved_mb=4000, state=HordeProcessState.PROCESS_ENDED)  # type: ignore[index]

        acted = scheduler.recalibrate_committed_ledger(device_index=0)

        assert asked == [1]  # only the idle, reporting, live lane
        assert acted == 1

    def test_device_index_scopes_recalibration(self) -> None:
        """Only lanes pinned to the named card are asked to release."""
        pm = make_testable_process_manager()
        scheduler = pm._inference_scheduler
        asked: list[int] = []
        scheduler.release_allocator_cache = Mock(  # type: ignore[attr-defined]
            side_effect=lambda pid: asked.append(pid) or True,
        )
        card0 = self._reporting_lane(1, reserved_mb=5000, state=HordeProcessState.WAITING_FOR_JOB)
        card1 = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=1)
        card1.process_reserved_mb = 5000
        pm._process_map.clear()
        pm._process_map[1] = card0  # type: ignore[index]
        pm._process_map[2] = card1  # type: ignore[index]

        acted = scheduler.recalibrate_committed_ledger(device_index=1)

        assert asked == [2]
        assert acted == 1


class TestManagerPhantomRecalibration:
    """The manager routes a device-contradicted committed over-commit to recalibration, never an eviction/wedge."""

    def _phantom_contributor(self, process_id: int, *, reserved_mb: int) -> object:
        info = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.report_sampled_at = time.time()  # type: ignore[attr-defined]
        return info

    def _prime_baseline(self, pm: HordeWorkerProcessManager) -> None:
        quiet = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        quiet.process_reserved_mb = None
        pm._process_map.clear()
        pm._process_map[0] = quiet  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=1499.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

    def test_phantom_overcommit_recalibrates_and_does_not_evict(self) -> None:
        """Episode-2 shape: empty map, idle lane, committed ~21926 but device ~1288 -> recalibrate, never evict."""
        pm = make_testable_process_manager()
        pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=243.0)  # type: ignore[attr-defined]
        reclaim = Mock(return_value=True)
        pm._inference_scheduler.reclaim_one_idle_model_under_pressure = reclaim  # type: ignore[attr-defined]
        recalibrate = Mock(return_value=1)
        pm._inference_scheduler.recalibrate_committed_ledger = recalibrate  # type: ignore[attr-defined]
        pm._process_map.get_reported_total_vram_mb = Mock(return_value=16375.0)  # type: ignore[attr-defined]

        self._prime_baseline(pm)

        def phantom_tick() -> None:
            # committed = 243 ctx + 21683 reserved = 21926, while the device physically holds ~1288 (phantom).
            pm._process_map[0] = self._phantom_contributor(0, reserved_mb=21926 - 243)  # type: ignore[index]
            pm._read_device_used_mb = Mock(return_value=1288.0)  # type: ignore[attr-defined]
            pm._last_vram_attribution_time = 0.0
            pm._evaluate_vram_attribution_drift()

        phantom_tick()
        assert recalibrate.call_count == 0  # one transient reading is not enough
        phantom_tick()
        assert recalibrate.call_count == 1  # streak confirmed: recalibrate the ledger
        assert reclaim.call_count == 0  # never evicts a (nonexistent) idle resident model

    def test_recalibrated_ledger_recovers_below_ceiling(self) -> None:
        """After the lanes re-report their released cache, committed falls to truth and the pressure path is quiet."""
        pm = make_testable_process_manager()
        pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=243.0)  # type: ignore[attr-defined]
        reclaim = Mock(return_value=True)
        pm._inference_scheduler.reclaim_one_idle_model_under_pressure = reclaim  # type: ignore[attr-defined]
        recalibrate = Mock(return_value=1)
        pm._inference_scheduler.recalibrate_committed_ledger = recalibrate  # type: ignore[attr-defined]
        pm._process_map.get_reported_total_vram_mb = Mock(return_value=16375.0)  # type: ignore[attr-defined]

        self._prime_baseline(pm)

        for _ in range(2):
            pm._process_map[0] = self._phantom_contributor(0, reserved_mb=21926 - 243)  # type: ignore[index]
            pm._read_device_used_mb = Mock(return_value=1288.0)  # type: ignore[attr-defined]
            pm._last_vram_attribution_time = 0.0
            pm._evaluate_vram_attribution_drift()
        assert recalibrate.call_count == 1

        # The recalibrated lane re-reports its true (near-zero) reservation: committed tracks device truth.
        pm._process_map[0] = self._phantom_contributor(0, reserved_mb=30)  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=1288.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        assert pm.latest_committed_vram_mb(0) == 273.0  # 243 ctx + 30 reserved: below any ceiling
        assert reclaim.call_count == 0

    def test_liveness_matrix_bookkeeping_never_permanently_denies(self) -> None:
        """Across lane/report/residency variations, a device-contradicted committed high never wedges admission.

        The invariant: whenever the committed ledger reads materially above the device-used truth, the manager
        recalibrates (and never latches suppressed), so a fits-alone candidate can never be permanently denied by
        bookkeeping alone. Each row is a distinct committed-vs-device shape.
        """
        # (reserved_mb reported by the single idle lane, device_used_mb) rows: all genuine physical over-commits
        # (committed + captured baseline > total) that are also phantoms (committed far above device-used truth),
        # across a spread of stale-reserved magnitudes. The captured baseline drops to the low device-used, so the
        # reserved figure carries the whole over-commit.
        rows = [
            (21683, 1288),  # the incident: ~21926 committed vs ~1288 device
            (18000, 1500),  # a smaller phantom, device near baseline
            (16500, 1000),  # just over the ceiling once the baseline drops to device-used
            (30000, 900),  # an extreme over-count
        ]
        for reserved_mb, device_used in rows:
            pm = make_testable_process_manager()
            pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=243.0)  # type: ignore[attr-defined]
            reclaim = Mock(return_value=True)
            pm._inference_scheduler.reclaim_one_idle_model_under_pressure = reclaim  # type: ignore[attr-defined]
            recalibrate = Mock(return_value=1)
            pm._inference_scheduler.recalibrate_committed_ledger = recalibrate  # type: ignore[attr-defined]
            pm._process_map.get_reported_total_vram_mb = Mock(return_value=16375.0)  # type: ignore[attr-defined]
            self._prime_baseline(pm)

            for _ in range(2):
                pm._process_map[0] = self._phantom_contributor(0, reserved_mb=reserved_mb)  # type: ignore[index]
                pm._read_device_used_mb = Mock(return_value=float(device_used))  # type: ignore[attr-defined]
                pm._last_vram_attribution_time = 0.0
                pm._evaluate_vram_attribution_drift()

            assert recalibrate.call_count >= 1, f"row reserved={reserved_mb} did not recalibrate"
            assert reclaim.call_count == 0, f"row reserved={reserved_mb} wrongly evicted"


class TestManagerPhysicalPressureUnload:
    """The manager acts on a streak-confirmed physical overcommit by commanding exactly one idle-model reclaim."""

    def _resident_contributor(self, process_id: int, *, reserved_mb: int, sampled_at: float) -> object:
        info = make_mock_process_info(process_id, state=HordeProcessState.INFERENCE_STARTING)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.report_sampled_at = sampled_at  # type: ignore[attr-defined]
        return info

    def test_streak_confirmed_overcommit_issues_one_reclaim_then_suppresses(self) -> None:
        """Two consecutive physical overcommits issue one reclaim; a third sustained tick does not re-issue."""
        pm = make_testable_process_manager()
        pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=200.0)  # type: ignore[attr-defined]
        reclaim = Mock(return_value=True)
        pm._inference_scheduler.reclaim_one_idle_model_under_pressure = reclaim  # type: ignore[attr-defined]
        pm._process_map.get_reported_total_vram_mb = Mock(return_value=16375.0)  # type: ignore[attr-defined]

        # Capture the baseline at a quiet moment (no model resident).
        quiet = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        quiet.process_reserved_mb = None
        pm._process_map.clear()
        pm._process_map[0] = quiet  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=1700.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        def over_commit_tick() -> None:
            # committed = 200 ctx + 15000 reserved = 15200; + baseline 1700 = 16900 > 16375 total.
            pm._process_map[0] = self._resident_contributor(0, reserved_mb=15000, sampled_at=time.time())  # type: ignore[index]
            pm._read_device_used_mb = Mock(return_value=16900.0)  # type: ignore[attr-defined]
            pm._last_vram_attribution_time = 0.0
            pm._evaluate_vram_attribution_drift()

        over_commit_tick()
        assert reclaim.call_count == 0  # one transient observation does not fire
        over_commit_tick()
        assert reclaim.call_count == 1  # streak confirmed: one reclaim issued
        assert pm.latest_measured_unloads_issued(0) == 1
        over_commit_tick()
        assert reclaim.call_count == 1  # hysteresis: sustained overcommit does not re-issue


class TestManagerDriftReconciliation:
    """The manager wires committed-ledger staleness and per-process report age into the drift evaluation."""

    def _resident_contributor(
        self,
        process_id: int,
        *,
        reserved_mb: int,
        sampled_at: float,
    ) -> object:
        info = make_mock_process_info(process_id, state=HordeProcessState.INFERENCE_STARTING)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.report_sampled_at = sampled_at  # type: ignore[attr-defined]
        return info

    def test_fresh_ledger_computes_drift(self) -> None:
        """With a fresh committed report the manager reconciles and records a numeric drift."""
        pm = make_testable_process_manager()
        pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=200.0)  # type: ignore[attr-defined]

        # Establish the baseline at a quiet moment (no model resident), then a resident, drifting reading.
        quiet = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        quiet.process_reserved_mb = None
        pm._process_map.clear()
        pm._process_map[0] = quiet  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=1000.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        pm._process_map[0] = self._resident_contributor(0, reserved_mb=5000, sampled_at=time.time())  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=8000.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        # drift = 8000 - (1000 baseline + (200 ctx + 5000 reserved)) = 1800
        assert pm.latest_vram_attribution_drift_mb(0) == 1800.0

    def test_stale_ledger_skips_drift(self) -> None:
        """A contributor whose report has aged past the staleness bound makes the drift uncomputable (None)."""
        pm = make_testable_process_manager()
        pm._inference_scheduler.resolved_context_constant_mb = Mock(return_value=200.0)  # type: ignore[attr-defined]

        quiet = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        quiet.process_reserved_mb = None
        pm._process_map.clear()
        pm._process_map[0] = quiet  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=1000.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        stale_sample = time.time() - (_REPORT_STALENESS_SECONDS + 5.0)
        pm._process_map[0] = self._resident_contributor(0, reserved_mb=5000, sampled_at=stale_sample)  # type: ignore[index]
        pm._read_device_used_mb = Mock(return_value=8000.0)  # type: ignore[attr-defined]
        pm._last_vram_attribution_time = 0.0
        pm._evaluate_vram_attribution_drift()

        assert pm.latest_vram_attribution_drift_mb(0) is None

    def test_snapshot_exposes_per_process_report_age(self) -> None:
        """The drift snapshot line surfaces each contributor's memory-report age."""
        pm = make_testable_process_manager()
        info = self._resident_contributor(0, reserved_mb=5000, sampled_at=100.0)
        pm._process_map.clear()
        pm._process_map[0] = info  # type: ignore[index]
        observation = DriftObservation(
            drift_mb=1800.0,
            device_used_mb=8000.0,
            baseline_estimate_mb=1000.0,
            committed_vram_mb=5200.0,
            consecutive_over_threshold=2,
            should_warn=True,
        )
        snapshot = pm._format_vram_attribution_snapshot(0, observation, 200.0, now=130.0)
        assert "age=30s" in snapshot
