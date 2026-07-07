"""Repro for a mis-measured per-process VRAM overhead that wedges a multi-process worker.

A 24 GB RTX 4090 on Linux running four inference processes with several resident SDXL models can fill the
card and then sit in a permanent queue deadlock ("Model causing deadlock: AlbedoBase XL 3.1", "Scale down:
no idle inference process available to stop right now") until the supervisor shuts it down. No fault, no
OOM, just a stall: the streaming forecast for the head model reports ``needs_exclusive=True,
needs_teardown=True``; it wants an *idle sibling inference process stopped* to reclaim a CUDA context,
but when every process is kept resident there is no idle process to stop and the demanded remedy can never
run.

The root cause is the per-process context overhead the forecast subtracts from total VRAM. It is sourced
once at startup from the accelerator probe, which records a *single fresh process's device-wide used VRAM*
(``get_torch_total_vram_mb() - get_torch_free_vram_mb()``). That first reading lands around ``4266 MB``,
because the first process to touch the device also pays a large one-time CUDA runtime/kernel allocation that
every later process *shares*. The forecast then treats that figure as a strictly per-process cost and
multiplies it by the live process count: ``free_after_model_evict = total - N * overhead``. With ``N=4`` that
is ``24074 - 4*4266 = 7010 MB``; yet the device's *measured* residency with all four idle contexts and no
models loaded is only ``5440 MB`` used (``~1360 MB`` marginal per process), i.e. ``18634 MB`` genuinely free.
The 3x over-count collapses the forecast's view of reclaimable VRAM, flipping a model that is servable by
evicting a sibling *model* into one that demands stopping a sibling *process*, and the worker wedges.

These tests reproduce that from the measured numbers, driving the scheduler's own ``_forecast_streaming``.
The fix derives a *marginal* per-additional-context cost from the device's measured all-contexts idle
residency (fed by the parent's attribution tick as truthful device-used net of the shared baseline and the
tenants' byte-exact reservations) and sizes ``free_after_model_evict`` as ``per_process_overhead +
(contexts - 1) * marginal`` instead of ``contexts * per_process_overhead``. The over-count corrupts
``free_after_model_evict`` for any multi-process worker; it surfaced first under a residency mode that kept
every process loaded, so no idle sibling existed to satisfy the bogus teardown demand and the worker had no
way out.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
)

# --- Ground truth, the device-wide memory figures a worker in this configuration reports. ---
_DEVICE_TOTAL_VRAM_MB = 24074
"""``total vram`` every inference process reports for the 4090."""

_NUM_INFERENCE_PROCESSES = 4
"""Four inference processes are launched (ids 1-4) alongside the safety process."""

_IDLE_DEVICE_USED_ALL_CONTEXTS_MB = 5440
"""Device-wide used VRAM with all four CUDA contexts materialised and *no model loaded*. This is the real
cost of the contexts, ~1360 MB marginal per process: the figure the forecast's ``free_after_model_evict``
should reflect."""

_PROBE_SINGLE_PROCESS_OVERHEAD_MB = 4266
"""What the startup accelerator probe measures: a single fresh process's device-wide used VRAM. Dominated by
the one-time CUDA runtime allocation the other processes share, so it is NOT a per-process cost. This is the
value fed to ``set_measured_per_process_overhead_mb`` and multiplied by the process count."""

# --- The head model that deadlocked: an SDXL checkpoint under a heavy activation/committed reserve. ---
_SDXL_WEIGHTS_MB = 4900.0
"""``weights ~4900 MB``, the AlbedoBase XL 3.1 checkpoint's stream-forecast weight estimate."""

_SDXL_SAMPLING_PEAK_MB = 17128.0
"""Chosen so the activation-inclusive reserve is ``17128 - 4900 = 12228 MB``: the reserve the
deadlock-causing forecast assembles ("weights ~4900 MB + 12228 MB reserve")."""

_BASE_INFERENCE_RESERVE_MB = 2000.0
"""A modest bounded weight-floor; the activation peak dominates, matching the live forecast."""


def _build_scheduler_at_idle_residency() -> InferenceScheduler:
    """Return a scheduler whose process map mirrors the four idle contexts at session start.

    Every inference process is WAITING_FOR_JOB with no model loaded, reporting the device-wide idle residency
    (``5440 MB`` used of ``24074 MB``). The startup-probed per-process overhead (``4266 MB``) is recorded just
    as the worker does, so the forecast reads exactly the inputs the live worker had.
    """
    process_map = ProcessMap({})
    for process_id in range(1, _NUM_INFERENCE_PROCESSES + 1):
        proc = make_mock_process_info(
            process_id=process_id,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.INFERENCE,
        )
        # Children report device-wide used / total, so every idle process carries the same figures.
        proc.vram_usage_mb = _IDLE_DEVICE_USED_ALL_CONTEXTS_MB
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        # An idle, model-less context holds no allocator reservation; a zero (not None) figure makes the
        # process a committed-ledger tenant, which the bare-context decomposition keys on.
        proc.process_reserved_mb = 0.0
        process_map[process_id] = proc

    bridge_data = make_mock_bridge_data(
        high_performance_mode=True,
        safety_on_gpu=False,
        max_threads=1,
        # Force the *measured* overhead path (no operator override), as on the live worker.
        vram_per_process_overhead_mb=0,
        vram_reserve_mb=0,
    )

    scheduler = InferenceScheduler(
        state=WorkerState(),
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=JobTracker(),
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        max_concurrent_inference_processes=1,
        max_inference_processes=_NUM_INFERENCE_PROCESSES,
        lru=LRUCache(_NUM_INFERENCE_PROCESSES),
    )
    scheduler.set_measured_per_process_overhead_mb(_PROBE_SINGLE_PROCESS_OVERHEAD_MB)
    return scheduler


class TestProbeOverheadTeardownWedge:
    """The startup-probed per-process overhead must not over-count the one-time CUDA cost across processes."""

    def test_free_after_model_evict_matches_measured_idle_free(self) -> None:
        """With every context resident and no model loaded, after-evict free must equal the measured free now.

        ``free_after_model_evict_mb`` is "free once every process's context has materialised but no model is
        loaded". Right now there *are* exactly four materialised contexts and zero resident models, so evicting
        the (zero) sibling models cannot change anything: this figure must equal the device's measured free
        VRAM (``24074 - 5440 = 18634 MB``). The forecast instead derives it as ``total - N * probed_overhead =
        24074 - 4*4266 = 7010 MB``, an ~11.6 GB phantom shortfall. This invariant is model-independent: it
        isolates the overhead over-count from any weight/reserve estimate.
        """
        scheduler = _build_scheduler_at_idle_residency()
        job = make_job_pop_response("AlbedoBase XL 3.1", width=1024, height=1024, n_iter=2)

        forecast = scheduler._forecast_streaming(job, "stable_diffusion_xl")

        measured_free_now = scheduler._measured_free_vram_mb()
        assert measured_free_now == pytest.approx(_DEVICE_TOTAL_VRAM_MB - _IDLE_DEVICE_USED_ALL_CONTEXTS_MB)
        assert forecast.free_after_model_evict_mb is not None
        # Evicting zero resident models can never leave *less* free than there is right now.
        assert forecast.free_after_model_evict_mb >= measured_free_now - 1.0

    def test_heavy_sdxl_does_not_demand_impossible_sibling_teardown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The SDXL head must be servable by evicting a sibling *model*, not by stopping a sibling *process*.

        ``high_memory_mode`` keeps every process resident, so ``requires_sibling_teardown`` /
        ``needs_exclusive_residency`` (both of which can only be satisfied by stopping an *idle* sibling
        process) are unsatisfiable and wedge the worker. With the device's true idle residency the model fits
        once a sibling model is evicted, a remedy ``high_memory_mode`` permits, so neither teardown flag
        should be set. They are today, purely because the probed overhead understates the reclaimable VRAM.
        """
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _SDXL_WEIGHTS_MB)
        monkeypatch.setattr(
            resource_budget,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: _SDXL_SAMPLING_PEAK_MB,
        )
        monkeypatch.setattr(
            resource_budget,
            "effective_inference_reserve_mb",
            lambda *args, **kwargs: _BASE_INFERENCE_RESERVE_MB,
        )

        scheduler = _build_scheduler_at_idle_residency()
        job = make_job_pop_response("AlbedoBase XL 3.1", width=1024, height=1024, n_iter=2)

        forecast = scheduler._forecast_streaming(job, "stable_diffusion_xl")

        # The reserve the forecast assembles must match the modeled sampling peak so the repro is faithful.
        assert forecast.reserve_mb == pytest.approx(_SDXL_SAMPLING_PEAK_MB - _SDXL_WEIGHTS_MB)
        # The wedge geometry: an over-counted overhead would demand a sibling-process teardown no idle process
        # could provide. With the true residency it must not.
        assert forecast.requires_sibling_teardown is False
        assert forecast.needs_exclusive_residency is False
        # The correct remedy under the true residency: evict a sibling model (and in fact it co-resides).
        assert forecast.fits_after_model_evict is True

    def test_marginal_derived_from_measured_idle_residency(self) -> None:
        """The scheduler derives marginal = (idle residency - first-context overhead) / (contexts - 1)."""
        scheduler = _build_scheduler_at_idle_residency()
        # The capture is fed by the parent's attribution tick with the truthful device-used reading and the
        # reconciled shared baseline; with a zero baseline and zero tenant reservations the bare-context
        # residual is the full idle reading.
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=0.0,
            device_index=0,
        )

        assert scheduler._overhead._idle_context_residency_mb == pytest.approx(_IDLE_DEVICE_USED_ALL_CONTEXTS_MB)
        assert scheduler._overhead._idle_residency_context_count == _NUM_INFERENCE_PROCESSES
        expected = (_IDLE_DEVICE_USED_ALL_CONTEXTS_MB - _PROBE_SINGLE_PROCESS_OVERHEAD_MB) / (
            _NUM_INFERENCE_PROCESSES - 1
        )
        assert scheduler._marginal_process_overhead_mb() == pytest.approx(expected)

    def test_capture_nets_out_baseline_and_reservations(self) -> None:
        """The bare-context residual excludes the shared baseline and the tenants' byte-exact reservations.

        Attributing either into the residual multiplies it across the process count and re-creates the
        inflated per-context marginal this repro exists to prevent (the 2272 MB/context phantom charged a
        16 GB card into a 15.9 GB committed ledger on a 5.2 GB-used device).
        """
        scheduler = _build_scheduler_at_idle_residency()
        baseline_mb = 1614.0
        reserved_each_mb = 100.0
        for process_info in scheduler._process_map.values():
            process_info.process_reserved_mb = reserved_each_mb
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=baseline_mb,
            device_index=0,
        )
        expected_residual = (
            _IDLE_DEVICE_USED_ALL_CONTEXTS_MB - baseline_mb - reserved_each_mb * _NUM_INFERENCE_PROCESSES
        )
        assert scheduler._overhead._idle_context_residency_mb == pytest.approx(expected_residual)

    def test_capture_skips_when_baseline_absorbed_contexts(self) -> None:
        """A non-positive residual (baseline captured with tenants already up) latches nothing.

        The marginal then correctly falls back to the probe or the platform seed instead of latching a
        meaningless figure.
        """
        scheduler = _build_scheduler_at_idle_residency()
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=float(_IDLE_DEVICE_USED_ALL_CONTEXTS_MB),
            device_index=0,
        )
        assert scheduler._overhead._idle_context_residency_mb is None

    def test_marginal_falls_back_without_clean_baseline(self) -> None:
        """With no clean all-idle baseline observed, marginal is None so the forecast reuses the overhead."""
        scheduler = _build_scheduler_at_idle_residency()
        # No capture has run, and a resident model means the all-idle-no-model window never holds.
        for process_info in scheduler._process_map.values():
            process_info.loaded_horde_model_name = "AMPonyXL"
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=0.0,
            device_index=0,
        )

        assert scheduler._overhead._idle_context_residency_mb is None
        assert scheduler._marginal_process_overhead_mb() is None

    def test_probe_marginal_takes_precedence_over_idle_residency(self) -> None:
        """The probe's directly-measured marginal wins over the idle-residency derivation when both exist.

        The probe figure is hard data available from the first tick (it covers the startup window), so the
        scheduler prefers it. Without it, the idle-residency derivation is the fallback; with neither, None.
        """
        scheduler = _build_scheduler_at_idle_residency()
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=0.0,
            device_index=0,
        )
        derived = (_IDLE_DEVICE_USED_ALL_CONTEXTS_MB - _PROBE_SINGLE_PROCESS_OVERHEAD_MB) / (
            _NUM_INFERENCE_PROCESSES - 1
        )
        # Both sources available: the probe figure (set by the manager) takes precedence.
        scheduler.set_measured_marginal_overhead_mb(412.0)
        assert scheduler._marginal_process_overhead_mb() == pytest.approx(412.0)
        # A zero/unmeasurable probe figure (e.g. Windows WDDM) falls back to the idle-residency derivation.
        scheduler.set_measured_marginal_overhead_mb(0)
        assert scheduler._marginal_process_overhead_mb() == pytest.approx(derived)

    def test_probe_marginal_alone_covers_startup_window(self) -> None:
        """With only the probe marginal (no idle baseline yet), the scheduler still has a marginal at startup."""
        scheduler = _build_scheduler_at_idle_residency()
        # Simulate the startup window: a sibling is still loading, so the clean all-idle baseline never holds.
        for process_info in scheduler._process_map.values():
            process_info.loaded_horde_model_name = "AMPonyXL"
        scheduler.capture_idle_context_residency(
            device_used_mb=_IDLE_DEVICE_USED_ALL_CONTEXTS_MB,
            baseline_mb=0.0,
            device_index=0,
        )
        assert scheduler._overhead._idle_context_residency_mb is None  # no idle-residency fallback available

        scheduler.set_measured_marginal_overhead_mb(455.0)
        assert scheduler._marginal_process_overhead_mb() == pytest.approx(455.0)
