"""Regression tests for system-RAM exhaustion OOM-kill scenarios in the admission and recovery paths.

The resource budget gates a *new* preload on the *marginal* job cost, but its best-effort /
head-starvation / over-budget admit paths deliberately load a head when "no live job holds memory" --
straight into a potential OS OOM kill if absolute system RAM is already critically low. Nothing in the
admission flow refuses a preload that the *absolute* available RAM says will trigger the kernel OOM-killer,
and nothing structurally reduces the worker's resident footprint when the host is already on the edge.

These tests cover the gap: when host RAM falls below a danger floor, every admission path (including
force-admit and whole-card terminal) must defer rather than load into a near-certain kill. The worker
should also shed idle resident processes to shrink its footprint and pause pops to stop adding pressure,
recovering gracefully instead of crash-looping under sustained RAM exhaustion.

* **Hard-refuse admit-into-OOM** -- when absolute available RAM is below the danger floor, the best-effort
  and head-starvation force-admit paths must defer the preload rather than send it into a near-certain
  OS kill, even for a starved head with no live job.
* **Throttle pops under pressure** -- sustained critical RAM engages the worker-initiated self-throttle so
  intake stops adding pressure and the host can recover.
* **Reduce resident footprint first** -- idle sibling inference processes each pin multiple GB of resident
  weights the allocator will not return without a respawn, so under RAM pressure the worker sheds process
  count (the RAM analogue of ``StreamForecast.needs_process_count_reduction``) to get back under the floor.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from .test_inference_scheduling import _make_inference_scheduler

# Test fixture parameters: a 16 GB-VRAM GPU on a 32 GB-RAM host (VRAM ample; RAM is the binding constraint).
_DEVICE_TOTAL_VRAM_MB = 15847.0
_TOTAL_RAM_MB = 32063.0  # 31.3 GiB

# A 16 GB card with one idle context resident reads ~12 GB free, so VRAM is *not* the binding constraint in
# these scenarios: the RAM ceiling is. Keeping VRAM ample forces the admission flow through the RAM branch.
_PER_PROCESS_OVERHEAD_MB = 1200.0
_AMPLE_FREE_VRAM_USED_MB = 3500.0  # device reads ~12.3 GB free

# A moderate, co-residable head (an SDXL checkpoint, one of the kinds the kills also landed on): it fits the
# ~12 GB free VRAM comfortably, so the admission flow passes the VRAM verdict and reaches the *RAM* branch --
# the gate under test here. Its weights still load through system RAM first, so a multi-GB RAM cost on a host
# with ~1.2 GB free is the OOM the budget must refuse. (A whole-card model like Flux short-circuits to the
# VRAM teardown path *before* the RAM check; that separate bypass is its own scenario, not this one.)
_HEAVY_MODEL = "AlbedoBase XL (SDXL)"
_HEAVY_RAM_MB = 9000.0
_HEAVY_SAMPLING_VRAM_MB = 3000.0  # well within the ~12 GB free: co-resident, not weight-dominant

# The 96%-RAM moment: available well under any sane danger floor (a 4 GB RAM reserve, ~12% of total).
_CRITICAL_AVAILABLE_RAM_MB = 1200.0
_RAM_RESERVE_MB = 4096.0
_VRAM_RESERVE_MB = 2048.0


def _idle_process_map(num_processes: int) -> ProcessMap:
    """A map of idle, model-free inference contexts on the 16 GB card reading ~12 GB free."""
    procs: dict[int, object] = {}
    for pid in range(1, num_processes + 1):
        proc = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _AMPLE_FREE_VRAM_USED_MB
        procs[pid] = proc
    return ProcessMap(procs)


def _ram_pressured_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_processes: int = 1,
    available_ram_mb: float = _CRITICAL_AVAILABLE_RAM_MB,
    reclaim_succeeds: bool = False,
) -> tuple[InferenceScheduler, ProcessMap]:
    """A budget-active scheduler with the 16 GB-VRAM / 32 GB-RAM fixture and ``available_ram_mb`` of system RAM left.

    VRAM is ample so the RAM ceiling is the binding constraint. ``reclaim_succeeds`` controls whether the
    idle-RAM reclaim / stale-slot cycle find anything to free; the exhausted case (nothing left to reclaim)
    is where the RAM floor guard matters most.
    """
    process_map = _idle_process_map(num_processes)
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        image_models_to_load=[_HEAVY_MODEL],
        max_threads=1,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=max(num_processes, 1),
    )

    # The heavy head's predicted costs: ample-VRAM-fitting sampling peak, multi-GB RAM load.
    monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _HEAVY_SAMPLING_VRAM_MB)
    monkeypatch.setattr(
        resource_budget,
        "predict_job_sampling_vram_mb",
        lambda job, baseline: _HEAVY_SAMPLING_VRAM_MB,
    )
    monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: _HEAVY_RAM_MB)

    # The host is on the RAM edge; the device has plenty of free VRAM. Total RAM is patched too so the
    # danger floor (a percentage of total) is deterministic regardless of the machine running the tests.
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_ram_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)

    # Reclaim is exhausted in the OOM case: gentle/head-of-queue unloads and the stale-slot cycle free
    # nothing (every resident copy is gone, or the RAM is allocator-stranded), exactly as in the bundle.
    monkeypatch.setattr(scheduler, "unload_models", Mock(return_value=reclaim_succeeds))
    monkeypatch.setattr(scheduler, "unload_models_from_vram", Mock(return_value=True))
    monkeypatch.setattr(scheduler, "_replace_stale_ram_unload_process", Mock(return_value=reclaim_succeeds))
    return scheduler, process_map


def _no_preload_sent(process_map: ProcessMap) -> bool:
    """Whether no process was sent a PRELOAD_MODEL control message (no admit reached the device)."""
    return all(
        proc.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        for proc in process_map.values()
        if proc.process_type == HordeProcessType.INFERENCE
    )


class TestBestEffortRamAdmitRefusesIntoOOM:
    """The RAM best-effort admit must not load a head into a host that is already out of RAM.

    Head of queue, no live job holding RAM, idle-RAM reclaim exhausted, host RAM at the critical floor: the
    scheduler must defer rather than best-effort admit into a near-certain OS kill. The worker should degrade
    to a slow pop rate instead of crash-looping.
    """

    async def test_critical_ram_head_is_not_admitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A heavy head must NOT be preloaded when absolute available RAM is below the danger floor."""
        scheduler, process_map = _ram_pressured_scheduler(monkeypatch)
        head_job = make_job_pop_response(_HEAVY_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)

        admitted = scheduler.preload_models()

        assert admitted is False, "the head must defer under critical RAM, not best-effort admit into an OOM"
        assert _no_preload_sent(process_map), "no PRELOAD_MODEL may be sent into a host that is out of RAM"

    async def test_admit_allowed_once_ram_recovers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard is pressure-scoped: with healthy available RAM the same head admits normally.

        Proves the refusal is driven by the absolute floor, not by something incidental to the setup (so the
        fix cannot be a blanket "never admit" that would wedge a healthy worker).
        """
        scheduler, process_map = _ram_pressured_scheduler(
            monkeypatch,
            available_ram_mb=_TOTAL_RAM_MB * 0.8,  # ~25 GB available: comfortably clear of the floor
        )
        head_job = make_job_pop_response(_HEAVY_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)

        admitted = scheduler.preload_models()

        assert admitted is True, "with ample RAM the head should preload as before"
        assert not _no_preload_sent(process_map), "a PRELOAD_MODEL should reach the device when RAM is healthy"


class TestHeadStarvationForceAdmitHonorsRamFloor:
    """The 15 s head-starvation backstop must not bypass the absolute RAM floor into an OOM.

    ``_HEAD_STARVATION_FORCE_ADMIT_SECONDS`` exists to rescue a head the *VRAM* verdict keeps rejecting on an
    idle card. It must not become a tunnel that force-loads a head onto a host that is out of *RAM*: that is
    the kill, not a rescue. A starved head on a RAM-exhausted host should keep deferring (while the governor
    sheds and throttles), not be force-admitted.
    """

    async def test_starved_head_still_deferred_under_critical_ram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even past the starvation horizon, a head is not force-admitted when RAM is critically low."""
        scheduler, process_map = _ram_pressured_scheduler(monkeypatch)
        head_job = make_job_pop_response(_HEAVY_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)

        # Drive the head past the force-admit horizon: the starvation timer reads as long-expired.
        monkeypatch.setattr(scheduler, "_head_starved_seconds", lambda job: 999.0)

        admitted = scheduler.preload_models()

        assert admitted is False, "the starvation backstop must not force a head into an out-of-RAM host"
        assert _no_preload_sent(process_map), "no PRELOAD_MODEL may be force-sent under critical RAM"


class TestWholeCardTerminalAdmitHonorsRamFloor:
    """A whole-card model bypasses the RAM check entirely -- a second, distinct route into the OOM.

    A weight-dominant head (whole-card exclusive-residency path) whose teardown is structurally exhausted is
    admitted "best-effort to load onto the cleared device" via ``whole_card_terminal`` -- a branch that
    short-circuits *before* the RAM verdict, so the RAM floor never even runs. On a host with critically-low
    RAM that route loads a multi-GB checkpoint into an OOM.

    The RAM floor must gate this path too: a whole-card head whose weights load through an out-of-RAM host
    must defer, not terminal-admit into the kill.
    """

    async def test_whole_card_terminal_admit_deferred_under_critical_ram(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A weight-dominant (whole-card) head is not terminal-admitted when the host is out of RAM."""
        scheduler, process_map = _ram_pressured_scheduler(monkeypatch)
        # Force the whole-card path: weights that stream co-resident but fit the measured free VRAM now, so
        # with a single process the teardown is immediately exhausted and the head reaches whole_card_terminal
        # (the "admit best-effort onto the cleared device" branch) -- which currently skips the RAM verdict.
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: 5000.0)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 5000.0)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=1)

        head_job = make_job_pop_response("Flux.1-Schnell fp8 (Compact)")
        await track_popped_job_async(scheduler._job_tracker, head_job)
        # Sanity: this really is the whole-card route, not the plain RAM branch.
        forecast = scheduler._forecast_streaming(head_job, "flux_1")
        assert forecast.needs_exclusive_residency is True

        admitted = scheduler.preload_models()

        assert admitted is False, "a whole-card head must not terminal-admit into an out-of-RAM host"
        assert _no_preload_sent(process_map), "the whole-card terminal path must also honor the RAM floor"


class TestMemoryPressureThrottlesPops:
    """Sustained critical RAM should engage the worker-initiated self-throttle (slow/stop pops).

    Pausing intake under RAM pressure (the same self-throttle the consecutive-failure backstop uses;
    in-flight checked jobs still submit, pops auto-resume on recovery) is how the worker stops taking on
    work it cannot hold.
    """

    async def test_critical_ram_engages_self_throttle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scheduling pass under critical RAM should pause pops via the self-throttle."""
        scheduler, _process_map = _ram_pressured_scheduler(monkeypatch)
        head_job = make_job_pop_response(_HEAVY_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)

        scheduler.preload_models()

        assert scheduler._state.self_throttle_paused is True, (
            "critical RAM should slow/stop pops so intake stops adding memory pressure"
        )


class TestMemoryPressureReducesResidentProcesses:
    """Under RAM pressure the worker should shed idle resident processes (the RAM footprint reduction).

    Each idle inference process pins multiple GB of resident weights that ``torch``/the allocator will not
    return to the OS without a respawn. When the host is over the danger floor and idle siblings hold
    reclaimable RAM, the structural remedy is to reduce the resident process count -- the RAM analogue of
    ``StreamForecast.needs_process_count_reduction`` -- rather than admit a new load on top.
    """

    async def test_idle_siblings_are_shed_under_critical_ram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With three idle resident contexts and critical RAM, the worker reduces the process count."""
        scheduler, _process_map = _ram_pressured_scheduler(monkeypatch, num_processes=3)
        # Each idle sibling holds a *different* resident model (not the head) whose RAM only a respawn
        # reclaims; the head must load a fresh model on top, which the host has no RAM left for.
        for proc in _process_map.values():
            proc.loaded_horde_model_name = "Juggernaut XL"
            proc.ram_usage_bytes = 5_000_000_000
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=1)

        head_job = make_job_pop_response(_HEAVY_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)

        scheduler.preload_models()

        assert scheduler._process_lifecycle.scale_inference_processes.called, (
            "critical RAM with reclaimable idle resident processes should reduce the process count"
        )
