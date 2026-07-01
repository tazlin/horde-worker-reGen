"""Regression tests for the multi-process resident-RAM growth that drove an OS OOM kill.

Field incident (2x 16 GB-VRAM cards, 64 GB host, also co-hosting an alchemist and a scribe): a single
inference process's resident system-RAM footprint grew monotonically to ~30-35 GB as it accumulated resident
weights. With one such process per card plus the co-tenants, the host crossed its RAM ceiling and the kernel
OOM-killer reaped the worker's inference child (taking the TUI/worker down with it). The bundle's own memory
reports show ``Process N ... ram: 30670069760`` (30.7 GB) and a peak of 35.4 GB in one process.

The absolute RAM danger-floor machinery shipped in the running version, yet the kill still happened. These
tests pin the two orchestration gaps that let it through, independent of any tuning constant:

* **The danger floor is dormant on a steady-state worker.** ``_govern_ram_pressure`` is only reached from
  inside the preload loop, which early-returns when every pending job's model is already resident (nothing
  to preload). A worker serving already-resident models across busy processes therefore never evaluates the
  floor and never sheds or throttles, so the resident set grows to OOM with the governor asleep. A scheduling
  pass must govern the floor every tick, not only when a new preload is attempted.

* **A lone over-ceiling process is never reclaimed.** The idle-shed reduction keeps at least one context per
  card, and the stale-unload recycle only fires for an idle, model-free slot already told to unload. So a
  single process whose retained RAM has ballooned past a per-process ceiling is never recycled while it is the
  card's only context: exactly the shape of the kill. Under pressure such a process must be reclaimed (drained
  first if busy) so the allocator returns its pages to the OS.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    run_scheduling_pass_with_dispatch_inert,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# A 16 GB card fixture matching the incident: VRAM is ample (residency reads ~12 GB free), so RAM is always
# the binding constraint and the admission flow is forced through the RAM branch.
_DEVICE_TOTAL_VRAM_MB = 15850.0
_AMPLE_FREE_VRAM_USED_MB = 3500.0
_TOTAL_RAM_MB = 64000.0

# Available RAM well under any sane danger floor (a 4 GB reserve on 64 GB is ~6%): the host is on the edge.
_CRITICAL_AVAILABLE_RAM_MB = 1500.0
_RAM_RESERVE_MB = 4096.0
_VRAM_RESERVE_MB = 2048.0

# One process's retained RAM in the incident: ~30 GB the allocator will not return without a respawn.
_OVER_CEILING_RAM_BYTES = 30_000_000_000
# A per-process ceiling comfortably below the incident's balloon but above a healthy single SDXL context.
_RAM_PER_PROCESS_CEILING_MB = 18432.0


def _resident_idle_proc(process_id: int, model_name: str, *, ram_bytes: int = 2_000_000_000) -> object:
    """An idle inference process with a model resident in RAM on the 16 GB card."""
    proc = make_mock_process_info(process_id, model_name=model_name, state=HordeProcessState.WAITING_FOR_JOB)
    proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
    proc.vram_usage_mb = _AMPLE_FREE_VRAM_USED_MB
    proc.ram_usage_bytes = ram_bytes
    return proc


def _ram_pressured_scheduler(
    process_map: ProcessMap,
    *,
    available_ram_mb: float = _CRITICAL_AVAILABLE_RAM_MB,
) -> InferenceScheduler:
    """A budget-active scheduler pinned to the 16 GB-VRAM / 64 GB-RAM fixture with ``available_ram_mb`` left."""
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        ram_per_process_max_mb=_RAM_PER_PROCESS_CEILING_MB,
        max_threads=1,
        image_models_to_load=[proc.loaded_horde_model_name for proc in process_map.values()],
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=max(1, len(process_map)),
    )
    scheduler._measured_available_ram_mb = lambda: available_ram_mb  # type: ignore[method-assign]
    scheduler._measured_total_ram_mb = lambda: _TOTAL_RAM_MB  # type: ignore[method-assign]
    return scheduler


class TestDangerFloorGovernsEveryTick:
    """The absolute RAM floor must be evaluated even when there is no new model to preload.

    The steady-state kill shape: every busy process is already serving a resident model, so the preload loop
    early-returns (``loaded_models == pending_models``) before it ever reaches the floor check. The floor must
    still govern on that tick, or a worker whose queue is full of resident-model jobs never throttles while its
    resident set grows into the OOM.
    """

    async def test_all_models_resident_still_engages_self_throttle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With every pending job's model resident and RAM below the floor, the pass must throttle pops."""
        process_map = ProcessMap(
            {
                1: _resident_idle_proc(1, "AlbedoBase XL (SDXL)"),
                2: _resident_idle_proc(2, "Juggernaut XL"),
            },
        )
        scheduler = _ram_pressured_scheduler(process_map)
        # Pending jobs for the *already-resident* models: this is the state that early-returns preload_models,
        # so only the cycle-level governor tick can throttle here.
        for model in ("AlbedoBase XL (SDXL)", "Juggernaut XL"):
            await track_popped_job_async(scheduler._job_tracker, make_job_pop_response(model))

        await run_scheduling_pass_with_dispatch_inert(scheduler)

        assert scheduler._state.self_throttle_paused is True, (
            "a steady-state worker under the RAM floor must throttle even when nothing new needs preloading"
        )


class TestLoneOverCeilingProcessIsReclaimed:
    """A single process whose retained RAM ballooned past the ceiling must be recycled under pressure.

    The idle-shed reduction keeps one context per card, so a lone ~30 GB process is never shed; the stale-unload
    recycle only fires for a model-free slot already told to unload. Under the danger floor the governor must
    reclaim such a process (return its allocator-retained pages) rather than leave it pinning the host at the OOM
    edge.
    """

    def test_over_ceiling_idle_process_is_cycled_to_return_ram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An idle 30 GB process on a card is recycled so the allocator returns its pages to the OS."""
        proc = _resident_idle_proc(1, "AlbedoBase XL (SDXL)", ram_bytes=_OVER_CEILING_RAM_BYTES)
        process_map = ProcessMap({1: proc})
        scheduler = _ram_pressured_scheduler(process_map)
        recycle = Mock()
        scheduler._process_lifecycle._replace_inference_process = recycle  # type: ignore[method-assign]

        scheduler._govern_ram_pressure(scheduler._ram_pressure_verdict())

        assert recycle.called, (
            "a lone over-ceiling process under the RAM floor must be recycled to return its retained RAM"
        )
        assert proc.process_type == HordeProcessType.INFERENCE

    def test_busy_over_ceiling_process_is_drained_not_killed_mid_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A BUSY over-ceiling process is drained (marked, no new work), not recycled out from under its job."""
        proc = _resident_idle_proc(1, "AlbedoBase XL (SDXL)", ram_bytes=_OVER_CEILING_RAM_BYTES)
        proc.last_process_state = HordeProcessState.INFERENCE_STARTING  # actively sampling: must not be killed
        process_map = ProcessMap({1: proc})
        scheduler = _ram_pressured_scheduler(process_map)
        recycle = Mock()
        scheduler._process_lifecycle._replace_inference_process = recycle  # type: ignore[method-assign]

        scheduler._govern_ram_pressure(scheduler._ram_pressure_verdict())

        assert not recycle.called, "a busy process must not be recycled mid-job; it must be drained first"
        assert 1 in scheduler._processes_draining_for_ram, (
            "a busy over-ceiling process under the RAM floor must be marked draining so it stops taking new work"
        )


class TestDrainOutlivesThePressureEpisode:
    """A drain marked under the floor must resolve after RAM recovers, or it wedges intake forever.

    The mark itself holds the soft pop hold engaged and blocks shed restore, and the degrade response that
    placed it (pop pause, idle-model eviction, context shedding) routinely lifts the pressure before the
    drained process finishes its in-flight job. If reclaim only ever runs under the floor, the mark can
    never resolve on the recovered host: the drained process idles over the ceiling untouched, its retained
    RAM is never returned, and the worker refuses every pop for the rest of the session.
    """

    def test_drained_process_is_recycled_once_idle_after_ram_recovers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A busy drain marked under the floor is recycled on a later healthy tick once its job finishes."""
        proc = _resident_idle_proc(1, "AlbedoBase XL (SDXL)", ram_bytes=_OVER_CEILING_RAM_BYTES)
        proc.last_process_state = HordeProcessState.INFERENCE_STARTING  # busy when the floor trips
        process_map = ProcessMap({1: proc})
        scheduler = _ram_pressured_scheduler(process_map)
        recycle = Mock()
        scheduler._process_lifecycle._replace_inference_process = recycle  # type: ignore[method-assign]

        scheduler._govern_ram_pressure_if_pressured()

        assert 1 in scheduler._processes_draining_for_ram, "the busy over-ceiling process is marked under the floor"
        assert not recycle.called

        # The degrade response worked: RAM recovered above the floor, and the drained job then finished.
        scheduler._measured_available_ram_mb = lambda: _TOTAL_RAM_MB * 0.7  # type: ignore[method-assign]
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB

        scheduler._govern_ram_pressure_if_pressured()

        assert recycle.called, "an idle drained process must still be recycled after the floor clears"
        assert 1 not in scheduler._processes_draining_for_ram, "the resolved drain must release its mark"

        scheduler._govern_ram_pressure_if_pressured()

        assert scheduler._state.ram_pressure_pop_hold is False, (
            "with the drain resolved and RAM recovered, the pop hold must release so intake reopens"
        )


class TestPopHeldBeforeTheFloorToAvoidStaleJobs:
    """Popping must pause as RAM *approaches* the floor, not only once it is breached.

    A job popped onto a worker already near its RAM ceiling can sit in-queue past its ttl while the worker
    degrades, and the horde then aborts it as too slow (a forced-maintenance driver). A soft pre-floor hold
    stops new jobs starting their ttl clock while RAM is within the marginal reserve of the danger floor
    and work is in flight; an idle worker whose stable footprint sits in the band is not held (nothing on an
    idle host frees RAM on its own, so a hold there would starve the worker permanently).
    """

    async def test_pop_hold_engages_in_the_approaching_band_with_work_in_flight(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With available RAM just above the floor and a job in flight, the soft pop hold engages."""
        process_map = ProcessMap({1: _resident_idle_proc(1, "AlbedoBase XL (SDXL)")})
        # Floor on 64 GB at 85% is ~9.6 GB; with a 4 GB reserve margin the band is [9.6, 13.6) GB. 11 GB sits
        # in it: above the hard floor (no self-throttle) but approaching it (soft hold).
        scheduler = _ram_pressured_scheduler(process_map, available_ram_mb=11000.0)
        job = make_job_pop_response("AlbedoBase XL (SDXL)")
        await track_popped_job_async(scheduler._job_tracker, job)
        await scheduler._job_tracker.mark_inference_started(job)

        scheduler._govern_ram_pressure_if_pressured()

        assert scheduler._state.self_throttle_paused is False, "the approaching band is above the hard floor"
        assert scheduler._state.ram_pressure_pop_hold is True, (
            "approaching the RAM floor must hold pops so a new job's ttl clock does not start on a degraded worker"
        )

    def test_pop_hold_stays_clear_in_the_approaching_band_on_an_idle_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An idle worker whose resident footprint merely sits in the band keeps popping.

        With no in-flight work nothing on the host frees RAM by itself, so a hold engaged here can never
        clear: the worker would sit above the floor, fully idle, refusing every pop for the rest of the
        session while the skipped-reason counter grows.
        """
        process_map = ProcessMap({1: _resident_idle_proc(1, "AlbedoBase XL (SDXL)")})
        scheduler = _ram_pressured_scheduler(process_map, available_ram_mb=11000.0)

        scheduler._govern_ram_pressure_if_pressured()

        assert scheduler._state.self_throttle_paused is False, "the approaching band is above the hard floor"
        assert scheduler._state.ram_pressure_pop_hold is False, (
            "an idle worker in the approaching band must not latch a hold nothing can ever release"
        )

    def test_pop_hold_clears_when_ram_is_ample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With plenty of RAM the soft hold is clear so the worker pops normally."""
        process_map = ProcessMap({1: _resident_idle_proc(1, "AlbedoBase XL (SDXL)")})
        scheduler = _ram_pressured_scheduler(process_map, available_ram_mb=_TOTAL_RAM_MB * 0.7)

        scheduler._govern_ram_pressure_if_pressured()

        assert scheduler._state.ram_pressure_pop_hold is False, "ample RAM must not hold pops"
