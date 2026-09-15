"""A saturated multi-card worker must keep most of its cards busy; a single card must stay unharmed.

Drives the real process manager (pop cadence, urgency, intake budgets, scheduler) against fake cards with
an inexhaustible mixed-model job supply, then measures average inference concurrency over the steady
window. The intake ceilings this guards are per-card sums: the pending-intake cap
(``JobPopper._intake_budget``), the megapixelstep budget (``enable_performance_mode``), and the busy-copy
duplicate escape (``InferenceScheduler._duplicate_copy_may_serve``). Sized flat for one card, those
ceilings held a 4-card worker under ~1.6 busy slots; per-card they sustain well over 2 on the same
config, while the single-card cases keep their original formulas exactly.

The floors are deliberately loose (well below the means observed while tuning) because fake-child IPC
timing varies run to run; they are chosen to separate the per-card regime from the flat one, not to pin a
throughput number.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import HarnessConfig, HarnessResult, run_harness_async
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.process_management.simulation._canned_scenarios import SoakImageTemplate

# The measured concurrency is a wall-clock figure, so these rows share one xdist worker under
# --dist loadgroup rather than competing with each other for the CPU.
pytestmark = [pytest.mark.slow, pytest.mark.e2e, pytest.mark.xdist_group("serial_timing")]

_MODELS = ["Deliberate", "Anything Diffusion", "Anything v5", "AbsoluteReality", "Abyss OrangeMix", "Dreamshaper"]
"""Several small models, so dispatch can spread across cards instead of pinning to one resident
process; a single-model soak measures model affinity, not intake."""
_JOB_SECONDS = 2.5
"""Fake inference duration per job: the fast end of observed field traffic, the harshest case for
intake since every busy window a slot finishes must be refilled within seconds."""
_SOAK_SECONDS = 40.0
_FED_AT_FINISH_FLOOR = 0.9
"""On one card, the share of steady-window finishes that must find the next job already accepted. An intake
ceiling sized to zero pending would score near 0 (every finish waits a pop cycle); a fed queue scores 1 apart
from the odd finish that lands in the moment between a pop being counted and the previous job's stamp."""


def _card_resources(card_count: int, *, card_gb: int = 12) -> SystemResources:
    """Fake hardware: ``card_count`` cards of ``card_gb`` GB and enough RAM that no RAM cap alters the plan."""
    return SystemResources(
        total_ram_bytes=128 * 1024**3,
        device_map=TorchDeviceMap(
            root={
                index: TorchDeviceInfo(
                    device_name=f"PopFeed CUDA {index}",
                    device_index=index,
                    total_memory=card_gb * 1024**3,
                    kind="cuda",
                )
                for index in range(card_count)
            },
        ),
        per_process_overhead_mb=900,
        marginal_process_overhead_mb=242,
    )


def _busy_intervals(result: HarnessResult) -> list[tuple[float, float]]:
    """Return (inference_start, inference_end) per image job, from the parent's stage stamps."""
    assert result.metrics is not None
    intervals: list[tuple[float, float]] = []
    for record in result.metrics.jobs:
        if record.workload is not WorkloadKind.IMAGE_GENERATION or record.stage is not None:
            continue
        start = record.stage_timestamps.get("INFERENCE_IN_PROGRESS")
        end = record.stage_timestamps.get("PENDING_SAFETY_CHECK") or record.stage_timestamps.get("PENDING_SUBMIT")
        if start is not None and end is not None and end > start:
            intervals.append((start, end))
    return intervals


def _fed_at_finish_ratio(result: HarnessResult, window: tuple[float, float]) -> float:
    """The share of inference finishes inside ``window`` at which another job was already accepted and waiting.

    A finish is "fed" when some other job had been popped (its ``PENDING_INFERENCE`` stamp) before the
    finish and had not yet started. This is what the intake ceilings decide: whether the queue holds the
    next job by the time a slot frees. It is an ordering of the parent's own stamps, so host load shifts
    every stamp together and leaves the ratio alone, where a wall-clock busy fraction would fall with it.
    """
    assert result.metrics is not None
    jobs = [
        record.stage_timestamps
        for record in result.metrics.jobs
        if record.workload is WorkloadKind.IMAGE_GENERATION
        and record.stage is None
        and "PENDING_INFERENCE" in record.stage_timestamps
    ]
    finishes = [
        stamps.get("PENDING_SAFETY_CHECK") or stamps.get("PENDING_SUBMIT")
        for stamps in jobs
        if "INFERENCE_IN_PROGRESS" in stamps
    ]
    lo, hi = window
    in_window = [finish for finish in finishes if finish is not None and lo <= finish <= hi]
    if not in_window:
        return 0.0
    fed = 0
    for finish in in_window:
        for stamps in jobs:
            popped = stamps["PENDING_INFERENCE"]
            started = stamps.get("INFERENCE_IN_PROGRESS")
            if popped <= finish and (started is None or started >= finish):
                fed += 1
                break
    return fed / len(in_window)


def _average_concurrency(intervals: list[tuple[float, float]], window: tuple[float, float]) -> float:
    """Sweep-line average of how many inference spans overlap each instant of ``window``."""
    lo, hi = window
    if hi <= lo:
        return 0.0
    busy_slot_seconds = 0.0
    for start, end in intervals:
        start, end = max(start, lo), min(end, hi)
        if end > start:
            busy_slot_seconds += end - start
    return busy_slot_seconds / (hi - lo)


@pytest.mark.parametrize(
    ("card_count", "card_gb", "queue_size", "high_performance", "concurrency_floor"),
    [
        # A field-observed 4x 12GB configuration. Under the flat (single-card-sized) intake ceilings this
        # sustained ~1.5-1.6 busy slots; the per-card ceilings sustain ~2.2-2.9. The floor sits between
        # the two regimes.
        pytest.param(4, 12, 1, True, 2.0, id="4card-field-config"),
        # Single-card invariance: the same soak on one card must still keep it fed. The card is 24 GB so the
        # VRAM-fit sizing keeps the configured spare process (a 12 GB card is sized down to one process, which
        # cannot pop while it runs, so the queue is only ever refilled between jobs). The busy fraction of a
        # 2.5s job lands ~0.85 on an idle host and falls with host load, so the single-card rows assert the
        # intake property directly: at every finish the next job was already accepted and waiting.
        pytest.param(1, 24, 1, False, None, id="1card-control"),
        # A high-end single card posture (higher threads/queue and high performance mode): the summed
        # budgets reduce to the original formulas, so it must stay as fed as the plain control.
        pytest.param(1, 24, 2, True, None, id="1card-high-end"),
    ],
)
async def test_saturated_worker_keeps_its_cards_busy(
    card_count: int,
    card_gb: int,
    queue_size: int,
    high_performance: bool,
    concurrency_floor: float | None,
) -> None:
    """Steady-window inference concurrency clears the floor, or on one card the queue is fed at every finish."""
    result = await run_harness_async(
        HarnessConfig(
            process_mode="fake",
            skip_api=True,
            soak_seconds=_SOAK_SECONDS,
            soak_drain_timeout_seconds=90.0,
            timeout_seconds=300.0,
            job_delay_seconds=_JOB_SECONDS,
            soak_image_templates=[SoakImageTemplate(model=name) for name in _MODELS],
            system_resources=_card_resources(card_count, card_gb=card_gb),
            bridge_data_overrides={
                "gpu_device_indices": list(range(card_count)),
                "max_threads": 1,
                "queue_size": queue_size,
                "models_to_load": list(_MODELS),
                "high_performance_mode": high_performance,
            },
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.num_jobs_faulted == 0, result.failure_summary()

    intervals = _busy_intervals(result)
    assert intervals, f"no inference intervals recorded: {result.failure_summary()}"

    first_start = min(start for start, _end in intervals)
    last_end = max(end for _start, end in intervals)
    # Steady window: skip the ramp (process spawn, first-job warmup gate) and the drain tail.
    steady = (first_start + 15.0, last_end - 5.0)
    avg_concurrency = _average_concurrency(intervals, steady)
    fed_ratio = _fed_at_finish_ratio(result, steady)

    print(
        f"\npop feed (cards={card_count}, queue_size={queue_size}, hp={high_performance}): "
        f"avg concurrency {avg_concurrency:.2f} of {card_count}, fed at finish {fed_ratio:.2f} "
        f"({result.num_jobs_completed} jobs over {_SOAK_SECONDS:.0f}s soak + drain)",
    )
    if concurrency_floor is not None:
        assert avg_concurrency >= concurrency_floor, (
            f"average inference concurrency {avg_concurrency:.2f} fell below the {concurrency_floor} floor for "
            f"{card_count} card(s): the intake ceilings are starving cards a saturated worker should be feeding"
        )
    else:
        assert fed_ratio >= _FED_AT_FINISH_FLOOR, (
            f"only {fed_ratio:.2f} of inference finishes had the next job already accepted and waiting: the "
            f"intake ceilings are letting a single saturated card run dry between jobs"
        )
