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
from horde_worker_regen.process_management.simulation._canned_scenarios import SoakImageTemplate

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

_MODELS = ["Deliberate", "Anything Diffusion", "Anything v5", "AbsoluteReality", "Abyss OrangeMix", "Dreamshaper"]
"""Several small models, so dispatch can spread across cards instead of pinning to one resident
process; a single-model soak measures model affinity, not intake."""
_JOB_SECONDS = 2.5
"""Fake inference duration per job: the fast end of observed field traffic, the harshest case for
intake since every busy window a slot finishes must be refilled within seconds."""
_SOAK_SECONDS = 60.0


def _card_resources(card_count: int) -> SystemResources:
    """Fake hardware: ``card_count`` 12 GB cards and enough RAM that no RAM cap alters the plan."""
    return SystemResources(
        total_ram_bytes=128 * 1024**3,
        device_map=TorchDeviceMap(
            root={
                index: TorchDeviceInfo(
                    device_name=f"PopFeed CUDA {index}",
                    device_index=index,
                    total_memory=12 * 1024**3,
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
        if record.is_alchemy or record.stage is not None:
            continue
        start = record.stage_timestamps.get("INFERENCE_IN_PROGRESS")
        end = record.stage_timestamps.get("PENDING_SAFETY_CHECK") or record.stage_timestamps.get("PENDING_SUBMIT")
        if start is not None and end is not None and end > start:
            intervals.append((start, end))
    return intervals


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
    ("card_count", "queue_size", "high_performance", "concurrency_floor"),
    [
        # A field-observed 4x 12GB configuration. Under the flat (single-card-sized) intake ceilings this
        # sustained ~1.5-1.6 busy slots; the per-card ceilings sustain ~2.2-2.9. The floor sits between
        # the two regimes.
        pytest.param(4, 1, True, 2.0, id="4card-field-config"),
        # Single-card invariance: the same soak on one card must still keep it busy. The duty cycle of a
        # 2.5s job (dispatch, safety, submit overheads between jobs) lands ~0.8; the floor only guards
        # against the intake changes idling the common case.
        pytest.param(1, 1, False, 0.7, id="1card-control"),
        # A high-end single card posture (higher threads/queue and high performance mode): the summed
        # budgets reduce to the original formulas, so it must stay as busy as the plain control.
        pytest.param(1, 2, True, 0.7, id="1card-high-end"),
    ],
)
async def test_saturated_worker_keeps_its_cards_busy(
    card_count: int,
    queue_size: int,
    high_performance: bool,
    concurrency_floor: float,
) -> None:
    """Average steady-window inference concurrency clears the floor for this card topology."""
    result = await run_harness_async(
        HarnessConfig(
            process_mode="fake",
            skip_api=True,
            soak_seconds=_SOAK_SECONDS,
            soak_drain_timeout_seconds=90.0,
            timeout_seconds=300.0,
            job_delay_seconds=_JOB_SECONDS,
            soak_image_templates=[SoakImageTemplate(model=name) for name in _MODELS],
            system_resources=_card_resources(card_count),
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

    print(
        f"\npop feed (cards={card_count}, queue_size={queue_size}, hp={high_performance}): "
        f"avg concurrency {avg_concurrency:.2f} of {card_count} "
        f"({result.num_jobs_completed} jobs over {_SOAK_SECONDS:.0f}s soak + drain)",
    )
    assert avg_concurrency >= concurrency_floor, (
        f"average inference concurrency {avg_concurrency:.2f} fell below the {concurrency_floor} floor for "
        f"{card_count} card(s): the intake ceilings are starving cards a saturated worker should be feeding"
    )
