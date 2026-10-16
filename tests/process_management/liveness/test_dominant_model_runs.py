"""A rung-1 run in the closed loop: the 15:05 to 15:25 replay window fed on demand.

Flux, Z-Image and Krea2 read rung 1 on the session's card once the foreign floor is learned, so the window runs
each of them in runs. The window is replayed once for the module and each row reads one property of it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.liveness._dispatch_world import _DispatchWorld
from tests.process_management.liveness.test_session_replay import _ReplayFigures, _run_w3_window

_RUN_MODEL = "Flux.1-Schnell fp8 (Compact)"
"""The model with the most jobs in the window, and so the most consecutive pairs to observe."""

_WEDGED_RECONCILIATION_SHARE = 0.05
"""A residency-reconciliation share of slot time above which the window reads as held, not merely busy.

A held head whose run job queued behind it wedged the window at about 80% of slot time in this bucket; a window
that serves its runs spends under 2% there."""


@dataclass
class _RunWindow:
    """The window's finished world and figures, and the lane state at each run grant."""

    world: _DispatchWorld
    figures: _ReplayFigures
    grant_lane_states: list[HordeProcessState | None]


@pytest.fixture(scope="module")
def run_window(tmp_path_factory: pytest.TempPathFactory) -> _RunWindow:
    """Replay the window on demand once, recording what the lane was doing whenever a run was asked for."""
    grant_lane_states: list[HordeProcessState | None] = []
    note_dominant_run = InferenceScheduler._note_dominant_run

    def recording_note(self: InferenceScheduler, model: str, process_with_model: HordeProcessInfo) -> None:
        grant_lane_states.append(process_with_model.last_process_state)
        note_dominant_run(self, model, process_with_model)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(InferenceScheduler, "_note_dominant_run", recording_note)
        world, figures = asyncio.run(_run_w3_window(Path(tmp_path_factory.mktemp("run_window")), on_demand=True))
    return _RunWindow(world=world, figures=figures, grant_lane_states=grant_lane_states)


def test_a_run_is_asked_for_only_by_a_sampling_job(run_window: _RunWindow) -> None:
    """The run, and the claim narrowing the offer with it, engages once the job reaches its first step.

    Engaged at the dispatch commit, the claim narrows intake for the whole load.
    """
    assert run_window.grant_lane_states, f"no run was asked for. {run_window.figures.describe()}"
    assert set(run_window.grant_lane_states) == {HordeProcessState.INFERENCE_STARTING}


def test_a_runs_next_job_on_the_same_lane_loads_nothing(run_window: _RunWindow) -> None:
    """Consecutive jobs of the run's model on one lane start on the weights the first left there."""
    windows = sorted(
        (window for window in run_window.world.job_windows.values() if window.model == _RUN_MODEL),
        key=lambda window: window.dispatched_at,
    )
    warm_successors = [
        later
        for earlier, later in zip(windows, windows[1:], strict=False)
        if later.lane_id == earlier.lane_id and not later.loaded_now
    ]
    assert warm_successors, f"every {_RUN_MODEL} job loaded its weights. {run_window.figures.describe()}"


def test_a_head_held_for_a_runs_weights_does_not_wedge_the_window(run_window: _RunWindow) -> None:
    """A head for another model waits for the run, and the run's queued jobs pass it so the run can end.

    The run's weights stay while it has a next job that can be seated, and it ends once its work drains or a
    different-model head is starved. A run job held at dispatch without release does not keep the copy, so a staged
    head of another model is cleared into the room the copy returns instead of waiting for its lease-acquire timeout.
    """
    figures = run_window.figures
    reconciliation_share = figures.slot_duty.get("residency_reconciliation", 0.0)
    assert reconciliation_share < _WEDGED_RECONCILIATION_SHARE, figures.describe()
    assert figures.overcommit_ticks == 0, figures.describe()
    assert not run_window.world.clearance_timeouts, figures.describe()
