"""Detectors that read the shared job lifecycle model: dispatch spread, churn, placement, loop health.

The marquee case is the one from the real incident: an eight-card host whose queue ran through two or
three cards while fourteen of sixteen lanes idled with the right weights already resident. The tool had
to be able to reach that on its own, and it must not reach it on a single-card worker (where one busy
card is full utilisation) or on a multi-card worker whose dispatch does spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.detectors import Finding, Severity, run_detectors
from horde_worker_regen.analysis.sessions import segment_sessions
from tests.analysis.test_job_lifecycle import (
    auxiliary_lane_safety_sharing,
    bridge,
    driving_cards,
    inference_dispatched,
    inference_finished,
    inference_lane_started,
    ipc_drain,
    model_preloading,
    model_unloaded,
    popped_job,
    post_process_lane_started,
    preload_cleared,
    safety_checked,
    safety_lane_started,
    status_block,
    utilities_lane_started,
)

_MODELS = [
    "Deliberate",
    "Nova Anime XL",
    "AlbedoBase XL 3.1",
    "CyberRealistic Pony",
    "Dreamshaper",
    "Rev Animated",
    "Pony Diffusion XL",
    "Anything v5",
    "Flat-2D Animerge",
    "Nova Furry XL",
    "Quiet Goodnight XL",
    "AMPonyXL",
    "AbsoluteReality",
    "Mistoon Anime",
    "Nova Flat XL",
    "ZavyChromaXL",
]
_BASE = datetime(2026, 9, 9, 1, 0, 0)


def diagnose(tmp_path: Path, bridge_log: str) -> dict[str, Finding]:
    """Write a synthetic bridge log and return its single session's findings, keyed by id."""
    (tmp_path / "bridge.log").write_text(bridge_log, encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    return {finding.id: finding for finding in run_detectors(build_session_context(session, bundle))}


def _stamp(offset_seconds: float) -> str:
    """A ``HH:MM:SS.mmm`` stamp this many seconds after the synthetic session's start."""
    return (_BASE + timedelta(seconds=offset_seconds)).strftime("%H:%M:%S.%f")[:-3]


def _job_id(index: int) -> str:
    """A stable synthetic job UUID for job number ``index``."""
    return f"{index:08x}-0000-4000-8000-000000000000"


def _fleet(cards: int, *, lanes_per_card: int = 2) -> tuple[list[str], dict[int, int]]:
    """Spawn lines for ``cards`` cards' inference lanes, and the resulting lane->card map."""
    lines: list[str] = []
    lane_card: dict[int, int] = {}
    process_id = 3
    for device in range(cards):
        for _ in range(lanes_per_card):
            lines.append(inference_lane_started(_stamp(0), process=process_id, device=device))
            lane_card[process_id] = device
            process_id += 1
    return lines, lane_card


def _multi_card_session(
    *,
    cards: int = 8,
    lanes_per_card: int = 2,
    busy_lanes: tuple[int, ...] = (3, 5),
    jobs: int = 24,
    job_seconds: float = 20.0,
    queue_depth: int | None = None,
) -> str:
    """A multi-card session whose dispatch runs through ``busy_lanes`` only, with a queue always full.

    Every other lane sits idle holding a model the pending queue is asking for, which is the seating the
    census must see: the fleet could have started those jobs without loading anything.
    """
    fleet_lines, lane_card = _fleet(cards, lanes_per_card=lanes_per_card)
    lanes = sorted(lane_card)
    lines: list[str] = [
        driving_cards(_stamp(0), cards=cards, processes_per_card=lanes_per_card),
        safety_lane_started(_stamp(0)),
        *fleet_lines,
    ]
    lane_model = {lane: _MODELS[index % len(_MODELS)] for index, lane in enumerate(lanes)}

    # The pending queue is the intake budget's worth of jobs, each for a model one lane holds. The head
    # asks for the model the first busy lane is sampling, which is the "resident but blocked" state.
    pending = [(_job_id(900 + index), lane_model[lane]) for index, lane in enumerate(lanes)]
    pending.sort(key=lambda entry: entry[1] != lane_model[busy_lanes[0]])
    if queue_depth is not None:
        pending = pending[:queue_depth]

    offset = 10.0
    for index in range(jobs):
        lane = busy_lanes[index % len(busy_lanes)]
        job_id = _job_id(index)
        start = offset + (index // len(busy_lanes)) * job_seconds
        lines.append(popped_job(_stamp(start - 5), job_id=job_id, model=lane_model[lane]))
        lines.append(inference_dispatched(_stamp(start), job_id=job_id, process=lane))
        lines.append(
            inference_finished(
                _stamp(start + job_seconds),
                job_id=job_id,
                model=lane_model[lane],
                process=lane,
                seconds=job_seconds,
            ),
        )

    total_seconds = offset + (jobs // len(busy_lanes)) * job_seconds
    tick = 0.0
    while tick < total_seconds:
        lines.extend(
            status_block(
                _stamp(tick),
                lanes=[
                    (
                        lane,
                        "50% of 20 steps using k_euler" if lane in busy_lanes else "WAITING_FOR_JOB",
                        lane_model[lane],
                    )
                    for lane in lanes
                ],
                aux=[(0, "SAFETY", "WAITING_FOR_JOB")],
                pending=pending,
            ),
        )
        for drain_offset in range(0, 20, 2):
            lines.append(ipc_drain(_stamp(tick + drain_offset), process=busy_lanes[0]))
        tick += 20.0
    return bridge(lines)


class TestMultiCardDispatchSerialization:
    """Dispatch collapsing onto a few cards of a fleet while the rest idle behind a full queue."""

    def test_fires_when_the_fleet_runs_through_two_of_eight_cards(self, tmp_path: Path) -> None:
        """The incident signature: two busy cards of eight, a full queue, and seatable idle lanes."""
        finding = diagnose(tmp_path, _multi_card_session())["multi_card_dispatch_serialization"]
        assert finding.severity is Severity.CRITICAL
        assert "8 cards" in finding.headline
        assert "2.0 of them" in finding.headline
        # The census, not a guess, is what makes this a scheduling verdict.
        joined = " ".join(finding.evidence)
        assert "sampling concurrency" in joined
        assert "resident on an idle lane on an unoccupied card" in joined
        # The head's model is resident on the lane that is sampling it, so the queue is blocked behind
        # work the rest of the fleet had the capacity to run.
        assert "had the head's model already resident but every holder's card busy" in finding.headline
        assert "100%" in finding.headline
        assert "parent loop was healthy" in finding.headline

    def test_silent_on_a_single_card_worker(self, tmp_path: Path) -> None:
        """A single-card worker emits no 'Driving N cards' line, and one busy card is not a shortfall."""
        lines: list[str] = [inference_lane_started(_stamp(0), process=3, device=0)]
        for index in range(24):
            job_id = _job_id(index)
            start = 10.0 + index * 20.0
            lines.append(popped_job(_stamp(start - 5), job_id=job_id, model="Deliberate"))
            lines.append(inference_dispatched(_stamp(start), job_id=job_id, process=3))
            lines.append(
                inference_finished(
                    _stamp(start + 20.0),
                    job_id=job_id,
                    model="Deliberate",
                    process=3,
                    seconds=20.0,
                ),
            )
        assert "multi_card_dispatch_serialization" not in diagnose(tmp_path, bridge(lines))

    def test_silent_when_dispatch_tracks_the_card_count(self, tmp_path: Path) -> None:
        """Eight cards each holding a job is the healthy case, however deep the queue is."""
        healthy = _multi_card_session(busy_lanes=(3, 5, 7, 9, 11, 13, 15, 17), jobs=32)
        assert "multi_card_dispatch_serialization" not in diagnose(tmp_path, healthy)

    def test_silent_without_the_driving_cards_line(self, tmp_path: Path) -> None:
        """With no card count the shortfall cannot be sized, so the detector degrades to silence."""
        text = _multi_card_session()
        without_shape = "\n".join(line for line in text.splitlines() if "Driving 8 cards" not in line)
        assert "multi_card_dispatch_serialization" not in diagnose(tmp_path, without_shape)

    def test_silent_when_the_queue_was_not_full(self, tmp_path: Path) -> None:
        """An idle fleet with nothing queued is the horde having no work, not a dispatch defect."""
        text = _multi_card_session()
        empty_queue = "\n".join(
            "  Jobs: " if line.rstrip().endswith(">") and "_print_job_info" in line else line
            for line in text.splitlines()
        )
        empty_queue = "\n".join(
            line.split("  Jobs: ")[0] + "  Jobs: " if "_print_job_info" in line else line
            for line in empty_queue.splitlines()
        )
        assert "multi_card_dispatch_serialization" not in diagnose(tmp_path, empty_queue)


class TestModelChurn:
    """Model weights moving through the lanes faster than the dispatched work justifies."""

    def _session(self, *, dispatches: int, preloads: int, cleared: int) -> str:
        lines: list[str] = [inference_lane_started(_stamp(0), process=3, device=0)]
        for index in range(dispatches):
            job_id = _job_id(index)
            lines.append(popped_job(_stamp(10 + index), job_id=job_id, model="Deliberate"))
            lines.append(inference_dispatched(_stamp(11 + index), job_id=job_id, process=3))
            lines.append(
                inference_finished(
                    _stamp(12 + index),
                    job_id=job_id,
                    model="Deliberate",
                    process=3,
                    seconds=1.0,
                ),
            )
        for index in range(preloads):
            lines.append(model_preloading(_stamp(200 + index), model=_MODELS[index % len(_MODELS)], process=3))
            lines.append(model_unloaded(_stamp(200 + index), model=_MODELS[index % len(_MODELS)], process=3))
        for index in range(cleared):
            lines.append(preload_cleared(_stamp(400 + index), model=_MODELS[index % len(_MODELS)], process=3))
        return bridge(lines)

    def test_fires_when_preloads_approach_a_third_of_dispatches(self, tmp_path: Path) -> None:
        """A preload for every other job means the resident set turns over faster than the queue."""
        finding = diagnose(tmp_path, self._session(dispatches=40, preloads=20, cleared=5))["model_churn"]
        assert finding.severity is Severity.WARNING
        assert "0.50 per job" in " ".join(finding.evidence)

    def test_is_critical_when_preloads_outnumber_dispatches(self, tmp_path: Path) -> None:
        """More model changes than jobs is thrash, not turnover."""
        finding = diagnose(tmp_path, self._session(dispatches=40, preloads=45, cleared=0))["model_churn"]
        assert finding.severity is Severity.CRITICAL

    def test_fires_on_cleared_preloads_alone(self, tmp_path: Path) -> None:
        """Weights fetched for a job the scheduler then seated elsewhere are waste worth naming."""
        assert "model_churn" in diagnose(tmp_path, self._session(dispatches=40, preloads=2, cleared=4))

    def test_silent_when_lanes_keep_their_models(self, tmp_path: Path) -> None:
        """A settled worker loads a model once and serves many jobs from it."""
        assert "model_churn" not in diagnose(tmp_path, self._session(dispatches=40, preloads=3, cleared=1))

    def test_silent_on_too_few_dispatches_to_form_a_ratio(self, tmp_path: Path) -> None:
        """A handful of jobs is a warm-up; every model load in it looks like churn."""
        assert "model_churn" not in diagnose(tmp_path, self._session(dispatches=5, preloads=5, cleared=3))


class TestLanePlacement:
    """Auxiliary lanes that moved cards mid-session, or that stacked onto the safety lane's card."""

    def test_fires_on_an_auxiliary_lane_migration(self, tmp_path: Path) -> None:
        """A post-processing lane re-spawned onto another card is a placement change worth naming."""
        finding = diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                post_process_lane_started(_stamp(0), process=1, device=2),
                post_process_lane_started(_stamp(600), process=1, device=3),
            ),
        )["lane_placement"]
        assert finding.severity is Severity.WARNING
        assert "moved from card 2 to card 3" in " ".join(finding.evidence)

    def test_fires_when_an_auxiliary_lane_stacks_onto_the_safety_card(self, tmp_path: Path) -> None:
        """Post-processing on card 0 shares the card GPU safety runs on, with other cards free."""
        finding = diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                safety_lane_started(_stamp(0)),
                post_process_lane_started(_stamp(0), process=1, device=0),
            ),
        )["lane_placement"]
        assert "alongside the safety lane" in finding.headline

    def test_quotes_the_workers_own_sharing_notice_when_the_session_carries_one(self, tmp_path: Path) -> None:
        """The edge-triggered INFO dates the sharing to a safety restore, which the occupancy map cannot."""
        finding = diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                safety_lane_started(_stamp(0)),
                post_process_lane_started(_stamp(0), process=1, device=0),
                auxiliary_lane_safety_sharing(_stamp(600), lanes="post-processing", device=0),
            ),
        )["lane_placement"]
        assert "now share device 0" in " ".join(finding.evidence)

    def test_the_sharing_notice_alone_does_not_trigger_the_finding(self, tmp_path: Path) -> None:
        """The line corroborates a placement the occupancy map already shows; it is not its own trigger."""
        assert "lane_placement" not in diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                safety_lane_started(_stamp(0)),
                post_process_lane_started(_stamp(0), process=1, device=2),
                auxiliary_lane_safety_sharing(_stamp(600), lanes="post-processing", device=2),
            ),
        )

    def test_silent_when_auxiliary_lanes_stay_put_off_card_zero(self, tmp_path: Path) -> None:
        """Stable placement away from the safety card is the intended arrangement."""
        assert "lane_placement" not in diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                safety_lane_started(_stamp(0)),
                post_process_lane_started(_stamp(0), process=1, device=2),
                utilities_lane_started(_stamp(0), process=2, device=3),
            ),
        )

    def test_silent_on_an_inference_lane_respawn(self, tmp_path: Path) -> None:
        """Inference lanes belong to their card's pool by construction; a re-spawn is not a migration."""
        assert "lane_placement" not in diagnose(
            tmp_path,
            bridge(
                driving_cards(_stamp(0), cards=4),
                inference_lane_started(_stamp(0), process=3, device=1),
                inference_lane_started(_stamp(600), process=3, device=1),
            ),
        )

    def test_silent_on_a_single_card_host(self, tmp_path: Path) -> None:
        """With one card there is nowhere else for an auxiliary lane to be."""
        assert "lane_placement" not in diagnose(
            tmp_path,
            bridge(safety_lane_started(_stamp(0)), post_process_lane_started(_stamp(0), process=1, device=0)),
        )


def _safety_stage_session(
    *,
    waits: Sequence[float],
    cards: int = 8,
    check_seconds: float = 1.2,
    finish_interval: float = 2.0,
    generation_seconds: float = 16.0,
    with_safety_lines: bool = True,
) -> str:
    """A fleet finishing a job every ``finish_interval`` seconds, each waiting ``waits[i]`` for safety.

    One job per entry in ``waits``. The cards only set the finish rate here: what the detector weighs is
    that rate against the single checker's own median check duration.
    """
    lines: list[str] = []
    if cards > 1:
        lines.append(driving_cards(_stamp(0), cards=cards))
        fleet_lines, lane_card = _fleet(cards, lanes_per_card=1)
        lines.extend(fleet_lines)
        lanes = sorted(lane_card)
    else:
        lines.append(inference_lane_started(_stamp(0), process=3, device=0))
        lanes = [3]
    lines.append(safety_lane_started(_stamp(0)))

    for index, wait in enumerate(waits):
        job_id = _job_id(index)
        finished = 100.0 + index * finish_interval
        lane = lanes[index % len(lanes)]
        model = _MODELS[index % len(_MODELS)]
        lines.append(popped_job(_stamp(finished - generation_seconds - 5), job_id=job_id, model=model))
        lines.append(inference_dispatched(_stamp(finished - generation_seconds), job_id=job_id, process=lane))
        lines.append(
            inference_finished(
                _stamp(finished),
                job_id=job_id,
                model=model,
                process=lane,
                seconds=generation_seconds,
            ),
        )
        if with_safety_lines:
            lines.append(safety_checked(_stamp(finished + wait), job_id=job_id, seconds=check_seconds))
    return bridge(sorted(lines))


class TestSafetyStageCapacity:
    """One safety process serving a whole fleet, and the queue that forms once it is the serial stage."""

    def test_fires_when_the_fleet_outruns_the_single_checker(self, tmp_path: Path) -> None:
        """Eight cards finishing every 2s against a 1.2s check: the wait grows until it dwarfs sampling."""
        session = _safety_stage_session(waits=[2.0 + index * 1.5 for index in range(40)])
        finding = diagnose(tmp_path, session)["safety_stage_capacity"]
        assert finding.severity is Severity.CRITICAL
        assert "8 cards" in finding.headline
        # Demand, capacity and the wait are all named, each from the session's own numbers.
        assert "0.50 job(s) per second" in finding.headline
        assert "0.83 per second" in finding.headline
        assert "1.20s per check" in finding.headline
        assert "costs more wall clock than the GPUs do" in finding.headline
        joined = " ".join(finding.evidence)
        assert "finished->safety wait" in joined
        assert "safety_on_gpu" in finding.action
        assert "max_power will not help" in finding.action

    def test_silent_when_the_wait_is_about_one_check(self, tmp_path: Path) -> None:
        """A checker keeping up costs each job one check duration; that is the stage working, not queuing."""
        session = _safety_stage_session(waits=[1.2] * 40)
        assert "safety_stage_capacity" not in diagnose(tmp_path, session)

    def test_silent_on_a_single_card_worker(self, tmp_path: Path) -> None:
        """One card cannot outrun the checker: whatever it waits for, it is not a fleet's finish rate."""
        session = _safety_stage_session(waits=[2.0 + index * 1.5 for index in range(40)], cards=1)
        assert "safety_stage_capacity" not in diagnose(tmp_path, session)

    def test_silent_without_safety_lines(self, tmp_path: Path) -> None:
        """With no check durations there is no capacity to compare a finish rate against."""
        session = _safety_stage_session(
            waits=[2.0 + index * 1.5 for index in range(40)],
            with_safety_lines=False,
        )
        assert "safety_stage_capacity" not in diagnose(tmp_path, session)

    def test_silent_on_too_few_checked_jobs(self, tmp_path: Path) -> None:
        """A handful of checks is a warm-up; the first cold check alone would set the median."""
        session = _safety_stage_session(waits=[2.0 + index * 1.5 for index in range(6)])
        assert "safety_stage_capacity" not in diagnose(tmp_path, session)


class TestParentLoopStall:
    """Silences in the parent's IPC drain, measured against the session's own status cadence."""

    def _session(self, *, gap_at: float | None) -> str:
        lines: list[str] = []
        tick = 0.0
        while tick < 600.0:
            lines.extend(status_block(_stamp(tick), lanes=[(3, "WAITING_FOR_JOB", None)], pending=[]))
            tick += 20.0
        drain_tick = 0.0
        while drain_tick < 600.0:
            if gap_at is None or not (gap_at <= drain_tick < gap_at + 120.0):
                lines.append(ipc_drain(_stamp(drain_tick), process=3))
            drain_tick += 2.0
        return bridge(sorted(lines))

    def test_fires_on_a_silence_longer_than_the_derived_threshold(self, tmp_path: Path) -> None:
        """Two minutes with no drained message, against a 20s status cadence, is a stalled loop."""
        finding = diagnose(tmp_path, self._session(gap_at=200.0))["parent_loop_stall"]
        assert finding.severity is Severity.CRITICAL
        assert "drained no child messages" in finding.headline
        assert "status cadence" in finding.headline

    def test_silent_on_a_healthy_loop(self, tmp_path: Path) -> None:
        """A loop that keeps draining produces no finding; findings are for problems."""
        assert "parent_loop_stall" not in diagnose(tmp_path, self._session(gap_at=None))

    def test_silent_when_the_capture_carries_no_drain_lines(self, tmp_path: Path) -> None:
        """A log without the parent's IPC debug stream says nothing about the loop either way."""
        lines: list[str] = []
        tick = 0.0
        while tick < 600.0:
            lines.extend(status_block(_stamp(tick), lanes=[(3, "WAITING_FOR_JOB", None)], pending=[]))
            tick += 20.0
        assert "parent_loop_stall" not in diagnose(tmp_path, bridge(lines))


class TestLifecycleDetectorsNeverRaise:
    """``run_detectors`` swallows detector exceptions, so a crash here would be invisible in the report."""

    def _ran(self, tmp_path: Path, bridge_log: str) -> list[str]:
        (tmp_path / "bridge.log").write_text(bridge_log, encoding="utf-8")
        bundle = LogBundle.from_path(tmp_path)
        session = segment_sessions(bundle.orchestrator_records())[0]
        context = build_session_context(session, bundle)
        from horde_worker_regen.analysis.detectors import (
            detect_lane_placement,
            detect_model_churn,
            detect_multi_card_dispatch_serialization,
            detect_parent_loop_stall,
        )

        findings: list[str] = []
        for detector in (
            detect_multi_card_dispatch_serialization,
            detect_model_churn,
            detect_lane_placement,
            detect_parent_loop_stall,
        ):
            findings.extend(finding.id for finding in detector(context))
        return findings

    def test_empty_session_produces_nothing_and_raises_nothing(self, tmp_path: Path) -> None:
        """A session with only its startup line is not an error; it is a session with no evidence."""
        assert self._ran(tmp_path, bridge()) == []

    def test_status_blocks_alone_produce_nothing_and_raise_nothing(self, tmp_path: Path) -> None:
        """Status prints with no jobs, lanes or shape lines exercise every derived view on empty input."""
        lines: list[str] = []
        for tick in range(0, 200, 20):
            lines.extend(
                status_block(
                    str(_stamp(tick)),
                    lanes=[(3, "PROCESS_STARTING", None), (4, "WAITING_FOR_JOB", None)],
                    aux=[(0, "SAFETY", "WAITING_FOR_JOB")],
                    pending=[],
                ),
            )
        assert self._ran(tmp_path, bridge(lines)) == []
