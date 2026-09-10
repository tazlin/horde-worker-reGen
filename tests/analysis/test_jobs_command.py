"""The ``horde-log jobs`` view: the per-job lifecycle table, its summary, and its JSON form."""

from __future__ import annotations

import json
from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.job_lifecycle import job_lifecycle_for
from horde_worker_regen.analysis.log_triage_cli import build_parser
from horde_worker_regen.analysis.sessions import segment_sessions
from horde_worker_regen.analysis.triage_report import job_lifecycle_to_dict, render_job_lifecycle
from tests.analysis.test_job_lifecycle import (
    bridge,
    driving_cards,
    inference_dispatched,
    inference_finished,
    inference_lane_started,
    popped_job,
    submitted_generation,
)

_JOBS = [
    ("11111111-0000-4000-8000-000000000000", "Deliberate", 3, 0),
    ("22222222-0000-4000-8000-000000000000", "Nova Anime XL", 4, 1),
    ("33333333-0000-4000-8000-000000000000", "AlbedoBase XL 3.1", 5, 2),
]


def _three_job_bridge() -> str:
    """A three-job session on a three-card host, one job per card, with every lifecycle line."""
    lines = [driving_cards("01:00:00.000", cards=3)]
    for index, (_job, _model, process, device) in enumerate(_JOBS):
        lines.append(inference_lane_started("01:00:00.000", process=process, device=device))
        del index
    for index, (job_id, model, process, _device) in enumerate(_JOBS):
        second = 10 + index
        lines.append(popped_job(f"01:00:{second:02d}.000", job_id=job_id, model=model))
        lines.append(inference_dispatched(f"01:00:{second + 5:02d}.000", job_id=job_id, process=process))
        lines.append(
            inference_finished(
                f"01:00:{second + 25:02d}.000",
                job_id=job_id,
                model=model,
                process=process,
                seconds=20.0,
            ),
        )
        lines.append(
            submitted_generation(
                f"01:00:{second + 28:02d}.000",
                job_id=job_id,
                model=model,
                popped_ago=28.0,
                generate=20.0,
            ),
        )
    return bridge(lines)


def _render(tmp_path: Path, *, limit: int | None = None) -> str:
    """Render the jobs view for a synthetic three-job session."""
    (tmp_path / "bridge.log").write_text(_three_job_bridge(), encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    model = job_lifecycle_for(build_session_context(session, bundle))
    return render_job_lifecycle(session, model, limit=limit)


def test_table_carries_a_row_per_job_with_its_lane_card_and_waits(tmp_path: Path) -> None:
    """Each job's row names the lane and card that ran it and its three wait segments."""
    rendered = _render(tmp_path)
    assert "job      popped    model" in rendered
    for job_id, model, process, device in _JOBS:
        row = next(line for line in rendered.splitlines() if line.startswith(job_id[:8]))
        assert model in row
        assert row.split()[-5:] == [str(process), str(device), "5.0", "20.0", "3.0"]


def test_summary_reports_percentiles_and_the_concurrency_histogram(tmp_path: Path) -> None:
    """The summary is the part that answers "where did the time go" for the whole session."""
    rendered = _render(tmp_path)
    assert "3 job(s); wait segments (seconds):" in rendered
    assert "pre_inference   n=3" in rendered
    assert "inference       n=3" in rendered
    assert "post_inference  n=3" in rendered
    assert "Cards driven: 3" in rendered
    assert "Sampling concurrency" in rendered


def test_limit_truncates_the_table_but_not_the_summary(tmp_path: Path) -> None:
    """A capped table still summarises every job, so the cap cannot mislead about the session."""
    rendered = _render(tmp_path, limit=1)
    assert "... 2 more job(s) not shown" in rendered
    assert "3 job(s); wait segments (seconds):" in rendered


def test_json_form_carries_the_rows_and_the_summary(tmp_path: Path) -> None:
    """The JSON form is what CI and further tooling read; it must serialize cleanly."""
    (tmp_path / "bridge.log").write_text(_three_job_bridge(), encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    model = job_lifecycle_for(build_session_context(session, bundle))
    payload = json.loads(json.dumps(job_lifecycle_to_dict(session, model)))
    assert payload["card_count"] == 3
    assert len(payload["jobs"]) == 3
    assert payload["jobs"][0]["pre_inference_seconds"] == 5.0
    assert {segment["name"] for segment in payload["wait_segments"]} == {
        "pre_inference",
        "inference",
        "post_inference",
    }


def test_a_session_with_no_job_lines_says_so(tmp_path: Path) -> None:
    """A capture with no lifecycle lines renders an explanation, not an empty table."""
    (tmp_path / "bridge.log").write_text(bridge(), encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    model = job_lifecycle_for(build_session_context(session, bundle))
    assert "(no job lifecycle lines in this session)" in render_job_lifecycle(session, model)


def test_the_subcommand_is_wired_with_session_selection(tmp_path: Path) -> None:
    """``jobs`` takes the same path/session/json selection flags as ``diagnose``."""
    args = build_parser().parse_args(["jobs", str(tmp_path), "--session", "2", "--json", "--limit", "0"])
    assert args.session == 2
    assert args.json is True
    assert args.limit == 0
    assert args.func.__name__ == "_run_jobs"
