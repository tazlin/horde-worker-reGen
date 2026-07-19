"""Tests for the pure demand-weighted pool ranker.

Exercises the ranking formula (demand times speed factor), the three-level ordering (score, then cluster size,
then name), neutral speed for unmeasured models, exclusions, models absent from the demand snapshot, and the
rescue-candidate eta threshold and max-eta pick, all over hand-built snapshots with no network or config.
"""

from __future__ import annotations

from collections.abc import Callable

from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot, ModelDemandRecord
from horde_worker_regen.process_management.scheduling.pool_ranker import (
    rank_candidates,
    select_rescue_candidate,
)


def _record(
    *,
    queued: float | None = None,
    worker_count: int | None = None,
    eta_seconds: int | None = None,
) -> ModelDemandRecord:
    """Build a demand record carrying only the fields the ranker reads."""
    return ModelDemandRecord(
        queued=queued,
        jobs=None,
        eta_seconds=eta_seconds,
        worker_count=worker_count,
        performance=None,
    )


def _snapshot(records: dict[str, ModelDemandRecord]) -> DemandSnapshot:
    """Wrap records in a snapshot with an arbitrary fetch time."""
    return DemandSnapshot(records=records, fetched_at=0.0)


def _constant_value(speed: float | None) -> Callable[[str], float | None]:
    """Return a speed lookup that reports the same value for every model."""
    return lambda _model: speed


class TestRankOrdering:
    """Score descending, then cluster size descending, then name ascending."""

    def test_orders_by_score_descending(self) -> None:
        """Higher queued-per-worker demand ranks first when speed is uniform."""
        snapshot = _snapshot(
            {
                "low": _record(queued=1.0, worker_count=0),
                "high": _record(queued=100.0, worker_count=0),
            },
        )
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["low", "high"],
            on_disk=frozenset(),
            expected_value=_constant_value(1.0),
            cluster_sizes={},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["high", "low"]
        assert ranked[0].score > ranked[1].score

    def test_cluster_size_breaks_score_ties(self) -> None:
        """Equal demand and speed: the larger shared-VAE cluster ranks first."""
        snapshot = _snapshot(
            {
                "small": _record(queued=10.0, worker_count=0),
                "big": _record(queued=10.0, worker_count=0),
            },
        )
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["small", "big"],
            on_disk=frozenset(),
            expected_value=_constant_value(1.0),
            cluster_sizes={"small": 1, "big": 4},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["big", "small"]

    def test_name_breaks_score_and_cluster_ties(self) -> None:
        """Equal score and cluster size: names decide, ascending."""
        snapshot = _snapshot(
            {
                "b": _record(queued=10.0, worker_count=0),
                "a": _record(queued=10.0, worker_count=0),
            },
        )
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["b", "a"],
            on_disk=frozenset(),
            expected_value=_constant_value(1.0),
            cluster_sizes={},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["a", "b"]


class TestSpeedWeighting:
    """Speed normalisation, neutral midpoint for unknowns, and its effect on the score."""

    def test_faster_model_outranks_slower_at_equal_demand(self) -> None:
        """With equal demand, the faster model scores higher via the speed factor."""
        snapshot = _snapshot(
            {
                "slow": _record(queued=10.0, worker_count=0),
                "fast": _record(queued=10.0, worker_count=0),
            },
        )
        speeds = {"slow": 1.0, "fast": 4.0}
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["slow", "fast"],
            on_disk=frozenset(),
            expected_value=lambda model: speeds[model],
            cluster_sizes={},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["fast", "slow"]

    def test_unknown_speed_takes_neutral_midpoint(self) -> None:
        """An unmeasured model sits between the fastest and a slow measured model, not excluded."""
        snapshot = _snapshot(
            {
                "fast": _record(queued=10.0, worker_count=0),
                "unknown": _record(queued=10.0, worker_count=0),
                "slow": _record(queued=10.0, worker_count=0),
            },
        )
        speeds: dict[str, float | None] = {"fast": 10.0, "unknown": None, "slow": 1.0}
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["fast", "unknown", "slow"],
            on_disk=frozenset(),
            expected_value=lambda model: speeds[model],
            cluster_sizes={},
            excluded=frozenset(),
        )
        # fast normalises to 1.0, unknown to the neutral 0.5, slow to 0.1: order is fast, unknown, slow.
        assert [model.name for model in ranked] == ["fast", "unknown", "slow"]

    def test_all_unknown_speeds_leave_ranking_demand_driven(self) -> None:
        """When no candidate has a known speed, demand alone orders them."""
        snapshot = _snapshot(
            {
                "more": _record(queued=50.0, worker_count=0),
                "less": _record(queued=5.0, worker_count=0),
            },
        )
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["more", "less"],
            on_disk=frozenset(),
            expected_value=_constant_value(None),
            cluster_sizes={},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["more", "less"]


class TestExclusionsAndAbsence:
    """Excluded models never appear; snapshot-absent models carry zero demand."""

    def test_excluded_models_never_appear(self) -> None:
        """An excluded model is dropped even when its demand is highest."""
        snapshot = _snapshot(
            {
                "banned": _record(queued=1000.0, worker_count=0),
                "kept": _record(queued=1.0, worker_count=0),
            },
        )
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["banned", "kept"],
            on_disk=frozenset(),
            expected_value=_constant_value(1.0),
            cluster_sizes={},
            excluded=frozenset({"banned"}),
        )
        assert [model.name for model in ranked] == ["kept"]

    def test_absent_from_snapshot_scores_zero(self) -> None:
        """A candidate the horde is not asking for carries zero demand and thus zero score."""
        snapshot = _snapshot({"wanted": _record(queued=10.0, worker_count=0)})
        ranked = rank_candidates(
            snapshot=snapshot,
            candidates=["wanted", "unwanted"],
            on_disk=frozenset({"unwanted"}),
            expected_value=_constant_value(1.0),
            cluster_sizes={},
            excluded=frozenset(),
        )
        assert [model.name for model in ranked] == ["wanted", "unwanted"]
        unwanted = next(model for model in ranked if model.name == "unwanted")
        assert unwanted.score == 0.0
        assert unwanted.on_disk is True
        assert unwanted.eta_seconds is None


class TestRescueCandidate:
    """The rescue picks the most-backed-up candidate past the eta threshold."""

    def test_picks_max_eta_over_threshold(self) -> None:
        """Among candidates over the threshold, the greatest eta is rescued."""
        snapshot = _snapshot(
            {
                "mild": _record(eta_seconds=120),
                "severe": _record(eta_seconds=600),
                "worst": _record(eta_seconds=900),
            },
        )
        rescue = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["mild", "severe", "worst"],
            excluded=frozenset(),
            eta_threshold_seconds=300.0,
        )
        assert rescue is not None
        assert rescue.name == "worst"
        assert rescue.eta_seconds == 900

    def test_below_threshold_and_missing_eta_ignored(self) -> None:
        """Candidates under the threshold, absent, or without an eta are not rescued."""
        snapshot = _snapshot(
            {
                "under": _record(eta_seconds=100),
                "no_eta": _record(eta_seconds=None),
            },
        )
        rescue = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["under", "no_eta", "absent"],
            excluded=frozenset(),
            eta_threshold_seconds=300.0,
        )
        assert rescue is None

    def test_excluded_never_rescued(self) -> None:
        """An excluded model is not rescued even when it is the most backed up."""
        snapshot = _snapshot(
            {
                "banned": _record(eta_seconds=5000),
                "ok": _record(eta_seconds=400),
            },
        )
        rescue = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["banned", "ok"],
            excluded=frozenset({"banned"}),
            eta_threshold_seconds=300.0,
        )
        assert rescue is not None
        assert rescue.name == "ok"

    def test_eta_ties_break_by_name(self) -> None:
        """Equal etas over the threshold resolve to the lexicographically smaller name."""
        snapshot = _snapshot(
            {
                "b": _record(eta_seconds=500),
                "a": _record(eta_seconds=500),
            },
        )
        rescue = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["b", "a"],
            excluded=frozenset(),
            eta_threshold_seconds=300.0,
        )
        assert rescue is not None
        assert rescue.name == "a"

    def test_on_disk_membership_carried_through(self) -> None:
        """The rescue result reports whether the chosen model's weights are already held locally."""
        snapshot = _snapshot({"starving": _record(eta_seconds=9000)})
        held = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["starving"],
            on_disk=frozenset({"starving"}),
            excluded=frozenset(),
            eta_threshold_seconds=300.0,
        )
        missing = select_rescue_candidate(
            snapshot=snapshot,
            candidates=["starving"],
            excluded=frozenset(),
            eta_threshold_seconds=300.0,
        )
        assert held is not None and held.on_disk is True
        assert missing is not None and missing.on_disk is False
