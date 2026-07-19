"""Pure ranking of candidate models by live demand weighted by expected local speed.

Given a :class:`~horde_worker_regen.process_management.scheduling.model_demand_poller.DemandSnapshot` of live per-model
queue demand and a set of locally-servable candidate models, this module orders those candidates by a single
score: how much work is queued per serving worker, weighted by how fast this worker expects to run the model.
The result tells pop-shaping and load-selection code which models to favour so the worker spends its time where
demand is high and it is fast, without loading models the horde is not asking for.

The scoring is deliberately simple and fully specified:

- ``demand(model) = log1p(queued_per_worker(model))`` where ``queued_per_worker = queued / (worker_count + 1)``.
  Models absent from the snapshot carry demand ``0`` (the horde is not asking for them).
- ``value(model)`` is the injected expected earning rate (kudos per wall second on this card, or any
  proportional stand-in) normalised across the candidate set to ``(0, 1]`` by dividing by the set maximum; a
  model with no known value takes the neutral midpoint ``0.5`` rather than being excluded.
- ``score = demand * value_normalised``, so a deep queue only wins in proportion to what serving it actually
  pays per second. Demand is log-compressed while value enters at full weight: a bottomless queue of cheap
  jobs must not outrank a healthy queue of well-paying ones (the log means a 100x deeper queue is worth
  under 2x, while a 2x better payer is worth 2x), and a zero-demand model always scores zero.

The module is pure: stdlib plus the :class:`DemandSnapshot` type only, no torch, no ``hordelib.api``, no config
object. The expected-value source is an injected callable, not an import, so the performance model, a seed
benchmark, or a test stub can all supply it. There is no adapter to a live performance model here on purpose:
turning a bare model name into an expected earning rate needs a job signature and a price estimate (resolution,
steps, baseline, kudos pricing), which is manager and config state this module must not reach. That resolution
belongs in wiring code, which passes the resulting ``name -> value`` callable in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot

__all__ = [
    "RankedModel",
    "rank_candidates",
    "select_rescue_candidate",
]

_NEUTRAL_VALUE = 0.5
"""Normalised value assigned to a candidate whose expected earning rate is unknown: the midpoint of the
``(0, 1]`` range, so an unmeasured model is neither favoured nor penalised on the value axis."""

_UNCLUSTERED_SIZE = 1
"""Cluster size charged to a model whose shared-VAE cluster is unknown to the caller (itself only)."""


@dataclass(frozen=True)
class RankedModel:
    """Represents one candidate model's place in the demand-weighted ranking.

    ``score`` is the demand-times-speed value the ordering is built on, ``on_disk`` whether the worker already
    holds the weights, ``eta_seconds`` the horde's queue-clear estimate (``None`` when unreported), and
    ``cluster_size`` the model's shared-VAE cluster size used as the first tie-break after score.
    """

    name: str
    score: float
    on_disk: bool
    eta_seconds: float | None
    cluster_size: int


def _normalised_values(
    candidate_names: list[str],
    expected_value: Callable[[str], float | None],
) -> dict[str, float]:
    """Return each candidate's expected earning rate normalised to ``(0, 1]``, unknowns at the neutral midpoint.

    Known values are divided by the maximum known value across the candidate set, so the best payer maps to
    ``1.0``; a candidate with no known value takes :data:`_NEUTRAL_VALUE`. When no candidate has a known value
    (or the maximum is non-positive) every candidate is neutral, leaving the ranking demand-driven.
    """
    known_values: dict[str, float] = {}
    for candidate_name in candidate_names:
        value = expected_value(candidate_name)
        if value is not None:
            known_values[candidate_name] = value

    max_value = max(known_values.values(), default=0.0)
    if max_value <= 0:
        return dict.fromkeys(candidate_names, _NEUTRAL_VALUE)

    normalised: dict[str, float] = {}
    for candidate_name in candidate_names:
        value = known_values.get(candidate_name)
        normalised[candidate_name] = _NEUTRAL_VALUE if value is None else value / max_value
    return normalised


def _demand(snapshot: DemandSnapshot, model: str) -> float:
    """Return the log-compressed queued-per-worker demand for ``model``, or ``0.0`` when it is absent."""
    queued_per_worker = snapshot.queued_per_worker(model)
    if queued_per_worker is None:
        return 0.0
    return math.log1p(queued_per_worker)


def rank_candidates(
    *,
    snapshot: DemandSnapshot,
    candidates: Iterable[str],
    on_disk: frozenset[str],
    expected_value: Callable[[str], float | None],
    cluster_sizes: Mapping[str, int],
    excluded: frozenset[str],
) -> list[RankedModel]:
    """Rank ``candidates`` by demand weighted by normalised expected earning rate, most valuable first.

    Each non-excluded candidate scores ``demand * value_normalised``, where demand is
    ``log1p(queued_per_worker)`` (zero when the model is absent from ``snapshot``) and ``value_normalised``
    is the candidate's expected earning rate divided by the candidate-set maximum (a neutral ``0.5`` when
    unknown). Value enters at full weight against log-compressed demand deliberately: seat selection must
    maximise what serving a queue pays per second of this card's time, so a bottomless queue of cheap jobs
    cannot outrank a healthy queue of well-paying ones. The result is ordered by score descending, then by
    cluster size descending, then by name ascending, so the order is total and deterministic.

    Args:
        snapshot: The live per-model demand reading.
        candidates: The locally-servable model names to rank.
        on_disk: The subset of names whose weights the worker already holds.
        expected_value: Injected lookup of a model's expected earning rate (kudos per wall second on this
            card, or any proportional stand-in), ``None`` when unknown.
        cluster_sizes: Per-model shared-VAE cluster size; a missing model is treated as unclustered (size 1).
        excluded: Names that must never appear in the ranking.

    Returns:
        The ranked candidates, best first.
    """
    candidate_names = [name for name in candidates if name not in excluded]
    normalised_values = _normalised_values(candidate_names, expected_value)

    ranked_models: list[RankedModel] = []
    for candidate_name in candidate_names:
        demand = _demand(snapshot, candidate_name)
        record = snapshot.records.get(candidate_name)
        ranked_models.append(
            RankedModel(
                name=candidate_name,
                score=demand * normalised_values[candidate_name],
                on_disk=candidate_name in on_disk,
                eta_seconds=None if record is None else record.eta_seconds,
                cluster_size=cluster_sizes.get(candidate_name, _UNCLUSTERED_SIZE),
            ),
        )

    ranked_models.sort(key=lambda ranked: (-ranked.score, -ranked.cluster_size, ranked.name))
    return ranked_models


def select_rescue_candidate(
    *,
    snapshot: DemandSnapshot,
    candidates: Iterable[str],
    on_disk: frozenset[str] = frozenset(),
    excluded: frozenset[str],
    eta_threshold_seconds: float,
) -> RankedModel | None:
    """Return the candidate whose queue is most backed up past ``eta_threshold_seconds``, or ``None``.

    A rescue targets the model the horde will take longest to clear: among non-excluded candidates whose
    reported eta is at least ``eta_threshold_seconds``, the one with the greatest eta is chosen. Candidates
    absent from the snapshot or without a reported eta are ignored (their backlog is unknown). Ties on eta
    resolve by name ascending for determinism. The returned :class:`RankedModel` carries the model's eta but a
    zero score and unclustered defaults, since a rescue is chosen on backlog, not the demand-weighted score.

    Args:
        snapshot: The live per-model demand reading.
        candidates: The locally-servable model names to consider rescuing.
        on_disk: The subset of names whose weights the worker already holds, carried through so the
            caller can distinguish a rescue that can start immediately from one needing a download.
        excluded: Names that must never be rescued.
        eta_threshold_seconds: Minimum reported eta for a model to be a rescue candidate.

    Returns:
        The most-backed-up eligible candidate, or ``None`` when none clears the threshold.
    """
    rescue_candidates: list[RankedModel] = []
    for candidate_name in candidates:
        if candidate_name in excluded:
            continue
        record = snapshot.records.get(candidate_name)
        if record is None or record.eta_seconds is None:
            continue
        if record.eta_seconds < eta_threshold_seconds:
            continue
        rescue_candidates.append(
            RankedModel(
                name=candidate_name,
                score=0.0,
                on_disk=candidate_name in on_disk,
                eta_seconds=record.eta_seconds,
                cluster_size=_UNCLUSTERED_SIZE,
            ),
        )

    if not rescue_candidates:
        return None
    return min(rescue_candidates, key=lambda ranked: (-(ranked.eta_seconds or 0.0), ranked.name))
