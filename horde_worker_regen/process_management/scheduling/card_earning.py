"""What each model earns per second of a card's time, measured on the worker.

:class:`CardEarningAccumulator` splits each card's wall clock between the models working on it, and holds the
kudos those models' finished jobs earned, per card and per model, in trailing time buckets. Two figures come
from it for a (model, card) pair:

- The model rate is the model's kudos over the card seconds attributed to it.
- The rest rate is the card's kudos less the model's, over the card's wall seconds less the model's seconds.

The scheduler feeds the time once per control tick (:meth:`CardEarningAccumulator.observe`) with each card's
lanes that have a job in flight, the card's whole-card grant and the card's standing pop claim. The run metrics
credit the kudos when a finished image record is folded into the rollups
(:meth:`CardEarningAccumulator.credit_finished_job`). The process manager owns the instance, persists it beside
``perf_model.json`` and saves it at shutdown. :meth:`CardEarningAccumulator.read_window` returns every pair's
figures, each card's time and the window's status in one read, for the supervisor snapshot.

The buckets hold figures for one capacity and workload configuration, named by its fingerprint
(:func:`~horde_worker_regen.process_management.ipc.supervisor_channel.capacity_workload_fingerprint`). A file
written under another fingerprint loads empty, and :meth:`CardEarningAccumulator.restart_window_if_changed`
drops the buckets when the fingerprint changes during a run. The accumulator keeps the settings it last
compared, so a restart during a run names the settings that changed. A restart found at load does not name
them, since the file holds only the fingerprint.

The module only measures. It does not hold a threshold or a verdict. The pop claim's value verdict in
:mod:`~horde_worker_regen.process_management.scheduling.governance.whole_card` reads it for disclosure, and
scheduling decisions do not read it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CapacityWorkloadSettings,
    capacity_workload_fingerprint,
    changed_capacity_workload_settings,
)
from horde_worker_regen.process_management.scheduling.performance_model import (
    _PERSIST_EVERY_N_OBSERVATIONS,
    _atomic_write_text,
)

CARD_EARNING_FILENAME = "card_earning.json"
"""Persisted beside ``perf_model.json`` in the app state directory."""

CARD_EARNING_SCHEMA_VERSION = 1
"""Bumped when the persisted schema changes incompatibly. A file of another version is discarded on load."""

CARD_EARNING_BUCKET_SECONDS = 15 * 60
"""Width of one trailing time bucket.

Old time leaves the window one bucket at a time, so the width is the step in which a rate changes as time
expires. One bucket is about 1% of the retention below, so a bucket leaving does not move a rate by much. A day
is 96 buckets, which keeps the persisted file small."""

CARD_EARNING_RETENTION_SECONDS = 24 * 60 * 60
"""How long a bucket counts toward the rates after it closes.

The models the horde sends change with the time of day, so a shorter window over-weights the hours in which a
model happened to be served. A day covers one cycle of that, and gives a model that is served rarely enough
jobs to read a rate from."""

_ZERO_SECONDS_TOLERANCE = 1e-9
"""A rate's denominator at or below this many seconds counts as zero.

A card's wall and the shares split from it are summed separately, so when one model took the whole wall the
difference of the two is a rounding residue rather than exactly zero."""

_FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_CardIndex = Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class CardTick:
    """Represents what one card was doing over one control tick, as the scheduler read it.

    Attributes:
        in_flight_models: The model of each lane on the card with a job in flight, one entry per lane. A model
            appears twice when two of the card's lanes run its jobs.
        held_model: The model of the card's whole-card grant, of either rung, or None when the card holds none.
        claimed_model: The model of the whole-card pop claim standing on the card, or None when none stands. A
            grant can outlast its claim, and a multi-card host holds grants without a claim, so this is read
            apart from ``held_model``.
    """

    in_flight_models: tuple[str, ...] = ()
    held_model: str | None = None
    claimed_model: str | None = None


@dataclass(frozen=True)
class CardEarningRate:
    """Represents the earning figures of one model on one card over the retained window.

    Attributes:
        model_kudos_per_second: The model's kudos over its attributed card seconds. None when the model has no
            attributed seconds on the card.
        rest_kudos_per_second: The card's kudos less the model's, over the card's wall seconds less the model's
            seconds. None when the model took all of the card's wall.
        jobs: How many of the model's finished jobs were credited on the card, faulted jobs included.
        model_seconds: The card seconds attributed to the model, the model rate's denominator.
        card_wall_seconds: The card's wall seconds over the same window.
        model_standard_error: The standard error of the model rate across the retained buckets. None when the
            rate is None or fewer than two buckets have attributed seconds for the model.
        rest_standard_error: The standard error of the rest rate across the retained buckets. None when the
            rate is None or fewer than two buckets have seconds left for the rest of the card.
        model_buckets: How many retained buckets have attributed seconds for the model. The model's standard
            error is estimated from this many readings.
        rest_buckets: How many retained buckets have seconds left for the rest of the card.
        claim_seconds: The card seconds during which a whole-card pop claim for the model stood on the card.
            These seconds are not part of either rate.
    """

    model_kudos_per_second: float | None
    rest_kudos_per_second: float | None
    jobs: int
    model_seconds: float
    card_wall_seconds: float
    model_standard_error: float | None = None
    rest_standard_error: float | None = None
    model_buckets: int = 0
    rest_buckets: int = 0
    claim_seconds: float = 0.0


@dataclass(frozen=True)
class CardTime:
    """Represents one card's time over the retained window.

    Attributes:
        device_index: The card, by the lane's device index.
        wall_seconds: The card's wall seconds.
        attributed_seconds: The card seconds attributed to any model. At most ``wall_seconds``.
    """

    device_index: int
    wall_seconds: float
    attributed_seconds: float

    @property
    def unattributed_seconds(self) -> float:
        """Return the card's wall seconds that did not go to any model.

        Clamped at zero, because the wall and its shares are summed separately and can differ by a rounding
        residue.
        """
        return max(0.0, self.wall_seconds - self.attributed_seconds)


@dataclass(frozen=True)
class CardEarningRestart:
    """Represents the last time the window started over because the capacity or workload settings changed.

    Attributes:
        at: The wall-clock time of the restart.
        changed_settings: The names of the settings that changed, in the order
            :class:`~horde_worker_regen.process_management.ipc.supervisor_channel.CapacityWorkloadSettings`
            declares them. None when they are unknown, which is the case for a restart found at load.
    """

    at: float
    changed_settings: tuple[str, ...] | None


@dataclass(frozen=True)
class CardEarningWindow:
    """Represents every figure of the retained window, read in one pass.

    Attributes:
        rates: Each (card, model) pair's figures, keyed by the lane's device index and the model. A pair is
            present when it has attributed seconds, counted jobs or claim seconds in the window.
        cards: Each card's time, by device index ascending.
        started_at: The start of the oldest retained bucket, or None when the window is empty.
        last_restart: The last restart during this run or at load, or None when there was none.
    """

    rates: dict[tuple[int, str], CardEarningRate]
    cards: list[CardTime]
    started_at: float | None
    last_restart: CardEarningRestart | None


@dataclass(frozen=True)
class _BucketFigures:
    """Represents one bucket's numerator and denominator of a rate."""

    kudos: float
    seconds: float


def _counted_buckets(by_bucket: list[_BucketFigures]) -> list[_BucketFigures]:
    """Return the buckets with seconds above zero, which are the ones a rate is read from."""
    return [figures for figures in by_bucket if figures.seconds > _ZERO_SECONDS_TOLERANCE]


def _ratio_standard_error(by_bucket: list[_BucketFigures], rate: float | None) -> float | None:
    """Return the standard error of ``rate`` across ``by_bucket``, or None with fewer than two counted buckets.

    A bucket counts when its seconds are above zero. Each residual is the bucket's kudos less the rate times its
    seconds, which is the bucket's deviation from the rate weighted by its seconds.
    """
    if rate is None:
        return None
    counted = _counted_buckets(by_bucket)
    count = len(counted)
    if count < 2:
        return None
    total_seconds = sum(figures.seconds for figures in counted)
    residual_sum = sum((figures.kudos - rate * figures.seconds) ** 2 for figures in counted)
    return math.sqrt(count / (count - 1) * residual_sum) / total_seconds


@dataclass
class _ModelTally:
    """Represents one (card, model) pair's figures inside one bucket."""

    seconds: float = 0.0
    kudos: float = 0.0
    jobs: int = 0
    claim_seconds: float = 0.0

    def is_empty(self) -> bool:
        """Return whether the tally's seconds, jobs and claim seconds are all zero."""
        return self.seconds <= 0.0 and self.jobs == 0 and self.claim_seconds <= 0.0


@dataclass(frozen=True)
class _BucketSummary:
    """Represents one bucket's figures with each card's kudos summed, which every pair's rate reads."""

    wall_seconds_by_card: Mapping[int, float]
    kudos_by_card: Mapping[int, float]
    tallies: Mapping[tuple[int, str], _ModelTally]


def _pair_rate(summaries: list[_BucketSummary], *, device_index: int, model: str) -> CardEarningRate:
    """Return the earning figures of ``model`` on ``device_index`` over ``summaries``."""
    card_wall = 0.0
    card_kudos = 0.0
    model_seconds = 0.0
    model_kudos = 0.0
    model_jobs = 0
    claim_seconds = 0.0
    model_by_bucket: list[_BucketFigures] = []
    rest_by_bucket: list[_BucketFigures] = []
    for summary in summaries:
        bucket_wall = summary.wall_seconds_by_card.get(device_index, 0.0)
        bucket_card_kudos = summary.kudos_by_card.get(device_index, 0.0)
        tally = summary.tallies.get((device_index, model))
        bucket_model_seconds = tally.seconds if tally is not None else 0.0
        bucket_model_kudos = tally.kudos if tally is not None else 0.0
        if tally is not None:
            model_jobs += tally.jobs
            claim_seconds += tally.claim_seconds
        card_wall += bucket_wall
        card_kudos += bucket_card_kudos
        model_seconds += bucket_model_seconds
        model_kudos += bucket_model_kudos
        model_by_bucket.append(_BucketFigures(kudos=bucket_model_kudos, seconds=bucket_model_seconds))
        rest_by_bucket.append(
            _BucketFigures(
                kudos=bucket_card_kudos - bucket_model_kudos,
                seconds=bucket_wall - bucket_model_seconds,
            ),
        )
    rest_seconds = card_wall - model_seconds
    model_rate = model_kudos / model_seconds if model_seconds > _ZERO_SECONDS_TOLERANCE else None
    rest_rate = (card_kudos - model_kudos) / rest_seconds if rest_seconds > _ZERO_SECONDS_TOLERANCE else None
    return CardEarningRate(
        model_kudos_per_second=model_rate,
        rest_kudos_per_second=rest_rate,
        jobs=model_jobs,
        model_seconds=model_seconds,
        card_wall_seconds=card_wall,
        model_standard_error=_ratio_standard_error(model_by_bucket, model_rate),
        rest_standard_error=_ratio_standard_error(rest_by_bucket, rest_rate),
        model_buckets=len(_counted_buckets(model_by_bucket)),
        rest_buckets=len(_counted_buckets(rest_by_bucket)),
        claim_seconds=claim_seconds,
    )


@dataclass
class _EarningBucket:
    """Represents one trailing time bucket."""

    wall_seconds_by_card: dict[int, float] = field(default_factory=dict)
    tallies: dict[tuple[int, str], _ModelTally] = field(default_factory=dict)

    def tally(self, device_index: int, model: str) -> _ModelTally:
        key = (device_index, model)
        existing = self.tallies.get(key)
        if existing is None:
            existing = _ModelTally()
            self.tallies[key] = existing
        return existing

    def summary(self) -> _BucketSummary:
        """Return the bucket's figures with each card's kudos summed over its tallies."""
        kudos_by_card: dict[int, float] = {}
        for (tally_card, _tally_model), tally in self.tallies.items():
            kudos_by_card[tally_card] = kudos_by_card.get(tally_card, 0.0) + tally.kudos
        return _BucketSummary(
            wall_seconds_by_card=self.wall_seconds_by_card,
            kudos_by_card=kudos_by_card,
            tallies=self.tallies,
        )


class _PersistedCardWall(BaseModel):
    """Represents one card's wall seconds in a persisted bucket."""

    model_config = ConfigDict(strict=True, frozen=True)

    card: _CardIndex
    seconds: _FiniteNonNegative


class _PersistedModelTally(BaseModel):
    """Represents one (card, model) tally in a persisted bucket."""

    model_config = ConfigDict(strict=True, frozen=True)

    card: _CardIndex
    model: Annotated[str, Field(min_length=1)]
    seconds: _FiniteNonNegative
    kudos: _FiniteNonNegative
    jobs: Annotated[int, Field(ge=0)]
    claim_seconds: _FiniteNonNegative = 0.0
    """Defaults to zero so a file written before the claim tally existed loads under the same schema version."""


class _PersistedBucketStart(BaseModel):
    """Represents the start time of a persisted bucket."""

    model_config = ConfigDict(strict=True, frozen=True)

    start: _FiniteNonNegative


class CardEarningAccumulator:
    """Accumulates attributed card seconds and kudos per card and per model, in trailing buckets.

    Each tick's elapsed time is split on each card by three rules. The lanes on the card with a job in
    flight share it equally, each share to its job's model. With no job in flight and a whole-card grant on the
    card, it goes to the held model whole, so establishing, releasing and holding the card idle are charged to the
    model that caused them. Otherwise it counts only in the card's wall. The card's wall grows by the whole
    elapsed time in every case, so a card's attributed seconds never exceed its wall.

    The time a whole-card pop claim stands on a card is tallied apart, to the claimed model. It does not change
    the split above.

    Thread Safety:
        Fed by the control loop only. Not safe for concurrent mutation.
    """

    def __init__(
        self,
        *,
        max_tick_seconds: float,
        settings: CapacityWorkloadSettings | None = None,
        path: Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Create an accumulator, loading a persisted file when ``path`` is given.

        Args:
            max_tick_seconds: The most one tick may add. A larger gap means the control loop stalled, and the
                stalled time is not evidence of what the card was doing.
            settings: The capacity and workload settings the figures are measured under. A persisted file
                written under another fingerprint loads empty. None stands for an empty fingerprint.
            path: Where the buckets persist. None keeps them in memory only.
            clock: Wall-clock source, read to age the persisted buckets at load, to place a kudos credit that
                arrives before the first tick and to time a restart. Defaults to ``time.time``.
        """
        self._max_tick_seconds = max_tick_seconds
        self._settings = settings
        self._config_fingerprint = capacity_workload_fingerprint(settings) if settings is not None else ""
        self._path = path
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._buckets: dict[float, _EarningBucket] = {}
        self._last_observation_time: float | None = None
        self._session_seconds_by_model: dict[str, float] = {}
        self._jobs_since_save = 0
        self._file_disabled = False
        self._last_restart: CardEarningRestart | None = None
        self._load()

    # region feeding

    def observe(self, now: float, cards: Mapping[int, CardTick]) -> None:
        """Attribute the time since the previous observation on every card in ``cards``.

        The interval is priced at this (closing) observation's facts, as the slot-duty accumulator prices it.
        The first observation only anchors the clock, and a reading that does not advance adds nothing.

        Args:
            now: The scheduler's wall clock.
            cards: Each card's facts for this tick, keyed by the lane's device index (0 on a single-card host).
        """
        previous = self._last_observation_time
        self._last_observation_time = now
        if previous is None or now <= previous:
            return
        elapsed = min(now - previous, self._max_tick_seconds)
        self._prune(now)
        bucket = self._bucket_at(now)
        for device_index, card_tick in cards.items():
            bucket.wall_seconds_by_card[device_index] = bucket.wall_seconds_by_card.get(device_index, 0.0) + elapsed
            if card_tick.in_flight_models:
                share = elapsed / len(card_tick.in_flight_models)
                for model in card_tick.in_flight_models:
                    self._credit_seconds(bucket, device_index, model, share)
            elif card_tick.held_model is not None:
                self._credit_seconds(bucket, device_index, card_tick.held_model, elapsed)
            if card_tick.claimed_model is not None:
                bucket.tally(device_index, card_tick.claimed_model).claim_seconds += elapsed

    def credit_finished_job(self, *, device_index: int, model: str, kudos: float | None) -> None:
        """Count one finished job for ``model`` on ``device_index`` and add its kudos.

        The credit lands in the bucket of the latest tick, so seconds and kudos are aged on one clock. A faulted
        job is passed with ``kudos=None``. It counts as a job and does not add kudos, and the seconds it took
        stay attributed to its model.

        Args:
            device_index: The card that sampled the job.
            model: The job's model.
            kudos: What the horde paid for the job, or None when it paid nothing or the figure is unknown.
        """
        when = self._last_observation_time if self._last_observation_time is not None else self._clock()
        bucket = self._bucket_at(when)
        tally = bucket.tally(device_index, model)
        tally.jobs += 1
        if kudos is not None and math.isfinite(kudos) and kudos > 0:
            tally.kudos += kudos
        self._jobs_since_save += 1
        if self._jobs_since_save >= _PERSIST_EVERY_N_OBSERVATIONS:
            self.save()

    def restart_window_if_changed(self, settings: CapacityWorkloadSettings) -> None:
        """Mutate the window to start over when the fingerprint of ``settings`` differs from the current one.

        The retained buckets are dropped and the new fingerprint is adopted. The session totals behind
        :meth:`session_seconds_by_model` are kept, since the rollups read them as session figures. The clock
        anchor is kept too, so the next tick accrues into the new window as usual. The restart is recorded with
        the settings that changed since the previous call. It is recorded even when the window was empty,
        because every figure from then on is measured under the new settings.

        Args:
            settings: The current capacity and workload settings.
        """
        previous_settings = self._settings
        self._settings = settings
        config_fingerprint = capacity_workload_fingerprint(settings)
        if config_fingerprint == self._config_fingerprint:
            return
        self._config_fingerprint = config_fingerprint
        self._last_restart = CardEarningRestart(
            at=self._clock(),
            changed_settings=(
                changed_capacity_workload_settings(previous_settings, settings)
                if previous_settings is not None
                else None
            ),
        )
        if not self._buckets:
            return
        self._buckets.clear()
        _log_window_restart()
        # Written at once so the file does not keep figures of the old settings until the next save.
        self.save()

    def _credit_seconds(self, bucket: _EarningBucket, device_index: int, model: str, seconds: float) -> None:
        bucket.tally(device_index, model).seconds += seconds
        self._session_seconds_by_model[model] = self._session_seconds_by_model.get(model, 0.0) + seconds

    def _bucket_at(self, when: float) -> _EarningBucket:
        start = math.floor(when / CARD_EARNING_BUCKET_SECONDS) * CARD_EARNING_BUCKET_SECONDS
        bucket = self._buckets.get(start)
        if bucket is None:
            bucket = _EarningBucket()
            self._buckets[start] = bucket
        return bucket

    def _prune(self, now: float) -> None:
        for start in [start for start in self._buckets if self._is_expired(start, now)]:
            del self._buckets[start]

    @staticmethod
    def _is_expired(start: float, now: float) -> bool:
        return start + CARD_EARNING_BUCKET_SECONDS <= now - CARD_EARNING_RETENTION_SECONDS

    # endregion

    # region reading

    def rate(self, *, model: str, device_index: int) -> CardEarningRate:
        """Return the earning figures of ``model`` on ``device_index`` over the retained buckets.

        A rate's standard error treats the rate as a ratio over the buckets. It is the square root of n/(n-1)
        times the sum over buckets of (bucket kudos - rate x bucket seconds) squared, divided by the total
        seconds, where n counts the buckets with seconds in the rate's denominator.

        Args:
            model: The model to read.
            device_index: The card to read, by the lane's device index.
        """
        summaries = [bucket.summary() for bucket in self._buckets.values()]
        return _pair_rate(summaries, device_index=device_index, model=model)

    @property
    def last_restart(self) -> CardEarningRestart | None:
        """The last window restart during this run or at load, or None when there was none.

        A reader that keeps a copy of :meth:`read_window` compares this object by identity to find out that the
        window restarted since the copy was taken, without paying for the read.
        """
        return self._last_restart

    def read_window(self) -> CardEarningWindow:
        """Return every pair's figures, each card's time and the window's status, from one pass over the buckets.

        Each pair's figures equal what :meth:`rate` returns for it. The buckets are summarised once and every
        pair reads the summaries, so the read does not walk each bucket's tallies once per pair.
        """
        summaries: list[_BucketSummary] = []
        pairs: set[tuple[int, str]] = set()
        wall_by_card: dict[int, float] = {}
        attributed_by_card: dict[int, float] = {}
        for bucket in self._buckets.values():
            summaries.append(bucket.summary())
            for device_index, wall_seconds in bucket.wall_seconds_by_card.items():
                wall_by_card[device_index] = wall_by_card.get(device_index, 0.0) + wall_seconds
            for pair, tally in bucket.tallies.items():
                if tally.is_empty():
                    continue
                pairs.add(pair)
                attributed_by_card[pair[0]] = attributed_by_card.get(pair[0], 0.0) + tally.seconds
        rates = {
            (device_index, model): _pair_rate(summaries, device_index=device_index, model=model)
            for device_index, model in sorted(pairs)
        }
        cards = [
            CardTime(
                device_index=device_index,
                wall_seconds=wall_by_card.get(device_index, 0.0),
                attributed_seconds=attributed_by_card.get(device_index, 0.0),
            )
            for device_index in sorted(wall_by_card.keys() | attributed_by_card.keys())
        ]
        return CardEarningWindow(
            rates=rates,
            cards=cards,
            started_at=min(self._buckets) if self._buckets else None,
            last_restart=self._last_restart,
        )

    def session_seconds_by_model(self) -> dict[str, float]:
        """Return the card seconds attributed to each model since construction, summed over cards.

        These totals only grow and are not aged or persisted, so a window's figure is the difference of two
        readings. The run metrics read them for the per-model rollup rows.
        """
        return dict(self._session_seconds_by_model)

    # endregion

    # region persistence

    def save(self) -> None:
        """Persist the retained buckets atomically. Does nothing without a path, and does not raise."""
        self._jobs_since_save = 0
        if self._path is None or self._file_disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(self._path, json.dumps(self._payload()))
        except OSError as write_error:
            # A measurement write must not take the worker down, so it degrades to memory and stops retrying.
            logger.debug(f"Could not persist card earning to {self._path} ({write_error}); continuing in memory.")
            self._file_disabled = True

    def _payload(self) -> dict[str, object]:
        buckets: list[dict[str, object]] = []
        for start, bucket in sorted(self._buckets.items()):
            walls = [
                _PersistedCardWall(card=card, seconds=seconds).model_dump()
                for card, seconds in sorted(bucket.wall_seconds_by_card.items())
            ]
            tallies = [
                _PersistedModelTally(
                    card=card,
                    model=model,
                    seconds=tally.seconds,
                    kudos=tally.kudos,
                    jobs=tally.jobs,
                    claim_seconds=tally.claim_seconds,
                ).model_dump()
                for (card, model), tally in sorted(bucket.tallies.items())
            ]
            buckets.append({"start": start, "walls": walls, "tallies": tallies})
        return {
            "schema_version": CARD_EARNING_SCHEMA_VERSION,
            "bucket_seconds": CARD_EARNING_BUCKET_SECONDS,
            "config_fingerprint": self._config_fingerprint,
            "buckets": buckets,
        }

    def _load(self) -> None:
        """Load the persisted buckets, dropping each entry that is malformed, non-finite, negative or expired.

        A file of another schema version or bucket width is discarded whole, since its buckets cannot be merged
        into this width. A file written under another configuration fingerprint is discarded whole too, and a
        file without a fingerprint counts as another one. Discarding buckets for another fingerprint is recorded
        as a restart whose changed settings are unknown.
        """
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as read_error:
            logger.debug(f"Could not read card earning at {self._path} ({read_error}); starting empty.")
            return
        if not isinstance(raw, dict):
            return
        if raw.get("schema_version") != CARD_EARNING_SCHEMA_VERSION:
            return
        if raw.get("bucket_seconds") != CARD_EARNING_BUCKET_SECONDS:
            return
        persisted_buckets = raw.get("buckets")
        if not isinstance(persisted_buckets, list):
            return
        if raw.get("config_fingerprint") != self._config_fingerprint:
            if persisted_buckets:
                self._last_restart = CardEarningRestart(at=self._clock(), changed_settings=None)
                _log_window_restart()
            return
        now = self._clock()
        dropped = 0
        for persisted_bucket in persisted_buckets:
            dropped += self._load_bucket(persisted_bucket, now)
        if dropped:
            logger.debug(f"Dropped {dropped} invalid or expired card earning entries from {self._path}.")

    def _load_bucket(self, persisted_bucket: object, now: float) -> int:
        """Fill one bucket from its persisted form and return how many entries were dropped."""
        if not isinstance(persisted_bucket, dict):
            return 1
        try:
            start = _PersistedBucketStart.model_validate({"start": persisted_bucket.get("start")}).start
        except ValidationError:
            return 1
        if self._is_expired(start, now):
            return 1
        walls = persisted_bucket.get("walls")
        tallies = persisted_bucket.get("tallies")
        if not isinstance(walls, list) or not isinstance(tallies, list):
            return 1
        bucket = self._bucket_at(start)
        dropped = 0
        for persisted_wall in walls:
            try:
                wall = _PersistedCardWall.model_validate(persisted_wall)
            except ValidationError:
                dropped += 1
                continue
            bucket.wall_seconds_by_card[wall.card] = bucket.wall_seconds_by_card.get(wall.card, 0.0) + wall.seconds
        for persisted_tally in tallies:
            try:
                loaded = _PersistedModelTally.model_validate(persisted_tally)
            except ValidationError:
                dropped += 1
                continue
            tally = bucket.tally(loaded.card, loaded.model)
            tally.seconds += loaded.seconds
            tally.kudos += loaded.kudos
            tally.jobs += loaded.jobs
            tally.claim_seconds += loaded.claim_seconds
        return dropped

    # endregion


def _log_window_restart() -> None:
    logger.info("Card earning figures restart because capacity or workload settings changed.")
