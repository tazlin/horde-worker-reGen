"""Tests for the pure concurrent-overlap rule."""

from __future__ import annotations

from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier
from horde_worker_regen.process_management.scheduling.concurrent_overlap import (
    OVERLAP_HEADWAY_AMPLE_VRAM,
    OVERLAP_HEADWAY_BOTH_HEAVY,
    OVERLAP_HEADWAY_MIXED_HEAVY,
    RunningSampler,
    concurrent_overlap_permitted,
)


class _Verdict:
    """A memory verdict that records how often it was consulted."""

    def __init__(self, value: bool | None) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> bool | None:
        self.calls += 1
        return self.value


def _permitted(
    *,
    candidate_tier: ModelSizeTier = ModelSizeTier.HEAVY,
    candidate_batched: bool = False,
    running: tuple[RunningSampler, ...] = (),
    headway_scale: float = 1.0,
    verdict: _Verdict | None = None,
) -> bool:
    return concurrent_overlap_permitted(
        candidate_tier=candidate_tier,
        candidate_batched=candidate_batched,
        running=running,
        headway_scale=headway_scale,
        memory_verdict=verdict if verdict is not None else _Verdict(None),
    )


def _running(tier: ModelSizeTier, progress: float, *, batched: bool = False) -> RunningSampler:
    return RunningSampler(tier=tier, batched=batched, progress_fraction=progress)


class TestHardBlocks:
    """Rules that decide before the memory question is asked."""

    def test_nothing_running_admits_without_consulting_memory(self) -> None:
        """With nothing in flight there is no memory question to ask."""
        verdict = _Verdict(False)
        assert _permitted(verdict=verdict) is True
        assert verdict.calls == 0

    def test_extra_large_candidate_never_joins(self) -> None:
        """An extra-large candidate neither joins a busy card nor asks about memory."""
        verdict = _Verdict(True)
        running = (_running(ModelSizeTier.LIGHT, 1.0),)
        assert _permitted(candidate_tier=ModelSizeTier.EXTRA_LARGE, running=running, verdict=verdict) is False
        assert verdict.calls == 0

    def test_extra_large_running_job_is_never_joined(self) -> None:
        """A running extra-large job is never shared, whatever the memory answer."""
        running = (_running(ModelSizeTier.EXTRA_LARGE, 1.0),)
        assert _permitted(candidate_tier=ModelSizeTier.LIGHT, running=running, verdict=_Verdict(True)) is False


class TestHeadway:
    """How far the running job must be before the candidate may join."""

    def test_two_light_jobs_thread_freely_then_memory_decides(self) -> None:
        """Two light jobs need no headway, so only the memory veto can hold the newcomer."""
        running = (_running(ModelSizeTier.LIGHT, 0.0),)
        assert _permitted(candidate_tier=ModelSizeTier.LIGHT, running=running, verdict=_Verdict(None)) is True
        assert _permitted(candidate_tier=ModelSizeTier.LIGHT, running=running, verdict=_Verdict(False)) is False

    def test_mixed_pairing_waits_for_strict_headway_when_memory_is_unpriced(self) -> None:
        """One heavy side keeps the strict mixed headway when the demand could not be priced."""
        below = (_running(ModelSizeTier.LIGHT, OVERLAP_HEADWAY_MIXED_HEAVY - 0.01),)
        at = (_running(ModelSizeTier.LIGHT, OVERLAP_HEADWAY_MIXED_HEAVY),)
        assert _permitted(running=below, verdict=_Verdict(None)) is False
        assert _permitted(running=at, verdict=_Verdict(None)) is True

    def test_both_heavy_needs_the_most_headway(self) -> None:
        """Two heavy jobs keep the strictest headway when the demand could not be priced."""
        below = (_running(ModelSizeTier.HEAVY, OVERLAP_HEADWAY_BOTH_HEAVY - 0.01),)
        at = (_running(ModelSizeTier.HEAVY, OVERLAP_HEADWAY_BOTH_HEAVY),)
        assert _permitted(running=below, verdict=_Verdict(None)) is False
        assert _permitted(running=at, verdict=_Verdict(None)) is True

    def test_confirmed_room_relaxes_headway_to_the_startup_beat(self) -> None:
        """Confirmed room drops a heavy pairing's headway to the startup beat."""
        running = (_running(ModelSizeTier.HEAVY, OVERLAP_HEADWAY_AMPLE_VRAM),)
        assert _permitted(running=running, verdict=_Verdict(True)) is True
        assert _permitted(running=running, verdict=_Verdict(None)) is False

    def test_denied_memory_vetoes_even_with_full_headway(self) -> None:
        """A memory denial withholds the overlap regardless of progress."""
        running = (_running(ModelSizeTier.HEAVY, 1.0),)
        assert _permitted(running=running, verdict=_Verdict(False)) is False

    def test_headway_scale_pulls_the_newcomer_in_sooner(self) -> None:
        """A performance-mode scale shrinks the headway the newcomer waits for."""
        running = (_running(ModelSizeTier.HEAVY, OVERLAP_HEADWAY_BOTH_HEAVY / 2),)
        assert _permitted(running=running, headway_scale=1.0, verdict=_Verdict(None)) is False
        assert _permitted(running=running, headway_scale=0.5, verdict=_Verdict(None)) is True

    def test_any_one_running_job_short_of_headway_blocks(self) -> None:
        """Every running job must have made its headway, not just one of them."""
        running = (_running(ModelSizeTier.HEAVY, 1.0), _running(ModelSizeTier.HEAVY, 0.1))
        assert _permitted(running=running, verdict=_Verdict(None)) is False


class TestBatches:
    """A batched side needs confirmed room and is then bounded by the strictest headway."""

    def test_batched_candidate_blocks_without_confirmed_room(self) -> None:
        """A batched candidate needs confirmed room before any headway is considered."""
        running = (_running(ModelSizeTier.LIGHT, 1.0),)
        assert _permitted(candidate_batched=True, running=running, verdict=_Verdict(None)) is False
        assert _permitted(candidate_batched=True, running=running, verdict=_Verdict(True)) is True

    def test_batched_running_job_uses_the_strictest_headway_not_the_relaxed_one(self) -> None:
        """A batched running job is bounded by the strictest headway even with confirmed room."""
        running = (_running(ModelSizeTier.LIGHT, OVERLAP_HEADWAY_BOTH_HEAVY - 0.01, batched=True),)
        assert _permitted(candidate_tier=ModelSizeTier.LIGHT, running=running, verdict=_Verdict(True)) is False
        running = (_running(ModelSizeTier.LIGHT, OVERLAP_HEADWAY_BOTH_HEAVY, batched=True),)
        assert _permitted(candidate_tier=ModelSizeTier.LIGHT, running=running, verdict=_Verdict(True)) is True


def test_memory_verdict_is_consulted_at_most_once() -> None:
    """The arbiter is asked once per decision however many rules need its answer."""
    verdict = _Verdict(True)
    running = (_running(ModelSizeTier.HEAVY, 1.0), _running(ModelSizeTier.HEAVY, 1.0))
    assert _permitted(candidate_batched=True, running=running, verdict=verdict) is True
    assert verdict.calls == 1
