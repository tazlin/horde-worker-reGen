"""Unit tests for the verified LIFO reclaim ladder: rung ordering, verification, escalation, exhaustion."""

from __future__ import annotations

import functools

from horde_worker_regen.process_management.resources.reclaim_ladder import (
    _SAFETY_RUNG_COOLDOWN_SECONDS,
    CacheReleaseTarget,
    IdleResidentModel,
    LadderCandidates,
    LaneReclaimCandidate,
    ReclaimRung,
    ReclaimRungKind,
    VerifiedReclaimLadder,
    build_reclaim_ladder,
)


class _FakeActuator:
    """Records the order of rung executions and calibration events; can fail specific unload/cache targets."""

    def __init__(self, *, fail_targets: frozenset[int] = frozenset()) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.calibration_events: list[tuple[ReclaimRungKind, float, float]] = []
        self._fail_targets = set(fail_targets)

    def unload_idle_model(self, process_id: int, device_index: int | None) -> bool:
        self.calls.append(("unload", process_id))
        return process_id not in self._fail_targets

    def release_idle_cache(self, process_id: int) -> bool:
        self.calls.append(("cache", process_id))
        return process_id not in self._fail_targets

    def pause_post_process_lane(self, device_index: int | None) -> bool:
        self.calls.append(("pp", None))
        return True

    def pause_vae_lane(self, device_index: int | None) -> bool:
        self.calls.append(("vae", None))
        return True

    def pause_component_lane(self, device_index: int | None) -> bool:
        self.calls.append(("component", None))
        return True

    def safety_off_gpu(self, device_index: int | None) -> bool:
        self.calls.append(("safety", None))
        return True

    def restore_post_process_lane(self, device_index: int | None) -> bool:
        self.calls.append(("restore_pp", None))
        return True

    def restore_vae_lane(self, device_index: int | None) -> bool:
        self.calls.append(("restore_vae", None))
        return True

    def restore_component_lane(self, device_index: int | None) -> bool:
        self.calls.append(("restore_component", None))
        return True

    def restore_live_contexts(self, device_index: int | None) -> bool:
        self.calls.append(("restore_contexts", device_index))
        return True

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        self.calibration_events.append((rung.kind, promised_mb, realized_mb))


def _resident(process_id: int, materialized: float, footprint: float = 1000.0) -> IdleResidentModel:
    return IdleResidentModel(
        process_id=process_id,
        tenant_label=f"model#{process_id}",
        materialized_monotonic=materialized,
        footprint_mb=footprint,
    )


class TestBuildReclaimLadder:
    """The pure builder orders candidates into the fixed sequence with LIFO ranking among like rungs."""

    def test_empty_candidates_yield_no_rungs(self) -> None:
        """A card with nothing to reclaim produces an empty (structurally exhausted) ladder."""
        assert build_reclaim_ladder(LadderCandidates(device_index=0)) == ()

    def test_newest_model_first_then_older_residents_lifo(self) -> None:
        """The newest idle model is the first rung; older residents follow newest-first (LIFO)."""
        candidates = LadderCandidates(
            device_index=0,
            idle_residents=(
                _resident(1, materialized=1.0),
                _resident(3, materialized=3.0),
                _resident(2, materialized=2.0),
            ),
        )
        ladder = build_reclaim_ladder(candidates)
        assert [(r.kind, r.target_process_id) for r in ladder] == [
            (ReclaimRungKind.UNLOAD_IDLE_MODEL, 3),
            (ReclaimRungKind.UNLOAD_IDLE_MODEL, 2),
            (ReclaimRungKind.UNLOAD_IDLE_MODEL, 1),
        ]

    def test_full_sequence_order(self) -> None:
        """Order is newest model, then caches (LIFO), then older models (LIFO), then lanes, then safety."""
        candidates = LadderCandidates(
            device_index=0,
            idle_residents=(_resident(1, materialized=1.0), _resident(2, materialized=5.0)),
            cache_targets=(
                CacheReleaseTarget(
                    process_id=7, tenant_label="lane#7", materialized_monotonic=2.0, reclaimable_mb=300.0
                ),
                CacheReleaseTarget(
                    process_id=8, tenant_label="lane#8", materialized_monotonic=4.0, reclaimable_mb=400.0
                ),
            ),
            lanes=(
                LaneReclaimCandidate(kind=ReclaimRungKind.PAUSE_PP_LANE, tenant_label="pp", promised_mb=500.0),
                LaneReclaimCandidate(kind=ReclaimRungKind.PAUSE_VAE_LANE, tenant_label="vae", promised_mb=600.0),
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_COMPONENT_LANE, tenant_label="component", promised_mb=700.0
                ),
            ),
            safety=LaneReclaimCandidate(
                kind=ReclaimRungKind.SAFETY_OFF_GPU, tenant_label="safety", promised_mb=3000.0
            ),
        )
        ladder = build_reclaim_ladder(candidates)
        assert [(r.kind, r.target_process_id) for r in ladder] == [
            (ReclaimRungKind.UNLOAD_IDLE_MODEL, 2),  # newest model
            (ReclaimRungKind.RELEASE_IDLE_CACHE, 8),  # caches newest-first
            (ReclaimRungKind.RELEASE_IDLE_CACHE, 7),
            (ReclaimRungKind.UNLOAD_IDLE_MODEL, 1),  # older resident
            (ReclaimRungKind.PAUSE_PP_LANE, None),
            (ReclaimRungKind.PAUSE_VAE_LANE, None),
            (ReclaimRungKind.PAUSE_COMPONENT_LANE, None),
            (ReclaimRungKind.SAFETY_OFF_GPU, None),
        ]


def _ladder(*rungs: ReclaimRung) -> tuple[ReclaimRung, ...]:
    return rungs


def _unload_rung(process_id: int, promised: float) -> ReclaimRung:
    return ReclaimRung(
        kind=ReclaimRungKind.UNLOAD_IDLE_MODEL,
        device_index=0,
        promised_freed_mb=promised,
        tenant_label=f"model#{process_id}",
        target_process_id=process_id,
    )


def _pause_rung(kind: ReclaimRungKind, promised: float = 500.0) -> ReclaimRung:
    return ReclaimRung(kind=kind, device_index=0, promised_freed_mb=promised, tenant_label=kind.value)


def _budget_for(rung: ReclaimRung) -> float:
    """The progress-free seconds the engine gives ``rung``, so a test states its clock in the engine's terms.

    Read from the engine rather than restated, because the budget is derived from the rung's promise: a test
    that hard-coded seconds would be asserting one card's arithmetic instead of the rule.
    """
    return VerifiedReclaimLadder._verification_budget_for(rung)


class TestVerifiedReclaimLadderEngine:
    """The engine issues one rung per tick, verifies realized frees, escalates on shortfall, flags exhaustion."""

    def test_one_rung_per_tick_and_verification_success_advances(self) -> None:
        """A rung that yields at least half its promise verifies and the next rung issues the same tick."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 1000.0), _unload_rung(2, 1000.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.rungs_issued == 1
        assert actuator.calls == [("unload", 1)]

        # Free rose by 600 (>= 50% of 1000): rung 1 verifies, then rung 2 issues this same tick.
        engine.on_tick(0, saturated=True, device_free_mb=700.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.rungs_issued == 2
        assert engine.verified_frees_mb == 600.0
        assert engine.verification_shortfalls == 0
        assert actuator.calls == [("unload", 1), ("unload", 2)]

    def test_a_rung_that_realizes_nothing_for_its_budget_records_calibration_and_escalates(self) -> None:
        """A rung that never yields half its promise escalates once its budget runs out, logging calibration."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 2000.0), _unload_rung(2, 500.0))
        budget = _budget_for(ladder[0])

        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        # A sample inside the budget: realized 30 << 1000 (half of 2000), and too small to count as progress,
        # but the rung has not had its time yet, so the engine waits rather than escalating.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=130.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget / 2.0,
        )
        assert engine.rungs_issued == 1
        assert actuator.calls == [("unload", 1)]

        # Past the budget with nothing further realized: shortfall recorded, calibration event, rung 2 escalated.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=150.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget + 1.0,
        )
        assert engine.verification_shortfalls == 1
        assert actuator.calibration_events == [(ReclaimRungKind.UNLOAD_IDLE_MODEL, 2000.0, 50.0)]
        assert engine.rungs_issued == 2
        assert actuator.calls == [("unload", 1), ("unload", 2)]

    def test_a_rung_still_realizing_free_keeps_its_budget(self) -> None:
        """A release that keeps arriving is never graded short, however long it takes to reach its promise.

        The budget is a ceiling on going nowhere, not a deadline: an actuation the driver is still servicing
        shows a rising device-free figure, and each new high restarts the clock. Grading it on elapsed time
        alone would escalate past exactly the rungs that are working, which is what a large release looks like
        on a control loop that samples faster than the hardware frees.
        """
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 4000.0), _unload_rung(2, 500.0))
        budget = _budget_for(ladder[0])

        engine.on_tick(
            0, saturated=True, device_free_mb=0.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        # Free climbs a few hundred megabytes at a time, well past the budget in elapsed seconds and still short
        # of half the promise. Every sample is a new high, so the rung keeps its clock.
        for step in range(1, 9):
            engine.on_tick(
                0,
                saturated=True,
                device_free_mb=200.0 * step,
                actuator=actuator,
                ladder_builder=lambda: ladder,
                now=budget * step,
            )
        assert actuator.calls == [("unload", 1)]
        assert engine.verification_shortfalls == 0

        # The release completes: the rung verifies on its own merits rather than on the clock.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=2500.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget * 9,
        )
        assert actuator.calls == [("unload", 1), ("unload", 2)]
        assert engine.verification_shortfalls == 0

    def test_the_verification_budget_scales_with_the_promised_release(self) -> None:
        """A rung promising more gigabytes is given proportionally longer, so no card size is privileged.

        A WDDM release settles at a rate set by how much is being returned, so a fixed budget is only ever right
        for the card it was measured on: generous for a small model's release and short enough for a large one
        that a working give-back is graded a failure. Both rungs here land at a pace proportional to their size
        and both must verify.
        """
        small = _unload_rung(1, 2048.0)
        large = _unload_rung(2, 12288.0)
        assert _budget_for(large) > _budget_for(small), (
            "a six-times-larger release is given no more time than a small one, so the budget is a constant "
            "sized for whichever card it was measured on"
        )

        for rung in (small, large):
            engine = VerifiedReclaimLadder()
            actuator = _FakeActuator()
            ladder = _ladder(rung, _unload_rung(9, 500.0))
            # The release arrives in one block, at a delay proportional to its size and shorter than the budget
            # the promise earns it.
            landing_at = _budget_for(rung) * 0.75
            builder = functools.partial(_ladder, *ladder)
            engine.on_tick(0, saturated=True, device_free_mb=0.0, actuator=actuator, ladder_builder=builder, now=0.0)
            engine.on_tick(
                0,
                saturated=True,
                device_free_mb=rung.promised_freed_mb,
                actuator=actuator,
                ladder_builder=builder,
                now=landing_at,
            )
            assert engine.verification_shortfalls == 0, (
                f"a release of {rung.promised_freed_mb:.0f}MB landing in {landing_at:.0f}s was graded short"
            )
            assert engine.verified_frees_mb == rung.promised_freed_mb

    def test_exhausted_ladder_while_saturated_marks_unresolved(self) -> None:
        """Once every rung has run and the card is still SATURATED, the episode is flagged unresolved."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 1000.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.is_saturation_unresolved(0) is False
        # Verify success on the only rung, then _issue_next finds the ladder exhausted -> unresolved.
        engine.on_tick(0, saturated=True, device_free_mb=2000.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.is_saturation_unresolved(0) is True

    def test_recovery_clears_the_episode(self) -> None:
        """A card leaving SATURATED clears its episode and its unresolved flag."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 1000.0))
        budget = _budget_for(ladder[0])
        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=120.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget / 2.0,
        )
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=130.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget + 1.0,
        )
        assert engine.is_saturation_unresolved(0) is True

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        assert engine.is_saturation_unresolved(0) is False

    def test_no_op_rung_is_skipped_and_the_next_issues_same_tick(self) -> None:
        """A rung whose target has gone away frees nothing to verify, so the engine advances immediately."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator(fail_targets=frozenset({1}))
        ladder = _ladder(_unload_rung(1, 1000.0), _unload_rung(2, 1000.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        # Rung 1 was attempted (returned False) and skipped; rung 2 issued the same tick and counts.
        assert actuator.calls == [("unload", 1), ("unload", 2)]
        assert engine.rungs_issued == 1

    def test_ladder_is_frozen_at_episode_start(self) -> None:
        """The ladder builder is called once per episode; later topology changes do not re-order a live episode."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        builds = 0

        def builder() -> tuple[ReclaimRung, ...]:
            nonlocal builds
            builds += 1
            return _ladder(_unload_rung(1, 1000.0), _unload_rung(2, 1000.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=builder)
        engine.on_tick(0, saturated=True, device_free_mb=2000.0, actuator=actuator, ladder_builder=builder)
        assert builds == 1

    def test_a_rung_with_no_promised_free_is_never_certified_on_its_first_sample(self) -> None:
        """A rung priced at zero (an unreported reservation) is unverifiable, so it must not self-certify.

        The yield-fraction test against a zero promise reduces to "realized at least nothing", which the first
        sample always satisfies. Certifying there would credit a rung that freed nothing and sprint the engine
        down the rest of the ladder, up to moving safety off the card, on evidence it never had.
        """
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 0.0), _unload_rung(2, 1000.0))
        budget = _budget_for(ladder[0])

        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        assert actuator.calls == [("unload", 1)]

        # A sample inside the budget: the unverifiable rung is held, not certified, and nothing is credited.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=100.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget / 2.0,
        )
        assert actuator.calls == [("unload", 1)]
        assert engine.verified_frees_mb == 0.0

        # Its full budget served, it resolves so the engine escalates, still crediting nothing and counting no
        # shortfall against a promise that was never a measurement.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=100.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=budget + 1.0,
        )
        assert actuator.calls == [("unload", 1), ("unload", 2)]
        assert engine.verified_frees_mb == 0.0
        assert engine.verification_shortfalls == 0
        assert actuator.calibration_events == []


class TestReclaimLadderVerifiedRestore:
    """Lane-pause rungs the engine issues are restored (LIFO) when the card returns HEALTHY, safety excepted."""

    def test_teardown_rung_gets_a_longer_verification_budget(self) -> None:
        """A lane pause is given longer than an in-process rung of the same promise before it counts as short.

        Its memory returns only once the lane process has exited, so it pays a whole process teardown before
        its first megabyte arrives; grading it on what an in-process release needs escalates past a pause that
        is still tearing down.
        """
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        pause = _pause_rung(ReclaimRungKind.PAUSE_PP_LANE, 5000.0)
        ladder = _ladder(pause, _unload_rung(2, 500.0))
        teardown_budget = _budget_for(pause)
        assert teardown_budget > _budget_for(_unload_rung(1, pause.promised_freed_mb)), (
            "a teardown rung is given no more time than an in-process rung returning the same memory, so the "
            "process exit it pays for is unaccounted"
        )

        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        assert actuator.calls == [("pp", None)]
        # Samples inside the teardown budget are short and too small to count as progress, but the pause holds:
        # an in-process rung of the same promise would already have been escalated past by the second of them.
        for elapsed, free_mb in ((teardown_budget / 3.0, 120.0), (teardown_budget * 2.0 / 3.0, 130.0)):
            engine.on_tick(
                0,
                saturated=True,
                device_free_mb=free_mb,
                actuator=actuator,
                ladder_builder=lambda: ladder,
                now=elapsed,
            )
        assert engine.rungs_issued == 1
        assert engine.verification_shortfalls == 0
        # Past the budget and still short: only now does it escalate, issuing the next rung.
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=140.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=teardown_budget + 1.0,
        )
        assert engine.verification_shortfalls == 1
        assert engine.rungs_issued == 2
        assert actuator.calls == [("pp", None), ("unload", 2)]

    def test_paused_lanes_restored_lifo_only_when_healthy(self) -> None:
        """Paused lanes are held through PRESSURE and restored newest-first once the card is HEALTHY."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(
            _pause_rung(ReclaimRungKind.PAUSE_PP_LANE),
            _pause_rung(ReclaimRungKind.PAUSE_VAE_LANE),
        )

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        # Free rose enough to verify the PP pause; the VAE pause issues the same tick.
        engine.on_tick(0, saturated=True, device_free_mb=400.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert actuator.calls == [("pp", None), ("vae", None)]

        # Saturation lifted but the card is only in the PRESSURE band (not HEALTHY): lanes stay paused.
        engine.on_tick(
            0, saturated=False, healthy=False, device_free_mb=500.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        assert actuator.calls == [("pp", None), ("vae", None)]

        # Fully HEALTHY: the engine unwinds its pauses in reverse order (VAE, the newest, first).
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        assert actuator.calls == [
            ("pp", None),
            ("vae", None),
            ("restore_vae", None),
            ("restore_pp", None),
        ]

    def test_context_reduction_is_regrown_only_when_healthy(self) -> None:
        """A booked live-context reduction is held through PRESSURE and regrown once the card is HEALTHY."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()

        engine.record_context_reduction(0)
        # A head that re-asks every cycle books the obligation once, so the pool is grown back once.
        engine.record_context_reduction(0)
        assert engine.has_context_reduction(0) is True
        assert actuator.calls == []

        engine.on_tick(
            0, saturated=False, healthy=False, device_free_mb=500.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == []

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == [("restore_contexts", 0)]
        assert engine.has_context_reduction(0) is False

    def test_context_restore_waits_for_the_callers_dwell(self) -> None:
        """A healthy sample the caller has not cleared holds the regrowth, and a later cleared one takes it."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()

        engine.record_context_reduction(0)
        engine.on_tick(
            0,
            saturated=False,
            healthy=True,
            device_free_mb=9000.0,
            actuator=actuator,
            ladder_builder=tuple,
            context_restore_ready=False,
        )
        assert actuator.calls == []
        assert engine.has_context_reduction(0) is True

        engine.on_tick(
            0,
            saturated=False,
            healthy=True,
            device_free_mb=9000.0,
            actuator=actuator,
            ladder_builder=tuple,
            context_restore_ready=True,
        )
        assert actuator.calls == [("restore_contexts", 0)]
        assert engine.has_context_reduction(0) is False

    def test_a_held_context_restore_does_not_hold_back_a_lane_restore(self) -> None:
        """A lane pause is cheap to undo, so it unwinds on the healthy sample the context reduction waits out."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_pause_rung(ReclaimRungKind.PAUSE_PP_LANE))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        engine.record_context_reduction(0)

        engine.on_tick(
            0,
            saturated=False,
            healthy=True,
            device_free_mb=9000.0,
            actuator=actuator,
            ladder_builder=tuple,
            context_restore_ready=False,
        )
        assert actuator.calls == [("pp", None), ("restore_pp", None)]
        assert engine.has_context_reduction(0) is True

    def test_worker_wide_reduction_is_regrown_by_the_governed_card(self) -> None:
        """A reduction booked against the card-agnostic scope unwinds on the one governed card's recovery."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()

        engine.record_context_reduction(None)

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == [("restore_contexts", None)]
        assert engine.has_context_reduction(None) is False

    def test_reduction_booked_mid_episode_unwinds_after_the_lane_it_followed(self) -> None:
        """Obligations unwind in one LIFO order regardless of which path recorded them."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_pause_rung(ReclaimRungKind.PAUSE_PP_LANE))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert actuator.calls == [("pp", None)]
        engine.record_context_reduction(0)

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == [("pp", None), ("restore_contexts", 0), ("restore_pp", None)]

    def test_only_lanes_that_actually_paused_are_restored(self) -> None:
        """A lane pause that was a no-op (already paused by another owner) is not restored by the engine."""
        engine = VerifiedReclaimLadder()

        class _PPNoOpActuator(_FakeActuator):
            def pause_post_process_lane(self, device_index: int | None) -> bool:
                self.calls.append(("pp_noop", None))
                return False  # already paused by the whole-card residency; the ladder's pause does not act

        actuator = _PPNoOpActuator()
        ladder = _ladder(
            _pause_rung(ReclaimRungKind.PAUSE_PP_LANE),
            _pause_rung(ReclaimRungKind.PAUSE_VAE_LANE),
        )

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        # PP pause was a no-op so the engine advanced to the VAE pause the same tick.
        assert actuator.calls == [("pp_noop", None), ("vae", None)]

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        # Only the VAE lane, which the engine actually stopped, is restored; the no-op PP is left to its owner.
        assert actuator.calls == [("pp_noop", None), ("vae", None), ("restore_vae", None)]

    def test_safety_rung_is_not_restored_by_the_ladder(self) -> None:
        """The engine restores lanes but never safety: the runtime placement policy owns safety's restore."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(
            ReclaimRung(
                kind=ReclaimRungKind.SAFETY_OFF_GPU,
                device_index=0,
                promised_freed_mb=3000.0,
                tenant_label="safety",
            ),
        )

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert actuator.calls == [("safety", None)]
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        # No restore call was appended: safety is not the ladder's to bring back.
        assert actuator.calls == [("safety", None)]


class _StandDownActuator(_FakeActuator):
    """An actuator whose context restore stands down while a whole-card residency owns the pool."""

    def __init__(self) -> None:
        super().__init__()
        self.residency_held = True

    def restore_live_contexts(self, device_index: int | None) -> bool:
        self.calls.append(("restore_contexts", device_index))
        return not self.residency_held


class TestRefusedRestoresStayOwed:
    """An obligation the actuator declined to act on is still owed and is retried later."""

    def test_declined_context_restore_survives_the_unwind(self) -> None:
        """A stood-down restore keeps the debt, and a later tick regrows the pool once the actuator can act.

        The context restore reports no-op for as long as a whole-card residency owns the pool. Discharging the
        obligation on that answer would leave the card at emergency depth with no owner left to regrow it.
        """
        engine = VerifiedReclaimLadder()
        actuator = _StandDownActuator()

        engine.record_context_reduction(0)
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == [("restore_contexts", 0)]
        assert engine.has_context_reduction(0) is True

        # The residency released, so the same obligation is retried and this time the actuator acts.
        actuator.residency_held = False
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple
        )
        assert actuator.calls == [("restore_contexts", 0), ("restore_contexts", 0)]
        assert engine.has_context_reduction(0) is False

    def test_retained_debt_does_not_resume_a_spent_ladder(self) -> None:
        """An episode kept alive by a refused restore builds a fresh ladder when the card saturates again."""
        engine = VerifiedReclaimLadder()
        actuator = _StandDownActuator()
        ladder = _ladder(_unload_rung(2, promised=1000.0))
        builds = 0

        def _build() -> tuple[ReclaimRung, ...]:
            nonlocal builds
            builds += 1
            return ladder

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=_build)
        engine.record_context_reduction(0)
        assert actuator.calls == [("unload", 2)]
        assert builds == 1

        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=_build
        )
        assert engine.has_context_reduction(0) is True

        # The card crosses the cliff again: the retained debt must not have consumed the new episode's ladder.
        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=_build)
        assert builds == 2
        assert actuator.calls == [("unload", 2), ("restore_contexts", 0), ("unload", 2)]

    def test_declined_lane_restore_remains_the_episodes_own_to_retry(self) -> None:
        """A lane whose restore was refused keeps its live claimant, so no backstop steals the restore."""
        engine = VerifiedReclaimLadder()

        class _LaneStandDownActuator(_FakeActuator):
            def __init__(self) -> None:
                super().__init__()
                self.can_restore = False

            def restore_vae_lane(self, device_index: int | None) -> bool:
                self.calls.append(("restore_vae", None))
                return self.can_restore

        actuator = _LaneStandDownActuator()
        ladder = _ladder(_pause_rung(ReclaimRungKind.PAUSE_VAE_LANE))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        assert engine.episode_holds_paused_lane(0) is True

        actuator.can_restore = True
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=lambda: ladder
        )
        assert engine.episode_holds_paused_lane(0) is False
        assert actuator.calls == [("vae", None), ("restore_vae", None), ("restore_vae", None)]


class TestSafetyRungCooldown:
    """The deepest rung is a whole process cycle, so a card may spend it only once per dwell."""

    def test_a_second_safety_rung_inside_the_dwell_is_skipped_like_an_inactive_one(self) -> None:
        """A card that saturates again soon after cycling safety is not allowed to cycle it a second time.

        The rung ends and rebuilds the safety process, and the placement policy restores it once the card fits
        it again, so spending it every episode is a process cycle per episode for relief the previous cycle had
        already failed to hold. Refused, it behaves exactly as a rung whose target has gone away: the engine
        moves on within the tick, and an episode with nothing left is unresolved rather than relieved.
        """
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        safety = _pause_rung(ReclaimRungKind.SAFETY_OFF_GPU, 3000.0)
        ladder = _ladder(safety)

        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        assert actuator.calls == [("safety", None)]

        # The card recovers and crosses the cliff again a minute later: the fresh episode builds the rung, and
        # the engine declines to spend it.
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple, now=30.0
        )
        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=60.0
        )
        assert actuator.calls == [("safety", None)]
        assert engine.safety_rungs_refused == 1
        assert engine.is_saturation_unresolved(0) is True

    def test_the_safety_rung_is_available_again_once_the_dwell_has_passed(self) -> None:
        """Past the dwell the rung is spendable again, so the cooldown paces the cycle rather than removing it."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_pause_rung(ReclaimRungKind.SAFETY_OFF_GPU, 3000.0))

        engine.on_tick(
            0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder, now=0.0
        )
        engine.on_tick(
            0, saturated=False, healthy=True, device_free_mb=9000.0, actuator=actuator, ladder_builder=tuple, now=30.0
        )
        engine.on_tick(
            0,
            saturated=True,
            device_free_mb=100.0,
            actuator=actuator,
            ladder_builder=lambda: ladder,
            now=_SAFETY_RUNG_COOLDOWN_SECONDS + 1.0,
        )
        assert actuator.calls == [("safety", None), ("safety", None)]
        assert engine.safety_rungs_refused == 0
