"""Unit tests for the verified LIFO reclaim ladder: rung ordering, verification, escalation, exhaustion."""

from __future__ import annotations

from horde_worker_regen.process_management.resources.reclaim_ladder import (
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

    def test_shortfall_after_two_samples_records_calibration_and_escalates(self) -> None:
        """A rung that never yields half its promise escalates after two samples, logging a calibration event."""
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_unload_rung(1, 2000.0), _unload_rung(2, 500.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        # First verification sample: realized 100 << 1000 (half of 2000); still within the window, waits.
        engine.on_tick(0, saturated=True, device_free_mb=200.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.rungs_issued == 1
        assert actuator.calls == [("unload", 1)]

        # Second verification sample: still short. Shortfall recorded, calibration event, rung 2 escalated.
        engine.on_tick(0, saturated=True, device_free_mb=250.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.verification_shortfalls == 1
        assert actuator.calibration_events == [(ReclaimRungKind.UNLOAD_IDLE_MODEL, 2000.0, 150.0)]
        assert engine.rungs_issued == 2
        assert actuator.calls == [("unload", 1), ("unload", 2)]

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
        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        engine.on_tick(0, saturated=True, device_free_mb=120.0, actuator=actuator, ladder_builder=lambda: ladder)
        engine.on_tick(0, saturated=True, device_free_mb=130.0, actuator=actuator, ladder_builder=lambda: ladder)
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


class TestReclaimLadderVerifiedRestore:
    """Lane-pause rungs the engine issues are restored (LIFO) when the card returns HEALTHY, safety excepted."""

    def test_teardown_rung_gets_a_longer_verification_window(self) -> None:
        """A lane pause is given three samples (not two) to free its promise before it counts as short.

        Its memory returns only once the lane process has exited, which takes longer than one governor sample,
        so the extra sample keeps the engine from escalating past a pause that is still tearing down.
        """
        engine = VerifiedReclaimLadder()
        actuator = _FakeActuator()
        ladder = _ladder(_pause_rung(ReclaimRungKind.PAUSE_PP_LANE, 5000.0), _unload_rung(2, 500.0))

        engine.on_tick(0, saturated=True, device_free_mb=100.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert actuator.calls == [("pp", None)]
        # Sample 1 and sample 2 are both short, but a teardown rung's window is three samples, so it holds.
        engine.on_tick(0, saturated=True, device_free_mb=150.0, actuator=actuator, ladder_builder=lambda: ladder)
        engine.on_tick(0, saturated=True, device_free_mb=160.0, actuator=actuator, ladder_builder=lambda: ladder)
        assert engine.rungs_issued == 1
        assert engine.verification_shortfalls == 0
        # Sample 3 is still short: only now does it escalate, issuing the next rung.
        engine.on_tick(0, saturated=True, device_free_mb=170.0, actuator=actuator, ladder_builder=lambda: ladder)
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
