"""Unit tests for the pure preload-admission decision functions.

Each test feeds plain values in and asserts the decision out; the scheduler-integrated behavior is
covered by the budget/whole-card/regression suites.
"""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.governance import (
    PreloadSlotSnapshot,
    RamReclaimOutcome,
    VramReclaimOutcome,
    card_preload_order,
    compute_preload_disallowed_processes,
    decide_ram_reclaim_outcome,
    decide_vram_reclaim_outcome,
    preload_concurrency_blocked,
    select_head_room_process_id,
)


def _slot(process_id: int, model_name: str | None, *, can_accept: bool = True) -> PreloadSlotSnapshot:
    """Build one preload slot snapshot."""
    return PreloadSlotSnapshot(process_id=process_id, model_name=model_name, can_accept_job=can_accept)


class TestPreloadDisallowed:
    """The three target-exclusion guards compose into one disallowed set."""

    def test_queued_model_guard_is_the_default(self) -> None:
        """Slots holding queued models are excluded when enough slots are loaded for the work."""
        disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=[1, 2],
            busy_process_ids=[3],
            prefer_busy_only=False,
            inference_process_models={},
            wanted_models=set(),
            max_inference_processes=2,
            draining_process_ids=frozenset(),
        )
        assert disallowed == {1, 2}

    def test_busy_only_regime_replaces_the_queued_guard(self) -> None:
        """With fewer loaded slots than work, only busy slots are protected so idle slots stay reachable."""
        disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=[1, 2],
            busy_process_ids=[3],
            prefer_busy_only=True,
            inference_process_models={},
            wanted_models=set(),
            max_inference_processes=2,
            draining_process_ids=frozenset(),
        )
        assert disallowed == {3}

    def test_affinity_protects_the_last_copy_of_a_wanted_model(self) -> None:
        """In the models<=processes regime the sole resident copy of a wanted model is protected."""
        disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=[],
            busy_process_ids=[],
            prefer_busy_only=False,
            inference_process_models={1: "model-a", 2: "model-b"},
            wanted_models={"model-a", "model-b"},
            max_inference_processes=2,
            draining_process_ids=frozenset(),
        )
        assert disallowed == {1, 2}

    def test_draining_slots_are_always_excluded(self) -> None:
        """A slot draining for RAM reclaim is excluded regardless of the other guards."""
        disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=[1],
            busy_process_ids=[],
            prefer_busy_only=False,
            inference_process_models={},
            wanted_models=set(),
            max_inference_processes=8,
            draining_process_ids=frozenset({7}),
        )
        assert disallowed == {1, 7}


class TestPreloadConcurrencyGate:
    """Model loads serialize per device unless very fast disk mode relaxes the gate."""

    def test_one_inflight_load_blocks_the_next_by_default(self) -> None:
        """The default gate blocks a second concurrent load."""
        assert preload_concurrency_blocked(
            num_preloading=1,
            max_concurrent_inference_processes=2,
            very_fast_disk_mode=False,
        )

    def test_no_inflight_load_does_not_block(self) -> None:
        """An idle device admits its first load."""
        assert not preload_concurrency_blocked(
            num_preloading=0,
            max_concurrent_inference_processes=2,
            very_fast_disk_mode=False,
        )

    def test_very_fast_disk_allows_up_to_ceiling_plus_one(self) -> None:
        """Fast-disk hosts may run ceiling+1 concurrent loads before blocking."""
        assert not preload_concurrency_blocked(
            num_preloading=2,
            max_concurrent_inference_processes=2,
            very_fast_disk_mode=True,
        )
        assert preload_concurrency_blocked(
            num_preloading=3,
            max_concurrent_inference_processes=2,
            very_fast_disk_mode=True,
        )


class TestHeadRoomSelection:
    """The starved-head fallback frees the cheapest displaceable slot without touching live work."""

    def test_prefers_an_empty_slot(self) -> None:
        """An empty slot beats displacing any resident model."""
        slots = (_slot(1, "model-a"), _slot(2, None))
        assert select_head_room_process_id(slots, in_progress_models=set(), pending_models={"model-a"}) == 2

    def test_prefers_an_unneeded_model_over_a_queued_one(self) -> None:
        """A resident model no pending job needs is cheaper to displace than a queued one."""
        slots = (_slot(1, "queued-model"), _slot(2, "cold-model"))
        assert select_head_room_process_id(slots, in_progress_models=set(), pending_models={"queued-model"}) == 2

    def test_never_displaces_an_in_progress_model(self) -> None:
        """A slot whose model a live job is using is never chosen."""
        slots = (_slot(1, "live-model"),)
        assert select_head_room_process_id(slots, in_progress_models={"live-model"}, pending_models=set()) is None

    def test_skips_slots_that_cannot_accept_work(self) -> None:
        """A slot that cannot take a job is not a displacement candidate."""
        slots = (_slot(1, None, can_accept=False),)
        assert select_head_room_process_id(slots, in_progress_models=set(), pending_models=set()) is None


class TestCardPreloadOrder:
    """A fresh load goes to a card already serving the model, then the least-loaded card."""

    def test_serving_card_first_then_least_loaded(self) -> None:
        """The sticky-then-least-loaded placement policy."""
        order = card_preload_order(
            {0, 1, 2},
            cards_already_serving_model={2},
            card_busy_counts={0: 1, 1: 0, 2: 2},
        )
        assert order == [2, 1, 0]


class TestVramReclaimOutcome:
    """The VRAM escalation policy resolves an exhausted reclaim pass."""

    def test_progress_defers(self) -> None:
        """Freed room is worth waiting for."""
        outcome = decide_vram_reclaim_outcome(
            freed=True,
            is_head_blocker=True,
            no_live_resource_consumer=True,
            model_unservable=False,
            context_reduction_demanded=False,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.DEFER

    def test_non_head_defers(self) -> None:
        """Only the head-of-queue blocker may escalate past reclamation."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=False,
            no_live_resource_consumer=True,
            model_unservable=False,
            context_reduction_demanded=False,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.DEFER

    def test_live_consumer_defers(self) -> None:
        """A live job holding the device always wins over an escalation."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=True,
            no_live_resource_consumer=False,
            model_unservable=False,
            context_reduction_demanded=False,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.DEFER

    def test_breaker_tripped_model_is_held(self) -> None:
        """A model that keeps faulting over the budget is held, not admitted."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
            model_unservable=True,
            context_reduction_demanded=False,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.HOLD_UNSERVABLE

    def test_warranted_context_reduction_wins_over_admit(self) -> None:
        """When contexts are the over-commit and the figures are trusted, reduce rather than admit."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
            model_unservable=False,
            context_reduction_demanded=True,
            whole_card_warranted=True,
        )
        assert outcome is VramReclaimOutcome.REDUCE_CONTEXTS

    def test_unwarranted_reduction_is_declined_into_an_admit(self) -> None:
        """A reduction demand resting on untrusted overhead figures is declined; the head still admits."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
            model_unservable=False,
            context_reduction_demanded=True,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.ADMIT_DECLINING_REDUCTION

    def test_exhausted_head_on_idle_device_admits_best_effort(self) -> None:
        """The final rung: nothing to reclaim, nothing live, admit the head over-budget."""
        outcome = decide_vram_reclaim_outcome(
            freed=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
            model_unservable=False,
            context_reduction_demanded=False,
            whole_card_warranted=False,
        )
        assert outcome is VramReclaimOutcome.BEST_EFFORT_ADMIT


class TestRamReclaimOutcome:
    """The RAM escalation policy mirrors the VRAM final rung."""

    def test_reclaim_progress_defers(self) -> None:
        """An eviction is worth waiting for."""
        outcome = decide_ram_reclaim_outcome(
            reclaimed=True,
            cycled_stale_slot=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
        )
        assert outcome is RamReclaimOutcome.DEFER

    def test_cycled_slot_defers(self) -> None:
        """Cycling an allocator-stuck slot reclaims RAM by respawn; wait for it."""
        outcome = decide_ram_reclaim_outcome(
            reclaimed=False,
            cycled_stale_slot=True,
            is_head_blocker=True,
            no_live_resource_consumer=True,
        )
        assert outcome is RamReclaimOutcome.DEFER

    def test_exhausted_head_admits_best_effort(self) -> None:
        """Nothing reclaimable and nothing live holding memory: admit the head rather than starve it."""
        outcome = decide_ram_reclaim_outcome(
            reclaimed=False,
            cycled_stale_slot=False,
            is_head_blocker=True,
            no_live_resource_consumer=True,
        )
        assert outcome is RamReclaimOutcome.BEST_EFFORT_ADMIT

    def test_non_head_never_admits(self) -> None:
        """A job that is not the head blocker only ever defers."""
        outcome = decide_ram_reclaim_outcome(
            reclaimed=False,
            cycled_stale_slot=False,
            is_head_blocker=False,
            no_live_resource_consumer=True,
        )
        assert outcome is RamReclaimOutcome.DEFER
