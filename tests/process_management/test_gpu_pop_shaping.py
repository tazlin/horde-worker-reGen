"""Tests for A6.1 union pop advertising (gpu_pop_shaping.advertised_capabilities).

A multi-GPU worker advertises one capability envelope that is the union of its cards: every model any card
serves, a feature/NSFW flag if any card allows it, the largest max_power, and the summed thread count. The
worker then routes each returned job to a card that can actually serve it. A single-card plan reduces to that
one card's config.
"""

from __future__ import annotations

from horde_worker_regen.process_management.gpu.gpu_pop_shaping import advertised_capabilities, under_fed_card

from .conftest import make_mock_bridge_data, make_test_card_runtimes


def _card(*, device_index: int, max_concurrent: int, **config_overrides: object) -> object:
    """A single CardRuntime whose effective config carries the given overrides."""
    config = make_mock_bridge_data(**config_overrides)
    runtimes = make_test_card_runtimes(
        device_indices=(device_index,),
        config=config,
        max_concurrent_inference=max_concurrent,
    )
    return runtimes[device_index]


class TestUnionAdvertising:
    """The advertised envelope is the most permissive value across the driven cards."""

    def test_unions_models_features_resolution_and_threads(self) -> None:
        """Models union, features OR, max_power max, and threads sum across the two cards."""
        card0 = _card(
            device_index=0,
            max_concurrent=2,
            image_models_to_load=["model_a", "shared"],
            nsfw=False,
            allow_controlnet=False,
            allow_lora=True,
            max_power=8,
        )
        card1 = _card(
            device_index=1,
            max_concurrent=1,
            image_models_to_load=["model_b", "shared"],
            nsfw=True,
            allow_controlnet=True,
            allow_lora=False,
            max_power=16,
        )

        envelope = advertised_capabilities({0: card0, 1: card1})

        assert envelope.models == frozenset({"model_a", "model_b", "shared"})
        assert envelope.nsfw is True  # card 1 serves NSFW
        assert envelope.allow_controlnet is True  # card 1 allows ControlNet
        assert envelope.allow_lora is True  # card 0 allows LoRA
        assert envelope.max_power == 16  # the bigger card's resolution ceiling
        assert envelope.threads == 3  # 2 + 1 summed concurrency

    def test_single_card_envelope_matches_that_card(self) -> None:
        """A one-card plan advertises exactly that card's config (the single-GPU reduction)."""
        card0 = _card(
            device_index=0,
            max_concurrent=2,
            image_models_to_load=["only_model"],
            nsfw=True,
            allow_post_processing=False,
            max_power=8,
        )

        envelope = advertised_capabilities({0: card0})

        assert envelope.models == frozenset({"only_model"})
        assert envelope.nsfw is True
        assert envelope.allow_post_processing is False
        assert envelope.max_power == 8
        assert envelope.threads == 2


class TestUnderFedCard:
    """Adaptive targeting picks the card the local queue is starving past the balance threshold."""

    def test_targets_the_starved_card_when_queue_is_lopsided(self) -> None:
        """Three of four held jobs run only on card 0, so card 1 is under-fed (75% >= 0.5 threshold)."""
        eligible_sets = [{0}, {0}, {0}, {0, 1}]
        assert under_fed_card(eligible_sets, [0, 1], balance_threshold=0.5) == 1

    def test_no_target_when_queue_is_balanced(self) -> None:
        """When most work is servable by both cards, neither is starved past the threshold."""
        eligible_sets = [{0, 1}, {0, 1}, {0}, {1}]
        assert under_fed_card(eligible_sets, [0, 1], balance_threshold=0.5) is None

    def test_picks_the_most_starved_of_several(self) -> None:
        """Card 2 (served by none of the held jobs) is more under-fed than card 1."""
        eligible_sets = [{0}, {0}, {0, 1}, {0, 1}]
        assert under_fed_card(eligible_sets, [0, 1, 2], balance_threshold=0.5) == 2

    def test_single_card_and_empty_queue_never_target(self) -> None:
        """One card cannot be lopsided, and an empty queue gives nothing to balance against."""
        assert under_fed_card([{0}, {0}], [0], balance_threshold=0.5) is None
        assert under_fed_card([], [0, 1], balance_threshold=0.5) is None

    def test_threshold_gates_targeting(self) -> None:
        """Half the queue unservable by card 1 targets it at threshold 0.5 but not at 0.75."""
        eligible_sets = [{0}, {0}, {0, 1}, {0, 1}]
        assert under_fed_card(eligible_sets, [0, 1], balance_threshold=0.5) == 1
        assert under_fed_card(eligible_sets, [0, 1], balance_threshold=0.75) is None
