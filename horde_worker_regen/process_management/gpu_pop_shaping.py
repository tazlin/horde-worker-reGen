"""Multi-GPU pop shaping: what a single worker driving several cards advertises to the AI-Horde.

The worker presents one identity and one pop stream, yet its cards can differ (heterogeneous VRAM, and
per-card feature / baseline / model overrides). So the pop advertises the **union** of what any card can
serve, and the worker then routes each returned job to an eligible card (see
:mod:`~horde_worker_regen.process_management.gpu_eligibility`); the horde never sees the per-card split. When
the local queue becomes lopsided toward a subset of cards, a later pop is instead **scoped** to an under-fed
card's capability set so the horde returns work that card can actually run (adaptive targeting).

Public:

- [`AdvertisedCapabilities`][horde_worker_regen.process_management.gpu_pop_shaping.AdvertisedCapabilities]:
  the union capability envelope advertised in one pop.
- [`advertised_capabilities`][horde_worker_regen.process_management.gpu_pop_shaping.advertised_capabilities]:
  build that envelope from the per-card runtime plan.

Pure and torch-free: it reads only the per-card config values, so the torch-free orchestrator can call it.
On a single-GPU worker every field equals that one card's config, so the pop is byte-identical to before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from horde_worker_regen.process_management.card_runtime import CardRuntime


@dataclass(frozen=True)
class AdvertisedCapabilities:
    """The union capability envelope a multi-GPU worker advertises in one pop.

    Represents the most permissive value across the driven cards for each axis, so the horde may return any
    job at least one card can serve; the worker's per-job routing then declines to dispatch a job to a card
    that cannot serve it. On a single-GPU worker every field equals that one card's effective config.
    """

    models: frozenset[str]
    """The union of every card's configured image models (the candidate set before stickiness/holdback)."""
    nsfw: bool
    """True if any card serves NSFW work."""
    allow_img2img: bool
    """True if any card allows img2img."""
    allow_inpainting: bool
    """True if any card allows inpainting."""
    allow_post_processing: bool
    """True if any card allows post-processing."""
    allow_controlnet: bool
    """True if any card allows ControlNet."""
    allow_sdxl_controlnet: bool
    """True if any card allows SDXL ControlNet."""
    allow_lora: bool
    """True if any card allows LoRAs."""
    max_power: int
    """The largest ``max_power`` across cards (the biggest resolution any card will accept)."""
    threads: int
    """The summed concurrent-inference ceiling across cards (the worker's total advertised thread count)."""


def advertised_capabilities(card_runtimes: Mapping[int, CardRuntime]) -> AdvertisedCapabilities:
    """Build the union pop envelope from the per-card runtime plan.

    Each axis is OR-ed (features/nsfw), max-ed (``max_power``), summed (threads), or unioned (models) across
    the driven cards. Empty input yields an all-false / zero envelope (the caller falls back to the global
    config in that case).

    Args:
        card_runtimes: The driven cards keyed by stable device index.

    Returns:
        The union envelope as an
        [`AdvertisedCapabilities`][horde_worker_regen.process_management.gpu_pop_shaping.AdvertisedCapabilities].
    """
    models: set[str] = set()
    nsfw = False
    allow_img2img = False
    allow_inpainting = False
    allow_post_processing = False
    allow_controlnet = False
    allow_sdxl_controlnet = False
    allow_lora = False
    max_power = 0
    threads = 0

    for card in card_runtimes.values():
        config = card.config
        models.update(config.image_models_to_load)
        nsfw = nsfw or bool(config.nsfw)
        allow_img2img = allow_img2img or bool(config.allow_img2img)
        allow_inpainting = allow_inpainting or bool(config.allow_inpainting)
        allow_post_processing = allow_post_processing or bool(config.allow_post_processing)
        allow_controlnet = allow_controlnet or bool(config.allow_controlnet)
        allow_sdxl_controlnet = allow_sdxl_controlnet or bool(config.allow_sdxl_controlnet)
        allow_lora = allow_lora or bool(config.allow_lora)
        max_power = max(max_power, int(config.max_power))
        threads += int(card.max_concurrent_inference)

    return AdvertisedCapabilities(
        models=frozenset(models),
        nsfw=nsfw,
        allow_img2img=allow_img2img,
        allow_inpainting=allow_inpainting,
        allow_post_processing=allow_post_processing,
        allow_controlnet=allow_controlnet,
        allow_sdxl_controlnet=allow_sdxl_controlnet,
        allow_lora=allow_lora,
        max_power=max_power,
        threads=threads,
    )


def under_fed_card(
    eligible_card_sets: Sequence[frozenset[int] | set[int]],
    card_indices: Iterable[int],
    *,
    balance_threshold: float,
) -> int | None:
    """Return the card most starved by the current local queue, when the imbalance crosses the threshold.

    For each driven card, this measures the fraction of held jobs that card *cannot* serve. A card that
    cannot serve at least ``balance_threshold`` of the held work is under-fed: the local queue is dominated
    by work only other cards can run, so the next pop should be scoped to this card's capabilities to draw
    work it can actually run. Returns the most under-fed such card, or None when the queue is empty, there is
    only one card, or no card is starved past the threshold (in which case the worker keeps union-popping).

    Args:
        eligible_card_sets: One eligible-card set per held job (from
            :func:`~horde_worker_regen.process_management.gpu_eligibility.eligible_card_indices_for`).
        card_indices: The driven cards' stable device indices.
        balance_threshold: The fraction of held work a card must be unable to serve to count as under-fed.

    Returns:
        The device index of the most under-fed card, or None.
    """
    cards = list(card_indices)
    total = len(eligible_card_sets)
    if total == 0 or len(cards) <= 1:
        return None

    worst_card: int | None = None
    worst_unservable_fraction = 0.0
    for card in cards:
        unservable_fraction = sum(1 for eligible in eligible_card_sets if card not in eligible) / total
        if unservable_fraction >= balance_threshold and unservable_fraction > worst_unservable_fraction:
            worst_unservable_fraction = unservable_fraction
            worst_card = card
    return worst_card
