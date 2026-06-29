"""Regression guard: a whole-card-intent model on a high-VRAM card keeps the sibling contexts it has room for.

The failure mode this guards: on a 24 GB card an ``Flux.1-Schnell fp8 (Compact)`` head reserves the whole
device, collapsing the live inference-process count to one and cycling safety off the GPU, even though the
card has ample headroom: free VRAM sits at 15-16 GB while it holds sole residency, against an fp8 weight
footprint of ~11.5 GB. A small-fraction model on a large card handed the card it does not need produces
sustained reservation churn (reload + safety cycling) that caps throughput.

The cause is tier classification, not a measurement gap. ``Flux.1-Schnell fp8 (Compact)`` is
named in :data:`horde_worker_regen.consts.VRAM_HEAVY_MODELS`, and every Flux baseline is in
``model_sizing.EXTRA_LARGE_BASELINE_VALUES``; either routes the size-tier classifier to ``EXTRA_LARGE``,
which the scheduler feeds to the forecast as ``wants_whole_card=True``. That flag then collapses
:meth:`StreamForecast.max_resident_processes` to a hard ``return 1`` regardless of how much of the card the
weights actually occupy, and ``_establish_whole_card_residency`` uses that as its teardown target. The
device's real free VRAM and the per-process context cost are never consulted: a Flux fp8 checkpoint that
genuinely co-resides with a sibling context on a 24 GB card is torn down to sole residency exactly as a Flux
fp16 checkpoint on a 16 GB card (which truly cannot share) would be.

The fix: :meth:`StreamForecast.max_resident_processes` no longer hard-collapses a ``wants_whole_card`` model
to one process. It sizes the residency target by the same budget arithmetic an ordinary model uses (total
VRAM minus weights minus the activation-inclusive reserve, against the per-context overhead), which is
hardware-relative: a Flux fp8 head on a 24GB card keeps an idle sibling context, the same head on a 16GB card
still collapses to sole residency. The ``wants_whole_card`` intent continues to govern that the head never
*co-samples* (the concurrency overlap gate), so the only thing that changed is the teardown depth.

Behavior these tests pin:

  * On a high-VRAM card where the weights plus the activation-inclusive reserve genuinely leave room for one
    or more additional contexts, the residency target preserves those processes instead of collapsing to one.
    The surviving sibling context is what lets the next non-Flux job pipeline rather than paying a full
    teardown + respawn each time the heavy head cycles. (The concurrency overlap gate, which already forbids
    an EXTRA_LARGE job from sharing a busy card, keeps two heavy jobs from sampling at once, so preserving the
    context is safe.)
  * The behavior stays hardware-relative: the same fp8 weights on a 16 GB card, where the budget genuinely
    leaves no room for a second context, still collapse to sole residency.
  * The budget math itself is willing: with ``wants_whole_card`` cleared, the identical forecast already sizes
    two resident contexts. The collapse-to-one is purely the intent override discarding that headroom.

The figures reflect observed ones: linux, a 24 GB card, the seeded Flux-schnell resident-weight estimate
(11.5 GB), and deliberately *pessimistic* per-additional-context marginal (the threads=2 reconciled ~3.4 GB,
far above a real idle context) so the surviving-process claim does not rest on an optimistic overhead reading.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast

# Figures representative of the 24 GB card the churn was observed on.
_CARD_24GB_MB = 24074.0
_CARD_16GB_MB = 16384.0

# Flux.1-Schnell fp8 (Compact): the hordelib seed for the flux_schnell baseline (vram_weights_mb=11500,
# min_recommended_vram_mb=14000). The activation working set folded into the reserve is the load peak minus
# the resident weights (14000 - 11500), bounded below by the inference-reserve floor.
_FLUX_FP8_WEIGHTS_MB = 11500.0
_FLUX_RESERVE_MB = 2500.0

# First-context cost (one-time CUDA runtime + one context): the figure a single fresh process measures, and
# what the worker reports as "per-process overhead". It is the floor, not the per-additional-context cost.
_FIRST_CONTEXT_OVERHEAD_MB = 4266.0
# A deliberately pessimistic per-additional-context marginal (the threads=2 reconciled figure). Even this
# leaves a 24 GB card room for one sibling context beyond Flux; a realistic idle context costs far less.
_PESSIMISTIC_MARGINAL_MB = 3431.0


def _flux_forecast(
    *,
    total_vram_mb: float,
    wants_whole_card: bool,
    marginal_mb: float = _PESSIMISTIC_MARGINAL_MB,
    live_contexts: int = 4,
) -> StreamForecast:
    """A Flux fp8 streaming forecast on a card of the given size, with measured (not fallback) overheads.

    ``free_now``/``free_if_alone``/``free_after_model_evict`` are sized as the scheduler sizes them so the
    forecast is ``known`` and its residency properties exercise the real code paths.
    """
    additional = max(0, live_contexts - 1)
    free_after_model_evict = total_vram_mb - _FIRST_CONTEXT_OVERHEAD_MB - marginal_mb * additional
    free_if_alone = total_vram_mb - _FIRST_CONTEXT_OVERHEAD_MB
    return StreamForecast(
        weights_mb=_FLUX_FP8_WEIGHTS_MB,
        reserve_mb=_FLUX_RESERVE_MB,
        base_reserve_mb=_FLUX_RESERVE_MB,
        free_now_mb=free_if_alone,
        free_if_alone_mb=free_if_alone,
        free_after_model_evict_mb=max(0.0, free_after_model_evict),
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=_FIRST_CONTEXT_OVERHEAD_MB,
        marginal_process_overhead_mb=marginal_mb,
        wants_whole_card=wants_whole_card,
    )


class TestHighVramWholeCardPreservesProcesses:
    """A whole-card-intent model on a 24 GB card must keep the processes the budget proves fit."""

    def test_flux_fp8_on_24gb_keeps_a_sibling_context(self) -> None:
        """The residency target for Flux fp8 on a 24 GB card keeps a sibling context (was: collapsed to one).

        Weights 11.5 GB + reserve 2.5 GB = 14 GB leaves ~10 GB on a 24 GB card. Even charged the pessimistic
        ~3.4 GB marginal, that holds the loading process's full first context (4.27 GB) plus one additional
        sibling context: a residency target of two. The current ``wants_whole_card`` collapse hard-returns
        one, tearing down a sibling process the card has room for: the reservation churn this guards against.
        """
        forecast = _flux_forecast(total_vram_mb=_CARD_24GB_MB, wants_whole_card=True)

        assert forecast.max_resident_processes() is not None
        assert forecast.max_resident_processes() >= 2, (
            "Flux fp8 on a 24 GB card has room for a sibling context; the residency target must not "
            "collapse to sole residency"
        )

    def test_budget_math_is_willing_without_the_intent_override(self) -> None:
        """The identical forecast sizes two contexts once ``wants_whole_card`` is cleared.

        Isolates the cause: the budget arithmetic (total - weights - reserve, against the measured marginal)
        already admits a second context. Only the intent override discards it, so the collapse is the
        override's doing, not a genuine shortfall.
        """
        forecast = _flux_forecast(total_vram_mb=_CARD_24GB_MB, wants_whole_card=False)

        assert forecast.max_resident_processes() == 2

    def test_flux_fp8_on_16gb_still_collapses_to_sole_residency(self) -> None:
        """Guard: on a 16 GB card the same fp8 weights genuinely leave no room, so one process is correct.

        The desired fix must stay hardware-relative: it preserves processes only where the budget proves
        they fit. Here weights 11.5 GB + reserve 2.5 GB = 14 GB leaves ~2.4 GB, below even the first context,
        so sole residency is right and must be unchanged.
        """
        forecast = _flux_forecast(total_vram_mb=_CARD_16GB_MB, wants_whole_card=True)

        assert forecast.max_resident_processes() == 1
