"""Unit smoke tests for the simulated-device-VRAM ledger that backs post-processing-pressure tests.

These pin the *mechanism* the harness rests on, deterministically and without a spawn: the cross-process
VRAM-visibility rule (a process can free only its own models, never a sibling's) and the post-processing
allocation decision built on it. The spawned end-to-end behaviour (a real process recovery) is covered by
``tests/e2e/test_sim_vram_post_processing_e2e.py``.
"""

from __future__ import annotations

from horde_worker_regen.process_management.simulation.sim_vram import (
    SimProcessVram,
    SimVramLedger,
    SimVramSpec,
    simulate_post_processing_allocation,
)

# A 16 GB card carrying two ~SDXL residents plus their contexts, the shape of the post-processing over-commit.
_TOTAL_16GB_MB = 16375.0
_WEIGHTS_MB = 4900.0
_CONTEXT_MB = 1354.0
_PP_PEAK_MB = 8533.0


def _two_resident_processes(total_mb: float) -> SimVramLedger:
    """Seed an in-process ledger: a card with processes 0 and 1 each holding weights + context."""
    ledger = SimVramLedger.in_process()
    SimVramSpec(
        device_index=0,
        total_vram_mb=total_mb,
        processes=[
            SimProcessVram(process_id=0, weights_mb=_WEIGHTS_MB, context_mb=_CONTEXT_MB),
            SimProcessVram(process_id=1, weights_mb=_WEIGHTS_MB, context_mb=_CONTEXT_MB),
        ],
    ).seed(ledger)
    return ledger


def test_device_free_is_total_minus_all_contributions() -> None:
    """Device-wide free VRAM is the card total minus every process's weights and context."""
    ledger = _two_resident_processes(_TOTAL_16GB_MB)

    used = 2 * (_WEIGHTS_MB + _CONTEXT_MB)
    assert ledger.device_used_mb(0) == used
    assert ledger.device_free_mb(0) == _TOTAL_16GB_MB - used


def test_free_own_models_releases_only_the_named_process() -> None:
    """A process frees its own weights but a sibling's weights and every context remain charged.

    This is the cross-process rule: ComfyUI's ``free_memory`` reaches only this process's loaded models.
    """
    ledger = _two_resident_processes(_TOTAL_16GB_MB)

    ledger.free_own_models(device_index=0, process_id=0)

    # Process 0's weights are gone; process 1's weights and both contexts are untouched.
    assert ledger.device_used_mb(0) == _WEIGHTS_MB + (2 * _CONTEXT_MB)


def test_post_processing_stalls_when_sibling_residency_overcommits_the_card() -> None:
    """The over-commit stall: freeing own weights is not enough because a sibling still over-commits.

    Three siblings' worth of residency leaves less than the upscaler peak free even after the running
    process evicts its own model, so the allocation cannot fit; the real child would thrash and be reaped.
    """
    ledger = SimVramLedger.in_process()
    SimVramSpec(
        device_index=0,
        total_vram_mb=_TOTAL_16GB_MB,
        processes=[SimProcessVram(process_id=pid, weights_mb=_WEIGHTS_MB, context_mb=_CONTEXT_MB) for pid in range(3)],
    ).seed(ledger)

    fits = simulate_post_processing_allocation(
        ledger,
        device_index=0,
        process_id=0,
        post_processing_peak_mb=_PP_PEAK_MB,
    )

    assert fits is False
    # Its own model was freed (the in-process reclaim), but the two siblings still hold the card.
    assert ledger.device_free_mb(0) < _PP_PEAK_MB


def test_post_processing_fits_on_a_roomy_card() -> None:
    """With ample total VRAM the same job's post-processing peak fits and the allocation succeeds."""
    ledger = _two_resident_processes(total_mb=49152.0)

    fits = simulate_post_processing_allocation(
        ledger,
        device_index=0,
        process_id=0,
        post_processing_peak_mb=_PP_PEAK_MB,
    )

    assert fits is True


def test_orchestrator_evicting_a_sibling_flips_a_stall_into_a_fit() -> None:
    """The dynamic the harness exists to show: only an orchestrator-driven sibling eviction frees room.

    The card is over-committed for process 0's upscaler. Process 0 cannot self-reclaim its way out, but
    once the orchestrator tells an *idle sibling* to unload (modeled by that sibling's own free call),
    cross-process VRAM is returned and the same allocation now fits.
    """
    ledger = SimVramLedger.in_process()
    SimVramSpec(
        device_index=0,
        total_vram_mb=_TOTAL_16GB_MB,
        processes=[SimProcessVram(process_id=pid, weights_mb=_WEIGHTS_MB, context_mb=_CONTEXT_MB) for pid in range(3)],
    ).seed(ledger)

    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is False
    )

    # The orchestrator reclaims cross-process VRAM by unloading idle siblings 1 and 2.
    ledger.free_own_models(device_index=0, process_id=1)
    ledger.free_own_models(device_index=0, process_id=2)

    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is True
    )


def test_per_card_isolation_of_free_vram() -> None:
    """A second card's residency does not affect the first card's free VRAM (per-device keying)."""
    ledger = _two_resident_processes(_TOTAL_16GB_MB)
    ledger.set_total(device_index=1, total_mb=_TOTAL_16GB_MB)
    ledger.set_resident_weights(device_index=1, process_id=0, weights_mb=_WEIGHTS_MB)

    card0_used = 2 * (_WEIGHTS_MB + _CONTEXT_MB)
    assert ledger.device_used_mb(0) == card0_used
    assert ledger.device_used_mb(1) == _WEIGHTS_MB


# A 16 GB card running post-processing with overlap, where the over-commit is a *concurrent* sample
# co-scheduled with a live upscale peak rather than stale idle residency. The card carries safety-on-GPU plus
# several inference contexts; one process holds an in-flight SDXL sample (its weights + a sampling-activation
# transient) while a sibling enters a 4x upscale. The shapes: an SD1.5 base whose upscale peak is still
# ~8.5 GB, a concurrent SDXL sample, and near-zero free VRAM once both are live.
_SAFETY_CONTEXT_MB = 1100.0
_CTX_MB = 1300.0
_SD15_WEIGHTS_MB = 2200.0
_SDXL_WEIGHTS_MB = 4900.0
_CONCURRENT_SAMPLE_MB = 2800.0
_UPSCALE_PEAK_MB = 8533.0


def _seed_overlap_saturated_card() -> SimVramLedger:
    """Seed an overlap-driven over-commit: safety-on-GPU, idle contexts, and a busy SDXL sample beside an upscaler.

    Process 1 holds the SD1.5 job about to upscale; process 3 holds a *concurrent* SDXL sample (weights plus
    a live sampling transient) co-scheduled by overlap; processes 2 and 4 are idle bare contexts (their models
    already unloaded). Safety-on-GPU is modeled as a context-only holder. Free VRAM is near zero.
    """
    ledger = SimVramLedger.in_process()
    ledger.set_total(0, _TOTAL_16GB_MB)
    ledger.set_context_overhead(0, "safety", _SAFETY_CONTEXT_MB)
    # Process 1: the SD1.5 job about to enter post-processing.
    ledger.set_resident_weights(0, 1, _SD15_WEIGHTS_MB)
    ledger.set_context_overhead(0, 1, _CTX_MB)
    # Process 3: a concurrent SDXL sample co-scheduled by overlap; busy, so its footprint cannot be evicted.
    ledger.set_resident_weights(0, 3, _SDXL_WEIGHTS_MB)
    ledger.set_context_overhead(0, 3, _CTX_MB)
    ledger.set_transient(0, 3, _CONCURRENT_SAMPLE_MB)
    # Processes 2 and 4: idle bare contexts (their warm models already unloaded).
    ledger.set_context_overhead(0, 2, _CTX_MB)
    ledger.set_context_overhead(0, 4, _CTX_MB)
    return ledger


def test_overlap_sample_makes_upscale_peak_unhostable_by_idle_reclaim_alone() -> None:
    """A concurrent overlap sample, not stale residency, is what the upscale peak cannot fit past.

    The upscaler does not fit even after the job frees its own weights and the orchestrator reclaims *every*
    idle increment the planner can touch (both idle sibling contexts and safety-off-GPU). The only footprint
    large enough to matter is the concurrent SDXL sample on process 3, a live job that idle reclaim must not
    abort. This is why a dispatch-time, evict-one-idle-sibling planner cannot prevent this class of stall: the
    over-commit emerges mid-flight from overlap, so the decisive lever is gating the overlapping sample rather
    than reclaiming idle room.
    """
    ledger = _seed_overlap_saturated_card()
    # Near-zero free with both the upscaler's owner and the concurrent sample live.
    assert ledger.device_free_mb(0) < 500.0

    # The job frees its own weights (ComfyUI in-child), still nowhere near the peak.
    ledger.free_own_models(device_index=0, process_id=1)
    assert ledger.device_free_mb(0) < _UPSCALE_PEAK_MB

    # The orchestrator reclaims every idle increment a reclaim plan could target: stop both idle bare
    # contexts and drop safety off-GPU. (Idle siblings hold no evictable model, so there is nothing for the
    # EVICT_SIBLING_MODEL rung; this is the most an aggressive idle reclaim could achieve.)
    ledger.set_context_overhead(0, 2, 0.0)
    ledger.set_context_overhead(0, 4, 0.0)
    ledger.set_context_overhead(0, "safety", 0.0)

    # Even after the maximal idle reclaim, the live concurrent sample keeps the peak from fitting.
    assert ledger.device_free_mb(0) < _UPSCALE_PEAK_MB


def test_gating_the_overlap_sample_lets_the_upscale_peak_fit() -> None:
    """Without the concurrent sample, the same peak fits the same card, isolating overlap as the cause.

    When the overlapping sample is withheld (or evicted) while the upscale peak is imminent, the card, even
    still carrying safety and the idle contexts, has room. This pins the overlap co-scheduling as the cause
    and the overlap gate as the lever, distinct from idle-residency reclaim.
    """
    ledger = _seed_overlap_saturated_card()

    # No overlapping sample is co-scheduled: process 3 holds no weights and no sampling transient.
    ledger.free_own_models(device_index=0, process_id=3)

    fits = simulate_post_processing_allocation(
        ledger,
        device_index=0,
        process_id=1,
        post_processing_peak_mb=_UPSCALE_PEAK_MB,
    )
    assert fits is True
