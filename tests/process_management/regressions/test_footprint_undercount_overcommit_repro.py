"""Reproduction and semantic validation of the resident-footprint under-count over-commit.

A worker on a 24 GB card popped a heavy multi-component checkpoint (a Qwen-Image fp8 job, batch 2) at the
head of the queue while a sibling process held a large model resident. The scheduler force-admitted the head
*shared* (not isolated) once reclamation was exhausted, and when it finally sampled the device was full: the
sampler raised ``torch.OutOfMemoryError`` with ~190 MB free. The fault was correctly classified as a resource
failure and requeued for a degraded, isolated retry, so the *classification* backstop worked; the defect is on
the *admission* side, which drove the co-resident over-commit in the first place.

The single root, traced from the live forecast line and confirmed against the burden seeds:

    A model's resident footprint is *core diffusion weights plus the support components* (text encoders, VAE)
    the engine force-loads onto the device for the duration of every job. The isolation and sibling-room
    judgments (:attr:`StreamForecast.admit_requires_isolation`, :attr:`needs_exclusive_residency`, the
    EXTRA_LARGE ``wants_whole_card`` intent) all key on :attr:`_has_room_for_coresident_model`, which asks
    whether sole-residency free VRAM would still hold *another* full model beside this one's footprint. When
    the footprint is under-counted to the bare core weights, that check reads phantom room on a card that is
    actually near-full, so a genuinely card-filling model reads as co-residable and is admitted beside a
    sibling that then pushes it into an out-of-memory sample.

The under-count was a *data* gap, not a logic gap: the worker's forecast is correct given an accurate
footprint. ``qwen_image`` carries a Qwen2.5-VL text encoder but its burden seed originally charged only part
of the DiT and none of the encoder, so :func:`predict_job_footprint_mb` returned ~12000 MB against a real
resident set of ~27 GB (the hordelib seed has since been corrected from an empirical measurement; these tests
guard that it, and the forecast built on it, keep the isolating verdict). ``flux_schnell`` always carried its
support seed; the Flux cases are the green contrast that shows the same verdict for a checkpoint that was
never mis-seeded.

The isolation verdict under test is :attr:`admit_requires_isolation`, which is what tags the over-budget
force-admit exclusive so concurrent sibling staging and dispatch are suppressed while the head loads and
samples. That exclusivity, not the co-resident-context *depth* (:meth:`max_resident_processes`, which by
design keeps cheap idle contexts on a roomy card since the overlap gate already bars co-sampling), is what
would have kept the sibling off the card during the Qwen sample.

The tests are grounded in the observed device geometry and the real burden seeds rather than hand-tuned
numbers: they construct forecasts from :func:`get_baseline_burden` and assert the *semantic* admission verdict
(isolate a card-filling model; co-reside a model that genuinely fits), which is what shakes out the same
class of failure for any other heavy multi-component baseline whose support components go unseeded.
"""

from __future__ import annotations

import pytest
from hordelib.feature_impact import get_baseline_burden

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async
from tests.process_management.gpu.test_whole_card_residency_repro import _build_context_overcommit_scheduler

# Device geometry read from the session's forecast line: a 24 GB RTX 4090 with two live inference contexts.
# per_process_overhead is the first/sole context's one-time CUDA runtime cost; marginal is each additional
# context once the runtime is shared. free_if_alone = total - one context; the sibling-room check keys on it.
_TOTAL_VRAM_MB_4090 = 24074.0
_PER_PROCESS_OVERHEAD_MB = 3515.0
_MARGINAL_OVERHEAD_MB = 1139.0
_BASE_RESERVE_MB = 2048.0  # the bounded inference-reserve floor sizing the persistent-weight judgments

# A second 16 GB card, the regime where even a correctly-seeded Flux needs the whole device.
_TOTAL_VRAM_MB_16GB = 16375.0
_OVERHEAD_16GB_MB = 1288.0

_QWEN_BASELINE = "qwen_image"
_FLUX_BASELINE = "flux_schnell"
_SDXL_BASELINE = "stable_diffusion_xl"
_SD15_BASELINE = "stable_diffusion_1"

# Ground-truth resident footprints (core weights + force-loaded support components) at the worker's typical
# dtype, independent of the burden seeds so the contract sweep is not circular. The Qwen-Image figure is the
# measured resident weight set (DiT ~19500 MB plus its Qwen2.5-VL text encoder and VAE ~8200 MB) that the
# corrected seed now reflects; the co-resident OOM followed from charging only ~12000 MB of it.
_TRUE_FOOTPRINT_MB: dict[str, float] = {
    _SD15_BASELINE: 3200.0,
    _SDXL_BASELINE: 6600.0,
    _FLUX_BASELINE: 16400.0,
    _QWEN_BASELINE: 27600.0,
}
# The historical under-count: the bare DiT weights the seed once charged, which read as co-residency-safe.
_QWEN_UNDERCOUNT_FOOTPRINT_MB = 12000.0


def _forecast(
    *,
    footprint_mb: float,
    weights_mb: float,
    total_vram_mb: float,
    per_process_overhead_mb: float,
    reserve_mb: float | None = None,
) -> StreamForecast:
    """Build a forecast for a head-of-queue model on an otherwise-established card.

    The free-VRAM readings are placed at their structural values for the given card so the classification
    keys on the persistent footprint (the quantity under test), not on a transient instantaneous reading:
    ``free_if_alone`` is sole residency, ``free_after_model_evict`` is siblings-alive-but-model-free.
    """
    free_if_alone = total_vram_mb - per_process_overhead_mb
    # Two live contexts, siblings model-free: total minus one full context minus one marginal context.
    free_after_evict = total_vram_mb - per_process_overhead_mb - _MARGINAL_OVERHEAD_MB
    return StreamForecast(
        weights_mb=weights_mb,
        footprint_mb=footprint_mb,
        reserve_mb=reserve_mb if reserve_mb is not None else _BASE_RESERVE_MB,
        base_reserve_mb=_BASE_RESERVE_MB,
        free_now_mb=free_after_evict,
        free_if_alone_mb=free_if_alone,
        free_after_model_evict_mb=free_after_evict,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=per_process_overhead_mb,
        marginal_process_overhead_mb=_MARGINAL_OVERHEAD_MB,
    )


def _seed_forecast(baseline: str, *, total_vram_mb: float, per_process_overhead_mb: float) -> StreamForecast:
    """Build a forecast whose footprint comes from the *live burden seed* for ``baseline``.

    This is the path the scheduler actually takes (:func:`predict_job_footprint_mb` wraps the same seed), so a
    seed that under-counts a model's support components produces exactly the forecast the worker acted on.
    """
    burden = get_baseline_burden(baseline)
    assert burden is not None, f"no burden seed for {baseline!r}"
    return _forecast(
        footprint_mb=float(burden.resident_footprint_estimate_mb()),
        weights_mb=float(burden.resident_weight_estimate_mb()),
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=per_process_overhead_mb,
    )


def _should_isolate(footprint_mb: float, *, total_vram_mb: float, per_process_overhead_mb: float) -> bool:
    """Independent oracle: does a model of this true footprint have to reserve the card?

    Derived from the raw device arithmetic, not from the forecast under test: a model must isolate when its
    footprint is a card-dominating share (the whole-card warrant) *and* sole-residency free VRAM cannot hold
    another full sibling model beside it (the co-resident sibling floor). Mirrors the physics the admission
    contract encodes without reusing the property being validated.
    """
    free_if_alone = total_vram_mb - per_process_overhead_mb
    card_demanding = (footprint_mb + _BASE_RESERVE_MB) >= total_vram_mb * 0.4
    room_for_sibling = (free_if_alone - footprint_mb - _BASE_RESERVE_MB) >= 5000.0
    return card_demanding and not room_for_sibling


class TestQwenImageFootprintUndercountRepro:
    """The observed scenario and its fix: the seed must charge the text encoder so a card-filling head isolates."""

    def test_seed_charges_support_components(self) -> None:
        """The corrected seed: the Qwen-Image footprint now exceeds the bare DiT weights.

        A Qwen-Image job force-loads its Qwen2.5-VL text encoder and VAE alongside the DiT. The footprint must
        charge them; a footprint equal to the bare DiT weights is the under-count that read as phantom
        co-residency room and drove the over-commit. This pins that the support mass stays seeded.
        """
        burden = get_baseline_burden(_QWEN_BASELINE)
        assert burden is not None
        assert burden.resident_footprint_estimate_mb() > burden.resident_weight_estimate_mb()

    def test_qwen_head_requires_isolation_on_24gb(self) -> None:
        """From the live (corrected) seed, the Qwen head reads card-demanding with no sibling room: isolate.

        The regression guard for the incident: were the support components dropped from the seed again, the
        head would read co-residable and be admitted shared into the out-of-memory sample.
        """
        forecast = _seed_forecast(
            _QWEN_BASELINE,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        assert forecast._has_room_for_coresident_model is False
        assert forecast.admit_requires_isolation is True

    def test_undercounted_footprint_reads_coresidency_safe(self) -> None:
        """The mechanism: charging only the bare DiT weights reads co-residable on the 24 GB card.

        Uses an explicit under-counted footprint rather than the live seed, so it documents the failure mode
        independently of the (now corrected) registry value: a model whose true resident set fills the card is
        judged to have room for a sibling, and the over-budget admit marks it shared.
        """
        forecast = _forecast(
            footprint_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            weights_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        assert forecast._has_room_for_coresident_model is True
        assert forecast.admit_requires_isolation is False
        assert forecast.needs_exclusive_residency is False
        # A resident count above one means the scheduler believes siblings can co-reside with the Qwen head.
        assert (forecast.max_resident_processes() or 0) > 1

    def test_true_footprint_requires_isolation(self) -> None:
        """With the accurate footprint supplied directly, the admission verdict is correct: isolate.

        Isolates the property logic from the seed: fed Qwen-Image's real resident set, the same forecast
        machinery classifies it card-demanding with no room for a sibling model beside it, so the over-budget
        admit tags it exclusive, the behavior the corrected seed produces.
        """
        forecast = _forecast(
            footprint_mb=_TRUE_FOOTPRINT_MB[_QWEN_BASELINE],
            weights_mb=19500.0,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        assert forecast._has_room_for_coresident_model is False
        assert forecast.admit_requires_isolation is True


class TestFluxCompactCorrectlySeededContrast:
    """Flux Schnell fp8 (Compact) carries its support seed, so it reaches the verdict the Qwen seed must.

    Flux and Qwen-Image are the same shape of model on this hardware: a heavy DiT plus a multi-GB text
    encoder, a resident set that fills most of a 24 GB card. The only difference the tests below turn on is
    that Flux's ``vram_support_weights_mb`` is seeded and Qwen's is not. With the support mass charged, the
    Flux head reads isolation-required (its over-budget admit would be tagged exclusive, keeping a sibling
    off the card); this is the exact verdict the Qwen head reaches once its own encoder is seeded.
    """

    def test_seed_includes_support_components(self) -> None:
        """The Flux seed charges its T5/CLIP support mass: footprint exceeds the bare DiT weights."""
        burden = get_baseline_burden(_FLUX_BASELINE)
        assert burden is not None
        assert burden.resident_footprint_estimate_mb() > burden.resident_weight_estimate_mb()

    def test_flux_head_requires_isolation_on_24gb(self) -> None:
        """From its live seed the Flux head reads card-demanding with no sibling room: isolate.

        The verdict the Qwen head cannot reach while its support components go unseeded. The co-resident
        context depth is left to the residency sizing (a roomy card may keep an idle context, the overlap
        gate barring co-sampling); the admission-side contract is that a sibling is never *staged and
        dispatched* beside this head.
        """
        forecast = _seed_forecast(
            _FLUX_BASELINE,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        assert forecast._has_room_for_coresident_model is False
        assert forecast.admit_requires_isolation is True


class TestCoResidencyIsolationContract:
    """Sweep the admission contract across cards and model classes with ground-truth footprints.

    The point is not to induce a failure but to validate the assumption the scheduler rests on: given an
    honest resident footprint, a model that genuinely fits beside a sibling co-resides (no wasteful whole-card
    reservation) and a model that genuinely fills the card isolates (no over-commit). Any baseline whose real
    footprint the seed fails to reflect will diverge from this contract, which is the general failure the
    Qwen case is one instance of.
    """

    @pytest.mark.parametrize(
        ("total_vram_mb", "overhead_mb"),
        [(_TOTAL_VRAM_MB_4090, _PER_PROCESS_OVERHEAD_MB), (_TOTAL_VRAM_MB_16GB, _OVERHEAD_16GB_MB)],
        ids=["24gb", "16gb"],
    )
    @pytest.mark.parametrize("baseline", list(_TRUE_FOOTPRINT_MB), ids=list(_TRUE_FOOTPRINT_MB))
    def test_isolation_verdict_matches_physics(
        self,
        baseline: str,
        total_vram_mb: float,
        overhead_mb: float,
    ) -> None:
        """The forecast's isolation verdict tracks the independent device-arithmetic oracle for every case.

        Light checkpoints (SD1.5, SDXL) co-reside on both cards; Flux and Qwen fill a 16 GB card and (with
        their full support set) a 24 GB card too. The verdict is derived, not asserted per-row, so a change to
        the warrant fraction or the sibling floor that breaks the relationship surfaces here.
        """
        footprint = _TRUE_FOOTPRINT_MB[baseline]
        forecast = _forecast(
            footprint_mb=footprint,
            weights_mb=footprint,
            total_vram_mb=total_vram_mb,
            per_process_overhead_mb=overhead_mb,
        )
        expected = _should_isolate(
            footprint,
            total_vram_mb=total_vram_mb,
            per_process_overhead_mb=overhead_mb,
        )
        assert forecast.admit_requires_isolation is expected


class TestOverBudgetAdmitExclusivityTracksIsolation:
    """The over-budget force-admit tags the head exclusive iff the forecast says isolate.

    This is the wiring that turned the Qwen head's mis-classification into the out-of-memory sample: when
    reclamation is exhausted and no live job holds the device, the head is admitted best-effort rather than
    wedging the queue, and :meth:`_mark_overbudget_admit` decides shared vs exclusive purely on
    :attr:`admit_requires_isolation`. Exclusive suppresses concurrent sibling staging and dispatch; shared
    leaves the sibling free to fill the card under the loading head. The observed admit logged *shared*.
    """

    async def test_undercounted_head_is_admitted_shared(self) -> None:
        """The reproduced fault path: an under-counted card-filling head is admitted shared, not isolated.

        With the forecast reading phantom co-residency room, the best-effort admit does not tag the head
        exclusive, so the concurrency gate keeps a sibling co-resident, the condition under which the head's
        sample ran the device out of memory.
        """
        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=2)
        job = make_job_pop_response(_QWEN_BASELINE)
        await track_popped_job_async(job_tracker, job)
        undercounted = _forecast(
            footprint_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            weights_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )

        scheduler._mark_overbudget_admit(job, undercounted)

        assert job_tracker.is_admitted_over_budget(job) is True
        assert job_tracker.is_admitted_exclusive(job) is False

    async def test_isolation_required_head_is_admitted_exclusive(self) -> None:
        """With an accurate footprint the same admit is tagged exclusive, keeping the sibling off the card.

        The corrected behavior the seed fix produces: a card-filling head force-admitted over budget is
        isolated, so no sibling is staged and dispatched beside it while it loads and samples.
        """
        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=2)
        job = make_job_pop_response(_QWEN_BASELINE)
        await track_popped_job_async(job_tracker, job)
        accurate = _forecast(
            footprint_mb=_TRUE_FOOTPRINT_MB[_QWEN_BASELINE],
            weights_mb=19500.0,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )

        scheduler._mark_overbudget_admit(job, accurate)

        assert job_tracker.is_admitted_over_budget(job) is True
        assert job_tracker.is_admitted_exclusive(job) is True


class TestCoResidencyIsolationContractInversion:
    """The under-count is the whole difference: same head, opposite verdict, on the two footprints."""

    def test_undercount_inverts_the_verdict(self) -> None:
        """Charging only the core weights flips a card-filling model's verdict from isolate to co-reside.

        The mechanism in one assertion: the same Qwen head reads isolation-required on its true resident set
        and co-residency-safe on the bare-DiT-weights under-count. The gap between the two is the over-commit
        the seed defect opens.
        """
        true_forecast = _forecast(
            footprint_mb=_TRUE_FOOTPRINT_MB[_QWEN_BASELINE],
            weights_mb=19500.0,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        undercounted_forecast = _forecast(
            footprint_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            weights_mb=_QWEN_UNDERCOUNT_FOOTPRINT_MB,
            total_vram_mb=_TOTAL_VRAM_MB_4090,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        )
        assert true_forecast.admit_requires_isolation is True
        assert undercounted_forecast.admit_requires_isolation is False
