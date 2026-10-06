"""RAM threshold ordering, conservation of admitted allocations and marginal load pricing."""

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.generation_parameters import KNOWN_FACEFIXERS, KNOWN_UPSCALERS
from hypothesis import assume, given
from hypothesis import strategies as st

from horde_worker_regen.process_management.resources.ram_footprints import LearnedRamStore
from horde_worker_regen.process_management.resources.resource_budget import (
    RamBudget,
    predict_checkpoint_staging_ram_mb,
    predict_context_ram_mb,
    predict_job_feature_ram_mb,
    ram_headroom,
)
from tests.process_management.conftest import make_job_pop_response

_MB = 1024 * 1024
_SEEDED_BASELINES = [baseline for baseline in KNOWN_IMAGE_GENERATION_BASELINE if predict_context_ram_mb(baseline)]
_POST_PROCESSORS = [KNOWN_UPSCALERS.RealESRGAN_x4plus.value, KNOWN_FACEFIXERS.CodeFormers.value]


@given(
    total=st.floats(min_value=16384, max_value=131072),
    reserve=st.floats(min_value=0, max_value=16384),
    pause=st.floats(min_value=50, max_value=100),
    charge=st.floats(min_value=0, max_value=32000),
    context=st.floats(min_value=1100, max_value=32000),
    outstanding=st.floats(min_value=0, max_value=16000),
)
def test_requirements_are_ordered_and_count_outstanding_work_once(
    total: float,
    reserve: float,
    pause: float,
    charge: float,
    context: float,
    outstanding: float,
) -> None:
    """Every requirement keeps max(floor, reserve) after outstanding work and the new allocation both land."""
    model = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        staging_charge_mb=charge,
        context_cost_mb=context,
        outstanding_mb=outstanding,
    )
    kept = max(reserve, model.hard_floor_mb)
    assert model.hard_floor_mb <= model.soft_hold_mb <= model.preload_requirement_mb <= model.restore_requirement_mb
    assert model.soft_hold_mb == pytest.approx(model.hard_floor_mb + outstanding)
    assert model.preload_requirement_mb == pytest.approx(kept + outstanding + charge)
    assert model.restore_requirement_mb >= kept + outstanding + context


@given(
    available=st.floats(min_value=0, max_value=80000),
    committed=st.floats(min_value=0, max_value=8000),
    outstanding=st.floats(min_value=0, max_value=12000),
    charge=st.floats(min_value=0, max_value=30000),
    credit=st.floats(min_value=0, max_value=20000),
    component=st.one_of(st.none(), st.floats(min_value=0, max_value=12000)),
    floor=st.floats(min_value=0, max_value=12000),
    reserve=st.floats(min_value=0, max_value=16384),
)
def test_every_accepted_admission_conserves_the_reserve(
    available: float,
    committed: float,
    outstanding: float,
    charge: float,
    credit: float,
    component: float | None,
    floor: float,
    reserve: float,
) -> None:
    """Whole, credited and component admissions all leave max(floor, reserve) once everything lands."""
    verdict = RamBudget(reserve_mb=reserve).check_job(
        make_job_pop_response("m"),
        None,
        available,
        committed_reserve_mb=committed,
        reusable_credit_mb=credit,
        danger_floor_mb=floor,
        disaggregated=component is not None,
        component_charge_mb=component,
        staging_charge_mb=charge,
        outstanding_planned_mb=outstanding,
    )
    assume(verdict.fits and verdict.predicted_mb is not None)
    assert verdict.predicted_mb is not None
    assert available - committed - outstanding - verdict.predicted_mb >= max(floor, reserve) - 1e-6


def test_outstanding_and_incoming_allocations_cannot_spend_the_same_headroom() -> None:
    """Outstanding feature RAM and an incoming stage that land together would leave 4.5 GB under a 6.3 GB floor."""
    verdict = RamBudget(reserve_mb=8192).check_job(
        make_job_pop_response("m"),
        None,
        15000,
        danger_floor_mb=6300,
        staging_charge_mb=6500,
        outstanding_planned_mb=4000,
    )
    assert not verdict.fits
    assert verdict.available_mb == 11000
    assert verdict.reserve_mb == 8192


@given(
    total=st.integers(32768, 131072),
    reserve=st.integers(0, 8192),
    pause=st.integers(80, 95),
    baseline=st.sampled_from(_SEEDED_BASELINES),
    resident=st.integers(1, 3),
    post_processing=st.lists(st.sampled_from(_POST_PROCESSORS), unique=True, max_size=2),
    file_mb=st.integers(2000, 24000),
)
def test_physically_feasible_plans_are_admitted_at_real_prices(
    total: int,
    reserve: int,
    pause: int,
    baseline: str,
    resident: int,
    post_processing: list[str],
    file_mb: int,
) -> None:
    """With no foreign occupancy, a plan that fits beside the reserve is admitted, so the worker cannot starve itself.

    Feasibility is conservation alone: resident contexts at their seed, the features of a job running on each,
    the incoming checkpoint with its features, and max(floor, reserve) fit the host together.
    """
    job = make_job_pop_response("m", width=1024, height=1024, post_processing=post_processing or None)
    context = predict_context_ram_mb(baseline)
    checkpoint = predict_checkpoint_staging_ram_mb(baseline, None, file_mb * _MB)
    assert context is not None and checkpoint is not None
    features = predict_job_feature_ram_mb(job, baseline).total_mb
    floor = ram_headroom(total, pause_percent=pause).hard_floor_mb
    available = total - context * resident
    outstanding = features * resident
    kept = max(floor, reserve)
    assume(available - outstanding - checkpoint - features >= kept)

    verdict = RamBudget(reserve_mb=reserve).check_job(
        job,
        baseline,
        available,
        danger_floor_mb=floor,
        staging_charge_mb=checkpoint + features,
        outstanding_planned_mb=outstanding,
    )
    assert verdict.fits
    thresholds = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        outstanding_mb=outstanding,
        staging_charge_mb=checkpoint + features,
        context_cost_mb=context,
    )
    if available - outstanding - context >= kept:
        assert available >= thresholds.restore_requirement_mb


@pytest.mark.parametrize(("total", "reserve", "pause"), [(63434, 8192, 90), (65536, 4096, 85)])
def test_two_fp8_contexts_on_a_64gb_host(total: int, reserve: int, pause: int) -> None:
    """Two fp8 contexts fit under a raised reserve and under defaults without charging a resident context twice."""
    model = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        staging_charge_mb=600,
        context_cost_mb=24000,
        outstanding_mb=1200,
    )
    assert total - 48000 >= model.preload_requirement_mb
    assert total - 24000 >= model.restore_requirement_mb


def test_checkpoint_price_excludes_features() -> None:
    """A checkpoint swap is priced at file bytes, the 12 GB seed only without a file, and never with features."""
    baseline = "stable_diffusion_xl"
    assert predict_checkpoint_staging_ram_mb(baseline, None, 6500 * _MB) == 6500
    assert predict_context_ram_mb(baseline) == 12000
    assert predict_checkpoint_staging_ram_mb(baseline, None, None) == 12000


def test_feature_ram_is_split_by_the_process_that_allocates_it() -> None:
    """Post-processors add RAM to the post-processing process only; sampling features stay on inference."""
    baseline = "stable_diffusion_xl"
    plain = predict_job_feature_ram_mb(make_job_pop_response("m", width=1024, height=1024), baseline)
    heavy = predict_job_feature_ram_mb(
        make_job_pop_response("m", width=1024, height=1024, post_processing=_POST_PROCESSORS),
        baseline,
    )
    assert plain.post_processing_mb == 0
    assert heavy.sampling_mb == plain.sampling_mb
    assert heavy.post_processing_mb > 0
    assert heavy.total_mb == heavy.sampling_mb + heavy.post_processing_mb


def test_learned_growth_is_trusted_bidirectionally_and_separated_by_load_kind() -> None:
    """Five samples permit a lower price, and a later larger sample raises it immediately."""
    store = LearnedRamStore()
    for _ in range(4):
        store.observe("sdxl", "whole", 6500)
    assert store.measured_estimate_mb("sdxl", "whole") is None
    store.observe("sdxl", "whole", 6500)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(7150)
    store.observe("sdxl", "whole", 10000)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(11000)
    assert store.measured_estimate_mb("sdxl", "component") is None
    for _ in range(20):
        store.observe("sdxl", "whole", 6000)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(6600)


def test_learned_price_follows_the_load_peak() -> None:
    """A load that touched more than it kept is priced at what it touched."""
    store = LearnedRamStore()
    for _ in range(5):
        store.observe("sdxl", "whole", 6500, 8000)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(8800)
    for _ in range(5):
        store.observe("flux", "whole", 9000, 5000)
    assert store.measured_estimate_mb("flux", "whole") == pytest.approx(9900)


def test_baseline_evidence_prices_a_sibling_checkpoint_by_its_size() -> None:
    """Five loads across a baseline's checkpoints price a sixth checkpoint from growth per staged megabyte."""
    store = LearnedRamStore()
    for model in ("sdxl_a", "sdxl_b", "sdxl_c", "sdxl_a", "sdxl_b"):
        store.observe(model, "whole", 6500, 7150, baseline="stable_diffusion_xl", size_mb=6500)
    assert store.measured_estimate_mb("sdxl_c", "whole") is None, "one model's own evidence is not yet trusted"
    priced = store.measured_estimate_mb("sdxl_d", "whole", baseline="stable_diffusion_xl", size_mb=3250)
    assert priced == pytest.approx(3250 * 1.1 * 1.1)
    assert store.measured_estimate_mb("flux_a", "whole", baseline="flux_1", size_mb=11500) is None
    assert store.measured_estimate_mb("sdxl_d", "component", baseline="stable_diffusion_xl", size_mb=3250) is None
    assert store.measured_estimate_mb("sdxl_d", "whole", baseline="stable_diffusion_xl") is None


def test_a_checkpoints_own_evidence_outranks_its_baseline() -> None:
    """A trusted per-model price is used even when the baseline's ratio would price it differently."""
    store = LearnedRamStore()
    for _ in range(5):
        store.observe("sdxl_a", "whole", 4000, baseline="stable_diffusion_xl", size_mb=6500)
    for _ in range(5):
        store.observe("sdxl_b", "whole", 9000, baseline="stable_diffusion_xl", size_mb=6500)
    assert store.measured_estimate_mb(
        "sdxl_a", "whole", baseline="stable_diffusion_xl", size_mb=6500
    ) == pytest.approx(4400)


def test_linux_private_reading_excludes_clean_checkpoint_mappings() -> None:
    """A 26 GB RSS/USS report with 8 GB anonymous pages cannot masquerade as 26 GB of allocator growth."""
    from horde_worker_regen.utils.private_memory import linux_private_ram_bytes

    assert (
        linux_private_ram_bytes(
            "Rss: 26680000 kB\nPrivate_Clean: 18000000 kB\nPrivate_Dirty: 8200000 kB\nAnonymous: 8200000 kB\n"
        )
        == 8200000 * 1024
    )
    assert linux_private_ram_bytes("Rss: 100 kB\n") is None


def test_peak_sampler_keeps_the_highest_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak survives a load whose settled reading is lower than what it touched."""
    from horde_worker_regen.utils import private_memory

    readings = iter([1000, 9000, 4000])
    monkeypatch.setattr(private_memory, "private_ram_usage_bytes", lambda _process: next(readings, 4000))
    sampler = private_memory.PrivateRamPeakSampler(interval_seconds=3600)
    with sampler:
        sampler._sample()
    assert sampler.peak_bytes == 9000


def test_host_cache_eviction_conserves_physical_ram_and_delays_the_next_load() -> None:
    """Growing foreign occupancy evicts clean checkpoint pages rather than inventing cache capacity."""
    from tests.process_management.liveness._host_ram import HostRamLedger

    ledger = HostRamLedger(65536, 30000)
    assert ledger.stage(0, "sdxl", 6500, 8500) > 0
    assert ledger.stage(0, "sdxl", 6500, 8500) == 0
    ledger.foreign_mb = 50000
    ledger.trim_cache()
    assert ledger.free_mb >= 0
    assert sum(ledger.cache_mb.values()) <= ledger.available_mb
    assert ledger.stage(0, "sdxl", 6500, 8500) > 0
    assert ledger.rss_mb(0) == 9600 + ledger.available_mb


def test_host_commit_charges_every_mapped_checkpoint_whole_until_the_lane_drops_it() -> None:
    """A mapped checkpoint charges its full size to commit even after its cached pages are reclaimed."""
    from tests.process_management.liveness._host_ram import HostRamLedger

    assert HostRamLedger(65536, 30000).available_commit_mb is None, "a row without a limit models no commit"
    ledger = HostRamLedger(65536, 10000, commit_limit_mb=90000)
    ledger.stage(0, "flux", 16000, 1000)
    ledger.stage(1, "qwen", 19000, 1000)
    private_mb = 10000 + 2 * (1100 + 1000)
    assert ledger.available_commit_mb == 90000 - 35000 - private_mb
    ledger.foreign_mb = 60000
    ledger.trim_cache()
    assert sum(ledger.cache_mb.values()) < 35000, "physical pressure reclaimed cached pages"
    assert ledger.available_commit_mb == 0.0, "the mappings still charge their whole size"
    ledger.evict(1)
    assert ledger.commit_charged_mb == 16000 + 60000 + 1100 + 2100
