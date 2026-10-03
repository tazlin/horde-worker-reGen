"""RAM threshold ordering, physical feasibility and marginal load pricing."""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from horde_worker_regen.process_management.resources.ram_footprints import LearnedRamStore
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_checkpoint_staging_ram_mb,
    predict_context_ram_mb,
    predict_job_transient_ram_mb,
    ram_headroom,
)
from tests.process_management.conftest import make_job_pop_response


@given(
    total=st.floats(min_value=16384, max_value=131072),
    reserve=st.floats(min_value=0, max_value=16384),
    pause=st.floats(min_value=50, max_value=100),
    charge=st.floats(min_value=0, max_value=32000),
    context=st.floats(min_value=1100, max_value=32000),
    risk=st.floats(min_value=0, max_value=8000),
)
def test_requirements_are_ordered_and_reserve_is_paid_once(
    total: float,
    reserve: float,
    pause: float,
    charge: float,
    context: float,
    risk: float,
) -> None:
    """All gates read one reserve and ordered absolute thresholds, even for unfit work."""
    model = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        staging_charge_mb=charge,
        context_cost_mb=context,
        in_flight_transient_mb=risk,
    )
    assert model.hard_floor_mb <= model.soft_hold_mb <= model.preload_requirement_mb <= model.restore_requirement_mb
    assert model.preload_requirement_mb == max(model.soft_hold_mb, max(reserve, model.hard_floor_mb) + charge)
    assert model.restore_requirement_mb == max(
        model.preload_requirement_mb, max(reserve, model.hard_floor_mb) + context
    )


@given(
    total=st.integers(16384, 131072),
    context=st.integers(1100, 28000),
    count=st.integers(1, 4),
    reserve=st.integers(0, 8192),
    pause=st.integers(80, 95),
)
def test_feasible_planned_contexts_do_not_starve_themselves(
    total: int,
    context: int,
    count: int,
    reserve: int,
    pause: int,
) -> None:
    """No foreign occupancy: resident work pays only new transients, never its context cost again."""
    thresholds = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        staging_charge_mb=600,
        context_cost_mb=context,
        in_flight_transient_mb=600 * count,
    )
    planned_free = total - context * count
    # Physical feasibility includes the floor, operator reserve, and outstanding transient risk.
    assume(planned_free >= max(thresholds.hard_floor_mb, reserve) + 600 * count)
    assert planned_free >= thresholds.soft_hold_mb
    assert planned_free >= thresholds.preload_requirement_mb
    before_restore = total - context * (count - 1)
    assert before_restore >= thresholds.restore_requirement_mb


@pytest.mark.parametrize(("total", "reserve", "pause"), [(63434, 8192, 90), (65536, 4096, 85)])
def test_two_fp8_contexts_on_a_64gb_host(total: int, reserve: int, pause: int) -> None:
    """Two fp8 contexts fit under a raised reserve and under defaults without charging a resident context twice."""
    model = ram_headroom(
        total,
        reserve_mb=reserve,
        pause_percent=pause,
        staging_charge_mb=600,
        context_cost_mb=24000,
        in_flight_transient_mb=1200,
    )
    assert total - 48000 >= model.preload_requirement_mb
    assert total - 24000 >= model.restore_requirement_mb


def test_sdxl_checkpoint_swap_uses_file_bytes_and_feature_deltas() -> None:
    """A checkpoint swap is around 7 GB even though a cold SDXL context is seeded at 12 GB."""
    job = make_job_pop_response("sdxl", width=1024, height=1024)
    baseline = "stable_diffusion_xl"
    delta = predict_job_transient_ram_mb(job, baseline)
    assert predict_checkpoint_staging_ram_mb(job, baseline, 6500 * 1024 * 1024) == 6500 + delta
    assert predict_context_ram_mb(baseline) == 12000
    assert predict_checkpoint_staging_ram_mb(job, baseline, None) == 12000 + delta


def test_learned_growth_is_trusted_bidirectionally_and_separated_by_load_kind() -> None:
    """Five samples permit a lower price, and a later larger sample raises it immediately."""
    store = LearnedRamStore()
    for _ in range(4):
        store.observe("sdxl", "whole", 6500)
    assert store.measured_estimate_mb("sdxl", "whole") is None
    store.observe("sdxl", "whole", 6500)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(7150)
    store.observe("sdxl", "whole", 10000)
    assert store.measured_estimate_mb("sdxl", "whole") == 11000
    assert store.measured_estimate_mb("sdxl", "component") is None
    for _ in range(20):
        store.observe("sdxl", "whole", 6000)
    assert store.measured_estimate_mb("sdxl", "whole") == pytest.approx(6600)


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
