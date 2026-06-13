"""Table-driven tests for the multi-model soak memory preflight."""

from __future__ import annotations

from horde_worker_regen.benchmark.memory_preflight import plan_soak_topology


class TestPlanSoakTopology:
    """`plan_soak_topology` trims the resident-model count to what memory can hold."""

    def test_full_pool_fits(self) -> None:
        """When every model fits, the plan reports the full desired count and `fits`."""
        plan = plan_soak_topology(
            desired_models=4,
            per_model_vram_mb=4000,
            total_vram_mb=24000,
            reserve_vram_mb=1500,
        )
        assert plan.fitting_models == 4
        assert plan.fits
        assert plan.is_viable
        assert "fit" in plan.reason

    def test_pool_trimmed_to_what_fits(self) -> None:
        """A pool that overflows VRAM is trimmed to the largest count that fits the budget."""
        # budget = 12000 - 1500 = 10500; 10500 // 5000 = 2
        plan = plan_soak_topology(
            desired_models=4,
            per_model_vram_mb=5000,
            total_vram_mb=12000,
            reserve_vram_mb=1500,
        )
        assert plan.fitting_models == 2
        assert not plan.fits
        assert plan.is_viable
        assert "only 2 of 4" in plan.reason

    def test_not_viable_when_one_model_cannot_fit(self) -> None:
        """When not even one model fits under the reserve, the plan is not viable."""
        plan = plan_soak_topology(
            desired_models=4,
            per_model_vram_mb=11000,
            total_vram_mb=12000,
            reserve_vram_mb=1500,
        )
        assert plan.fitting_models == 0
        assert not plan.is_viable
        assert "no soak model fits" in plan.reason

    def test_unknown_per_model_does_not_block(self) -> None:
        """A non-positive (unknown) per-model estimate must not block the soak."""
        plan = plan_soak_topology(
            desired_models=4,
            per_model_vram_mb=0,
            total_vram_mb=12000,
        )
        assert plan.fitting_models == 4
        assert plan.fits

    def test_ram_is_the_tighter_bound(self) -> None:
        """When RAM holds fewer models than VRAM, RAM governs the fitting count."""
        # VRAM allows 4 (24000-1500)//4000=5 -> capped at 4; RAM allows (10000-0)//4000=2
        plan = plan_soak_topology(
            desired_models=4,
            per_model_vram_mb=4000,
            total_vram_mb=24000,
            reserve_vram_mb=1500,
            per_model_ram_mb=4000,
            total_ram_mb=10000,
        )
        assert plan.fitting_models == 2
        assert not plan.fits
