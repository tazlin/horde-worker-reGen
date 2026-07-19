"""End-to-end fixed-model-pool flow: seeded demand seats models and the lanes shape real pop traffic.

Drives the full worker lifecycle (fake children, generating pop source) with the pool enabled and a
synthetic demand snapshot seeded through the harness. The generating source honors the advertised model
list exactly as the live API does, so these runs prove the whole chain: demand snapshot -> ranker ->
seats -> fixed/free advertising lanes -> the traffic mix that actually arrives and completes.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import HarnessConfig, SyntheticModelDemand, run_harness_async
from horde_worker_regen.process_management.simulation._canned_scenarios import SoakImageTemplate

# Every scenario spawns real OS child processes through the harness, so the module is opt-in via -m slow.
pytestmark = pytest.mark.slow

_HOT_MODELS = ("Deliberate", "Anything Diffusion")
_COLD_MODELS = ("Dreamshaper", "stable_diffusion")


def _equal_weight_templates() -> list[SoakImageTemplate]:
    return [SoakImageTemplate(model=model, weight=1.0) for model in (*_HOT_MODELS, *_COLD_MODELS)]


def _hot_demand() -> dict[str, SyntheticModelDemand]:
    demand = {model: SyntheticModelDemand(queued=400.0, worker_count=1) for model in _HOT_MODELS}
    demand.update({model: SyntheticModelDemand(queued=1.0, worker_count=5) for model in _COLD_MODELS})
    return demand


@pytest.mark.e2e
async def test_pool_concentrates_traffic_on_seated_high_demand_models() -> None:
    """With the pool on and demand concentrated, completed traffic concentrates on the seated models.

    All four templates carry equal generation weight, so without lane narrowing the mix would split
    evenly; the seated models dominating the completed set is the advertising lanes doing their job.
    """
    result = await run_harness_async(
        HarnessConfig(
            process_mode="fake",
            skip_api=True,
            soak_seconds=30.0,
            timeout_seconds=180.0,
            soak_image_templates=_equal_weight_templates(),
            synthetic_demand=_hot_demand(),
            bridge_data_overrides={
                "queue_size": 2,
                "max_threads": 1,
                "model_pool": {
                    "enabled": True,
                    "seats": 2,
                    "ranker_enabled": True,
                },
            },
        ),
    )

    assert result.audit_failures == []
    completed_models = [job.model_name for job in (result.metrics.jobs if result.metrics else []) if not job.faulted]
    assert len(completed_models) >= 4, f"soak completed too few jobs to judge the mix: {completed_models}"
    hot_share = sum(1 for model in completed_models if model in _HOT_MODELS) / len(completed_models)
    assert hot_share >= 0.7, (
        f"expected the seated high-demand models to dominate the completed mix, got {hot_share:.0%} "
        f"across {len(completed_models)} jobs: {completed_models}"
    )


@pytest.mark.e2e
async def test_pool_disabled_soak_spreads_traffic() -> None:
    """The identical mix with the pool disabled completes work across the whole template set."""
    result = await run_harness_async(
        HarnessConfig(
            process_mode="fake",
            skip_api=True,
            soak_seconds=30.0,
            timeout_seconds=180.0,
            soak_image_templates=_equal_weight_templates(),
            synthetic_demand=_hot_demand(),
            bridge_data_overrides={"queue_size": 2, "max_threads": 1},
        ),
    )

    assert result.audit_failures == []
    completed_models = {job.model_name for job in (result.metrics.jobs if result.metrics else []) if not job.faulted}
    assert len(completed_models) >= 3, (
        f"expected an un-pooled soak to complete work across most of the template set, got {completed_models}"
    )
