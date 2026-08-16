"""Closed-loop e2e for the GPU denoise clearance lease over the disaggregated pipeline.

The real process manager, scheduler and clearance controller run against fake children that honour the
same lease handshake the real child does: stage, report primed, wait for the parent's grant, release when
the sampling window closes. That makes the parent's grant accounting observable end to end, which is what
this pins: every sample stage a pinned sampler runs is a window the parent granted, none is entered
unpriced, and the tail-overlap handoff fires at least once so the pipeline's next window opens before the
current one closes.

The workload alternates two models across two pinned samplers. Each sampler then serves a same-model streak
of its own (the retention and resident-credit path), while the other sampler supplies the staged waiter the
handoff needs: a single-model queue pins every job to one sampler, so no second child is ever staged and the
handoff has nothing to hand to.
"""

from __future__ import annotations

import pytest

from horde_worker_regen import harness as harness_module
from horde_worker_regen.harness import HarnessConfig, HarnessResult, run_harness_async
from horde_worker_regen.process_management.ipc.messages import PipelineStageTag
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager, SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.simulation._canned_scenarios import make_canned_job

# Real OS child processes and multi-second sampling windows: opt-in via -m slow, like the other e2e sims.
pytestmark = pytest.mark.slow

_MODELS = ("Deliberate", "Anything Diffusion")
_NUM_JOBS = 12

_SAMPLE_SECONDS = 2.5
"""How long a fake sampling window lasts.

Sized against the child heartbeat throttle (1s), not against test duration: the parent estimates a
sampler's remaining time from the step positions its beats carry, and a window shorter than a couple of
throttle intervals delivers one beat at step one. The handoff then denies every tick as progress-unknown,
which measures the beat rate rather than the gate.
"""


def _single_card_resources() -> SystemResources:
    """A 16GB single-card host with room for two samplers plus the pipeline's stage lanes."""
    return SystemResources(
        total_ram_bytes=32 * 1024 * 1024 * 1024,
        device_map=TorchDeviceMap(
            root={
                0: TorchDeviceInfo(
                    device_name="Clearance lease sim 0",
                    device_index=0,
                    total_memory=16 * 1024 * 1024 * 1024,
                    kind="cuda",
                ),
            },
        ),
        per_process_overhead_mb=1000,
        marginal_process_overhead_mb=500,
    )


def _sample_stage_count(result: HarnessResult) -> int:
    """How many disaggregated sample stages the run actually ran."""
    if result.metrics is None:
        return 0
    return sum(1 for record in result.metrics.stage_metrics if record.stage is PipelineStageTag.SAMPLE)


@pytest.mark.e2e
async def test_disaggregated_streak_grants_every_sampling_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every sample stage runs inside a granted window, none unpriced, with the handoff firing and no stall."""
    managers: list[HordeWorkerProcessManager] = []
    build_manager = harness_module.build_harness_process_manager

    def _capture_manager(config: HarnessConfig) -> tuple[HordeWorkerProcessManager, int]:
        manager, num_jobs_expected = build_manager(config)
        managers.append(manager)
        return manager, num_jobs_expected

    monkeypatch.setattr(harness_module, "build_harness_process_manager", _capture_manager)

    scenario = [
        make_canned_job(_MODELS[index % len(_MODELS)], width=512, height=512, ddim_steps=30)
        for index in range(_NUM_JOBS)
    ]
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            job_delay_seconds=_SAMPLE_SECONDS,
            timeout_seconds=240.0,
            bridge_data_overrides={
                "max_threads": 2,
                "queue_size": 4,
                "enable_pipeline_disaggregation": True,
                "gpu_sampling_lease_enabled": True,
                "gpu_sampling_lease_slots": 1,
                "gpu_sampling_lease_tail_overlap": True,
                "enable_vram_budget": True,
                "vram_reserve_mb": 2048,
                "ram_reserve_mb": 4096,
                # Retention is what leaves a sampler's weights on the card between its same-model jobs, and
                # the eager evictor would return them after every one.
                "unload_models_from_vram_often": False,
            },
            system_resources=_single_card_resources(),
        ),
    )

    # The pipeline drained: no wedge, no job left to the harness timeout, nothing faulted.
    assert not result.timed_out, result.failure_summary()
    assert result.num_jobs_completed == len(scenario), result.failure_summary()
    assert result.num_jobs_faulted == 0, result.failure_summary()
    assert result.audit_failures == [], result.failure_summary()

    sample_stages = _sample_stage_count(result)
    assert sample_stages == len(scenario), f"expected one sample stage per job, saw {sample_stages}"

    assert len(managers) == 1
    controllers = managers[0]._clearance_controllers
    assert controllers, "the lease was enabled, so the card must have a clearance controller"

    unpriced = sum(controller.unpriced_sampling_windows for controller in controllers.values())
    grants = sum(controller.grants_issued for controller in controllers.values())
    handoffs = sum(controller.tail_overlap_grant_count for controller in controllers.values())

    # Grants track sampling windows: a sampler that entered a denoise loop the parent had not granted is
    # exactly the unpriced window the controller flags, and a stage-per-grant shortfall is the same failure
    # seen from the other side.
    assert unpriced == 0, f"{unpriced} sampling window(s) ran unpriced"
    assert grants >= sample_stages, f"{grants} grant(s) for {sample_stages} sample stage(s)"
    # The handoff is the pipeline's overlap: without it the next window opens only after the current one
    # closes, which is the inter-job GPU stall the lease exists to remove.
    assert handoffs >= 1, "the tail-overlap handoff never fired across the streak"
