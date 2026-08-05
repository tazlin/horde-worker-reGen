"""End-to-end tests that run the full worker lifecycle through the harness.

These spawn real OS child processes (running the protocol-faithful fakes) and run the
real asyncio orchestration loop with the API faked out; no GPU, no network, no
hordelib/torch in any process.

All tests in this module are async so they use pytest-asyncio's managed event loop
instead of ``asyncio.run()``.  Calling ``asyncio.run()`` from inside a test creates a
nested event loop whose ``ProactorEventLoop`` teardown on Windows can race with the
vscode-pytest named-pipe server, causing the pipe to disappear before the final test
report can be sent.
"""

from __future__ import annotations

import os
import threading

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness, run_harness_async
from horde_worker_regen.process_management.lifecycle import shutdown_manager as shutdown_manager_module
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    make_alchemy_scenario,
    make_simple_scenario,
)

# Every scenario spawns real OS child processes through the harness, so the module is opt-in via -m slow.
pytestmark = pytest.mark.slow


@pytest.mark.e2e
async def test_full_lifecycle_fake_processes_no_api() -> None:
    """Every job in a small scenario must complete pop → inference → safety → submit."""
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=3,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
        ),
    )

    assert not result.timed_out, f"Harness run timed out before the scenario completed ({result.failure_summary()})"
    assert result.num_jobs_faulted == 0, (
        f"Expected 0 faulted jobs, got {result.num_jobs_faulted} ({result.failure_summary()})"
    )
    assert result.num_jobs_completed == 3, (
        f"Expected 3 completed jobs, got {result.num_jobs_completed} ({result.failure_summary()})"
    )
    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"


@pytest.mark.e2e
async def test_run_metrics_flow_through_fake_processes() -> None:
    """Verify per-job records carry stage latencies and the fakes' synthetic phase metrics.

    Exercises the pipe → dispatcher → run-metrics chain end-to-end.
    """
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=2,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.metrics is not None
    assert len(result.metrics.jobs) == 2

    for record in result.metrics.jobs:
        assert record.e2e_seconds is not None and record.e2e_seconds > 0
        assert record.queue_wait_seconds is not None
        assert record.stage_timestamps.get("FINALIZED") is not None
        assert record.phase_metrics is not None, "fake-process job metrics were not correlated"
        assert record.phase_metrics.sampling is not None
        assert record.phase_metrics.vram_used_high_water_mb == 1234

    assert result.metrics.vram_used_high_water_mb_per_process, "no per-process VRAM high-water recorded"
    assert result.metrics.num_process_recoveries == 0
    assert result.metrics.process_crash_events == []


@pytest.mark.e2e
async def test_mixed_image_and_alchemy_scenario() -> None:
    """Image jobs and canned alchemy forms must both complete in the same fake-mode run."""
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=2,
            alchemy_forms=make_alchemy_scenario(["caption", "RealESRGAN_x4plus"], 2),
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
            bridge_data_overrides={"alchemy_allow_concurrent": True},
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.num_jobs_completed == 2
    assert result.num_alchemy_forms_completed == 2
    assert result.num_alchemy_forms_faulted == 0

    # Alchemy form metrics flow through the same run-metrics chain as image jobs.
    assert result.metrics is not None
    alchemy_records = [record for record in result.metrics.jobs if record.is_alchemy]
    assert len(alchemy_records) == 2


@pytest.mark.e2e
async def test_full_lifecycle_with_simulated_inference_time() -> None:
    """Jobs that take nonzero (fake) inference time must still all complete."""
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            job_delay_seconds=0.5,
            timeout_seconds=90.0,
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.num_jobs_completed == len(scenario), (
        f"Expected {len(scenario)} completed jobs, got {result.num_jobs_completed} ({result.failure_summary()})"
    )


@pytest.mark.e2e
def test_sequential_run_harness_calls_complete_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two fixed-scenario runs in one interpreter both succeed and leave no process-terminating thread.

    The force-kill backstop is scoped to a single run: once ``run_harness`` returns it must have cancelled
    and joined any backstop it armed, so a thread created by the first run can never terminate the shared
    process during the second. ``_force_exit_process`` is replaced with a recorder so a regression (a
    leaked backstop reaching the exit lever) is caught as a recorded call instead of killing the test
    process, and the ``run_harness`` (asyncio.run) entry point is used so the sequential-lifecycle path is
    exercised exactly as the gate driver drives it.
    """
    force_exit_calls: list[int] = []
    monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", force_exit_calls.append)

    for run_index in range(2):
        result = run_harness(
            HarnessConfig(
                num_jobs=2,
                process_mode="fake",
                skip_api=True,
                timeout_seconds=90.0,
            ),
        )
        assert result.succeeded, f"run {run_index} did not succeed ({result.failure_summary()})"
        assert result.num_jobs_completed == 2, f"run {run_index}: {result.failure_summary()}"
        assert result.boot_failed_no_progress is False, f"run {run_index} was misread as a boot failure"
        assert force_exit_calls == [], f"a backstop reached the force-exit lever by run {run_index}"

        live_backstops = [
            thread for thread in threading.enumerate() if thread.name == "shutdown-backstop" and thread.is_alive()
        ]
        assert not live_backstops, f"run {run_index} left a live backstop thread: {live_backstops}"


class TestHarnessBridgeDataCapabilities:
    """The harness bridge data must advertise every capability the workload actually needs.

    The simulated pop matching honours the request exactly as the live API does, so a capability the
    bridge does not advertise silently filters that traffic out of the run instead of failing loudly.
    """

    def test_lora_soak_templates_enable_lora_advertising(self) -> None:
        """A soak mix carrying LoRA references produces bridge data that advertises LoRA support."""
        from horde_worker_regen.benchmark.soak import build_lora_storm_soak_scenario
        from horde_worker_regen.harness import build_harness_bridge_data

        scenario = build_lora_storm_soak_scenario(soak_seconds=30.0)
        config = HarnessConfig.from_scenario(scenario, process_mode="fake", timeout_seconds=60.0)
        bridge_data = build_harness_bridge_data(config, [])

        assert bridge_data.allow_lora is True

    def test_plain_scenario_leaves_lora_advertising_at_default(self) -> None:
        """A workload with no auxiliary references does not force LoRA advertising on."""
        from horde_worker_regen.harness import build_harness_bridge_data

        config = HarnessConfig(scenario=make_simple_scenario(1), timeout_seconds=60.0)
        bridge_data = build_harness_bridge_data(config, config.scenario or [])

        # Validation resolves the unset worker-level tri-state; the contract is only that a plain
        # workload never has LoRA support forced on.
        assert not bridge_data.allow_lora

    def test_max_power_covers_the_largest_workload_job(self) -> None:
        """The bridge advertises enough power for the mix's largest job, or heavy templates never pop."""
        from horde_worker_regen.benchmark.soak import build_production_replay_soak_scenario
        from horde_worker_regen.harness import build_harness_bridge_data

        scenario = build_production_replay_soak_scenario(soak_seconds=30.0)
        config = HarnessConfig.from_scenario(scenario, process_mode="fake", timeout_seconds=60.0)
        bridge_data = build_harness_bridge_data(config, [])

        largest = max(t.width * t.height for t in config.soak_image_templates)
        assert bridge_data.max_power is not None
        assert bridge_data.max_power * 8 * 64 * 64 >= largest


class TestHarnessStatsExport:
    """Only a real-mode run exports stats, and the knob can still turn that off.

    The stats stream is the machine-readable record of what each job cost, so a run whose children
    fabricate their durations must not write one.
    """

    def test_real_mode_enables_stats_export(self) -> None:
        """A real-mode run writes its session stats stream without any caller opt-in."""
        from horde_worker_regen.harness import build_harness_bridge_data

        config = HarnessConfig(scenario=make_simple_scenario(1), process_mode="real", timeout_seconds=60.0)
        bridge_data = build_harness_bridge_data(config, config.scenario or [])

        assert bridge_data.stats_export_enabled is True

    def test_fake_mode_leaves_stats_export_off(self) -> None:
        """Fake and dry-run timings are synthetic, so those runs never write a stats stream."""
        from horde_worker_regen.harness import build_harness_bridge_data

        for process_mode in ("fake", "dry_run"):
            config = HarnessConfig(
                scenario=make_simple_scenario(1),
                process_mode=process_mode,  # type: ignore[arg-type]
                timeout_seconds=60.0,
            )
            bridge_data = build_harness_bridge_data(config, config.scenario or [])

            assert bridge_data.stats_export_enabled is False, process_mode

    def test_knob_disables_stats_export_in_real_mode(self) -> None:
        """A real-mode run can be asked to leave no stats file behind."""
        from horde_worker_regen.harness import build_harness_bridge_data

        config = HarnessConfig(
            scenario=make_simple_scenario(1),
            process_mode="real",
            timeout_seconds=60.0,
            stats_export=False,
        )
        bridge_data = build_harness_bridge_data(config, config.scenario or [])

        assert bridge_data.stats_export_enabled is False


class TestHarnessScenarioProvenance:
    """A scenario-driven run publishes its workload identity for the session's stats snapshot."""

    def test_scenario_identity_reaches_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scenario's name and revision are stamped where the manager's snapshot reads them."""
        from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario
        from horde_worker_regen.harness import _apply_scenario_provenance_env
        from horde_worker_regen.process_management.resources.run_metrics import (
            SCENARIO_ID_ENV_VAR,
            SCENARIO_REVISION_ENV_VAR,
        )

        monkeypatch.delenv(SCENARIO_ID_ENV_VAR, raising=False)
        monkeypatch.delenv(SCENARIO_REVISION_ENV_VAR, raising=False)
        scenario = Scenario(name="pricing_corpus", revision="3", image_jobs=[CannedImageJobSpec()])
        config = HarnessConfig.from_scenario(scenario, process_mode="fake", timeout_seconds=60.0)

        assert config.scenario_id == "pricing_corpus"
        assert config.scenario_revision == "3"

        _apply_scenario_provenance_env(config)
        assert os.environ[SCENARIO_ID_ENV_VAR] == "pricing_corpus"
        assert os.environ[SCENARIO_REVISION_ENV_VAR] == "3"

    def test_run_without_a_scenario_clears_a_prior_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ad-hoc run cannot inherit the identity an earlier run left in the same process."""
        from horde_worker_regen.harness import _apply_scenario_provenance_env
        from horde_worker_regen.process_management.resources.run_metrics import (
            SCENARIO_ID_ENV_VAR,
            SCENARIO_REVISION_ENV_VAR,
        )

        monkeypatch.setenv(SCENARIO_ID_ENV_VAR, "stale_scenario")
        monkeypatch.setenv(SCENARIO_REVISION_ENV_VAR, "1")

        _apply_scenario_provenance_env(HarnessConfig(num_jobs=1, timeout_seconds=60.0))

        assert SCENARIO_ID_ENV_VAR not in os.environ
        assert SCENARIO_REVISION_ENV_VAR not in os.environ
