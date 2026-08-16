"""Tests for the `horde-benchmark soak` subcommand (the standalone before/after soak driver)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from horde_worker_regen.benchmark.cli import main
from horde_worker_regen.benchmark.soak import (
    OPERATOR_THROUGHPUT_BRIDGE_FIELDS,
    SOAK_MIXES,
    build_soak_mix_scenario,
    operator_throughput_overrides,
    strip_auxiliary_references,
)


class TestSoakMixSelection:
    """`--mix` picks a named mix and `--no-loras` strips its auxiliary references."""

    def test_both_mixes_are_offered(self) -> None:
        """The CLI choices are the mixes the builders implement, production cadence first."""
        assert SOAK_MIXES == ("production_replay", "lora_storm")

    @pytest.mark.parametrize("mix", SOAK_MIXES)
    def test_scenario_is_a_soak_named_after_its_mix(self, mix: str) -> None:
        """Each mix builds a soak scenario carrying its own name as the scenario id."""
        scenario = build_soak_mix_scenario(mix, soak_seconds=42.0)  # type: ignore[arg-type]

        assert scenario.name == mix
        assert scenario.soak_seconds == 42.0
        assert scenario.image_jobs

    def test_production_replay_is_the_default_mix(self) -> None:
        """The default arm replays production cadence rather than the adversarial storm."""
        assert SOAK_MIXES[0] == "production_replay"

    @pytest.mark.parametrize("mix", SOAK_MIXES)
    def test_no_loras_strips_every_auxiliary_reference(self, mix: str) -> None:
        """Stripping removes LoRA and TI references while leaving the traffic shape intact."""
        full = build_soak_mix_scenario(mix, soak_seconds=42.0)  # type: ignore[arg-type]
        stripped = build_soak_mix_scenario(  # type: ignore[arg-type]
            mix,
            soak_seconds=42.0,
            include_auxiliary_references=False,
        )

        assert any(spec.lora_names for spec in full.image_jobs)
        assert not any(spec.lora_names or spec.ti_names for spec in stripped.image_jobs)
        assert [spec.model for spec in stripped.image_jobs] == [spec.model for spec in full.image_jobs]
        assert [spec.count for spec in stripped.image_jobs] == [spec.count for spec in full.image_jobs]
        assert [spec.post_processing for spec in stripped.image_jobs] == [
            spec.post_processing for spec in full.image_jobs
        ]

    def test_supplied_reference_pools_reach_the_templates(self) -> None:
        """Operator-supplied pools replace the synthetic default names a real-mode run cannot resolve."""
        shared = [f"real-shared-{index}" for index in range(3)]
        unique = [f"real-unique-{index}" for index in range(8)]

        scenario = build_soak_mix_scenario(
            "production_replay",
            soak_seconds=42.0,
            shared_lora_references=shared,
            unique_lora_references=unique,
        )

        referenced = {name for spec in scenario.image_jobs for name in spec.lora_names}
        assert referenced
        assert referenced <= set(shared) | set(unique)

    def test_undersized_supplied_pools_are_rejected(self) -> None:
        """A partial pool fails fast rather than building a mix that measures the failure path."""
        with pytest.raises(ValueError, match="at least 3 shared and 8 unique"):
            build_soak_mix_scenario("lora_storm", soak_seconds=42.0, shared_lora_references=["only-one"])

    def test_stripping_is_idempotent(self) -> None:
        """Stripping an already-stripped scenario is a no-op, so a doubled pass cannot lose templates."""
        once = strip_auxiliary_references(build_soak_mix_scenario("lora_storm", soak_seconds=1.0))
        twice = strip_auxiliary_references(once)

        assert twice.image_jobs == once.image_jobs


class TestOperatorThroughputOverrides:
    """The soak carries the operator's throughput-shaping config and nothing else."""

    def test_only_throughput_fields_present_in_the_file_are_carried(self, tmp_path: Path) -> None:
        """Fields outside the throughput set are ignored, and absent fields are not defaulted in."""
        config = tmp_path / "bridgeData.yaml"
        config.write_text(
            "api_key: secret\n"
            "dreamer_name: some-worker\n"
            "max_power: 64\n"
            "max_threads: 2\n"
            "queue_size: 3\n"
            "unload_models_from_vram_often: false\n"
            "models_to_load:\n"
            "  - ALL\n",
            encoding="utf-8",
        )

        overrides = operator_throughput_overrides(config)

        assert overrides == {"max_threads": 2, "queue_size": 3, "unload_models_from_vram_often": False}

    def test_workload_owned_fields_are_never_carried(self) -> None:
        """`max_power` and the model set come from the mix; carrying them could make its jobs ineligible."""
        assert "max_power" not in OPERATOR_THROUGHPUT_BRIDGE_FIELDS
        assert "models_to_load" not in OPERATOR_THROUGHPUT_BRIDGE_FIELDS
        assert "allow_post_processing" not in OPERATOR_THROUGHPUT_BRIDGE_FIELDS

    def test_missing_config_yields_no_overrides(self, tmp_path: Path) -> None:
        """A soak without a readable config still runs, on harness defaults."""
        assert operator_throughput_overrides(tmp_path / "absent.yaml") == {}

    def test_malformed_config_yields_no_overrides(self, tmp_path: Path) -> None:
        """An unreadable config degrades the run's fidelity rather than failing it."""
        config = tmp_path / "bridgeData.yaml"
        config.write_text("max_threads: [unclosed\n", encoding="utf-8")

        assert operator_throughput_overrides(config) == {}


class TestSoakSubcommandPlumbing:
    """The subcommand resolves its arguments into one harness run and a run manifest."""

    def _capture_harness_call(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Replace `run_harness` with a recorder so the plumbing is testable without a worker."""
        captured: dict[str, Any] = {}

        class _Result:
            num_jobs_completed = 7
            num_jobs_faulted = 0
            elapsed_seconds = 12.0
            exit_reason = "completed"
            succeeded = True

        def _fake_run_harness(config: Any) -> Any:  # noqa: ANN401 - a HarnessConfig recorder
            captured["config"] = config
            return _Result()

        monkeypatch.setattr("horde_worker_regen.harness.run_harness", _fake_run_harness)
        monkeypatch.setattr("horde_worker_regen.benchmark.worker_env.ensure_worker_env", lambda *_a, **_k: None)
        return captured

    def test_overrides_layer_over_the_operator_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--override` wins over the carried config, and both land in the manifest and the harness config."""
        captured = self._capture_harness_call(monkeypatch)
        config = tmp_path / "bridgeData.yaml"
        config.write_text("max_threads: 2\nqueue_size: 3\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        rc = main(
            [
                "soak",
                "--mix",
                "lora_storm",
                "--minutes",
                "0.5",
                "--process-mode",
                "fake",
                "--no-loras",
                "--label",
                "arm-b",
                "--bridge-data",
                str(config),
                "--override",
                "queue_size=5",
                "--override",
                "legacy_comfy_vram_unload=true",
                "--out",
                str(out_dir),
            ],
        )
        capsys.readouterr()

        assert rc == 0
        overrides = captured["config"].bridge_data_overrides
        assert overrides == {"max_threads": 2, "queue_size": 5, "legacy_comfy_vram_unload": True}

        manifest = json.loads((out_dir / "soak.json").read_text(encoding="utf-8"))
        assert manifest["label"] == "arm-b"
        assert manifest["mix"] == "lora_storm"
        assert manifest["scenario_id"] == "lora_storm"
        assert manifest["loras_included"] is False
        assert manifest["bridge_overrides"] == overrides
        assert manifest["jobs_completed"] == 7

    def test_soak_seconds_and_timeout_derive_from_minutes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The scenario sustains for `--minutes` and the run timeout adds boot and drain headroom."""
        captured = self._capture_harness_call(monkeypatch)

        rc = main(
            [
                "soak",
                "--minutes",
                "2",
                "--process-mode",
                "fake",
                "--ignore-bridge-data",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        capsys.readouterr()

        assert rc == 0
        config = captured["config"]
        assert config.soak_seconds == 120.0
        assert config.timeout_seconds > 120.0
        assert config.scenario_id == "production_replay"
        assert config.process_mode == "fake"

    def test_ignore_bridge_data_drops_the_operator_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`--ignore-bridge-data` runs the mix on harness defaults even with a config present."""
        captured = self._capture_harness_call(monkeypatch)
        config = tmp_path / "bridgeData.yaml"
        config.write_text("max_threads: 4\n", encoding="utf-8")

        rc = main(
            [
                "soak",
                "--minutes",
                "0.1",
                "--process-mode",
                "fake",
                "--ignore-bridge-data",
                "--bridge-data",
                str(config),
                "--out",
                str(tmp_path / "out"),
            ],
        )
        capsys.readouterr()

        assert rc == 0
        assert captured["config"].bridge_data_overrides == {}

    def test_malformed_override_is_rejected_before_any_worker_starts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `--override` that is not `key=value` exits non-zero without running the harness."""
        captured = self._capture_harness_call(monkeypatch)

        rc = main(
            [
                "soak",
                "--process-mode",
                "fake",
                "--override",
                "not_a_pair",
                "--out",
                str(tmp_path / "out"),
            ],
        )

        assert rc == 2
        assert "config" not in captured


@pytest.mark.e2e
@pytest.mark.slow
def test_short_fake_soak_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A few seconds of fake-mode soak boots the worker, mints jobs, and writes its manifest.

    Fake mode disables the stats export (its children fabricate durations), so the run reports that it
    cannot be scored offline rather than producing a duty summary; the scoring path itself is covered by
    the session-duty tests. Marked ``slow``: it spawns the real worker process tree, and the soak period
    has to outlast that boot for any job to complete.
    """
    out_dir = tmp_path / "out"

    rc = main(
        [
            "soak",
            "--minutes",
            "0.25",
            "--process-mode",
            "fake",
            "--no-loras",
            "--ignore-bridge-data",
            "--label",
            "smoke",
            "--out",
            str(out_dir),
        ],
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "Soak finished" in output
    manifest = json.loads((out_dir / "soak.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "smoke"
    assert manifest["process_mode"] == "fake"
    assert manifest["jobs_completed"] >= 1
    assert manifest["jobs_faulted"] == 0
