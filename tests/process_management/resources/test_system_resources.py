"""Tests for backend-agnostic hardware detection in ``SystemResources.detect``.

These assert that device discovery flows through the out-of-process accelerator probe (which itself uses
hordelib's backend-agnostic ``enumerate_accelerators``, covering every ComfyUI backend) rather than
``torch.cuda`` directly, so non-NVIDIA backends - including a CPU-only machine - still yield a populated
device map. The probe is mocked so the tests need no GPU, no network, and no subprocess: enumeration runs
out-of-process precisely to keep the orchestrator torch-free (see ``test_orchestrator_torch_free.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator

import pytest

import horde_worker_regen.utils.accelerator_probe as accelerator_probe_module
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.utils.accelerator_probe import (
    _RESULT_PREFIX,
    MAX_CONTEXT_PROBE_RUNS,
    ContextReadingVerdict,
    ProbedAccelerator,
    _first_context_overhead_mb,
    probe_accelerators,
    probe_session_accelerators,
    select_context_readings,
)

_MB = 1024 * 1024


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ProbedAccelerator]]:
    """Patch ``probe_accelerators`` to return a mutable list, so ``detect`` runs without a real subprocess."""
    accelerators: list[ProbedAccelerator] = []
    monkeypatch.setattr(
        accelerator_probe_module,
        "probe_accelerators",
        lambda **_kwargs: list(accelerators),
    )
    yield accelerators


def test_detect_maps_multiple_devices(fake_probe: list[ProbedAccelerator]) -> None:
    """Every probed accelerator becomes a TorchDeviceInfo keyed by its index, MB converted to bytes."""
    fake_probe.extend(
        [
            ProbedAccelerator(index=0, name="NVIDIA RTX 4090", total_vram_mb=24564),
            ProbedAccelerator(index=1, name="NVIDIA RTX 3090", total_vram_mb=24576),
        ],
    )

    resources = SystemResources.detect()

    assert set(resources.device_map.root) == {0, 1}
    assert resources.device_map.root[0].device_name == "NVIDIA RTX 4090"
    # MB are converted back to bytes for the TorchDeviceInfo contract.
    assert resources.device_map.root[0].total_memory == 24564 * _MB
    assert resources.device_map.root[1].device_index == 1


def test_detect_yields_cpu_pseudo_device_without_gpu(fake_probe: list[ProbedAccelerator]) -> None:
    """A CPU-only machine must still produce a device, where a bare torch.cuda loop would yield none."""
    fake_probe.append(ProbedAccelerator(index=0, name="CPU", total_vram_mb=65455))

    resources = SystemResources.detect()

    assert list(resources.device_map.root) == [0]
    assert resources.device_map.root[0].device_name == "CPU"
    assert resources.total_ram_bytes > 0


def test_detect_carries_overhead_and_marginal(fake_probe: list[ProbedAccelerator]) -> None:
    """detect() surfaces the probe's first-context overhead and the (smaller) per-additional-context marginal.

    These are the two figures the streaming forecast needs: the overhead sizes free-if-alone, the marginal
    sizes free-after-model-evict. Across multiple devices it takes the max of each (single-GPU is the norm;
    the max is the conservative choice).
    """
    fake_probe.extend(
        [
            ProbedAccelerator(
                index=0,
                name="GPU0",
                total_vram_mb=24564,
                runtime_overhead_mb=4112,
                marginal_overhead_mb=455,
            ),
            ProbedAccelerator(
                index=1,
                name="GPU1",
                total_vram_mb=24564,
                runtime_overhead_mb=3000,
                marginal_overhead_mb=480,
            ),
        ],
    )

    resources = SystemResources.detect()

    assert resources.per_process_overhead_mb == 4112
    assert resources.marginal_process_overhead_mb == 480


def test_detect_marginal_defaults_zero_for_old_probe(fake_probe: list[ProbedAccelerator]) -> None:
    """A probe result without the marginal (older serialisation) leaves it 0 -> forecast falls back."""
    fake_probe.append(ProbedAccelerator(index=0, name="GPU0", total_vram_mb=16375, runtime_overhead_mb=1288))

    resources = SystemResources.detect()

    assert resources.per_process_overhead_mb == 1288
    assert resources.marginal_process_overhead_mb == 0


@pytest.mark.gpu
def test_probe_measures_overhead_and_marginal_on_real_device() -> None:
    """On real hardware the session probe reports a positive first-context cost that agrees with the marginal.

    Both figures are device-wide deltas around one new context, so a clean reading gives nearly equal values.
    The session probe re-measures a reading whose figures disagree, so the reading it returns is consistent
    unless a foreign tenant changed its allocation during every run. A card whose marginal could not be
    measured is not compared.
    """
    accelerators = probe_session_accelerators().accelerators
    assert accelerators, "probe found no accelerators on a GPU box"
    primary = accelerators[0]
    assert primary.total_vram_mb > 0
    assert primary.runtime_overhead_mb > 0, "a fresh process must show some context/runtime VRAM"
    assert primary.marginal_overhead_mb >= 0

    if sys.platform != "win32":
        assert primary.marginal_overhead_mb > 0, (
            "on Linux the device-wide second-context delta must be measurable; "
            "a 0 here means the marginal measurement regressed"
        )

    verdict = select_context_readings([[primary]])[0].verdict
    assert verdict is not ContextReadingVerdict.INCONSISTENT, (
        "the first-context and marginal figures of the selected reading must agree "
        f"(got first={primary.runtime_overhead_mb} marginal={primary.marginal_overhead_mb})"
    )


class TestFirstContextOverheadExcludesTheDeviceBaseline:
    """The first-context overhead is the context's own cost, not everything the card already held.

    A device-wide *used* reading taken after the probe's context materialises contains every other tenant on
    the card: a desktop compositor, a browser, another application. Charged whole as the worker's per-process
    overhead, that figure measures the host's business rather than the worker's, and applied to a small card
    it removes more budget than the models the card is being asked to serve. Netting the pre-context baseline
    out leaves the term its name claims.
    """

    _BASELINE_MB = 3600
    """A desktop host's pre-existing device usage: nothing to do with the worker."""
    _CONTEXT_COST_MB = 278
    """What the context itself actually costs."""

    def _payload(self, **overrides: object) -> str:
        """One probe result line carrying a synthetic before/after pair."""
        entry: dict[str, object] = {
            "index": 0,
            "name": "GPU0",
            "total_vram_mb": 8192,
            "kind": "cuda",
            "context_device_used_mb": self._BASELINE_MB + self._CONTEXT_COST_MB,
            "device_baseline_mb": self._BASELINE_MB,
            "marginal_overhead_mb": 240,
        }
        entry.update(overrides)
        return _RESULT_PREFIX + json.dumps([entry])

    def _run_probe(self, monkeypatch: pytest.MonkeyPatch, payload: str) -> list[ProbedAccelerator]:
        """Run ``probe_accelerators`` against a canned child result, with no subprocess and no GPU."""
        monkeypatch.setattr(
            accelerator_probe_module.subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr=""),
        )
        return probe_accelerators()

    def test_pre_existing_baseline_is_not_charged_to_the_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reported overhead is the delta, so a large synthetic baseline leaves no trace in it."""
        accelerators = self._run_probe(monkeypatch, self._payload())
        assert len(accelerators) == 1
        assert accelerators[0].runtime_overhead_mb == self._CONTEXT_COST_MB
        # The raw readings survive for diagnostics; only the derived term excludes the baseline.
        assert accelerators[0].device_baseline_mb == self._BASELINE_MB
        assert accelerators[0].context_device_used_mb == self._BASELINE_MB + self._CONTEXT_COST_MB

    def test_overhead_stays_a_small_fraction_of_a_small_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the delta: a desktop host's baseline no longer eats an 8 GB card's budget."""
        accelerators = self._run_probe(monkeypatch, self._payload())
        overhead_mb = accelerators[0].runtime_overhead_mb
        assert overhead_mb < 0.1 * accelerators[0].total_vram_mb, (
            f"a first-context overhead of {overhead_mb} MB on an 8 GB card is a device-wide reading, not a "
            "context cost"
        )

    def test_unmeasurable_baseline_keeps_the_raw_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a baseline (a non-NVIDIA backend) the raw reading stands: over-counting, never under."""
        accelerators = self._run_probe(monkeypatch, self._payload(device_baseline_mb=None))
        assert accelerators[0].runtime_overhead_mb == self._BASELINE_MB + self._CONTEXT_COST_MB

    def test_inverted_samples_never_credit_negative_overhead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tenant releasing memory between the two samples yields 0, never a negative charge."""
        accelerators = self._run_probe(monkeypatch, self._payload(context_device_used_mb=100))
        assert accelerators[0].runtime_overhead_mb == 0

    def test_payload_without_the_pair_keeps_its_reported_overhead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A result predating the before/after pair is passed through rather than zeroed."""
        accelerators = self._run_probe(
            monkeypatch,
            _RESULT_PREFIX
            + json.dumps([{"index": 0, "name": "GPU0", "total_vram_mb": 8192, "runtime_overhead_mb": 1288}]),
        )
        assert accelerators[0].runtime_overhead_mb == 1288

    def test_derivation_is_a_plain_subtraction(self) -> None:
        """The arithmetic itself, independent of any probe plumbing."""
        assert _first_context_overhead_mb(context_device_used_mb=3878, device_baseline_mb=3600) == 278
        assert _first_context_overhead_mb(context_device_used_mb=3878, device_baseline_mb=None) == 3878


class TestPerDeviceProbeFigures:
    """The probe measures every card, so ``detect`` carries each card's own overhead alongside the maxima."""

    def _payload(self, entries: list[dict[str, object]]) -> str:
        """One probe result line carrying several devices' before/after pairs."""
        return _RESULT_PREFIX + json.dumps(entries)

    def _run_probe(self, monkeypatch: pytest.MonkeyPatch, payload: str) -> list[ProbedAccelerator]:
        """Run ``probe_accelerators`` against a canned child result, with no subprocess and no GPU."""
        monkeypatch.setattr(accelerator_probe_module, "_nvml_device_count", lambda: 2)
        monkeypatch.setattr(
            accelerator_probe_module.subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr=""),
        )
        return probe_accelerators()

    def test_each_device_keeps_its_own_derived_overhead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two cards with different baselines and different context costs stay distinct through parsing."""
        accelerators = self._run_probe(
            monkeypatch,
            self._payload(
                [
                    {
                        "index": 0,
                        "name": "GPU0",
                        "total_vram_mb": 24564,
                        "context_device_used_mb": 3600 + 278,
                        "device_baseline_mb": 3600,
                        "marginal_overhead_mb": 240,
                    },
                    {
                        "index": 1,
                        "name": "GPU1",
                        "total_vram_mb": 8192,
                        "context_device_used_mb": 120 + 412,
                        "device_baseline_mb": 120,
                        "marginal_overhead_mb": 190,
                    },
                ],
            ),
        )
        assert [a.runtime_overhead_mb for a in accelerators] == [278, 412]
        assert [a.marginal_overhead_mb for a in accelerators] == [240, 190]

    def test_an_unmeasurable_card_reports_no_figures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card the child could not read leaves both figures at 0, which consumers treat as unmeasured."""
        accelerators = self._run_probe(
            monkeypatch,
            self._payload(
                [
                    {
                        "index": 0,
                        "name": "GPU0",
                        "total_vram_mb": 24564,
                        "context_device_used_mb": 3600 + 278,
                        "device_baseline_mb": 3600,
                        "marginal_overhead_mb": 240,
                    },
                    {
                        "index": 1,
                        "name": "GPU1",
                        "total_vram_mb": 8192,
                        "context_device_used_mb": 0,
                        "device_baseline_mb": None,
                        "marginal_overhead_mb": 0,
                        "marginal_note": "no device-wide reading for this card",
                    },
                ],
            ),
        )
        assert accelerators[1].runtime_overhead_mb == 0
        assert accelerators[1].marginal_overhead_mb == 0

    def test_single_device_payload_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One card parses to exactly the figures it did before the probe walked every device."""
        accelerators = self._run_probe(
            monkeypatch,
            self._payload(
                [
                    {
                        "index": 0,
                        "name": "GPU0",
                        "total_vram_mb": 16375,
                        "context_device_used_mb": 3600 + 1288,
                        "device_baseline_mb": 3600,
                        "marginal_overhead_mb": 455,
                    },
                ],
            ),
        )
        assert len(accelerators) == 1
        assert accelerators[0].runtime_overhead_mb == 1288
        assert accelerators[0].marginal_overhead_mb == 455

    def test_timeout_scales_with_the_device_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A four-card host gets four times the per-card budget; a single-card host is bounded as before."""
        seen: list[float] = []

        def _capture(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.append(float(kwargs["timeout"]))  # type: ignore[arg-type]
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        monkeypatch.setattr(accelerator_probe_module.subprocess, "run", _capture)
        monkeypatch.setattr(accelerator_probe_module, "_nvml_device_count", lambda: 4)
        probe_accelerators(timeout_seconds=120.0)
        monkeypatch.setattr(accelerator_probe_module, "_nvml_device_count", lambda: 1)
        probe_accelerators(timeout_seconds=120.0)
        monkeypatch.setattr(accelerator_probe_module, "_nvml_device_count", lambda: 0)
        probe_accelerators(timeout_seconds=120.0)
        assert seen == [480.0, 120.0, 120.0]


class TestDetectCarriesPerCardFigures:
    """``SystemResources.detect`` includes both the per-card figures and the worker-wide maxima."""

    def test_per_card_map_and_maxima(self, fake_probe: list[ProbedAccelerator]) -> None:
        """The maxima remain the worker-wide reduction; the maps let a card-scoped forecast price its card."""
        fake_probe.extend(
            [
                ProbedAccelerator(
                    index=0,
                    name="GPU0",
                    total_vram_mb=24564,
                    runtime_overhead_mb=1288,
                    marginal_overhead_mb=455,
                ),
                ProbedAccelerator(
                    index=1,
                    name="GPU1",
                    total_vram_mb=8192,
                    runtime_overhead_mb=4112,
                    marginal_overhead_mb=480,
                ),
            ],
        )
        resources = SystemResources.detect()
        assert resources.per_process_overhead_mb_by_device == {0: 1288, 1: 4112}
        assert resources.marginal_process_overhead_mb_by_device == {0: 455, 1: 480}
        assert resources.per_process_overhead_mb == 4112
        assert resources.marginal_process_overhead_mb == 480

    def test_unmeasured_card_is_absent_from_the_maps(self, fake_probe: list[ProbedAccelerator]) -> None:
        """A card with no measurement contributes no key, so its consumers fall back to the maxima."""
        fake_probe.extend(
            [
                ProbedAccelerator(
                    index=0,
                    name="GPU0",
                    total_vram_mb=24564,
                    runtime_overhead_mb=1288,
                    marginal_overhead_mb=455,
                ),
                ProbedAccelerator(index=1, name="GPU1", total_vram_mb=8192),
            ],
        )
        resources = SystemResources.detect()
        assert resources.per_process_overhead_mb_by_device == {0: 1288}
        assert resources.marginal_process_overhead_mb_by_device == {0: 455}


def _context_reading(
    index: int,
    *,
    first_mb: int,
    marginal_mb: int,
    baseline_mb: int = 3600,
) -> ProbedAccelerator:
    """Return one card's probe reading built from its two figures, with the raw NVML readings behind them."""
    after_context_mb = baseline_mb + first_mb
    return ProbedAccelerator(
        index=index,
        name=f"GPU{index}",
        total_vram_mb=24564,
        device_baseline_mb=baseline_mb,
        context_device_used_mb=after_context_mb,
        after_holder_device_used_mb=after_context_mb + marginal_mb,
        runtime_overhead_mb=first_mb,
        marginal_overhead_mb=marginal_mb,
    )


class TestContextReadingSelection:
    """A foreign allocation inside the probe's window inflates one figure, and a re-measure replaces it.

    On a clean launch the first-context and marginal figures read nearly equal, since both are the cost of
    one context. A reading where they disagree by more than the tolerance had another program allocate or
    release on the card between two of the readings.
    """

    def test_consistent_first_run_is_accepted(self) -> None:
        """Agreeing figures need no second run."""
        (selection,) = select_context_readings([[_context_reading(0, first_mb=487, marginal_mb=488)]])
        assert selection.verdict is ContextReadingVerdict.CONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 487
        assert selection.accelerator.marginal_overhead_mb == 488

    def test_inflated_first_run_is_replaced_by_a_consistent_one(self) -> None:
        """The first run that agrees supplies both figures."""
        (selection,) = select_context_readings(
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=472, marginal_mb=506)],
            ],
        )
        assert selection.verdict is ContextReadingVerdict.CONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 472
        assert selection.accelerator.marginal_overhead_mb == 506

    def test_no_consistent_run_takes_the_median_first_context(self) -> None:
        """With no agreeing run, the median first-context figure and the marginal from the same run stand."""
        (selection,) = select_context_readings(
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=900, marginal_mb=300)],
                [_context_reading(0, first_mb=1000, marginal_mb=200)],
            ],
        )
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 1000
        assert selection.accelerator.marginal_overhead_mb == 200
        assert len(selection.readings) == 3

    def test_one_deflated_run_does_not_beat_two_inflated_ones(self) -> None:
        """A release inside one run's window lowers that run's figure, and work that cannot fit is then admitted."""
        (selection,) = select_context_readings(
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=120, marginal_mb=480)],
                [_context_reading(0, first_mb=900, marginal_mb=300)],
            ],
        )
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 900

    def test_a_first_figure_clamped_to_zero_is_passed_over(self) -> None:
        """A first figure of 0 is the clamp, so it does not count toward the median while another reading exists."""
        (selection,) = select_context_readings(
            [
                [_context_reading(0, first_mb=0, marginal_mb=480)],
                [_context_reading(0, first_mb=0, marginal_mb=470)],
                [_context_reading(0, first_mb=900, marginal_mb=300)],
            ],
        )
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 900
        assert selection.accelerator.marginal_overhead_mb == 300

    def test_two_disagreeing_runs_take_the_larger_first_context(self) -> None:
        """With an even count, as after a failed re-measure, the larger of the middle two stands."""
        (selection,) = select_context_readings(
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=900, marginal_mb=300)],
            ],
        )
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT
        assert selection.accelerator.runtime_overhead_mb == 1161

    def test_a_zero_first_figure_is_measured_again(self) -> None:
        """A first figure clamped to 0 disagrees with a measured marginal, so the card is suspect."""
        (selection,) = select_context_readings([[_context_reading(0, first_mb=0, marginal_mb=480)]])
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT

    def test_unmeasured_marginal_is_never_suspect(self) -> None:
        """A card with the 0 marginal fallback keeps its single reading and is not measured again."""
        unmeasured = _context_reading(0, first_mb=1161, marginal_mb=0).model_copy(
            update={"after_holder_device_used_mb": None, "marginal_note": "holder never signalled ready"},
        )
        (selection,) = select_context_readings([[unmeasured]])
        assert selection.verdict is ContextReadingVerdict.UNCHECKED
        assert selection.accelerator.runtime_overhead_mb == 1161
        assert selection.accelerator.marginal_overhead_mb == 0

    def test_only_the_suspect_card_takes_a_later_run(self) -> None:
        """A card that agreed on the first run keeps that run, whatever later runs read."""
        selections = select_context_readings(
            [
                [
                    _context_reading(0, first_mb=487, marginal_mb=488),
                    _context_reading(1, first_mb=1161, marginal_mb=426),
                ],
                [
                    _context_reading(0, first_mb=470, marginal_mb=520),
                    _context_reading(1, first_mb=472, marginal_mb=506),
                ],
            ],
        )
        by_index = {selection.accelerator.index: selection for selection in selections}
        assert by_index[0].verdict is ContextReadingVerdict.CONSISTENT
        assert (by_index[0].accelerator.runtime_overhead_mb, by_index[0].accelerator.marginal_overhead_mb) == (
            487,
            488,
        )
        assert by_index[1].verdict is ContextReadingVerdict.CONSISTENT
        assert (by_index[1].accelerator.runtime_overhead_mb, by_index[1].accelerator.marginal_overhead_mb) == (
            472,
            506,
        )

    def test_small_figures_use_the_absolute_floor(self) -> None:
        """On a small card a few tens of MB of noise stays inside the floor."""
        (selection,) = select_context_readings([[_context_reading(0, first_mb=150, marginal_mb=200)]])
        assert selection.verdict is ContextReadingVerdict.CONSISTENT

    def test_child_payload_includes_the_after_holder_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The third raw reading reaches the parent, so the startup line can show all three."""
        payload = _RESULT_PREFIX + json.dumps(
            [
                {
                    "index": 0,
                    "name": "GPU0",
                    "total_vram_mb": 24564,
                    "context_device_used_mb": 3600 + 487,
                    "device_baseline_mb": 3600,
                    "after_holder_device_used_mb": 3600 + 487 + 488,
                    "marginal_overhead_mb": 488,
                },
            ],
        )
        monkeypatch.setattr(accelerator_probe_module, "_nvml_device_count", lambda: 1)
        monkeypatch.setattr(
            accelerator_probe_module.subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr=""),
        )
        (accelerator,) = probe_accelerators()
        assert accelerator.after_holder_device_used_mb == 3600 + 487 + 488


class TestSessionProbeRetry:
    """``probe_session_accelerators`` repeats the probe only while a card reads inconsistent, at most three times."""

    def _install_runs(self, monkeypatch: pytest.MonkeyPatch, runs: list[list[ProbedAccelerator]]) -> list[int]:
        """Substitute the single probe run with canned runs, returning a list that records each call."""
        calls: list[int] = []

        def _next_run(**_kwargs: object) -> list[ProbedAccelerator]:
            calls.append(len(calls))
            return list(runs[min(len(calls) - 1, len(runs) - 1)])

        monkeypatch.setattr(accelerator_probe_module, "probe_accelerators", _next_run)
        return calls

    def test_stops_at_the_first_consistent_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A consistent second run ends the loop."""
        calls = self._install_runs(
            monkeypatch,
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=472, marginal_mb=506)],
                [_context_reading(0, first_mb=480, marginal_mb=490)],
            ],
        )
        (accelerator,) = probe_session_accelerators().accelerators
        assert len(calls) == 2
        assert accelerator.runtime_overhead_mb == 472

    def test_a_consistent_first_run_is_the_only_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The usual launch pays for one probe."""
        calls = self._install_runs(monkeypatch, [[_context_reading(0, first_mb=487, marginal_mb=488)]])
        probe_session_accelerators()
        assert len(calls) == 1

    def test_never_exceeds_the_run_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card that never agrees costs at most three probes, then takes the median first-context run."""
        calls = self._install_runs(
            monkeypatch,
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=900, marginal_mb=300)],
                [_context_reading(0, first_mb=1000, marginal_mb=200)],
            ],
        )
        (accelerator,) = probe_session_accelerators().accelerators
        assert MAX_CONTEXT_PROBE_RUNS == 3
        assert len(calls) == MAX_CONTEXT_PROBE_RUNS
        assert accelerator.runtime_overhead_mb == 1000

    def test_logs_every_raw_reading_per_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The startup line includes the baseline, both later readings and both figures."""
        from loguru import logger

        self._install_runs(monkeypatch, [[_context_reading(0, first_mb=487, marginal_mb=488, baseline_mb=3600)]])
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
        try:
            probe_session_accelerators()
        finally:
            logger.remove(sink_id)
        (line,) = [message for message in messages if message.startswith("Accelerator 0 context cost")]
        for fragment in ("baseline 3600MB", "after context 4087MB", "after holder 4575MB", "first 487MB"):
            assert fragment in line
        assert "marginal 488MB" in line
        assert "1 probe run" in line

    def test_detect_uses_the_checked_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Startup detection takes the re-measured figures, never the contaminated first run."""
        self._install_runs(
            monkeypatch,
            [
                [_context_reading(0, first_mb=1161, marginal_mb=426)],
                [_context_reading(0, first_mb=472, marginal_mb=506)],
            ],
        )
        resources = SystemResources.detect()
        assert resources.per_process_overhead_mb_by_device == {0: 472}
        assert resources.marginal_process_overhead_mb_by_device == {0: 506}
        (selection,) = resources.context_readings
        assert selection.verdict is ContextReadingVerdict.CONSISTENT
        assert len(selection.readings) == 2


class TestSessionProbeKeepsTheSelection:
    """The session probe returns each card's verdict and every run's readings, so the dashboard can show them."""

    def _install_runs(self, monkeypatch: pytest.MonkeyPatch, runs: list[list[ProbedAccelerator]]) -> None:
        """Substitute the single probe run with canned runs, the last one repeated."""
        calls: list[int] = []

        def _next_run(**_kwargs: object) -> list[ProbedAccelerator]:
            calls.append(len(calls))
            return list(runs[min(len(calls) - 1, len(runs) - 1)])

        monkeypatch.setattr(accelerator_probe_module, "probe_accelerators", _next_run)

    def test_a_consistent_first_run_keeps_one_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One agreeing run gives a consistent verdict, a run count of one and that run's reading."""
        first = _context_reading(0, first_mb=487, marginal_mb=488)
        self._install_runs(monkeypatch, [[first]])

        probe = probe_session_accelerators()

        assert probe.run_count == 1
        (selection,) = probe.selections
        assert selection.verdict is ContextReadingVerdict.CONSISTENT
        assert selection.readings == [first]
        assert selection.accelerator == first
        assert probe.accelerators == [first]

    def test_a_replaced_run_keeps_both_readings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A disagreeing first run replaced by an agreeing second keeps both readings and picks the second."""
        inflated = _context_reading(0, first_mb=1161, marginal_mb=426)
        clean = _context_reading(0, first_mb=472, marginal_mb=506)
        self._install_runs(monkeypatch, [[inflated], [clean]])

        probe = probe_session_accelerators()

        assert probe.run_count == 2
        (selection,) = probe.selections
        assert selection.verdict is ContextReadingVerdict.CONSISTENT
        assert selection.readings == [inflated, clean]
        assert selection.accelerator == clean

    def test_a_card_that_never_agrees_keeps_every_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three disagreeing runs give an inconsistent verdict, all three readings and the median first figure."""
        runs = [
            [_context_reading(0, first_mb=1161, marginal_mb=426)],
            [_context_reading(0, first_mb=900, marginal_mb=300)],
            [_context_reading(0, first_mb=1000, marginal_mb=200)],
        ]
        self._install_runs(monkeypatch, runs)

        probe = probe_session_accelerators()

        assert probe.run_count == MAX_CONTEXT_PROBE_RUNS
        (selection,) = probe.selections
        assert selection.verdict is ContextReadingVerdict.INCONSISTENT
        assert [reading.runtime_overhead_mb for reading in selection.readings] == [1161, 900, 1000]
        assert selection.accelerator.runtime_overhead_mb == 1000

    def test_no_accelerator_gives_an_empty_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host without a card gives no selections and one run."""
        self._install_runs(monkeypatch, [[]])

        probe = probe_session_accelerators()

        assert probe.selections == []
        assert probe.accelerators == []
        assert probe.run_count == 1
