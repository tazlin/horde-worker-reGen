"""Unit tests for capability detection and the bridge-data coercion it drives.

The feature-availability probe (``capabilities._available_features``) is monkeypatched so these tests
need none of the optional packages (rembg / onnxruntime) installed and assert the coercion behaviour
directly: a worker must never advertise a feature whose backend packages are absent.
"""

from types import SimpleNamespace

import pytest
from hordelib.feature_impact import FEATURE_KIND

from horde_worker_regen import capabilities
from horde_worker_regen.process_management.alchemy_popper import expand_offered_forms

_ALL_FEATURES = frozenset(FEATURE_KIND)
_NO_FEATURES: frozenset[FEATURE_KIND] = frozenset()
_ONLY_PURE_TORCH = frozenset(
    f for f in FEATURE_KIND if f not in {FEATURE_KIND.strip_background, FEATURE_KIND.controlnet}
)


def _patch_features(monkeypatch: pytest.MonkeyPatch, available: frozenset[FEATURE_KIND]) -> None:
    monkeypatch.setattr(capabilities, "_available_features", lambda: available)


def _bridge_data(**overrides: object) -> SimpleNamespace:
    bd = SimpleNamespace(
        allow_post_processing=True,
        allow_controlnet=True,
        allow_sdxl_controlnet=True,
        dry_run_skip_inference=False,
    )
    for key, value in overrides.items():
        setattr(bd, key, value)
    return bd


def test_no_coercion_when_all_features_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every feature installed, no flag is touched and nothing is reported."""
    _patch_features(monkeypatch, _ALL_FEATURES)
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert coercions == []
    assert bd.allow_post_processing is True
    assert bd.allow_controlnet is True
    assert bd.allow_sdxl_controlnet is True


def test_post_processing_coerced_off_when_strip_background_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing rembg coerces the whole post-processing bucket off but leaves controlnet alone."""
    # Only strip_background is missing; the controlnet annotators are present.
    _patch_features(monkeypatch, _ALL_FEATURES - {FEATURE_KIND.strip_background})
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.allow_post_processing is False
    # The atomic post-processing bucket is independent of controlnet, which stays on.
    assert bd.allow_controlnet is True
    assert len(coercions) == 1
    assert "allow_post_processing" in coercions[0]
    assert "post-processing" in coercions[0]


def test_controlnet_coerced_off_when_annotators_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing onnxruntime coerces both controlnet flags off but leaves post-processing alone."""
    _patch_features(monkeypatch, _ALL_FEATURES - {FEATURE_KIND.controlnet})
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.allow_controlnet is False
    assert bd.allow_sdxl_controlnet is False
    assert bd.allow_post_processing is True
    assert len(coercions) == 1
    assert "controlnet" in coercions[0]


def test_both_buckets_coerced_off_on_lean_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lean base install (neither optional package) coerces both buckets off."""
    _patch_features(monkeypatch, _ONLY_PURE_TORCH)
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.allow_post_processing is False
    assert bd.allow_controlnet is False
    assert bd.allow_sdxl_controlnet is False
    assert len(coercions) == 2


def test_no_coercion_when_flags_already_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent packages but already-off flags report nothing (idempotent on the hot-reload path)."""
    _patch_features(monkeypatch, _NO_FEATURES)
    bd = _bridge_data(allow_post_processing=False, allow_controlnet=False, allow_sdxl_controlnet=False)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    # Nothing to flip, so nothing is reported even though the packages are absent.
    assert coercions == []


def test_dry_run_skips_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run leaves flags untouched since it never runs real inference."""
    _patch_features(monkeypatch, _NO_FEATURES)
    bd = _bridge_data(dry_run_skip_inference=True)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    # Dry-run does not run real inference, so capabilities are irrelevant and flags are untouched.
    assert coercions == []
    assert bd.allow_post_processing is True
    assert bd.allow_controlnet is True


def test_helpers_reflect_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The availability helpers track the probe result both ways."""
    _patch_features(monkeypatch, _NO_FEATURES)
    assert capabilities.strip_background_available() is False
    assert capabilities.controlnet_available() is False

    _patch_features(monkeypatch, _ALL_FEATURES)
    assert capabilities.strip_background_available() is True
    assert capabilities.controlnet_available() is True


def test_expand_offered_forms_drops_strip_background_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alchemy drops only strip_background on a lean install, keeping the pure-torch forms."""
    _patch_features(monkeypatch, _ONLY_PURE_TORCH)
    bd = SimpleNamespace(forms=["post-process"], alchemy_caption_enabled=False)

    offered = expand_offered_forms(bd)  # type: ignore[arg-type]

    assert "strip_background" not in offered
    # The pure-torch graph forms remain on offer.
    assert len(offered) > 0


def test_expand_offered_forms_keeps_strip_background_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alchemy offers strip_background when rembg is installed."""
    _patch_features(monkeypatch, _ALL_FEATURES)
    bd = SimpleNamespace(forms=["post-process"], alchemy_caption_enabled=False)

    offered = expand_offered_forms(bd)  # type: ignore[arg-type]

    assert "strip_background" in offered
