"""Unit tests for the pure feature-readiness logic that decides which gated features the worker offers.

The module is a pure function of plain values, so these assert the readiness state machine directly: a
feature is offered only when it is enabled with its models on disk; it is withheld while those models are
still downloading; and an unknown presence (no download process yet) never withholds it.
"""

from __future__ import annotations

from horde_worker_regen.process_management.feature_readiness import (
    FeatureInputs,
    FeatureReadinessState,
    GatedFeature,
    build_feature_readiness,
    is_offered,
)


def test_enabled_with_models_present_is_offered() -> None:
    """An enabled feature whose models are on disk is advertised to the horde."""
    readiness = build_feature_readiness({GatedFeature.CONTROLNET: FeatureInputs(enabled=True, present=True)})
    assert readiness[0].state is FeatureReadinessState.OFFERED
    assert readiness[0].offered is True


def test_enabled_but_downloading_is_withheld() -> None:
    """An enabled feature whose models are confirmed absent (still downloading) is withheld."""
    readiness = build_feature_readiness({GatedFeature.CONTROLNET: FeatureInputs(enabled=True, present=False)})
    assert readiness[0].state is FeatureReadinessState.WAITING
    assert readiness[0].offered is False
    assert "downloading" in readiness[0].detail


def test_unknown_presence_never_withholds() -> None:
    """A None presence (no download process / not yet reported) advertises, mirroring image availability."""
    readiness = build_feature_readiness({GatedFeature.POST_PROCESSING: FeatureInputs(enabled=True, present=None)})
    assert readiness[0].state is FeatureReadinessState.OFFERED
    assert readiness[0].offered is True


def test_disabled_feature_with_deps_present_reads_as_off() -> None:
    """A feature that is simply not enabled (deps fine) shows as disabled, not as a deps problem."""
    readiness = build_feature_readiness(
        {GatedFeature.CONTROLNET: FeatureInputs(enabled=False, present=None, deps_available=True)},
    )
    assert readiness[0].state is FeatureReadinessState.DISABLED
    assert readiness[0].offered is False


def test_missing_deps_surfaces_the_install_hint() -> None:
    """A disabled feature whose packages are absent reads as missing-deps and carries the install hint."""
    readiness = build_feature_readiness(
        {
            GatedFeature.POST_PROCESSING: FeatureInputs(
                enabled=False,
                present=None,
                deps_available=False,
                deps_hint="install horde-worker-reGen[post-processing]",
            ),
        },
    )
    assert readiness[0].state is FeatureReadinessState.MISSING_DEPS
    assert readiness[0].offered is False
    assert "post-processing" in readiness[0].detail


def test_build_preserves_feature_order_and_subset() -> None:
    """Readiness is returned in the GatedFeature declaration order, for exactly the supplied features."""
    readiness = build_feature_readiness(
        {
            GatedFeature.POST_PROCESSING: FeatureInputs(enabled=True, present=True),
            GatedFeature.CONTROLNET: FeatureInputs(enabled=True, present=True),
        },
    )
    assert [entry.feature for entry in readiness] == [GatedFeature.CONTROLNET, GatedFeature.POST_PROCESSING]


def test_is_offered_reads_the_tracked_feature_and_defaults_open() -> None:
    """``is_offered`` returns a tracked feature's decision, and defaults to True for an untracked one."""
    readiness = build_feature_readiness({GatedFeature.CONTROLNET: FeatureInputs(enabled=True, present=False)})
    assert is_offered(readiness, GatedFeature.CONTROLNET) is False
    # SDXL-ControlNet was not built into this readiness set, so it is not gated (default open).
    assert is_offered(readiness, GatedFeature.SDXL_CONTROLNET) is True
