"""Parent-side, torch-free model of which gated features the worker can actually serve right now.

A feature is offered to the Horde only when both halves of its readiness hold: the Python packages that
back it are installed (the *deps* half) and the models/annotators it needs are present on disk (the
*presence* half). The deps half is enforced upstream by
[`coerce_bridge_data_to_capabilities`][horde_worker_regen.capabilities.coerce_bridge_data_to_capabilities],
which turns the opt-in flag off when the packages are missing; the presence half is reported up from the
download process (the only torch-free on-disk authority) and held in
[`ModelAvailability`][horde_worker_regen.process_management.models.model_availability.ModelAvailability].

This module is a pure function of plain values: a caller injects, per feature, the opt-in flag, whether
the deps are installed, and the on-disk presence, and it returns the readiness. Keeping it free of
hordelib/torch imports lets the orchestrator gate pops on it and the snapshot builder render it without
dragging a backend into the parent, and makes it trivially unit-testable.

Presence is tri-state: ``True`` (on disk), ``False`` (confirmed absent or still downloading), or ``None``
(not yet reported, e.g. no download process). An unknown presence never withholds a feature, mirroring
image-model availability: a worker that pre-downloads everything keeps its long-standing behaviour and is
never gated before the download process has had its say.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel


class GatedFeature(enum.StrEnum):
    """A worker feature whose Horde offer is withheld until its models/annotators are present on disk."""

    CONTROLNET = "controlnet"
    SDXL_CONTROLNET = "sdxl_controlnet"
    POST_PROCESSING = "post_processing"


class FeatureReadinessState(enum.StrEnum):
    """The serving state of a gated feature, as the operator would understand it."""

    OFFERED = "offered"
    """Advertised to the Horde now: deps installed and models present (or presence not yet reported)."""
    WAITING = "waiting"
    """Enabled with deps installed, but its models/annotators are still downloading, so it is withheld."""
    MISSING_DEPS = "missing_deps"
    """The backing packages are not installed, so the feature cannot run and is not advertised."""
    DISABLED = "disabled"
    """Not enabled in the worker config."""
    FAILED = "failed"
    """Enabled with models present, but a runtime verify failed (e.g. annotators download but do not run),
    so the feature is disabled until the operator intervenes. Distinct from WAITING, which recovers itself."""


CONTROLNET_ANNOTATOR_FAILED_DETAIL = "annotators failed to load; ControlNet disabled; restart the worker to retry"
"""Operator-facing reason for a ControlNet feature withheld by a permanent annotator verify failure."""


_FEATURE_LABELS: dict[GatedFeature, str] = {
    GatedFeature.CONTROLNET: "ControlNet",
    GatedFeature.SDXL_CONTROLNET: "SDXL ControlNet",
    GatedFeature.POST_PROCESSING: "Post-processing",
}


@dataclass(frozen=True)
class FeatureInputs:
    """The facts that decide one feature's readiness, injected by the caller.

    ``enabled`` is the post-coercion opt-in flag, so when it is True the deps are necessarily installed
    (coercion would otherwise have turned it off). ``deps_available`` and ``deps_hint`` only refine the
    label shown when the feature is *not* enabled, distinguishing "you lack the packages" from "you did
    not turn it on"; they never change whether the feature is offered.
    """

    enabled: bool
    present: bool | None
    deps_available: bool = True
    deps_hint: str = ""
    failed: bool = False
    """A runtime verify permanently failed (the models are present but do not run). Overrides presence to
    withhold the feature in a distinct FAILED state until the operator acts. Defaults False."""
    failed_detail: str = ""
    """Operator-facing reason shown in the FAILED state (e.g. which verify failed and what to do)."""


class FeatureReadiness(BaseModel):
    """One gated feature's readiness: its state and a short human detail (a hint when blocked).

    Serialized into the worker snapshot so the TUI renders the same readiness the pop gate enforces;
    there is a single source of truth for "is this feature offered?".
    """

    feature: GatedFeature
    label: str
    state: FeatureReadinessState
    detail: str = ""

    @property
    def offered(self) -> bool:
        """Whether this feature is advertised to the Horde right now (the pop gate)."""
        return self.state is FeatureReadinessState.OFFERED


def _readiness_for(feature: GatedFeature, inputs: FeatureInputs) -> FeatureReadiness:
    """Resolve one feature's :class:`FeatureReadiness` from its injected inputs."""
    label = _FEATURE_LABELS[feature]
    if not inputs.enabled:
        # A disabled flag is ambiguous between "not opted in" and "opted in but coerced off for missing
        # deps"; the live deps probe disambiguates without needing the pre-coercion intent.
        if not inputs.deps_available:
            detail = inputs.deps_hint or "required packages are not installed"
            return FeatureReadiness(
                feature=feature,
                label=label,
                state=FeatureReadinessState.MISSING_DEPS,
                detail=detail,
            )
        return FeatureReadiness(
            feature=feature,
            label=label,
            state=FeatureReadinessState.DISABLED,
            detail="not enabled in config",
        )
    # A runtime verify that permanently failed disables the feature outright, regardless of on-disk
    # presence: the models are there but do not run, so advertising would only fault every job.
    if inputs.failed:
        return FeatureReadiness(
            feature=feature,
            label=label,
            state=FeatureReadinessState.FAILED,
            detail=inputs.failed_detail or "verification failed; disabled until restart",
        )
    # Enabled implies the deps are present (coercion would have disabled it otherwise). Withhold only
    # when the models are confirmed not on disk; an unknown presence advertises, so the worker is never
    # gated before the download process reports.
    if inputs.present is False:
        return FeatureReadiness(
            feature=feature,
            label=label,
            state=FeatureReadinessState.WAITING,
            detail="models still downloading",
        )
    return FeatureReadiness(feature=feature, label=label, state=FeatureReadinessState.OFFERED, detail="")


def build_feature_readiness(inputs: Mapping[GatedFeature, FeatureInputs]) -> tuple[FeatureReadiness, ...]:
    """Compute readiness for each supplied gated feature, in a stable display order."""
    return tuple(_readiness_for(feature, inputs[feature]) for feature in GatedFeature if feature in inputs)


def is_offered(readiness: Iterable[FeatureReadiness], feature: GatedFeature) -> bool:
    """Whether *feature* is offered, defaulting to True (do not gate) when it is not tracked."""
    for entry in readiness:
        if entry.feature is feature:
            return entry.offered
    return True
