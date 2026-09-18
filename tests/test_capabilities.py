"""Unit tests for capability detection and the bridge-data coercion it drives.

ControlNet annotation and background removal both run on the out-of-venv image-utilities lane, so their
availability collapses to one probe (``capabilities.utilities_available``): is that lane provisioned and
enabled. The probe (and, for the coercion matrix, ``utilities_available`` itself) is monkeypatched so these
tests need no provisioned venv and assert the coercion behaviour directly: a worker must never advertise a
feature the utilities lane cannot serve.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen import capabilities, compute_mode
from horde_worker_regen.process_management.jobs.alchemy_popper import expand_offered_forms
from horde_worker_regen.process_management.lifecycle.horde_process import DEFAULT_CAPABILITIES, HordeProcessType
from horde_worker_regen.process_management.scheduling.workload_flow import (
    WorkloadKind,
    capabilities_for_workload,
    process_types_for_workloads,
)


def _patch_utilities(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    """Force the image-utilities availability probe (both the no-arg and config-aware call forms)."""
    monkeypatch.setattr(capabilities, "utilities_available", lambda bridge_data=None: available)


def _bridge_data(**overrides: object) -> SimpleNamespace:
    bd = SimpleNamespace(
        allow_post_processing=True,
        allow_controlnet=True,
        allow_sdxl_controlnet=True,
        enable_image_utilities=True,
        dry_run_skip_inference=False,
        dreamer=True,
        alchemist=False,
        scribe=False,
    )
    for key, value in overrides.items():
        setattr(bd, key, value)
    return bd


def test_no_coercion_when_utilities_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the utilities lane available, no flag is touched and nothing is reported."""
    _patch_utilities(monkeypatch, True)
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert coercions == []
    assert bd.allow_post_processing is True
    assert bd.allow_controlnet is True
    assert bd.allow_sdxl_controlnet is True


def test_both_buckets_coerced_off_when_utilities_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the utilities lane is unavailable, both the post-processing and controlnet buckets are dropped.

    The two features share one execution home now, so they are gated together (unlike the old independent
    rembg/onnxruntime probes): post-processing off because strip_background cannot run, controlnet off
    because annotation cannot run.
    """
    _patch_utilities(monkeypatch, False)
    bd = _bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.allow_post_processing is False
    assert bd.allow_controlnet is False
    assert bd.allow_sdxl_controlnet is False
    assert len(coercions) == 2
    assert any("allow_post_processing" in c for c in coercions)
    assert any("allow_controlnet" in c for c in coercions)


def test_no_coercion_when_flags_already_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable lane but already-off flags report nothing (idempotent on the hot-reload path)."""
    _patch_utilities(monkeypatch, False)
    bd = _bridge_data(allow_post_processing=False, allow_controlnet=False, allow_sdxl_controlnet=False)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert coercions == []


def test_dry_run_skips_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run leaves flags untouched since it never runs real inference."""
    _patch_utilities(monkeypatch, False)
    bd = _bridge_data(dry_run_skip_inference=True)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert coercions == []
    assert bd.allow_post_processing is True
    assert bd.allow_controlnet is True


def test_dependency_helpers_track_utilities_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-arg dependency helpers both resolve to the utilities-lane probe."""
    _patch_utilities(monkeypatch, False)
    assert capabilities.strip_background_available() is False
    assert capabilities.controlnet_available() is False

    _patch_utilities(monkeypatch, True)
    assert capabilities.strip_background_available() is True
    assert capabilities.controlnet_available() is True


def test_utilities_available_requires_provisioned_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """utilities_available is False when the venv interpreter is missing, True once it exists."""
    from worker_bootstrap import paths

    missing = SimpleNamespace(is_file=lambda: False)
    monkeypatch.setattr(paths, "utilities_python", lambda *_a, **_k: missing)
    assert capabilities.utilities_available() is False

    present = SimpleNamespace(is_file=lambda: True)
    monkeypatch.setattr(paths, "utilities_python", lambda *_a, **_k: present)
    assert capabilities.utilities_available() is True


def test_utilities_available_honours_enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the venv present, an explicit enable_image_utilities: false makes the lane unavailable."""
    from worker_bootstrap import paths

    present = SimpleNamespace(is_file=lambda: True)
    monkeypatch.setattr(paths, "utilities_python", lambda *_a, **_k: present)

    assert capabilities.utilities_available(_bridge_data(enable_image_utilities=False)) is False  # type: ignore[arg-type]
    assert capabilities.utilities_available(_bridge_data(enable_image_utilities=True)) is True  # type: ignore[arg-type]


def test_utilities_available_treats_mock_bridge_data_as_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mock bridge-data object (every attribute truthy) reads as enabled, never spuriously disabled.

    Guards the Mock-truthiness gotcha: ``enable_image_utilities is not False`` must not trip for a Mock
    whose ``enable_image_utilities`` is itself a truthy Mock.
    """
    from worker_bootstrap import paths

    present = SimpleNamespace(is_file=lambda: True)
    monkeypatch.setattr(paths, "utilities_python", lambda *_a, **_k: present)

    assert capabilities.utilities_available(Mock()) is True  # type: ignore[arg-type]


def test_expand_offered_forms_drops_strip_background_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alchemy drops only strip_background when the utilities lane is unavailable, keeping pure-torch forms."""
    _patch_utilities(monkeypatch, False)
    bd = SimpleNamespace(forms=["post-process"], alchemy_caption_enabled=False)

    offered = expand_offered_forms(bd)  # type: ignore[arg-type]

    assert "strip_background" not in offered
    assert len(offered) > 0


def test_expand_offered_forms_keeps_strip_background_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alchemy offers strip_background when the utilities lane is available."""
    _patch_utilities(monkeypatch, True)
    bd = SimpleNamespace(forms=["post-process"], alchemy_caption_enabled=False)

    offered = expand_offered_forms(bd)  # type: ignore[arg-type]

    assert "strip_background" in offered


def _patch_cpu_install(monkeypatch: pytest.MonkeyPatch, *, cpu: bool) -> None:
    """Force the CPU-only install gate (the bin/backend sentinel reader) for a coercion test."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: cpu)


def _cpu_bridge_data(**overrides: object) -> SimpleNamespace:
    bd = _bridge_data(
        image_models_to_load=["Deliberate", "AlbedoBase XL (SDXL)"],
        dynamic_models=True,
        alchemist=True,
    )
    for key, value in overrides.items():
        setattr(bd, key, value)
    return bd


def test_cpu_install_disables_image_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only install clears the image model list and dynamic loading, keeping alchemist on."""
    _patch_utilities(monkeypatch, True)
    _patch_cpu_install(monkeypatch, cpu=True)
    bd = _cpu_bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.image_models_to_load == []
    assert bd.dynamic_models is False
    assert bd.alchemist is True  # the CPU-friendly role is never forced off
    assert any("image_models_to_load" in c for c in coercions)
    assert any("dynamic_models" in c for c in coercions)


def test_non_cpu_install_leaves_image_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU install with the utilities lane available does not touch the image model list."""
    _patch_utilities(monkeypatch, True)
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _cpu_bridge_data()

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.image_models_to_load == ["Deliberate", "AlbedoBase XL (SDXL)"]
    assert bd.dynamic_models is True
    assert coercions == []


def test_cpu_install_idempotent_when_already_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running on an already-disabled CPU config reports nothing (hot-reload safe)."""
    _patch_utilities(monkeypatch, True)
    _patch_cpu_install(monkeypatch, cpu=True)
    bd = _cpu_bridge_data(image_models_to_load=[], dynamic_models=False)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert coercions == []


def test_enabled_workloads_dreamer_and_alchemist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal GPU dreamer+alchemist worker serves both workloads."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=True, alchemist=True)

    assert capabilities.enabled_workloads(bd) == frozenset(  # type: ignore[arg-type]
        {WorkloadKind.IMAGE_GENERATION, WorkloadKind.ALCHEMY},
    )


def test_enabled_workloads_dreamer_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The historical default (dreamer on, alchemist off) serves only image generation."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=True, alchemist=False)

    assert capabilities.enabled_workloads(bd) == frozenset({WorkloadKind.IMAGE_GENERATION})  # type: ignore[arg-type]


def test_enabled_workloads_alchemist_only_via_dreamer_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deselecting the dreamer role on a GPU box yields an alchemist-only worker."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=True)

    assert capabilities.enabled_workloads(bd) == frozenset({WorkloadKind.ALCHEMY})  # type: ignore[arg-type]


def test_enabled_workloads_cpu_forces_alchemist_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU install cannot serve image generation even with dreamer left on."""
    _patch_cpu_install(monkeypatch, cpu=True)
    bd = _bridge_data(dreamer=True, alchemist=True)

    assert capabilities.enabled_workloads(bd) == frozenset({WorkloadKind.ALCHEMY})  # type: ignore[arg-type]


def test_enabled_workloads_empty_when_nothing_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker with every role off serves nothing (the warning case)."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=False, scribe=False)

    assert capabilities.enabled_workloads(bd) == frozenset()  # type: ignore[arg-type]


def test_enabled_workloads_scribe_adds_text_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scribe role adds text generation beside whatever else the worker serves."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=True, scribe=True)

    assert capabilities.enabled_workloads(bd) == frozenset(  # type: ignore[arg-type]
        {WorkloadKind.IMAGE_GENERATION, WorkloadKind.TEXT_GENERATION},
    )


def test_enabled_workloads_scribe_off_serves_no_text_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text generation is offered only when the operator asked for it."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=True, scribe=False)

    assert WorkloadKind.TEXT_GENERATION not in capabilities.enabled_workloads(bd)  # type: ignore[arg-type]


def test_enabled_workloads_scribe_only_is_a_valid_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scribe with no other role serves text generation and is not an empty worker."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=False, scribe=True)

    assert capabilities.enabled_workloads(bd) == frozenset({WorkloadKind.TEXT_GENERATION})  # type: ignore[arg-type]


def test_enabled_workloads_scribe_survives_a_cpu_only_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text generation runs in an external program, so a CPU-only box is a perfectly good scribe."""
    _patch_cpu_install(monkeypatch, cpu=True)
    bd = _bridge_data(dreamer=True, alchemist=False, scribe=True)

    assert capabilities.enabled_workloads(bd) == frozenset({WorkloadKind.TEXT_GENERATION})  # type: ignore[arg-type]


def test_a_text_only_worker_wants_no_child_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text generation runs in an external program, so only the backend itself is wanted."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=False, scribe=True)

    assert capabilities.wanted_process_types(bd) == frozenset({HordeProcessType.TEXT_BACKEND})  # type: ignore[arg-type]


def test_an_alchemy_only_worker_wants_no_inference_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """No alchemy form routes to an inference process, and the disaggregation lanes are image stages."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=True)

    assert capabilities.wanted_process_types(bd) == frozenset(  # type: ignore[arg-type]
        {
            HordeProcessType.SAFETY,
            HordeProcessType.POST_PROCESS,
            HordeProcessType.UTILITIES,
            HordeProcessType.DOWNLOAD,
        },
    )


def test_an_image_worker_wants_every_type_but_the_text_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image generation is the workload every in-worker process type exists for."""
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=True, alchemist=False)

    assert capabilities.wanted_process_types(bd) == frozenset(HordeProcessType) - {  # type: ignore[arg-type]
        HordeProcessType.TEXT_BACKEND,
    }


@pytest.mark.parametrize("workload", list(WorkloadKind))
def test_a_process_type_that_executes_a_workload_is_wanted_by_it(workload: WorkloadKind) -> None:
    """The start table may be wider than the routing maps and never narrower: a routed unit needs its executor."""
    wanted = process_types_for_workloads(frozenset({workload}))

    for process_type, declared in DEFAULT_CAPABILITIES.items():
        if declared & capabilities_for_workload(workload):
            assert process_type in wanted


def test_every_process_type_states_which_workloads_need_it() -> None:
    """A type missing from the start table would never start for any worker."""
    assert process_types_for_workloads(frozenset(WorkloadKind)) == frozenset(HordeProcessType)


def test_scribe_only_worker_is_not_warned_as_having_nothing_to_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scribe-only worker has work, so the coercion pass must not report it as serving nothing."""
    _patch_utilities(monkeypatch, True)
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _bridge_data(dreamer=False, alchemist=False, scribe=True, image_models_to_load=[], dynamic_models=False)

    logged: list[str] = []
    sink_id = logger.add(lambda message: logged.append(message.record["message"]), level="WARNING")
    try:
        coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=True)  # type: ignore[arg-type]
    finally:
        logger.remove(sink_id)

    assert coercions == []
    assert not [line for line in logged if "nothing to" in line], logged


def test_dreamer_false_disables_image_generation_on_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberate dreamer: false opt-out clears image models on a GPU install too."""
    _patch_utilities(monkeypatch, True)
    _patch_cpu_install(monkeypatch, cpu=False)
    bd = _cpu_bridge_data(dreamer=False, alchemist=True)

    coercions = capabilities.coerce_bridge_data_to_capabilities(bd, log=False)  # type: ignore[arg-type]

    assert bd.image_models_to_load == []
    assert bd.dynamic_models is False
    assert bd.alchemist is True
    assert any("dreamer" in c for c in coercions)
