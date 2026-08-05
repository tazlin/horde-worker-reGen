"""The harness's model reference resolves real records, so exported baselines name the real model class."""

from __future__ import annotations

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord

import horde_worker_regen.harness as harness
from horde_worker_regen.harness import build_harness_model_reference
from horde_worker_regen.process_management.simulation._canned_scenarios import make_canned_job

_SDXL_MODEL = "AlbedoBase XL (SDXL)"


def _sdxl_record() -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_SDXL_MODEL,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        nsfw=False,
        description="a real reference record",
    )


class _StubReferenceManager:
    """A reference manager returning one real image-generation record."""

    def __init__(self, records: dict[str, ImageGenerationModelRecord]) -> None:
        self._records = records

    def get_model_reference(self, _category: object) -> dict[str, ImageGenerationModelRecord]:
        return self._records


def _install_singleton(monkeypatch: pytest.MonkeyPatch, instance: object | None) -> None:
    """Present (or withhold) a process-wide reference singleton to the harness."""

    class _Singleton:
        @staticmethod
        def has_instance() -> bool:
            return instance is not None

        @staticmethod
        def get_instance() -> object:
            assert instance is not None
            return instance

    monkeypatch.setattr(harness, "ModelReferenceManager", _Singleton)


def test_supplied_manager_resolves_the_real_baseline() -> None:
    """A model the reference knows keeps its own baseline rather than the synthetic fallback."""
    scenario = [make_canned_job(_SDXL_MODEL)]

    reference = build_harness_model_reference(
        scenario,
        _StubReferenceManager({_SDXL_MODEL: _sdxl_record()}),  # type: ignore[arg-type]
    )

    assert reference[_SDXL_MODEL].baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def test_reference_singleton_is_used_when_no_manager_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any driver that initialized the reference gets real baselines without threading the manager through.

    This is the path a canned (``skip_api``) corpus run takes: its config carries no reference manager,
    and before the singleton was consulted every job was exported as stable_diffusion_1.
    """
    _install_singleton(monkeypatch, _StubReferenceManager({_SDXL_MODEL: _sdxl_record()}))

    reference = build_harness_model_reference([make_canned_job(_SDXL_MODEL)])

    assert reference[_SDXL_MODEL].baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def test_models_absent_from_the_reference_keep_the_synthetic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A synthetic test-only model still gets a record, so fake-process scenarios keep running."""
    _install_singleton(monkeypatch, _StubReferenceManager({_SDXL_MODEL: _sdxl_record()}))

    reference = build_harness_model_reference([make_canned_job("a-model-no-reference-knows")])

    assert reference["a-model-no-reference-knows"].baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1


def test_no_reference_at_all_falls_back_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any reference the harness still builds a usable record set."""
    _install_singleton(monkeypatch, None)

    reference = build_harness_model_reference([make_canned_job(_SDXL_MODEL)])

    assert reference[_SDXL_MODEL].baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
