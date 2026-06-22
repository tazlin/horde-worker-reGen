"""The parent's disk plan refreshes live (throttled) so readiness tracks downloads completing.

The TUI's readiness is single-sourced from this plan's ``num_present`` (see ``DownloadsView._readiness``),
so the plan must re-read disk as files land rather than staying pinned to its first computation. These
drive ``HordeWorkerProcessManager._get_download_plan_summary`` against a real on-disk tree through a light
stub, exercising the throttle and the live recompute without standing up a whole worker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


def _record(name: str, file_name: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        description="live-refresh test record",
        size_on_disk_bytes=1000,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=file_name, file_url=f"https://example/{file_name}")],
        ),
    )


def _model_tree(tmp_path: Path) -> Path:
    """Create a cache tree resolve_weights_root will find (has compvis + clip)."""
    (tmp_path / "compvis").mkdir(parents=True)
    (tmp_path / "clip").mkdir(parents=True)
    return tmp_path


def _stub(reference: dict[str, ImageGenerationModelRecord], names: list[str]) -> SimpleNamespace:
    """A minimal stand-in carrying just the attributes ``_get_download_plan_summary`` reads."""
    return SimpleNamespace(
        _download_plan_summary=None,
        _download_plan_refreshed_at=0.0,
        _DOWNLOAD_PLAN_REFRESH_SECONDS=HordeWorkerProcessManager._DOWNLOAD_PLAN_REFRESH_SECONDS,
        stable_diffusion_reference=reference,
        bridge_data=SimpleNamespace(image_models_to_load=names, extra_model_directories=None),
    )


def test_plan_recomputes_after_file_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """num_present climbs once a model lands on disk and the throttle window has elapsed."""
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(_model_tree(tmp_path)))
    reference = {"M": _record("M", "m.safetensors")}
    stub = _stub(reference, ["M"])

    first = HordeWorkerProcessManager._get_download_plan_summary(stub)
    assert first is not None
    assert first.num_present == 0
    assert first.num_to_download == 1

    # The file lands, but within the throttle window the cached plan is still served (no disk churn).
    (tmp_path / "compvis" / "m.safetensors").write_bytes(b"weights")
    assert HordeWorkerProcessManager._get_download_plan_summary(stub).num_present == 0

    # Past the throttle window it re-reads disk and now counts the model present.
    stub._download_plan_refreshed_at -= stub._DOWNLOAD_PLAN_REFRESH_SECONDS + 1.0
    refreshed = HordeWorkerProcessManager._get_download_plan_summary(stub)
    assert refreshed.num_present == 1
    assert refreshed.num_to_download == 0


def test_plan_holds_last_value_when_reference_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no reference yet, the last-known plan is returned rather than dropping to None mid-run."""
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(_model_tree(tmp_path)))
    reference = {"M": _record("M", "m.safetensors")}
    stub = _stub(reference, ["M"])

    computed = HordeWorkerProcessManager._get_download_plan_summary(stub)
    assert computed is not None

    # Reference goes away (e.g. a reload in flight); a stale-but-real plan beats a flicker to None.
    stub.stable_diffusion_reference = None
    stub._download_plan_refreshed_at -= stub._DOWNLOAD_PLAN_REFRESH_SECONDS + 1.0
    assert HordeWorkerProcessManager._get_download_plan_summary(stub) is computed
