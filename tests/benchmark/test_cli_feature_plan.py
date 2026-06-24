"""The benchmark download plan accounts for *all* the files a feature needs, with real on-disk state.

These guard the user-reported defect: a machine whose controlnet files are only partly present must NOT be
told the plan has nothing to download. The plan is built torch-free from the model reference (no cold
hordelib import), so the dry-run preview reports the genuine on-disk picture for controlnet checkpoints,
post-processing models and the controlnet annotators alike.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)

from horde_worker_regen.benchmark import requirements
from horde_worker_regen.benchmark.cli import main
from horde_worker_regen.benchmark.download_progress import DownloadEvent, decode_download_events
from horde_worker_regen.benchmark.requirements import (
    FeatureModelFile,
    controlnet_checkpoint_files,
    post_processor_model_files,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _record(
    category: MODEL_REFERENCE_CATEGORY,
    name: str,
    file_name: str,
    *,
    size: int | None = 100,
) -> GenericModelRecord:
    """A minimal reference record declaring one downloadable file, for presence/path resolution."""
    download = DownloadRecord(file_name=file_name, file_url=f"https://example.com/{file_name}", size_bytes=size)
    return GenericModelRecord(
        name=name,
        record_type=category,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.auxiliary_or_patch),
        config=GenericModelRecordConfig(download=[download]),
    )


def _place(root: Path, category: str, file_name: str) -> None:
    """Write a stub file at ``<root>/<category>/<file_name>`` so the presence check finds it."""
    target = root / category / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"weights")


def _fake_references(
    monkeypatch: pytest.MonkeyPatch,
    references: dict[str, Mapping[str, GenericModelRecord]],
) -> None:
    """Make the offline-reference lookup return controlled references (None for unlisted categories)."""
    monkeypatch.setattr(requirements, "_offline_category_reference", references.get)


def _controlnet_reference(*control_types: str) -> dict[str, GenericModelRecord]:
    """A controlnet reference holding a ``control_<type>`` record for each requested control type."""
    return {
        f"control_{control_type}": _record(
            MODEL_REFERENCE_CATEGORY.controlnet,
            f"control_{control_type}",
            f"control_{control_type}.safetensors",
        )
        for control_type in control_types
    }


def test_missing_controlnet_checkpoint_is_reported_to_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlnet checkpoint the reference knows but disk lacks is surfaced as not-on-disk (to download)."""
    _fake_references(monkeypatch, {"controlnet": _controlnet_reference("canny")})

    rows = controlnet_checkpoint_files(["canny"], cache_home=str(tmp_path))

    assert [row.name for row in rows] == ["control_canny"]
    assert rows[0].on_disk is False  # the user's case: a missing controlnet must not read as present


def test_present_controlnet_checkpoint_is_reported_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the controlnet file is on disk, the same row reads present (not a redundant fetch)."""
    _fake_references(monkeypatch, {"controlnet": _controlnet_reference("canny")})
    _place(tmp_path, "controlnet", "control_canny.safetensors")

    rows = controlnet_checkpoint_files(["canny"], cache_home=str(tmp_path))

    assert rows[0].on_disk is True


def test_controlnet_type_absent_from_the_reference_is_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control type with no matching record (e.g. an SDXL-only tier) yields no row: it cannot be planned."""
    _fake_references(monkeypatch, {"controlnet": {}})
    assert controlnet_checkpoint_files(["openpose"], cache_home=str(tmp_path)) == []


def test_controlnet_rows_are_undeterminable_when_the_reference_cannot_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the whole reference is unreadable, presence is None (unknown), never a false 'present'."""
    _fake_references(monkeypatch, {})  # "controlnet" maps to None
    rows = controlnet_checkpoint_files(["canny"], cache_home=str(tmp_path))
    assert rows[0].on_disk is None


def test_post_processor_resolves_to_its_category_and_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-processor name resolves to whichever category owns its record, with real on-disk state."""
    esrgan = {
        "RealESRGAN_x4plus": _record(MODEL_REFERENCE_CATEGORY.esrgan, "RealESRGAN_x4plus", "RealESRGAN_x4plus.pth"),
    }
    codeformer = {"CodeFormers": _record(MODEL_REFERENCE_CATEGORY.codeformer, "CodeFormers", "CodeFormers.pth")}
    _fake_references(monkeypatch, {"esrgan": esrgan, "gfpgan": {}, "codeformer": codeformer})
    _place(tmp_path, "esrgan", "RealESRGAN_x4plus.pth")  # present; CodeFormers left absent

    rows = post_processor_model_files(["RealESRGAN_x4plus", "CodeFormers"], cache_home=str(tmp_path))

    by_name = {row.name: row for row in rows}
    assert by_name["RealESRGAN_x4plus"].category == "esrgan"
    assert by_name["RealESRGAN_x4plus"].on_disk is True
    assert by_name["CodeFormers"].category == "codeformer"
    assert by_name["CodeFormers"].on_disk is False


def test_post_processor_without_a_model_record_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """strip_background/rembg has no horde-managed model, so it contributes no plan row (fetched lazily)."""
    _fake_references(monkeypatch, {"esrgan": {}, "gfpgan": {}, "codeformer": {}})
    assert post_processor_model_files(["strip_background"], cache_home=str(tmp_path)) == []


def _planned_event(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> DownloadEvent:
    """Run ``download --dry-run`` with controlled feature files and return the single planned event."""
    monkeypatch.setattr(requirements, "models_disk_plan", lambda _names: None)  # unsized: avoid the network
    monkeypatch.setattr(requirements, "model_present_on_disk", lambda _name: True)
    monkeypatch.setattr(requirements, "annotators_present_offline", lambda _types: True)
    monkeypatch.setattr(
        requirements,
        "controlnet_checkpoint_files",
        lambda _types, **_kwargs: [
            FeatureModelFile(name="control_canny", category="controlnet", size_bytes=500, on_disk=False),
        ],
    )
    monkeypatch.setattr(requirements, "post_processor_model_files", lambda _names, **_kwargs: [])

    rc = main(["download", "--tiers", "sd15", "--dry-run", "--json-progress"])
    assert rc == 0
    events = decode_download_events(capsys.readouterr().out)
    return next(event for event in events if event.kind == "planned")


def test_dry_run_plan_lists_a_missing_controlnet_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end: the dry-run plan a non-technical operator sees includes the missing controlnet, to download.

    This is the reported bug in miniature: with a controlnet checkpoint absent, the plan must show it as a
    pending download rather than implying everything is ready.
    """
    planned = _planned_event(monkeypatch, capsys)

    controlnet_rows = [row for row in planned.models if row.name == "control_canny"]
    assert controlnet_rows, "the missing controlnet checkpoint must appear in the plan"
    assert controlnet_rows[0].on_disk is False


def test_dry_run_plan_does_not_import_torch() -> None:
    """The dry-run plan stays torch-free: a short-lived preview must never pay the cold torch import.

    Resolving feature on-disk state moved to the model reference's torch-free helpers, so the preview no
    longer needs the controlnet sizing import that dragged hordelib's heavy stack. hordelib's own torch-free
    helpers may still load to build the ladder; torch itself must not (its cold import is what once timed the
    plan out on packaged installs). Run in a clean subprocess so it reflects a fresh interpreter.
    """
    code = (
        "import sys\n"
        "from horde_worker_regen.benchmark.cli import main\n"
        "main(['download', '--tiers', 'sd15', '--dry-run', '--json-progress'])\n"
        "assert 'torch' not in sys.modules, 'the dry-run plan imported torch'\n"
        "print('torch-free-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "AI_HORDE_TESTING": "True"},
    )
    assert result.returncode == 0, result.stderr
    assert "torch-free-ok" in result.stdout
