"""Repro: a beta model on disk reported as "will download" because a surface's reference lacks it.

User report (Z-Image-Turbo): one surface said the model "will download", the Downloads tab said "all
requested models are present", and no download fired. Z-Image-Turbo is a *beta* (pending-queue) model,
so whether a given surface even holds its record depends on whether that surface loaded the reference
with the beta source. ``compute_download_plan`` answers "on disk?" against whichever reference copy it
is handed: when the record is absent it cannot resolve the files and reports the model as to-download,
*even though the weights are on disk*. The beta-aware download subsystem, holding the record, correctly
reports it present. This test reproduces that contradiction deterministically (no GPU, no network) by
running the same on-disk files through a reference that has the record vs one that does not.
"""

from __future__ import annotations

from pathlib import Path

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.model_download_plan import compute_download_plan

_MODEL_NAME = "Z-Image-Turbo"
# (file_name, file_purpose) for the real 3-file Z-Image-Turbo layout.
_ZIMAGE_FILES: tuple[tuple[str, str], ...] = (
    ("z_image_turbo_bf16.safetensors", "unet"),
    ("ae.safetensors", "vae"),
    ("qwen_3_4b.safetensors", "text_encoders"),
)


def _zimage_record() -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_MODEL_NAME,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        nsfw=True,
        description="Z-Image-Turbo (beta/pending) repro record",
        size_on_disk_bytes=20_430_635_136,
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name=file_name,
                    file_url=f"https://example.com/{file_name}",
                    file_purpose=file_purpose,
                )
                for file_name, file_purpose in _ZIMAGE_FILES
            ],
        ),
    )


def _place_zimage_on_disk(tmp_path: Path) -> None:
    """Create a model tree (compvis + clip markers) with all three Z-Image files present."""
    (tmp_path / "compvis").mkdir(parents=True)
    (tmp_path / "clip").mkdir(parents=True)
    (tmp_path / "compvis" / "z_image_turbo_bf16.safetensors").write_bytes(b"x")
    (tmp_path / "vae").mkdir(parents=True)
    (tmp_path / "vae" / "ae.safetensors").write_bytes(b"x")
    (tmp_path / "text_encoders").mkdir(parents=True)
    (tmp_path / "text_encoders" / "qwen_3_4b.safetensors").write_bytes(b"x")


def test_present_record_sees_zimage_on_disk(tmp_path: Path) -> None:
    """With the record present, the on-disk Z-Image is correctly counted present (the Downloads tab view)."""
    _place_zimage_on_disk(tmp_path)
    reference = {_MODEL_NAME: _zimage_record()}

    plan = compute_download_plan([_MODEL_NAME], reference, cache_home=str(tmp_path))

    assert plan.num_present == 1
    assert plan.num_to_download == 0
    assert plan.models[0].on_disk is True


def test_missing_record_reports_will_download_despite_files_on_disk(tmp_path: Path) -> None:
    """The bug: the SAME on-disk files are reported as to-download when the surface's reference lacks the record.

    This is the "z-image-turbo will download" half of the contradiction: a non-beta reference copy has no
    Z-Image-Turbo record, so the planner cannot resolve its files and treats it as not-on-disk, regardless
    of the weights actually sitting in the model folder.
    """
    _place_zimage_on_disk(tmp_path)
    reference_without_beta_record: dict = {}

    plan = compute_download_plan([_MODEL_NAME], reference_without_beta_record, cache_home=str(tmp_path))

    assert plan.num_present == 0
    assert plan.num_to_download == 1
    only = plan.models[0]
    assert only.on_disk is False
    assert only.category is None  # No record -> no category -> no resolvable path, despite files on disk.
    assert only.target_path == ""


def test_presence_answer_flips_only_because_the_reference_differs(tmp_path: Path) -> None:
    """Pin the root cause: identical disk state, opposite answers, driven solely by the reference copy."""
    _place_zimage_on_disk(tmp_path)

    with_record = compute_download_plan([_MODEL_NAME], {_MODEL_NAME: _zimage_record()}, cache_home=str(tmp_path))
    without_record = compute_download_plan([_MODEL_NAME], {}, cache_home=str(tmp_path))

    assert with_record.models[0].on_disk is True
    assert without_record.models[0].on_disk is False
