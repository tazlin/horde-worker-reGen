"""Tests for the torch-free model-download planning utility (now backed by horde_model_reference)."""

from __future__ import annotations

from pathlib import Path

import pytest
from horde_model_reference import category_folder
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    ControlNetModelRecord,
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen import model_download_plan
from horde_worker_regen.model_download_plan import (
    ENV_EXTRA_MODEL_DIRECTORIES,
    compute_download_plan,
    is_model_present,
)


def _record(name: str, file_name: str, size: int | None) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        description="test record",
        size_on_disk_bytes=size,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=file_name, file_url=f"https://example/{file_name}")],
        ),
    )


def _make_model_tree(tmp_path: Path, present_files: list[str]) -> Path:
    """Create a cache tree resolve_weights_root will find (has compvis + clip), with some files."""
    compvis = tmp_path / "compvis"
    compvis.mkdir(parents=True)
    (tmp_path / "clip").mkdir(parents=True)
    for file_name in present_files:
        (compvis / file_name).write_bytes(b"x")
    return tmp_path


def test_plan_splits_present_and_to_download(tmp_path: Path) -> None:
    """Present and missing models are split, and their byte totals computed from record sizes."""
    _make_model_tree(tmp_path, ["present.safetensors"])
    reference = {
        "Present": _record("Present", "present.safetensors", 1000),
        "Missing": _record("Missing", "missing.safetensors", 2000),
    }
    plan = compute_download_plan(["Present", "Missing"], reference, cache_home=str(tmp_path))

    assert plan.num_present == 1
    assert plan.num_to_download == 1
    assert plan.present_bytes == 1000
    assert plan.to_download_bytes == 2000
    assert plan.total_bytes == 3000
    assert plan.sizes_complete is True
    present = next(model for model in plan.models if model.name == "Present")
    assert present.on_disk is True
    assert present.target_path.endswith("present.safetensors")


def test_unknown_size_is_flagged_and_excluded_from_totals(tmp_path: Path) -> None:
    """A model without size metadata contributes nothing to the byte totals and is flagged."""
    _make_model_tree(tmp_path, [])
    reference = {"NoSize": _record("NoSize", "nosize.safetensors", None)}
    plan = compute_download_plan(["NoSize"], reference, cache_home=str(tmp_path))

    assert plan.to_download_bytes == 0
    assert plan.unknown_size_models == ["NoSize"]
    assert plan.sizes_complete is False


def test_over_budget_reports_shortfall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When to-download exceeds free space, the plan does not fit and reports the shortfall."""
    _make_model_tree(tmp_path, [])
    reference = {"Big": _record("Big", "big.safetensors", 10_000)}

    monkeypatch.setattr(model_download_plan, "free_bytes_for", lambda _root: 4_000)

    plan = compute_download_plan(["Big"], reference, cache_home=str(tmp_path))
    assert plan.free_disk_bytes == 4_000
    assert plan.fits is False
    assert plan.shortfall_bytes == 6_000


def test_is_model_present(tmp_path: Path) -> None:
    """is_model_present is an existence-only check of the record's declared files."""
    _make_model_tree(tmp_path, ["here.safetensors"])
    reference = {
        "Here": _record("Here", "here.safetensors", 1),
        "Gone": _record("Gone", "gone.safetensors", 1),
    }
    assert is_model_present("Here", reference, cache_home=str(tmp_path)) is True
    assert is_model_present("Gone", reference, cache_home=str(tmp_path)) is False


def test_missing_record_is_not_on_disk_and_unsized(tmp_path: Path) -> None:
    """A configured name absent from the reference is not on disk, has no size, and no category."""
    _make_model_tree(tmp_path, [])
    plan = compute_download_plan(["Unknown"], {}, cache_home=str(tmp_path))
    assert plan.num_to_download == 1
    assert plan.unknown_size_models == ["Unknown"]
    only = plan.models[0]
    assert only.on_disk is False
    assert only.category is None
    assert only.target_path == ""


def test_component_file_routes_to_sibling_folder(tmp_path: Path) -> None:
    """A component file with a routed purpose is located in its sibling folder, not beside the unet."""
    _make_model_tree(tmp_path, ["unet.safetensors"])
    record = ImageGenerationModelRecord(
        name="Qwen",
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        nsfw=False,
        description="multi-file",
        size_on_disk_bytes=100,
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(file_name="unet.safetensors", file_url="https://example/unet"),
                DownloadRecord(file_name="ae.safetensors", file_url="https://example/ae", file_purpose="vae"),
            ],
        ),
    )
    reference = {"Qwen": record}

    # The VAE component lives under <root>/vae, not <root>/compvis, so it is missing until placed there.
    assert is_model_present("Qwen", reference, cache_home=str(tmp_path)) is False

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "ae.safetensors").write_bytes(b"x")
    assert is_model_present("Qwen", reference, cache_home=str(tmp_path)) is True


def test_generic_category_uses_its_own_folder(tmp_path: Path) -> None:
    """A non-image record is planned against its own category folder (existence + no size)."""
    _make_model_tree(tmp_path, [])
    controlnet_dir = tmp_path / "controlnet"
    controlnet_dir.mkdir()
    (controlnet_dir / "canny.safetensors").write_bytes(b"x")
    record = ControlNetModelRecord(
        name="Canny",
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name="canny.safetensors", file_url="https://example/canny")],
        ),
    )

    plan = compute_download_plan(["Canny"], {"Canny": record}, cache_home=str(tmp_path))
    only = plan.models[0]
    assert only.category is MODEL_REFERENCE_CATEGORY.controlnet
    assert only.on_disk is True
    assert only.size_bytes is None  # ControlNet records carry no size; the byte totals stay a lower bound.
    assert plan.sizes_complete is False


def test_record_accessors_feed_the_plan() -> None:
    """The plan reads the typed record's own category/size accessors (no worker-local bridge)."""
    image_record = _record("Image", "x.safetensors", 42)
    controlnet_record = ControlNetModelRecord(name="cn", config=GenericModelRecordConfig())

    assert image_record.category is MODEL_REFERENCE_CATEGORY.image_generation
    assert controlnet_record.category is MODEL_REFERENCE_CATEGORY.controlnet
    assert category_folder(MODEL_REFERENCE_CATEGORY.image_generation) == "compvis"
    assert category_folder(MODEL_REFERENCE_CATEGORY.controlnet) == "controlnet"
    assert image_record.declared_total_size_bytes == 42
    assert controlnet_record.declared_total_size_bytes is None


def test_extra_model_directory_locates_files_on_another_root(tmp_path: Path) -> None:
    """A file absent from the primary root but present in an extra root counts as on disk."""
    primary = _make_model_tree(tmp_path / "primary", [])
    extra = tmp_path / "extra"
    (extra / "compvis").mkdir(parents=True)
    (extra / "compvis" / "spread.safetensors").write_bytes(b"x")
    reference = {"Spread": _record("Spread", "spread.safetensors", 5)}

    assert is_model_present("Spread", reference, cache_home=str(primary)) is False
    assert (
        is_model_present(
            "Spread",
            reference,
            cache_home=str(primary),
            extra_model_directories=[str(extra)],
        )
        is True
    )

    plan = compute_download_plan(
        ["Spread"],
        reference,
        cache_home=str(primary),
        extra_model_directories=[str(extra)],
    )
    assert plan.num_present == 1
    assert plan.models[0].target_path.endswith("spread.safetensors")


def test_extra_model_directories_read_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the argument is omitted, extra roots are read from the environment variable."""
    primary = _make_model_tree(tmp_path / "primary", [])
    extra = tmp_path / "extra"
    (extra / "compvis").mkdir(parents=True)
    (extra / "compvis" / "fromenv.safetensors").write_bytes(b"x")
    reference = {"FromEnv": _record("FromEnv", "fromenv.safetensors", 5)}

    monkeypatch.setenv(ENV_EXTRA_MODEL_DIRECTORIES, str(extra))
    assert is_model_present("FromEnv", reference, cache_home=str(primary)) is True


def test_symlinked_model_file_counts_as_present(tmp_path: Path) -> None:
    """A symlink to a real weight on another path counts as present (existence follows symlinks)."""
    primary = _make_model_tree(tmp_path / "primary", [])
    real_target = tmp_path / "elsewhere" / "real.safetensors"
    real_target.parent.mkdir(parents=True)
    real_target.write_bytes(b"x")
    link = primary / "compvis" / "linked.safetensors"
    try:
        link.symlink_to(real_target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    reference = {"Linked": _record("Linked", "linked.safetensors", 5)}
    assert is_model_present("Linked", reference, cache_home=str(primary)) is True
