"""Tests for kudos training recorder."""

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from pytest import MonkeyPatch

from horde_worker_regen.reporting.kudos_training_recorder import KudosTrainingRecorder


@pytest.fixture
def temp_dir() -> Iterator[str]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_model_reference() -> dict[str, ImageGenerationModelRecord]:
    """Create a mock model reference."""
    ref = MagicMock(spec=dict)
    ref.root = {
        "test_model": Mock(baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
    }
    return cast(dict[str, ImageGenerationModelRecord], ref)


@pytest.fixture
def mock_job_info() -> MagicMock:
    """Create a mock job info object."""
    job_info = MagicMock()
    job_info.sdk_api_job_info.id_ = "test_job_123"
    job_info.sdk_api_job_info.model = "test_model"
    job_info.sdk_api_job_info.payload.n_iter = 1
    job_info.sdk_api_job_info.payload.karras = True
    job_info.sdk_api_job_info.payload.loras = []
    job_info.sdk_api_job_info.payload.tis = []
    job_info.sdk_api_job_info.extra_source_images = None
    job_info.sdk_api_job_info._downloaded_source_image = None
    job_info.sdk_api_job_info._downloaded_source_mask = None

    # Mock model_dump to return a simple dict
    job_info.model_dump.return_value = {
        "sdk_api_job_info": {
            "model": "test_model",
            "payload": {
                "karras": True,
                "loras": [],
                "tis": [],
            },
        },
    }

    return job_info


def test_kudos_training_recorder_init_with_file(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
) -> None:
    """Test initialization with a specified file."""
    recorder = KudosTrainingRecorder(
        training_data_file="custom_file.json",
        stable_diffusion_reference=mock_model_reference,
    )

    assert recorder.training_data_file == "custom_file.json"
    assert recorder.stable_diffusion_reference == mock_model_reference
    assert recorder.base_directory == "kudos_model_training"


def test_kudos_training_recorder_init_default_file(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
) -> None:
    """Test initialization with default file."""
    recorder = KudosTrainingRecorder(
        training_data_file=None,
        stable_diffusion_reference=mock_model_reference,
    )

    assert recorder.training_data_file == "kudos_training_data.json"


def test_record_job_data_creates_new_file(
    temp_dir: str,
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that recording creates a new file when it doesn't exist."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=mock_model_reference,
    )

    recorder.record_job_data(mock_job_info)

    # Verify file was created
    file_path = Path(temp_dir) / "kudos_model_training" / "test_data.json"
    assert file_path.exists()

    # Verify content
    with open(file_path) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    assert "sdk_api_job_info" in data[0]


def test_record_job_data_appends_to_existing_file(
    temp_dir: str,
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that recording appends to an existing file."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Record first job
    recorder.record_job_data(mock_job_info)

    # Record second job
    mock_job_info.sdk_api_job_info.id_ = "test_job_456"
    recorder.record_job_data(mock_job_info)

    # Verify file has both entries
    file_path = Path(temp_dir) / "kudos_model_training" / "test_data.json"
    with open(file_path) as f:
        data = json.load(f)

    assert len(data) == 2


def test_record_job_data_skips_batched_jobs(
    temp_dir: str,
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that batched jobs (n_iter > 1) are not appended to existing files."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Record first job (n_iter=1)
    recorder.record_job_data(mock_job_info)

    # Try to record batched job (n_iter=2)
    mock_job_info.sdk_api_job_info.payload.n_iter = 2
    recorder.record_job_data(mock_job_info)

    # Verify file still has only one entry (batched job was skipped)
    file_path = Path(temp_dir) / "kudos_model_training" / "test_data.json"
    with open(file_path) as f:
        data = json.load(f)

    # Should still be 1 because batched jobs don't get appended
    assert len(data) == 1


def test_prepare_model_dump_adds_scheduler(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
) -> None:
    """Test that _prepare_model_dump adds scheduler field."""
    recorder = KudosTrainingRecorder(
        training_data_file="test.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Test with karras=True
    mock_job_info.sdk_api_job_info.payload.karras = True
    dump = recorder._prepare_model_dump(mock_job_info)
    assert dump["sdk_api_job_info"]["payload"]["scheduler"] == "karras"
    assert "karras" not in dump["sdk_api_job_info"]["payload"]

    # Reset for next test
    mock_job_info.model_dump.return_value = {
        "sdk_api_job_info": {
            "model": "test_model",
            "payload": {
                "karras": False,
                "loras": [],
                "tis": [],
            },
        },
    }

    # Test with karras=False
    mock_job_info.sdk_api_job_info.payload.karras = False
    dump = recorder._prepare_model_dump(mock_job_info)
    assert dump["sdk_api_job_info"]["payload"]["scheduler"] == "simple"


def test_prepare_model_dump_adds_counts(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
) -> None:
    """Test that _prepare_model_dump adds lora and TI counts."""
    recorder = KudosTrainingRecorder(
        training_data_file="test.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Add some loras and TIs to the mock
    mock_job_info.model_dump.return_value = {
        "sdk_api_job_info": {
            "model": "test_model",
            "payload": {
                "karras": True,
                "loras": [{"name": "lora1"}, {"name": "lora2"}],
                "tis": [{"name": "ti1"}],
            },
        },
    }
    mock_job_info.sdk_api_job_info.payload.loras = [{"name": "lora1"}, {"name": "lora2"}]
    mock_job_info.sdk_api_job_info.payload.tis = [{"name": "ti1"}]

    dump = recorder._prepare_model_dump(mock_job_info)

    assert dump["sdk_api_job_info"]["payload"]["lora_count"] == 2
    assert dump["sdk_api_job_info"]["payload"]["ti_count"] == 1


def test_prepare_model_dump_adds_model_baseline(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
) -> None:
    """Test that _prepare_model_dump adds model baseline."""
    recorder = KudosTrainingRecorder(
        training_data_file="test.json",
        stable_diffusion_reference=mock_model_reference,
    )

    dump = recorder._prepare_model_dump(mock_job_info)

    assert "model_baseline" in dump["sdk_api_job_info"]
    assert dump["sdk_api_job_info"]["model_baseline"] == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1


def test_prepare_model_dump_handles_extra_source_images(
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
) -> None:
    """Test that _prepare_model_dump handles extra source images."""
    recorder = KudosTrainingRecorder(
        training_data_file="test.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Create mock extra source images
    mock_esi1 = Mock(image=b"x" * 100)
    mock_esi2 = Mock(image=b"y" * 200)
    mock_job_info.sdk_api_job_info.extra_source_images = [mock_esi1, mock_esi2]

    dump = recorder._prepare_model_dump(mock_job_info)

    assert dump["sdk_api_job_info"]["extra_source_images_count"] == 2
    assert dump["sdk_api_job_info"]["extra_source_images_combined_size"] == 300


def test_record_job_data_skips_unknown_models(
    temp_dir: str,
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    mock_job_info: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that jobs with unknown models are skipped."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Set model to one that's not in the reference
    mock_job_info.sdk_api_job_info.model = "unknown_model"

    recorder.record_job_data(mock_job_info)

    # Verify file was not created
    file_path = Path(temp_dir) / "kudos_model_training" / "test_data.json"
    assert not file_path.exists()


def test_record_job_data_handles_none_reference(
    temp_dir: str,
    mock_job_info: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that jobs are skipped when model reference is None."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=None,
    )

    recorder.record_job_data(mock_job_info)

    # Verify file was not created
    file_path = Path(temp_dir) / "kudos_model_training" / "test_data.json"
    assert not file_path.exists()


def test_get_file_path_with_rotation(
    temp_dir: str,
    mock_model_reference: dict[str, ImageGenerationModelRecord],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test file rotation when file exceeds 2MB."""
    # Change to temp directory
    monkeypatch.chdir(temp_dir)

    recorder = KudosTrainingRecorder(
        training_data_file="test_data.json",
        stable_diffusion_reference=mock_model_reference,
    )

    # Create base directory
    os.makedirs(recorder.base_directory, exist_ok=True)

    # Create a file that's larger than 2MB
    base_file = Path(temp_dir) / recorder.base_directory / "test_data.json"
    with open(base_file, "w") as f:
        f.write("x" * (2 * 1024 * 1024 + 1))

    # Get file path (should rotate to .1)
    file_path = recorder._get_file_path_with_rotation()

    expected_path = str(Path(temp_dir) / recorder.base_directory / "test_data.json.1")
    assert file_path == expected_path
