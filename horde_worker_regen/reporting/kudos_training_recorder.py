"""Kudos training data recording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.process_management.job_models import HordeJobInfo


_excludes_for_job_dump = {
    "job_image_results": True,
    "sdk_api_job_info": {
        "payload": {"prompt": True, "special": True},
        "skipped": True,
        "source_image": True,
        "source_mask": True,
        "extra_source_images": True,
        "r2_upload": True,
        "r2_uploads": True,
    },
}


class KudosTrainingRecorder:
    """Handles recording of job data for kudos model training."""

    def __init__(
        self,
        training_data_file: str | None,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord] | None,
    ) -> None:
        """Initialize the kudos training recorder.

        Args:
            training_data_file: Path to the training data file (relative name).
            stable_diffusion_reference: Reference to stable diffusion models.
        """
        self.training_data_file = training_data_file or "kudos_training_data.json"
        self.stable_diffusion_reference = stable_diffusion_reference
        self.base_directory = "kudos_model_training"

        # Warn if default file is being used
        if training_data_file is None:
            logger.warning(
                "Kudos training data capture is enabled but no file has been specified. "
                f"Defaulting to {self.training_data_file}",
            )

    def record_job_data(self, job_info: HordeJobInfo) -> None:
        """Record job data for kudos training.

        Args:
            job_info: The job information to record.
        """
        # Skip if model is not in the reference or if we don't have a reference
        if (
            self.stable_diffusion_reference is None
            or job_info.sdk_api_job_info.model is None
            or job_info.sdk_api_job_info.model not in self.stable_diffusion_reference
        ):
            return

        try:
            with logger.catch(reraise=False):
                # Get the file to use (with rotation if needed)
                file_path = self._get_file_path_with_rotation()

                # Prepare the model dump with additional fields
                model_dump = self._prepare_model_dump(job_info)

                # Write to file
                self._write_to_file(file_path, model_dump, job_info.sdk_api_job_info.payload.n_iter)

        except Exception as e:
            logger.error(
                f"Failed to write kudos training data for job {job_info.sdk_api_job_info.id_} {type(e)}: {e}",
            )

    def _get_file_path_with_rotation(self) -> str:
        """Get the file path to use, rotating to a new file if the current one is too large.

        Returns:
            The file path to use for writing.
        """
        base_directory_path = Path.cwd() / self.base_directory
        base_directory_path.mkdir(parents=True, exist_ok=True)

        file_path = base_directory_path / self.training_data_file

        # Check if file exists and is larger than 2MB
        if file_path.exists() and file_path.stat().st_size > 2 * 1024 * 1024:
            # Find next available file number
            for i in range(1, 10000):
                new_file_path = base_directory_path / f"{self.training_data_file}.{i}"
                if new_file_path.exists() and new_file_path.stat().st_size > 2 * 1024 * 1024:
                    continue

                file_path = new_file_path
                break

        return str(file_path)

    def _prepare_model_dump(self, job_info: HordeJobInfo) -> dict[str, Any]:
        """Prepare the model dump with additional fields.

        Args:
            job_info: The job information to dump.

        Returns:
            A dictionary containing the job data ready for JSON serialization.
        """
        model_dump = job_info.model_dump(
            exclude=_excludes_for_job_dump,  # type: ignore
            mode="json",
        )

        api_job = model_dump["sdk_api_job_info"]
        payload = api_job["payload"]

        # Add model baseline
        if self.stable_diffusion_reference is not None and job_info.sdk_api_job_info.model is not None:
            api_job["model_baseline"] = self.stable_diffusion_reference[job_info.sdk_api_job_info.model].baseline

        # Add scheduler information (preparation for multiple schedulers)
        payload["scheduler"] = "karras" if job_info.sdk_api_job_info.payload.karras else "simple"
        payload.pop("karras", None)

        # Add lora and TI counts
        payload["lora_count"] = len(payload["loras"]) if payload["loras"] else 0
        payload["ti_count"] = len(payload["tis"]) if payload["tis"] else 0

        # Add extra source images count and size
        extra_images = job_info.sdk_api_job_info.extra_source_images
        api_job["extra_source_images_count"] = len(extra_images) if extra_images else 0
        api_job["extra_source_images_combined_size"] = (
            sum(len(esi.image) for esi in extra_images) if extra_images else 0
        )

        # Add source image and mask sizes
        api_job["source_image_size"] = (
            len(job_info.sdk_api_job_info._downloaded_source_image)
            if job_info.sdk_api_job_info._downloaded_source_image
            else 0
        )
        api_job["source_mask_size"] = (
            len(job_info.sdk_api_job_info._downloaded_source_mask)
            if job_info.sdk_api_job_info._downloaded_source_mask
            else 0
        )

        return model_dump

    def _write_to_file(self, file_path: str, model_dump: dict[str, Any], n_iter: int) -> None:
        """Write the model dump to the file.

        Args:
            file_path: Path to the file to write to.
            model_dump: The model data to write.
            n_iter: Number of iterations (used to skip batched jobs).
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            # Create new file with first entry
            with open(path_obj, "w") as f:
                json.dump([model_dump], f, indent=4)
        elif n_iter == 1:
            # Append to existing file (only for non-batched jobs)
            data = []
            with open(path_obj) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning(f"Kudos training data file {file_path} is not a list")
                    data = []

            data.append(model_dump)
            with open(path_obj, "w") as f:
                json.dump(data, f, indent=4)
