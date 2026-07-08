"""The heavy-model-and-workflow dispatch gate classifies through the shared size-tier authority.

The gate holds a heavy batch head back while a thread is already busy, so stacked weight loads and
activation peaks do not thrash a running sampler. "Heavy" must mean the same thing here as it does to the
whole-card machinery: any EXTRA_LARGE-tier model qualifies (resolved by name or by baseline through the
shared size-tier authority), alongside the SDXL known-slow-workflow combination read from the loaded
model reference.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord

from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from tests.process_management.conftest import make_job_pop_response, make_mock_model_reference_record
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _gate(
    reference: dict[str, ImageGenerationModelRecord],
    model: str,
    workflow: str | None = None,
) -> bool:
    """Run the gate for a job against a scheduler whose metadata holds the given reference."""
    model_metadata = ModelMetadata()
    model_metadata.set_reference(reference)
    scheduler = _make_inference_scheduler(model_metadata=model_metadata)
    job = make_job_pop_response(model, workflow=workflow)
    return scheduler._is_heavy_model_and_workflow(job, reference)


class TestHeavyModelAndWorkflowGate:
    """EXTRA_LARGE models and SDXL slow workflows serialise; ordinary models and workflows do not."""

    def test_extra_large_by_baseline_is_heavy(self) -> None:
        """A model whose reference baseline is EXTRA_LARGE tier (qwen) is heavy with any workflow."""
        reference = {
            "Qwen-Image_fp8": make_mock_model_reference_record(
                "Qwen-Image_fp8",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
            ),
        }
        assert _gate(reference, "Qwen-Image_fp8") is True

    def test_extra_large_by_name_is_heavy_without_a_reference_entry(self) -> None:
        """A named very large checkpoint classifies heavy even before its reference record resolves."""
        assert _gate({}, "Flux.1-Schnell fp8 (Compact)") is True

    def test_sdxl_with_known_slow_workflow_is_heavy(self) -> None:
        """An SDXL model running a known-slow workflow serialises behind in-flight work."""
        reference = {
            "AlbedoBase XL (SDXL)": make_mock_model_reference_record(
                "AlbedoBase XL (SDXL)",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        assert _gate(reference, "AlbedoBase XL (SDXL)", workflow="qr_code") is True

    def test_sdxl_with_ordinary_workflow_is_not_heavy(self) -> None:
        """An SDXL model on an ordinary workflow does not serialise."""
        reference = {
            "AlbedoBase XL (SDXL)": make_mock_model_reference_record(
                "AlbedoBase XL (SDXL)",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        assert _gate(reference, "AlbedoBase XL (SDXL)") is False

    def test_sd15_with_slow_workflow_is_not_heavy(self) -> None:
        """The slow-workflow branch is SDXL-scoped: an SD 1.5 model on the same workflow does not serialise."""
        reference = {
            "Deliberate": make_mock_model_reference_record(
                "Deliberate",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            ),
        }
        assert _gate(reference, "Deliberate", workflow="qr_code") is False
