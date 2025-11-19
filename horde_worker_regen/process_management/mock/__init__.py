"""Mock process implementations for GPU-free testing.

This package provides mock versions of inference and safety processes that simulate
realistic worker behavior without requiring GPU hardware or heavy dependencies.

Mock processes send the same message sequences as real processes, making them perfect
for testing the terminal UI, event system, and worker orchestration logic.
"""

from __future__ import annotations

from horde_worker_regen.process_management.mock.mock_config import MockConfig, MockScenario
from horde_worker_regen.process_management.mock.mock_data_generator import (
    calculate_mock_inference_time,
    calculate_mock_kudos,
    generate_fake_csam_score,
    generate_fake_image,
    generate_fake_nsfw_score,
)
from horde_worker_regen.process_management.mock.mock_inference_process import MockInferenceProcess
from horde_worker_regen.process_management.mock.mock_safety_process import MockSafetyProcess
from horde_worker_regen.process_management.mock.mock_worker_entry_points import (
    start_mock_inference_process,
    start_mock_safety_process,
)

__all__ = [
    # Process classes
    "MockInferenceProcess",
    "MockSafetyProcess",
    # Configuration
    "MockConfig",
    "MockScenario",
    # Data generators
    "generate_fake_image",
    "calculate_mock_kudos",
    "calculate_mock_inference_time",
    "generate_fake_nsfw_score",
    "generate_fake_csam_score",
    # Entry points
    "start_mock_inference_process",
    "start_mock_safety_process",
]
