"""Mock process implementations for GPU-free testing.

This package provides mock versions of inference and safety processes that simulate
realistic worker behavior without requiring GPU hardware or heavy dependencies.

Mock processes send the same message sequences as real processes, making them perfect
for testing the terminal UI, event system, and worker orchestration logic.
"""

from __future__ import annotations

__all__ = [
    "MockInferenceProcess",
    "MockSafetyProcess",
    "MockConfig",
    "MockScenario",
    "generate_fake_image",
    "start_mock_inference_process",
    "start_mock_safety_process",
]
