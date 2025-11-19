"""Configuration for mock processes.

Defines settings that control mock process behavior, timing, and scenario simulation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from enum import auto


class MockScenario(enum.Enum):
    """Predefined scenarios for testing different behaviors."""

    HAPPY_PATH = auto()
    """All jobs succeed quickly with no issues."""

    RANDOM_FAILURES = auto()
    """Random job failures at configured rate."""

    SLOW_INFERENCE = auto()
    """All jobs take significantly longer."""

    STUCK_PROCESS = auto()
    """Process gets stuck and stops responding."""

    DOWNLOAD_FAILURES = auto()
    """Model downloads fail randomly."""

    MEMORY_PRESSURE = auto()
    """Simulate high memory usage."""

    RAPID_FIRE = auto()
    """Very fast execution for stress testing."""


@dataclass
class MockConfig:
    """Configuration for mock process behavior.

    This controls timing, failure rates, and other aspects of mock process simulation.
    """

    # Performance
    speed_multiplier: float = 1.0
    """Speed multiplier for all operations (10.0 = 10x faster, 0.1 = 10x slower)."""

    # Failure simulation
    enable_failures: bool = False
    """Enable random job failures."""

    failure_rate: float = 0.05
    """Probability of job failure (0.0-1.0)."""

    failure_types: list[str] = field(
        default_factory=lambda: [
            "Out of memory",
            "Model load failed",
            "Inference timeout",
            "CUDA error",
        ],
    )
    """List of failure messages to randomly choose from."""

    # Slowdown simulation
    enable_slowdowns: bool = False
    """Enable random job slowdowns."""

    slowdown_rate: float = 0.1
    """Probability of slow job (0.0-1.0)."""

    slowdown_multiplier: float = 3.0
    """How much slower a slow job should be (3.0 = 3x slower)."""

    # Download simulation
    enable_download_failures: bool = False
    """Enable random download failures."""

    download_failure_rate: float = 0.05
    """Probability of download failure (0.0-1.0)."""

    download_speed_mbps: float = 50.0
    """Simulated download speed in megabits per second."""

    # Memory simulation
    mock_vram_usage_mb: int = 8192
    """Simulated VRAM usage in megabytes."""

    mock_ram_usage_mb: int = 4096
    """Simulated RAM usage in megabytes."""

    simulate_memory_fluctuation: bool = True
    """Whether to simulate realistic memory usage fluctuations."""

    # Process behavior
    enable_stuck_simulation: bool = False
    """Enable process getting stuck (for deadlock testing)."""

    stuck_after_jobs: int | None = None
    """Number of jobs to complete before getting stuck (None = don't get stuck)."""

    # Model simulation
    model_download_size_mb: dict[str, float] = field(
        default_factory=lambda: {
            "stable_diffusion_1": 2000.0,  # 2GB
            "stable_diffusion_2": 2000.0,
            "sdxl": 6000.0,  # 6GB
            "flux": 12000.0,  # 12GB
            "cascade": 9000.0,  # 9GB
        },
    )
    """Simulated download sizes for different model types (in MB)."""

    model_load_time_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "stable_diffusion_1": 3.0,
            "stable_diffusion_2": 3.0,
            "sdxl": 8.0,
            "flux": 15.0,
            "cascade": 10.0,
        },
    )
    """Simulated load times for different model types (in seconds)."""

    # Safety process
    safety_check_time_seconds: float = 1.0
    """Time to perform safety check on one image."""

    # Scenario
    scenario: MockScenario | None = None
    """Active scenario (overrides individual settings)."""

    def apply_scenario(self, scenario: MockScenario) -> None:
        """Apply a predefined scenario configuration.

        Args:
            scenario: The scenario to apply.
        """
        self.scenario = scenario

        if scenario == MockScenario.HAPPY_PATH:
            self.enable_failures = False
            self.enable_slowdowns = False
            self.enable_download_failures = False
            self.enable_stuck_simulation = False
            self.speed_multiplier = 1.0

        elif scenario == MockScenario.RANDOM_FAILURES:
            self.enable_failures = True
            self.failure_rate = 0.1  # 10%
            self.enable_slowdowns = False

        elif scenario == MockScenario.SLOW_INFERENCE:
            self.enable_failures = False
            self.enable_slowdowns = True
            self.slowdown_rate = 1.0  # All jobs
            self.slowdown_multiplier = 3.0

        elif scenario == MockScenario.STUCK_PROCESS:
            self.enable_stuck_simulation = True
            self.stuck_after_jobs = 5  # Get stuck after 5 jobs

        elif scenario == MockScenario.DOWNLOAD_FAILURES:
            self.enable_download_failures = True
            self.download_failure_rate = 0.2  # 20%

        elif scenario == MockScenario.MEMORY_PRESSURE:
            self.mock_vram_usage_mb = 22000  # 22GB (near limit for 24GB cards)
            self.mock_ram_usage_mb = 30000  # 30GB
            self.simulate_memory_fluctuation = True

        elif scenario == MockScenario.RAPID_FIRE:
            self.speed_multiplier = 100.0  # 100x faster
            self.enable_failures = False
            self.enable_slowdowns = False

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config.
        """
        return {
            "speed_multiplier": self.speed_multiplier,
            "enable_failures": self.enable_failures,
            "failure_rate": self.failure_rate,
            "enable_slowdowns": self.enable_slowdowns,
            "slowdown_rate": self.slowdown_rate,
            "slowdown_multiplier": self.slowdown_multiplier,
            "mock_vram_usage_mb": self.mock_vram_usage_mb,
            "mock_ram_usage_mb": self.mock_ram_usage_mb,
            "scenario": self.scenario.name if self.scenario else None,
        }

    @classmethod
    def from_bridge_data(cls, bridge_data) -> MockConfig:
        """Create MockConfig from bridge data.

        Args:
            bridge_data: The bridge data configuration.

        Returns:
            MockConfig instance populated from bridge data.
        """
        config = cls()

        # Extract mock-related settings from bridge_data if they exist
        if hasattr(bridge_data, "mock_speed_multiplier"):
            config.speed_multiplier = bridge_data.mock_speed_multiplier

        if hasattr(bridge_data, "mock_enable_failures"):
            config.enable_failures = bridge_data.mock_enable_failures

        if hasattr(bridge_data, "mock_failure_rate"):
            config.failure_rate = bridge_data.mock_failure_rate

        if hasattr(bridge_data, "mock_enable_slowdowns"):
            config.enable_slowdowns = bridge_data.mock_enable_slowdowns

        if hasattr(bridge_data, "mock_slowdown_rate"):
            config.slowdown_rate = bridge_data.mock_slowdown_rate

        if hasattr(bridge_data, "mock_slowdown_multiplier"):
            config.slowdown_multiplier = bridge_data.mock_slowdown_multiplier

        if hasattr(bridge_data, "mock_vram_usage_mb"):
            config.mock_vram_usage_mb = bridge_data.mock_vram_usage_mb

        if hasattr(bridge_data, "mock_ram_usage_mb"):
            config.mock_ram_usage_mb = bridge_data.mock_ram_usage_mb

        # Apply scenario if specified
        if hasattr(bridge_data, "mock_scenario") and bridge_data.mock_scenario:
            scenario_name = bridge_data.mock_scenario.upper()
            try:
                scenario = MockScenario[scenario_name]
                config.apply_scenario(scenario)
            except KeyError:
                pass  # Invalid scenario name, ignore

        return config
