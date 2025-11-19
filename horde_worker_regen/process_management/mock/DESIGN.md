# Mock Process System Design

## Overview

The mock process system allows testing the worker's terminal UI, event system, and orchestration logic without requiring actual GPU hardware or expensive inference operations. Mock processes simulate realistic worker behavior with configurable timing and scenarios.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         HordeWorkerProcessManager (Main)            │
│                                                       │
│  Chooses: Real or Mock processes based on config    │
└───────────────┬─────────────────────────────────────┘
                │
                ├─── Real Mode ────────┬──── Mock Mode
                │                      │
        ┌───────▼─────────┐    ┌──────▼──────────┐
        │ start_inference │    │ start_mock_     │
        │ _process()      │    │ inference_      │
        │                 │    │ process()       │
        │ ↓               │    │ ↓               │
        │ HordeInference  │    │ MockInference   │
        │ Process         │    │ Process         │
        │ (GPU required)  │    │ (CPU only)      │
        └─────────────────┘    └─────────────────┘
```

## Key Design Principles

1. **Drop-in Replacement**: Mock processes use same message protocol as real processes
2. **Configurable Timing**: Speed up or slow down simulation via multiplier
3. **Scenario Support**: Simulate various behaviors (success, failure, stuck, slow)
4. **No GPU Dependencies**: Pure Python, no hordelib/torch/CUDA imports
5. **Realistic Behavior**: Send same message sequences as real processes
6. **Fake Data Generation**: Generate placeholder images that pass through pipeline

## Mock Process Behavior

### MockInferenceProcess

**Simulates:**
- Model download (with progress updates)
- Model loading (RAM → VRAM)
- Inference steps (with heartbeat progress)
- Post-processing
- Result generation (fake base64 images)
- Memory usage reporting

**Message Sequence:**
```
1. PROCESS_STARTING
2. WAITING_FOR_JOB
3. [Receive DOWNLOAD_MODEL]
4. DOWNLOADING_MODEL (with progress: 0% → 100%)
5. [Receive PRELOAD_MODEL]
6. PRELOADING_MODEL
7. PRELOADED_MODEL
8. [Receive START_INFERENCE]
9. INFERENCE_STARTING
10. [Send heartbeats with step progress: 0% → 100%]
11. INFERENCE_POST_PROCESSING
12. INFERENCE_COMPLETE
13. [Send result with fake images]
14. Back to WAITING_FOR_JOB
```

**Timing (configurable via speed_multiplier):**
- Download: 5-15 seconds (faster for subsequent downloads)
- Loading: 2-8 seconds (depends on model size/baseline)
- Inference: 0.1s per step × steps (e.g., 20 steps = 2s)
- Post-processing: 0.5-1.5 seconds

### MockSafetyProcess

**Simulates:**
- Safety model initialization
- NSFW checking
- CSAM detection
- Image censoring (returns censor images)

**Message Sequence:**
```
1. PROCESS_STARTING
2. WAITING_FOR_JOB
3. [Receive EVALUATE_SAFETY]
4. EVALUATING_SAFETY
5. [Send heartbeat]
6. [Send safety result]
7. Back to WAITING_FOR_JOB
```

**Timing:**
- Safety check: 0.5-2 seconds per image

## Configuration

### Bridge Data Fields

```python
class reGenBridgeData:
    # Mock mode enablement
    enable_mock_processes: bool = False
    """Enable mock processes instead of real GPU processes (for testing)."""

    # Performance tuning
    mock_speed_multiplier: float = 1.0
    """Speed multiplier for mock processes (10.0 = 10x faster, 0.1 = 10x slower)."""

    # Scenario simulation
    mock_enable_failures: bool = False
    """Enable random failures in mock processes."""

    mock_failure_rate: float = 0.05
    """Probability of job failure (0.0-1.0, default 5%)."""

    mock_enable_slowdowns: bool = False
    """Enable random slowdowns in mock processes."""

    mock_slowdown_rate: float = 0.1
    """Probability of slow job (0.0-1.0, default 10%)."""

    mock_slowdown_multiplier: float = 3.0
    """How much slower a slow job should be (3.0 = 3x slower)."""

    # Memory simulation
    mock_vram_usage_mb: int = 8192
    """Simulated VRAM usage in MB."""

    mock_ram_usage_mb: int = 4096
    """Simulated RAM usage in MB."""
```

### Environment Variables

```bash
# Quick enable via environment
export AIWORKER_ENABLE_MOCK_PROCESSES=true
export AIWORKER_MOCK_SPEED_MULTIPLIER=10.0  # 10x faster for rapid testing
export AIWORKER_MOCK_ENABLE_FAILURES=true
```

## Implementation Structure

```
horde_worker_regen/process_management/
├── mock/
│   ├── __init__.py
│   ├── mock_inference_process.py  # MockInferenceProcess class
│   ├── mock_safety_process.py     # MockSafetyProcess class
│   ├── mock_data_generator.py     # Fake image/data generation
│   ├── mock_scenarios.py          # Predefined failure/success scenarios
│   └── README.md                   # Documentation
└── worker_entry_points.py         # Modified to support mock mode
```

## Mock Data Generation

### Fake Images

Instead of real images, generate:
- Solid color images with text overlay
- Include job ID, model name, seed in image
- Correct dimensions for the request
- Valid PNG/JPEG encoding
- Proper base64 encoding

```python
def generate_fake_image(width: int, height: int, job_id: str, model: str) -> str:
    """Generate a fake image for testing.

    Returns a base64-encoded PNG with:
    - Solid color background (varies by job_id hash)
    - Text overlay with job details
    - Correct dimensions
    """
```

### Realistic Metadata

Generate realistic generation metadata:
- Approximate kudos calculations
- Realistic inference times (based on params)
- Proper model names and baselines
- Valid NSFW scores

## Scenarios System

Pre-configured scenarios for testing edge cases:

```python
class MockScenario(Enum):
    """Predefined mock scenarios."""

    HAPPY_PATH = "happy_path"           # All jobs succeed quickly
    RANDOM_FAILURES = "random_failures"  # 5% failure rate
    SLOW_INFERENCE = "slow_inference"    # All jobs take 3x longer
    STUCK_PROCESS = "stuck_process"      # One process gets stuck
    DOWNLOAD_FAILURES = "download_fail"  # Model downloads fail
    MEMORY_PRESSURE = "memory_pressure"  # Simulate high memory usage
    RAPID_FIRE = "rapid_fire"            # Very fast for stress testing
```

### Usage

```python
# In config or CLI
bridge_data.mock_scenario = "random_failures"

# Or programmatically
mock_process.apply_scenario(MockScenario.SLOW_INFERENCE)
```

## Process Factory Pattern

Modify `worker_entry_points.py` to support factory pattern:

```python
def start_inference_process_or_mock(
    enable_mock: bool,
    mock_config: MockConfig,
    **process_args
) -> None:
    """Start either a real or mock inference process."""

    if enable_mock:
        start_mock_inference_process(mock_config=mock_config, **process_args)
    else:
        start_inference_process(**process_args)
```

Alternatively, use a cleaner factory:

```python
class ProcessFactory:
    """Factory for creating real or mock processes."""

    @staticmethod
    def create_inference_process(
        bridge_data: reGenBridgeData,
        **args
    ) -> HordeProcess:
        if bridge_data.enable_mock_processes:
            return MockInferenceProcess(**args, mock_config=...)
        else:
            return HordeInferenceProcess(**args)
```

## Testing Benefits

With mock processes, you can:

1. **Rapid UI Iteration**: Test terminal UI without waiting for real inference
2. **Event System Testing**: Verify all events are emitted correctly
3. **Edge Case Simulation**: Test failure handling, stuck processes, etc.
4. **CI/CD Integration**: Run worker tests without GPU infrastructure
5. **Performance Profiling**: Measure overhead of worker orchestration
6. **Stress Testing**: Run many jobs quickly to find race conditions

## Example Usage

### Quick Test Run

```bash
# Start worker in mock mode with 10x speed
python run_worker.py --mock --mock-speed=10.0

# Or via env vars
export AIWORKER_ENABLE_MOCK_PROCESSES=true
export AIWORKER_MOCK_SPEED_MULTIPLIER=20.0
python run_worker.py
```

### Config File

```yaml
# bridgeData.yaml
enable_mock_processes: true
mock_speed_multiplier: 10.0
mock_enable_failures: true
mock_failure_rate: 0.1  # 10% failure rate
```

### Programmatic

```python
from horde_worker_regen.bridge_data import reGenBridgeData
from horde_worker_regen.process_management.mock import MockConfig

bridge_data = reGenBridgeData(
    enable_mock_processes=True,
    mock_speed_multiplier=5.0,
)

# Worker will use mock processes automatically
```

## Comparison: Real vs Mock

| Aspect | Real Process | Mock Process |
|--------|-------------|--------------|
| **GPU Required** | Yes (CUDA/ROCm) | No |
| **Startup Time** | 30-60 seconds | <1 second |
| **Inference Time** | 5-30 seconds | 0.5-3 seconds (adjustable) |
| **Memory Usage** | 4-24 GB VRAM | ~100 MB RAM |
| **Image Quality** | Photo-realistic | Placeholder with metadata |
| **Dependencies** | torch, hordelib, comfyui | Pure Python stdlib |
| **Scenarios** | Real failures only | Configurable scenarios |
| **Testing Speed** | Slow (real-time) | Fast (10-100x faster) |

## Migration Path

### Phase 1: Basic Mock Implementation
- ✅ Design mock architecture (this document)
- [ ] Implement MockInferenceProcess
- [ ] Implement MockSafetyProcess
- [ ] Add basic configuration support

### Phase 2: Enhanced Features
- [ ] Implement scenario system
- [ ] Add fake data generators
- [ ] Create factory pattern
- [ ] Add CLI flags

### Phase 3: Integration
- [ ] Wire up to main process manager
- [ ] Add tests using mock processes
- [ ] Document usage
- [ ] Add CI/CD examples

## Security Considerations

⚠️ **IMPORTANT**: Mock mode should NEVER be used in production!

- Add clear warnings when mock mode is enabled
- Log prominently at startup
- Consider adding `--i-understand-this-is-for-testing-only` flag
- Disable mock mode if API key is production key (optional safety check)

## Implementation Notes

### Message Protocol Compatibility

Mock processes MUST send identical message types and sequences:
- Use real `HordeProcessMessage` subclasses
- Maintain correct message timing/ordering
- Include all required fields in messages

### Fake Image Format

```python
# Example fake image generation
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def create_fake_image(width, height, text):
    img = Image.new('RGB', (width, height), color='#336699')
    draw = ImageDraw.Draw(img)
    # Add text with job info
    draw.text((10, 10), text, fill='white')

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

### Timing Calculations

```python
def calculate_mock_inference_time(steps, width, height, speed_multiplier):
    """Calculate realistic mock inference time."""
    # Base time per step (milliseconds)
    base_time_per_step = 100  # 0.1s

    # Adjust for resolution (higher res = slower)
    megapixels = (width * height) / 1_000_000
    resolution_multiplier = 0.5 + (megapixels * 0.5)

    # Total time
    total_ms = steps * base_time_per_step * resolution_multiplier

    # Apply speed multiplier
    return total_ms / speed_multiplier / 1000.0  # Convert to seconds
```

## Questions & Decisions

### Q: Should mock processes download real models?
**A**: No. Mock processes should skip all model downloads and loading. Use a registry of "known models" and pretend they exist.

### Q: Should safety process use real NSFW detection?
**A**: No. Return randomized but realistic NSFW scores. Or always safe for testing.

### Q: How to handle LoRAs and ControlNet in mock mode?
**A**: Acknowledge them in logs but skip actual loading. Add small timing delays.

### Q: Should we simulate process crashes?
**A**: Yes, as an optional scenario. Process can exit with error code after N jobs.

## Summary

The mock process system enables:
- ✅ GPU-free testing
- ✅ Rapid development iteration
- ✅ CI/CD integration
- ✅ Edge case simulation
- ✅ Performance profiling
- ✅ Terminal UI development

All while maintaining compatibility with the real process protocol and message flow.
