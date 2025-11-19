# Mock Process System for GPU-Free Testing

## Overview

The mock process system allows testing the worker's terminal UI, event system, and orchestration logic **without requiring actual GPU hardware**. Mock processes simulate realistic worker behavior with configurable timing and failure scenarios.

## Current Status

### ✅ Completed

- [x] Architecture design (see `DESIGN.md`)
- [x] Mock data generator (`mock_data_generator.py`)
  - Fake image generation with metadata overlay
  - Kudos calculation
  - Inference time estimation
  - NSFW/CSAM score generation
- [x] Configuration system (`mock_config.py`)
  - `MockConfig` dataclass with all settings
  - `MockScenario` enum for predefined test scenarios
  - Bridge data integration

### 🚧 To Be Implemented

- [ ] `MockInferenceProcess` - Simulates image generation
- [ ] `MockSafetyProcess` - Simulates safety checking
- [ ] Entry point functions (`start_mock_inference_process`, etc.)
- [ ] Integration with `worker_entry_points.py`
- [ ] CLI flags (`--mock`, `--mock-speed`, etc.)
- [ ] Bridge data field additions

## Features

### Fake Image Generation

```python
from horde_worker_regen.process_management.mock import generate_fake_image

# Generate a placeholder image
image_b64 = generate_fake_image(
    width=512,
    height=512,
    job_id="abc123",
    model_name="SDXL",
    seed=42,
    steps=20,
)
```

**Output**: Base64-encoded PNG with:
- Colored background (varies by job ID)
- Text overlay with job metadata
- "MOCK DATA" watermark
- Correct dimensions

### Configuration

```python
from horde_worker_regen.process_management.mock import MockConfig, MockScenario

# Create config
config = MockConfig(
    speed_multiplier=10.0,  # 10x faster
    enable_failures=True,
    failure_rate=0.1,  # 10% failure rate
)

# Or use a scenario
config.apply_scenario(MockScenario.RAPID_FIRE)  # 100x speed
```

### Available Scenarios

| Scenario | Description | Use Case |
|----------|-------------|----------|
| `HAPPY_PATH` | All jobs succeed, normal speed | Basic functionality testing |
| `RANDOM_FAILURES` | 10% job failure rate | Error handling testing |
| `SLOW_INFERENCE` | 3x slower jobs | Timeout testing |
| `STUCK_PROCESS` | Process hangs after 5 jobs | Deadlock detection testing |
| `DOWNLOAD_FAILURES` | 20% download failures | Download retry testing |
| `MEMORY_PRESSURE` | High memory usage | Memory management testing |
| `RAPID_FIRE` | 100x faster | Stress testing, rapid iteration |

## Implementation Plan

### Phase 1: Basic Mock Processes ✅ (Foundation Complete)

Foundation is laid with design docs, data generators, and configuration.

### Phase 2: Mock Process Implementation (Next Steps)

Need to implement:

#### 2.1 MockInferenceProcess

```python
class MockInferenceProcess(HordeProcess):
    """Mock inference process that simulates image generation."""

    def __init__(self, ..., mock_config: MockConfig):
        super().__init__(...)
        self.mock_config = mock_config
        self.jobs_completed = 0

    @override
    def worker_cycle(self) -> None:
        """Handle control messages and simulate work."""
        # Process downloads, preloads, inference jobs
        # Send realistic message sequences
        # Use sleep() with configurable timing
        # Generate fake images using mock_data_generator

    def _handle_download_model(self, message: HordeControlModelMessage):
        """Simulate model download with progress updates."""
        # Send download progress: 0% → 100%
        # Use configured download speed
        # Simulate failures if enabled

    def _handle_preload_model(self, message: HordePreloadInferenceModelMessage):
        """Simulate model loading into memory."""
        # Send state changes: PRELOADING → PRELOADED
        # Simulate load time based on model type
        # Update simulated memory usage

    def _handle_start_inference(self, message: HordeInferenceControlMessage):
        """Simulate inference job."""
        # Send heartbeats with step progress
        # Calculate realistic timing
        # Generate fake images
        # Send result message
```

Key implementation notes:
- Inherit from `HordeProcess` (NOT from `HordeInferenceProcess`)
- Skip all HordeLib/ComfyUI imports
- Send identical message sequences as real process
- Use `time.sleep()` for timing simulation
- Track state internally to know what messages to send

#### 2.2 MockSafetyProcess

```python
class MockSafetyProcess(HordeProcess):
    """Mock safety process that simulates NSFW checking."""

    def __init__(self, ..., mock_config: MockConfig):
        super().__init__(...)
        self.mock_config = mock_config

    @override
    def worker_cycle(self) -> None:
        """Handle safety evaluation requests."""
        # Wait for EVALUATE_SAFETY messages
        # Sleep for configured safety_check_time
        # Generate fake NSFW/CSAM scores
        # Send safety result
```

#### 2.3 Entry Points

```python
# In worker_entry_points.py or new file

def start_mock_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    mock_config: MockConfig,
    ...,
) -> None:
    """Start a mock inference process."""
    from horde_worker_regen.process_management.mock import MockInferenceProcess

    worker_process = MockInferenceProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        mock_config=mock_config,
        # No GPU-related parameters
    )

    worker_process.main_loop()
```

### Phase 3: Integration

#### 3.1 Bridge Data Fields

Add to `bridge_data/data_model.py`:

```python
class reGenBridgeData(CombinedHordeBridgeData):
    # ... existing fields ...

    # Mock mode
    enable_mock_processes: bool = False
    mock_speed_multiplier: float = 1.0
    mock_enable_failures: bool = False
    mock_failure_rate: float = 0.05
    mock_enable_slowdowns: bool = False
    mock_slowdown_rate: float = 0.1
    mock_slowdown_multiplier: float = 3.0
    mock_vram_usage_mb: int = 8192
    mock_ram_usage_mb: int = 4096
    mock_scenario: str | None = None  # "RAPID_FIRE", "RANDOM_FAILURES", etc.
```

#### 3.2 Process Manager Integration

Modify `HordeWorkerProcessManager` to choose real vs mock:

```python
def _launch_inference_process(self, process_id: int, ...):
    if self.bridge_data.enable_mock_processes:
        # Launch mock process
        mock_config = MockConfig.from_bridge_data(self.bridge_data)
        self.mp_context.Process(
            target=start_mock_inference_process,
            args=(process_id, ..., mock_config),
        ).start()
    else:
        # Launch real process (existing code)
        self.mp_context.Process(
            target=start_inference_process,
            args=(process_id, ...),
        ).start()
```

#### 3.3 CLI Flags

Add to `run_worker.py`:

```python
parser.add_argument("--mock", action="store_true", help="Use mock processes (no GPU required)")
parser.add_argument("--mock-speed", type=float, default=1.0, help="Mock speed multiplier")
parser.add_argument("--mock-scenario", type=str, help="Mock scenario (RAPID_FIRE, etc.)")

# Later, apply to bridge_data
if args.mock:
    bridge_data.enable_mock_processes = True
    bridge_data.mock_speed_multiplier = args.mock_speed
    if args.mock_scenario:
        bridge_data.mock_scenario = args.mock_scenario
```

## Usage Examples

### Quick Test Run

```bash
# Start worker in mock mode, 10x speed
python run_worker.py --mock --mock-speed=10.0

# Rapid fire mode for UI testing
python run_worker.py --mock --mock-scenario=RAPID_FIRE

# Test failure handling
python run_worker.py --mock --mock-scenario=RANDOM_FAILURES
```

### Config File

```yaml
# bridgeData.yaml
enable_mock_processes: true
mock_speed_multiplier: 10.0
mock_scenario: "RAPID_FIRE"
```

### Testing Terminal UI

```bash
# Start mock worker with terminal UI enabled
python run_worker.py --mock --mock-speed=20.0 --disable-terminal-ui=false

# You should see:
# - Rapid job processing
# - Fake images generated
# - Realistic kudos accumulation
# - No GPU usage
```

## Benefits

✅ **GPU-Free Testing** - No CUDA/ROCm required
✅ **Rapid Iteration** - 10-100x faster than real inference
✅ **Edge Case Simulation** - Test failures, stuck processes, etc.
✅ **CI/CD Integration** - Run worker tests in GitHub Actions
✅ **Terminal UI Development** - Quickly test UI without waiting
✅ **Event System Validation** - Verify all events fire correctly

## Testing the Current Implementation

Even without the full mock processes implemented, you can test the data generator:

```python
# Test fake image generation
from horde_worker_regen.process_management.mock.mock_data_generator import (
    generate_fake_image,
    calculate_mock_kudos,
    calculate_mock_inference_time,
)

# Generate a fake image
image_b64 = generate_fake_image(512, 512, job_id="test", model_name="SDXL")

# Calculate mock kudos
kudos = calculate_mock_kudos(512, 512, 20, has_controlnet=True)
print(f"Kudos: {kudos}")  # ~15.6

# Calculate timing
time_sec = calculate_mock_inference_time(512, 512, 20, speed_multiplier=10.0)
print(f"Time: {time_sec}s")  # ~0.2s (10x faster)

# Test config
from horde_worker_regen.process_management.mock.mock_config import MockConfig, MockScenario

config = MockConfig()
config.apply_scenario(MockScenario.RAPID_FIRE)
print(config.to_dict())
```

## Next Steps

1. **Implement `MockInferenceProcess`** (highest priority)
   - State machine for handling control messages
   - Realistic message sequences
   - Timing simulation
   - Fake image generation

2. **Implement `MockSafetyProcess`** (medium priority)
   - Safety evaluation simulation
   - NSFW/CSAM scoring

3. **Add Bridge Data Fields** (easy)
   - Add mock-related config fields
   - Document in config examples

4. **Create Entry Points** (easy)
   - `start_mock_inference_process()`
   - `start_mock_safety_process()`

5. **Integrate with Process Manager** (medium)
   - Conditional process selection
   - Warning messages when mock mode enabled

6. **Add CLI Flags** (easy)
   - `--mock`, `--mock-speed`, `--mock-scenario`

7. **Testing & Documentation** (ongoing)
   - Test mock processes with event system
   - Test with terminal UI
   - Document limitations

## Limitations

⚠️ **Mock processes are for testing only!**

- No real image generation
- No actual model loading
- Fake kudos (not submitted to real API)
- Should never be used in production
- API calls would still go to real server (unless mocked separately)

## Architecture Diagram

```
                    ┌─────────────────────────┐
                    │   Process Manager       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   enable_mock_processes?│
                    └───────┬─────────┬───────┘
                            │         │
                        No  │         │ Yes
                            │         │
                   ┌────────▼──┐   ┌─▼──────────────┐
                   │  Real     │   │  Mock          │
                   │  Process  │   │  Process       │
                   │           │   │                │
                   │ HordeLib  │   │ Fake Images    │
                   │ GPU       │   │ Sleep Timing   │
                   │ Models    │   │ No GPU         │
                   └───────────┘   └────────────────┘
                         │               │
                         └───────┬───────┘
                                 │
                        Same Message Protocol
                                 │
                         ┌───────▼────────┐
                         │  Event System  │
                         │  Terminal UI   │
                         └────────────────┘
```

## Questions?

See `DESIGN.md` for detailed architecture documentation.
