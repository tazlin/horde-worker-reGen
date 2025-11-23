# Hot Path Diagrams - reGen Worker

This directory contains hierarchical diagrams documenting the hot paths (critical execution flows) in the reGen Worker system. The diagrams are organized by level of detail, from high-level system overview to detailed component implementations.

## Quick Navigation

### By Level of Detail

- **[Level 1: System Overview](level-1-system-overview.md)** - Highest level view of the worker in the Horde ecosystem
- **[Level 2: Major Subsystems](level-2-major-subsystems.md)** - Main components and async control loops
- **[Level 3: Hot Path Details](level-3-hot-paths/)** - Detailed flows for each critical operation
- **[Level 4: Component Details](level-4-components/)** - Implementation-level component diagrams

### By Hot Path

1. **[Job Acquisition](level-3-hot-paths/job-pop-flow.md)** - How jobs are fetched from the API
2. **[Image Generation](level-3-hot-paths/inference-flow.md)** - How images are generated
3. **[Safety Checking](level-3-hot-paths/safety-check-flow.md)** - How NSFW/CSAM detection works
4. **[Job Submission](level-3-hot-paths/job-submit-flow.md)** - How results are returned to the API

### By Component

- **[Model Management](level-4-components/model-management.md)** - Model loading, unloading, and stickiness
- **[Process State Machine](level-4-components/process-state-machine.md)** - Process lifecycle and states
- **[Inter-Process Communication](level-4-components/inter-process-communication.md)** - Messages, pipes, and queues
- **[Semaphore Control](level-4-components/semaphore-control.md)** - Concurrency and VRAM management

---

## Documentation Hierarchy

The diagrams are organized in a hierarchical structure, with each level providing progressively more detail:

```
Level 1: System Overview (1 diagram)
    ↓
Level 2: Major Subsystems (1 diagram)
    ↓
Level 3: Hot Path Details (4 diagrams)
    ├── Job Pop Flow
    ├── Inference Flow
    ├── Safety Check Flow
    └── Job Submit Flow
    ↓
Level 4: Component Details (4 diagrams)
    ├── Model Management
    ├── Process State Machine
    ├── Inter-Process Communication
    └── Semaphore Control
```

---

## Level 1: System Overview

**Purpose**: Understand the worker's place in the AI Horde ecosystem

**Audience**: Anyone new to the project, product managers, stakeholders

**What you'll learn**:
- The three main system groups (External Systems, Worker System, Local Resources)
- High-level data flow (jobs in, images out)
- Main components (Process Manager, Inference Processes, Safety Processes)

**Start here if**: You want a bird's-eye view of the entire system

**[→ View Level 1 Diagram](level-1-system-overview.md)**

---

## Level 2: Major Subsystems

**Purpose**: Understand the internal architecture and control flow

**Audience**: Developers, architects, technical leads

**What you'll learn**:
- The five async control loops that orchestrate the worker
- Job queue progression (pending → in progress → safety → submit)
- State management (ProcessMap, ModelMap, JobsLookup)
- Inter-process communication patterns

**Start here if**: You want to understand the overall architecture before diving into code

**[→ View Level 2 Diagram](level-2-major-subsystems.md)**

---

## Level 3: Hot Path Details

**Purpose**: Understand the detailed flow of each critical operation

**Audience**: Developers implementing features, debugging issues, optimizing performance

**What you'll learn**:
- Step-by-step execution flow with decision points
- Error handling and fault recovery
- Performance characteristics and timing
- Key files and line numbers

**Start here if**: You're working on a specific hot path or debugging an issue

### 3.1 Job Pop Flow (Job Acquisition)

**File**: `process_manager.py:3606-3850` (`api_job_pop()`)

**Flow**:
1. Pre-flight checks (capacity, processes available, throttling)
2. Model selection (model stickiness optimization)
3. Build job request with worker capabilities
4. Call AI Horde API (`POST /api/v2/generate/pop`)
5. Download source images if needed
6. Enqueue job for processing

**Key Optimizations**:
- Model stickiness (prefer loaded models)
- Capacity checking (prevent overload)
- Throttle respect (megapixelsteps rate limiting)

**[→ View Job Pop Flow Diagram](level-3-hot-paths/job-pop-flow.md)**

### 3.2 Inference Flow (Image Generation)

**Files**:
- Process Manager: `process_manager.py:4150-4300` (`_process_control_loop()`)
- Inference Process: `inference_process.py:507-829` (`start_inference()`)

**Flow**:
1. Match job with available process
2. Preload model if needed (download, load to RAM, load to VRAM)
3. Acquire inference semaphore
4. Run sampling loop (denoising steps)
5. Release inference semaphore
6. Acquire VAE decode semaphore
7. VAE decode and post-processing
8. Release VAE semaphore
9. Send results to manager

**Key Optimizations**:
- Semaphore overlap (start new job while post-processing)
- Model preloading (async ahead of job)
- Memory modes (high/very high memory)

**[→ View Inference Flow Diagram](level-3-hot-paths/inference-flow.md)**

### 3.3 Safety Check Flow (NSFW/CSAM Detection)

**Files**:
- Process Manager: `process_manager.py:2800-2950` (`start_evaluate_safety()`)
- Safety Process: `safety_process.py:171-250` (`evaluate_safety()`)

**Flow**:
1. Get job from safety queue
2. Find available safety process
3. Send images to safety process
4. Stage 1: Interrogator (CLIP-based NSFW detection)
5. Stage 2: Deep Danbooru (tag-based confirmation, if needed)
6. CSAM detection (if enabled)
7. Apply censorship (blur NSFW, replace CSAM with black)
8. Return censored images and faults

**Key Features**:
- Two-stage NSFW detection (reduces false positives)
- Highly sensitive CSAM detection (zero tolerance)
- Configurable censoring (per job and per worker)

**[→ View Safety Check Flow Diagram](level-3-hot-paths/safety-check-flow.md)**

### 3.4 Job Submit Flow (Result Submission)

**Files**:
- Submit Loop: `process_manager.py:4097-4110` (`_job_submit_loop()`)
- Submit Function: `process_manager.py:2954-3170` (`submit_single_generation()`)

**Flow**:
1. Get job from submit queue
2. Upload images to R2 (if configured)
3. Build job submit request
4. Call AI Horde API (`POST /api/v2/generate/submit`)
5. Handle errors (retry 5xx, log 4xx)
6. Parse kudos reward from response
7. Update kudos tracking and statistics
8. Record training data (if enabled)
9. Remove job from queues

**Key Features**:
- R2 image upload (reduces API bandwidth)
- Kudos tracking (earn rewards)
- Error recovery (retries with backoff)
- Statistics recording

**[→ View Job Submit Flow Diagram](level-3-hot-paths/job-submit-flow.md)**

---

## Level 4: Component Details

**Purpose**: Understand the implementation details of key components

**Audience**: Developers implementing features, optimizing performance, debugging complex issues

**What you'll learn**:
- Data structures and algorithms
- State machines and transitions
- Concurrency patterns
- Performance characteristics

**Start here if**: You're implementing a feature or need to understand a specific component deeply

### 4.1 Model Management

**Topics**:
- Model state lifecycle (NotLoaded → Downloading → InRAM → InVRAM)
- Model stickiness algorithm (prioritize loaded models)
- Model preloading flow (download, load to RAM, load to VRAM)
- Model unloading strategies (memory modes)
- Performance optimizations (caching, aux model locks)

**Key Data Structures**:
- `HordeModelMap`: Tracks which models are loaded in which processes
- `LoadedModelInfo`: Metadata about loaded models
- `HordeModelState`: Model state enum

**[→ View Model Management Diagram](level-4-components/model-management.md)**

### 4.2 Process State Machine

**Topics**:
- Inference process state machine (all states and transitions)
- Safety process state machine
- State change message flow
- Heartbeat mechanism (detect hung processes)
- Error handling and recovery
- Process lifecycle (startup, work loop, shutdown)

**Key States**:
- `WAITING_FOR_JOB`: Idle, ready for work
- `INFERENCE_RUNNING`: Actively generating images
- `EVALUATING_SAFETY`: Checking for NSFW/CSAM
- `ERROR`: Temporary error state

**[→ View Process State Machine Diagram](level-4-components/process-state-machine.md)**

### 4.3 Inter-Process Communication

**Topics**:
- IPC architecture (pipes and queues)
- Message types (Control, State, Result, Heartbeat)
- Message flow examples (preload, inference, safety)
- Message serialization (pickle)
- Error handling (broken pipes, queue errors)

**Communication Channels**:
- **Pipe (Manager → Child)**: Send commands to child processes
- **Queue (Child → Manager)**: Receive status and results from children

**[→ View Inter-Process Communication Diagram](level-4-components/inter-process-communication.md)**

### 4.4 Semaphore Control

**Topics**:
- Semaphore architecture (inference and VAE decode semaphores)
- High performance mode (job overlap)
- VRAM usage analysis
- Semaphore acquisition order (prevent deadlock)
- Error handling with semaphores
- Multi-process concurrency

**Semaphore Types**:
- **inference_semaphore**: Limit concurrent sampling (most VRAM-intensive)
- **vae_decode_semaphore**: Limit concurrent VAE decode (allows overlap)

**[→ View Semaphore Control Diagram](level-4-components/semaphore-control.md)**

---

## Diagram Format

All diagrams are written in **Mermaid** format, which renders natively in:
- GitHub
- GitLab
- VS Code (with Mermaid extension)
- Many documentation platforms

To view diagrams locally:
1. Install a Mermaid-compatible viewer (VS Code extension, browser extension, etc.)
2. Or use the Mermaid Live Editor: https://mermaid.live/

---

## Suggested Reading Paths

### Path 1: New to the Project
1. [Level 1: System Overview](level-1-system-overview.md)
2. [Level 2: Major Subsystems](level-2-major-subsystems.md)
3. Pick a hot path from Level 3 that interests you

### Path 2: Debugging an Issue
1. Identify which hot path is affected
2. Read the corresponding Level 3 diagram
3. Dive into Level 4 components as needed

### Path 3: Implementing a Feature
1. [Level 2: Major Subsystems](level-2-major-subsystems.md) - understand the architecture
2. Relevant Level 3 hot paths - understand the flow
3. Relevant Level 4 components - understand the implementation

### Path 4: Performance Optimization
1. [Level 3: Inference Flow](level-3-hot-paths/inference-flow.md) - main bottleneck
2. [Level 4: Model Management](level-4-components/model-management.md) - loading optimizations
3. [Level 4: Semaphore Control](level-4-components/semaphore-control.md) - concurrency optimizations

---

## Key Performance Metrics

**Typical Job Timeline** (512x512, 30 steps, SD 1.5):
- Job pop: 0.5-2s
- Model preload (first time): 15-40s
- Model preload (cached): 3-10s
- Inference sampling: 10-40s
- VAE decode: 1-3s
- Post-processing: 2-20s (if enabled)
- Safety check: 0.5-3s
- Job submit: 0.5-3s
- **Total first job**: 30-110s
- **Total subsequent jobs**: 15-70s

**Throughput** (with high performance mode):
- Single process: 90-103 jobs/hour
- Two processes: 180-206 jobs/hour

---

## Related Documentation

- **Main README**: `../../README.md` - Project overview and setup
- **Configuration**: `../../bridgeData_template.yaml` - Configuration options
- **API Documentation**: AI Horde API docs - https://aihorde.net/api/

---

## Contributing

When updating these diagrams:
1. Ensure Mermaid syntax is valid (test at https://mermaid.live/)
2. Keep the hierarchy consistent (Level 1 → 2 → 3 → 4)
3. Update cross-references when adding new diagrams
4. Include file paths and line numbers for code references
5. Add performance metrics where applicable

---

## Feedback

Found an issue or have suggestions for improving these diagrams?
- Open an issue on GitHub
- Submit a pull request with corrections
- Ask questions in discussions

---

**Last Updated**: 2025-11-23
**Diagram Count**: 10 diagrams across 4 levels
**Coverage**: Complete coverage of all hot paths
