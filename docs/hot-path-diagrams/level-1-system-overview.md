# Level 1: System Overview

This diagram shows the highest-level view of the reGen Worker's place in the AI Horde ecosystem and the main system groups.

```mermaid
flowchart TB
    subgraph External["External Systems"]
        API["AI Horde API<br/>(horde.koboldai.net)"]
        R2["R2 Storage<br/>(Optional Image Uploads)"]
        Users["Horde Users<br/>(Requesting Images)"]
    end

    subgraph Worker["reGen Worker System"]
        PM["Process Manager<br/>(Main Coordinator)"]
        INF["Inference Processes<br/>(1-N Workers)"]
        SAFE["Safety Processes<br/>(1-N Checkers)"]
    end

    subgraph Resources["Local Resources"]
        GPU["GPU / VRAM<br/>(CUDA/DirectML/ROCm)"]
        Models["Model Storage<br/>(Checkpoints, LoRAs, TIs)"]
        Config["Configuration<br/>(bridgeData.yaml)"]
    end

    Users -->|"Submit Requests"| API
    API <-->|"Pop Jobs<br/>Submit Results"| PM
    PM -->|"Orchestrate"| INF
    PM -->|"Safety Check"| SAFE
    PM -->|"Upload Images"| R2

    INF -->|"Use"| GPU
    INF -->|"Load"| Models
    SAFE -->|"Use"| GPU
    PM -->|"Read"| Config

    classDef external fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef worker fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    classDef resource fill:#e8f5e9,stroke:#4caf50,stroke-width:2px

    class API,R2,Users external
    class PM,INF,SAFE worker
    class GPU,Models,Config resource
```

## System Groups

### External Systems
- **AI Horde API**: Centralized job distribution service that coordinates work across distributed workers
- **R2 Storage**: Optional cloud storage for generated images (if configured)
- **Horde Users**: End users submitting image generation requests to the Horde

### reGen Worker System
- **Process Manager**: Main coordinator process that orchestrates all worker activities
- **Inference Processes**: GPU worker processes (1-N) that generate images using Stable Diffusion
- **Safety Processes**: Checker processes (1-N) that detect NSFW/CSAM content

### Local Resources
- **GPU / VRAM**: Graphics hardware (supports CUDA, DirectML, ROCm)
- **Model Storage**: Local filesystem storage for AI models (checkpoints, LoRAs, textual inversions)
- **Configuration**: YAML configuration file defining worker capabilities and settings

## Main Hot Path Overview

The core workflow consists of four stages:

1. **Job Acquisition**: Process Manager polls API for available jobs
2. **Image Generation**: Inference processes generate images using GPU
3. **Safety Checking**: Safety processes scan for prohibited content
4. **Job Submission**: Process Manager returns completed images to API

For detailed flows, see:
- [Level 2: Major Subsystems](level-2-major-subsystems.md)
- [Level 3: Hot Path Details](level-3-hot-paths/)
