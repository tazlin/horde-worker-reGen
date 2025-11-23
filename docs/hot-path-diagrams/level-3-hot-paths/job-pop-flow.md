# Level 3: Job Pop Flow (Job Acquisition)

This diagram shows the detailed flow of how the worker acquires jobs from the AI Horde API.

**Primary File**: `process_manager.py:3606-3850` (`api_job_pop()`)

```mermaid
flowchart TD
    Start([Job Pop Loop<br/>Continuous]) --> CheckShutdown{Shutting down?}
    CheckShutdown -->|Yes| End([Exit Loop])
    CheckShutdown -->|No| CheckMaintenance{In Maintenance<br/>Mode?}

    CheckMaintenance -->|Yes| WaitMaintenance[Wait 10s]
    WaitMaintenance --> CheckShutdown

    CheckMaintenance -->|No| CheckCapacity{Has Queue<br/>Capacity?}

    CheckCapacity -->|No| WaitNoCapacity[Wait 2s]
    WaitNoCapacity --> CheckShutdown

    CheckCapacity -->|Yes| CheckProcesses{Inference/Safety<br/>Processes Available?}

    CheckProcesses -->|No| WaitNoProcesses[Wait 1s]
    WaitNoProcesses --> CheckShutdown

    CheckProcesses -->|Yes| CheckThrottle{Megapixelsteps<br/>Throttle Active?}

    CheckThrottle -->|Yes| WaitThrottle[Wait Based on<br/>Remaining Time]
    WaitThrottle --> CheckShutdown

    CheckThrottle -->|No| SelectModels[Select Models to Request<br/>Model Stickiness Logic]

    SelectModels --> BuildRequest[Build ImageGenerateJobPopRequest<br/>- Worker capabilities<br/>- Loaded models<br/>- Max pixels/batch<br/>- Threads available]

    BuildRequest --> APICall[POST /api/v2/generate/pop<br/>AI Horde API]

    APICall --> CheckResponse{Response<br/>Status?}

    CheckResponse -->|Error 429| HandleThrottle[Parse megapixelsteps_limit<br/>Calculate wait time]
    HandleThrottle --> WaitThrottle

    CheckResponse -->|Error 503| HandleMaintenance[Enter Maintenance Mode]
    HandleMaintenance --> WaitMaintenance

    CheckResponse -->|Other Error| LogError[Log Error]
    LogError --> WaitError[Wait 5s]
    WaitError --> CheckShutdown

    CheckResponse -->|Empty/No Job| NoJob[No job available]
    NoJob --> WaitNoJob[Wait 1s]
    WaitNoJob --> CheckShutdown

    CheckResponse -->|Success| ParseJob[Parse ImageGenerateJobPopResponse<br/>Job ID, model, params, etc.]

    ParseJob --> CheckSourceImage{Has Source<br/>Image?}

    CheckSourceImage -->|Yes| DownloadSource[Download Source Image<br/>_get_source_images]
    DownloadSource --> CheckDownload{Download<br/>Success?}

    CheckDownload -->|No| SubmitFault[Submit Fault to API<br/>Job Failed]
    SubmitFault --> CheckShutdown

    CheckDownload -->|Yes| CreateJobInfo
    CheckSourceImage -->|No| CreateJobInfo[Create HordeJobInfo<br/>Metadata object]

    CreateJobInfo --> EnqueueJob[Add to jobs_pending_inference<br/>Add to jobs_lookup]

    EnqueueJob --> LogSuccess[Log Job Acquired<br/>Model, Resolution, Steps]

    LogSuccess --> UpdateStats[Update Statistics<br/>Jobs Popped Counter]

    UpdateStats --> CheckShutdown

    classDef decision fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    classDef process fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef api fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef wait fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef success fill:#e8f5e9,stroke:#4caf50,stroke-width:2px

    class CheckShutdown,CheckMaintenance,CheckCapacity,CheckProcesses,CheckThrottle,CheckResponse,CheckSourceImage,CheckDownload decision
    class SelectModels,BuildRequest,ParseJob,CreateJobInfo,HandleThrottle,HandleMaintenance,DownloadSource,SubmitFault process
    class APICall api
    class WaitMaintenance,WaitNoCapacity,WaitNoProcesses,WaitThrottle,WaitError,WaitNoJob wait
    class EnqueueJob,LogSuccess,UpdateStats success
```

## Flow Stages

### 1. Pre-flight Checks (Lines 3606-3680)

**Checks performed before requesting a job:**

- **Shutdown Check**: Exit loop if worker is shutting down
- **Maintenance Mode**: Wait if API is in maintenance (auto-recovery)
- **Queue Capacity**: Ensure we have room for more jobs
  - Formula: `current_queue_size < (queue_size + max_threads)`
  - Prevents overloading the worker
- **Process Availability**: Ensure at least one inference and safety process is ready
  - Checks: `WAITING_FOR_JOB` or `PROCESS_ENDED` state
- **Throttle Check**: Respect megapixelsteps rate limiting
  - If throttled, calculate wait time and sleep

### 2. Model Selection (Lines 3680-3730)

**Model Stickiness Logic**: Prefer models already loaded to reduce load times

```
Priority Order:
1. Models loaded in VRAM (fastest)
2. Models loaded in RAM (fast)
3. Models with matching base name (medium)
4. All configured models (slow - requires download/load)
```

**Special Handling**:
- Alchemy models (upscaling/post-processing)
- SDXL vs SD 1.5 separation
- Model form support (stable diffusion, flux, etc.)

### 3. Request Building (Lines 3730-3780)

**ImageGenerateJobPopRequest Construction**:

```python
{
    "apikey": worker_api_key,
    "name": worker_name,
    "models": selected_models,  # From model stickiness
    "max_pixels": configured_max_pixels,
    "threads": available_threads,  # Based on free processes
    "allow_img2img": true/false,
    "allow_painting": true/false,
    "allow_post_processing": true/false,
    "allow_controlnet": true/false,
    "max_batch": configured_max_batch,
    # ... more capabilities
}
```

**Key Optimization**: Include loaded models in request to get prioritized jobs

### 4. API Call (Lines 3780-3800)

**HTTP Request**:
- **Method**: POST
- **Endpoint**: `/api/v2/generate/pop`
- **Timeout**: Configurable (default 30s)
- **Transport**: aiohttp async HTTP

**Response Handling**:
- **200 OK + Job**: Proceed to parse job
- **200 OK + Empty**: No job available, wait 1s
- **429 Too Many Requests**: Parse throttle info, wait
- **503 Service Unavailable**: Enter maintenance mode
- **Other Errors**: Log and retry

### 5. Error Handling (Lines 3800-3850)

**Megapixelsteps Throttling**:
```python
if status == 429:
    # Parse response
    megapixelsteps_limit = response["megapixelsteps_limit"]
    wait_time = calculate_wait_time(limit)
    # Wait before next request
```

**Maintenance Mode**:
- Triggered by 503 response
- Worker pauses job acquisition
- Continues submitting completed jobs
- Auto-recovery when API is back
- Displays status message to user

**Fault Submission**:
- If source image download fails
- If job is malformed
- Immediately submit fault to API
- Job is NOT processed

### 6. Source Image Handling (Lines 3680-3730)

**_get_source_images() Flow**:

```mermaid
flowchart LR
    Check{Source Image<br/>in Job?} -->|Yes| Download[Download from URL]
    Download --> Validate{Valid Image?}
    Validate -->|Yes| Attach[Attach to Job Info]
    Validate -->|No| Fault[Return Error]
    Check -->|No| Continue[Continue]

    classDef process fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef decision fill:#fff4e1,stroke:#ff9900,stroke-width:2px

    class Download,Attach,Fault process
    class Check,Validate decision
```

**Image Types**:
- **img2img**: Source image to transform
- **inpainting**: Image with mask
- **ControlNet**: Conditioning image

### 7. Job Enqueue (Lines 3840-3850)

**Final Steps**:
1. Create `HordeJobInfo` object with metadata
2. Add to `jobs_pending_inference` list
3. Add to `jobs_lookup` dict (keyed by job ID)
4. Log job details (model, resolution, steps)
5. Update statistics counters

## Performance Characteristics

### Timing
- **Typical API call**: 0.5-2 seconds
- **With source image download**: +1-3 seconds
- **Throttled wait**: 1-60 seconds (based on megapixelsteps)
- **No job wait**: 1 second
- **Maintenance wait**: 10 seconds

### Concurrency
- Loop runs continuously in async context
- Only one API call at a time (no parallel pops)
- Downloads can timeout after configured duration

### Optimization Features
1. **Model Stickiness**: Prioritizes jobs for loaded models
2. **Capacity Checking**: Prevents queue overload
3. **Throttle Respect**: Avoids API rate limiting
4. **Early Validation**: Checks processes before API call

## Key Variables

**File**: `process_manager.py`

- `jobs_pending_inference`: `List[ImageGenerateJobPopResponse]`
- `jobs_lookup`: `dict[str, HordeJobInfo]` (job_id → metadata)
- `_process_map`: `ProcessMap` (tracks all processes)
- `_horde_model_map`: `HordeModelMap` (tracks loaded models)
- `_api_call_loop_interval`: `float` (delay between requests)

## Related Flows

**Next Steps**:
- Jobs in `jobs_pending_inference` → [Inference Flow](inference-flow.md)

**See Also**:
- [Level 4: Model Stickiness Logic](../level-4-components/model-selection.md)
- [Level 4: Throttle Management](../level-4-components/throttle-handling.md)
