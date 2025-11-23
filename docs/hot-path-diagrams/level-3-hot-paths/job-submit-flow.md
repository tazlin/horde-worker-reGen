# Level 3: Job Submit Flow (Result Submission)

This diagram shows the detailed flow of how completed jobs are submitted back to the AI Horde API, including image upload and kudos tracking.

**Primary Files**:
- Process Manager: `process_manager.py:4097-4110` (`_job_submit_loop()`)
- Submit Function: `process_manager.py:2954-3170` (`submit_single_generation()`)

```mermaid
flowchart TD
    Start([Job Submit Loop<br/>Every 0.1s]) --> CheckShutdown{Shutting down?}

    CheckShutdown -->|Yes| DrainQueue{Jobs still in<br/>pending_submit?}
    DrainQueue -->|Yes| ProcessJob
    DrainQueue -->|No| End([Exit Loop])

    CheckShutdown -->|No| CheckQueue{Jobs in<br/>jobs_pending_submit?}

    CheckQueue -->|No| WaitLoop[Wait 0.1s]
    WaitLoop --> CheckShutdown

    CheckQueue -->|Yes| ProcessJob[Get Next Job<br/>from Queue]

    ProcessJob --> CheckR2{R2 Upload<br/>Configured?}

    CheckR2 -->|Yes| UploadImages[Upload Images to R2<br/>_upload_images_to_r2]

    subgraph R2Upload["R2 Image Upload"]
        StartUpload[For Each Image] --> EncodeImage[Ensure Base64 Encoded]

        EncodeImage --> BuildKey[Build R2 Key<br/>job_id/image_index.webp]

        BuildKey --> ConvertWebP[Convert PNG to WebP<br/>Compression]

        ConvertWebP --> S3Put[PUT to S3-Compatible API<br/>R2 Bucket]

        S3Put --> CheckUpload{Upload<br/>Success?}

        CheckUpload -->|No| RetryUpload{Retry<br/>Count < 3?}
        RetryUpload -->|Yes| S3Put
        RetryUpload -->|No| UploadFailed[Log Error<br/>Continue Without Upload]

        CheckUpload -->|Yes| StoreURL[Store R2 URL<br/>in Job Info]

        StoreURL --> NextImage{More<br/>Images?}
        NextImage -->|Yes| StartUpload
        NextImage -->|No| UploadComplete
    end

    UploadImages -.-> UploadComplete[R2 Upload Complete]

    CheckR2 -->|No| BuildSubmit
    UploadComplete --> BuildSubmit[Build JobSubmitRequest]

    BuildSubmit --> AddMetadata[Add Metadata<br/>- Job ID<br/>- State (ok/faulted)<br/>- Faults list<br/>- Seed<br/>- Generation metadata]

    AddMetadata --> AddImages{Images<br/>Uploaded to R2?}

    AddImages -->|Yes| AddR2URLs[Add R2 URLs<br/>to Request]
    AddR2URLs --> APISubmit

    AddImages -->|No| AddBase64[Add Base64 Images<br/>to Request]
    AddBase64 --> APISubmit[POST /api/v2/generate/submit<br/>AI Horde API]

    APISubmit --> CheckResponse{Response<br/>Status?}

    CheckResponse -->|Error 410| HandleGone[Job Already Complete<br/>or Timed Out]
    HandleGone --> LogWarning[Log Warning]
    LogWarning --> RemoveJob

    CheckResponse -->|Error 404| HandleNotFound[Job Not Found<br/>API Issue]
    HandleNotFound --> LogError1[Log Error]
    LogError1 --> RemoveJob

    CheckResponse -->|Error 403| HandleForbidden[Worker Not Authorized]
    HandleForbidden --> LogError2[Log Error<br/>Check API Key]
    LogError2 --> RemoveJob

    CheckResponse -->|Error 5xx| HandleServerError[API Server Error<br/>Temporary]
    HandleServerError --> CheckRetry{Retry<br/>Count < 3?}
    CheckRetry -->|Yes| WaitRetry[Wait 2s]
    WaitRetry --> APISubmit
    CheckRetry -->|No| LogError3[Log Error<br/>Give Up]
    LogError3 --> RemoveJob

    CheckResponse -->|Success 200| ParseResponse[Parse JobSubmitResponse]

    ParseResponse --> ExtractReward[Extract Kudos Reward<br/>reward.kudos]

    ExtractReward --> UpdateKudos[Update Kudos Tracking<br/>Total Earned<br/>Session Earned]

    UpdateKudos --> LogSuccess[Log Completion<br/>Job ID, Kudos, Time]

    LogSuccess --> RecordStats[Record Statistics<br/>- Total jobs completed<br/>- Total kudos earned<br/>- Job time breakdown]

    RecordStats --> CheckTraining{Kudos Training<br/>Mode Enabled?}

    CheckTraining -->|Yes| RecordTraining[Record to Training CSV<br/>job_params → kudos]
    RecordTraining --> RemoveJob

    CheckTraining -->|No| RemoveJob[Remove from<br/>jobs_pending_submit<br/>jobs_lookup]

    RemoveJob --> DisplayStatus[Update Status Display<br/>Show Current Stats]

    DisplayStatus --> CheckShutdown

    classDef decision fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    classDef process fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef subprocess fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef api fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef success fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef error fill:#ffebee,stroke:#f44336,stroke-width:2px

    class CheckShutdown,DrainQueue,CheckQueue,CheckR2,CheckUpload,RetryUpload,NextImage,AddImages,CheckResponse,CheckRetry,CheckTraining decision
    class ProcessJob,BuildSubmit,AddMetadata,ParseResponse,ExtractReward,UpdateKudos,RecordStats,RemoveJob,DisplayStatus process
    class StartUpload,EncodeImage,BuildKey,ConvertWebP,S3Put,StoreURL subprocess
    class UploadImages,AddR2URLs,AddBase64,APISubmit api
    class LogSuccess,RecordTraining success
    class HandleGone,HandleNotFound,HandleForbidden,HandleServerError,LogError1,LogError2,LogError3,LogWarning,UploadFailed error
```

## Flow Stages

### 1. Submit Loop Control (Lines 4097-4110)

**_job_submit_loop() Function**:

```python
async def _job_submit_loop():
    while True:
        if shutting_down and not jobs_pending_submit:
            break  # Exit after draining queue

        if not jobs_pending_submit:
            await asyncio.sleep(0.1)
            continue

        job_info = jobs_pending_submit[0]
        await submit_single_generation(job_info)
        await asyncio.sleep(0.01)  # Small delay between submits
```

**Loop Characteristics**:
- Runs continuously in async event loop
- Checks queue every 0.1s if empty
- Processes jobs sequentially (one at a time)
- Drains queue before shutdown

### 2. R2 Image Upload (Lines 2954-3050)

**_upload_images_to_r2() Function**:

**Configuration**:
```yaml
# bridgeData.yaml
r2_upload: true
r2_account_id: "your_account_id"
r2_access_key: "your_access_key"
r2_secret_key: "your_secret_key"
r2_bucket_name: "horde-images"
```

**Upload Process**:
```python
def _upload_images_to_r2(job_info: HordeJobInfo):
    for idx, image_result in enumerate(job_info.job_image_results):
        # Decode base64 to PIL Image
        image = base64_to_image(image_result.image)

        # Convert to WebP (smaller size)
        webp_buffer = convert_to_webp(image, quality=90)

        # Build R2 key
        key = f"{job_info.job_id}/{idx}.webp"

        # Upload to R2
        s3_client.put_object(
            Bucket=r2_bucket_name,
            Key=key,
            Body=webp_buffer,
            ContentType="image/webp"
        )

        # Store public URL
        url = f"https://{r2_bucket_name}.r2.dev/{key}"
        image_result.r2_url = url
```

**WebP Conversion Benefits**:
- Smaller file size (30-50% reduction vs PNG)
- Faster upload
- Faster download for end users
- Lower storage costs

**Error Handling**:
- Retry up to 3 times on failure
- Continue without upload if all retries fail
- Fall back to base64 submission

### 3. Submit Request Building (Lines 3050-3100)

**JobSubmitRequest Construction**:

```python
submit_request = JobSubmitRequest(
    apikey=worker_api_key,
    id=job_info.job_id,
    state=job_info.state,  # "ok" or "faulted"
    seed=job_info.seed,
    generation_time=job_info.time_elapsed,

    # Image data (either R2 URLs or base64)
    generations=[
        {
            "img": r2_url or base64_image,
            "seed": image_seed,
            "state": "ok" or "faulted",
            "generation_faults": ["nsfw", "csam", ...],
            "censored": true/false,
        }
        for each image in job
    ]
)
```

**State Values**:
- `"ok"`: Job completed successfully
- `"faulted"`: Job completed with faults (NSFW, CSAM, errors)

**Generation Metadata**:
- Seed used for each image
- Faults detected (NSFW, CSAM, etc.)
- Censorship flag
- Generation time

### 4. API Submission (Lines 3100-3150)

**HTTP Request**:
```python
response = await api_client.submit_generation(submit_request)
# POST /api/v2/generate/submit
```

**Request Size**:
- **With R2**: ~1-5 KB (just URLs and metadata)
- **Without R2**: 1-10 MB (base64 images + metadata)

**Timeout**: 30s (configurable)

### 5. Error Handling (Lines 3150-3200)

**HTTP Status Codes**:

**410 Gone** - Job Already Complete:
```python
# Job was already submitted or timed out
# This is not an error, just log and continue
logger.warning(f"Job {job_id} already complete")
```

**404 Not Found** - Job Not Found:
```python
# API doesn't know about this job
# Possible API database issue
logger.error(f"Job {job_id} not found in API")
```

**403 Forbidden** - Not Authorized:
```python
# Worker API key invalid or worker suspended
logger.error(f"Worker not authorized for job {job_id}")
# Check API key configuration
```

**5xx Server Error** - API Temporary Error:
```python
# API is having issues
# Retry up to 3 times with exponential backoff
for retry in range(3):
    await asyncio.sleep(2 ** retry)
    response = await api_client.submit_generation(submit_request)
    if response.ok:
        break
```

**Retry Strategy**:
- **410, 404, 403**: No retry (permanent errors)
- **5xx**: Retry with backoff (2s, 4s, 8s)
- **Network errors**: Retry with backoff

### 6. Kudos Tracking (Lines 3200-3250)

**JobSubmitResponse Parsing**:
```python
response = JobSubmitResponse(
    reward={
        "kudos": 123.45  # Kudos earned for this job
    }
)
```

**Kudos Calculation** (done by API):
- Base kudos = resolution * steps * batch_size / 1000
- Modifiers:
  - Model complexity (SDXL = higher)
  - Post-processing (upscale, face fix)
  - ControlNet usage
  - Trusted user bonus
  - Worker priority

**Tracking Updates**:
```python
# Update session stats
total_kudos_earned += kudos
session_kudos_earned += kudos
jobs_completed += 1

# Update moving averages
avg_kudos_per_job = total_kudos / jobs_completed
avg_job_time = total_time / jobs_completed
kudos_per_hour = kudos / (uptime / 3600)
```

### 7. Statistics Recording (Lines 3250-3300)

**Logged Statistics**:
- **Job Info**: Job ID, model, resolution, steps
- **Timing**: Total time, preload time, inference time, safety time
- **Rewards**: Kudos earned, total kudos
- **Faults**: Any faults detected
- **Success**: True/false

**Status Display Update**:
```
┌─────────────────────────────────────┐
│ Worker: worker-name                 │
│ Status: Working                     │
│                                     │
│ Jobs Completed: 1234                │
│ Total Kudos: 12345.67              │
│ Session Kudos: 234.56              │
│ Avg Kudos/Job: 10.01               │
│ Kudos/Hour: 89.23                  │
│                                     │
│ Queue: 2 pending, 1 in progress    │
└─────────────────────────────────────┘
```

### 8. Kudos Training Mode (Lines 3300-3350)

**Training Data Collection** (optional feature):

```python
if kudos_training_mode:
    # Record job parameters and kudos earned
    record = {
        "model": job_info.model,
        "width": job_info.width,
        "height": job_info.height,
        "steps": job_info.steps,
        "sampler": job_info.sampler,
        "cfg_scale": job_info.cfg_scale,
        "post_processing": job_info.post_processing,
        "kudos_earned": kudos,
        "time_elapsed": job_info.time_elapsed,
    }

    # Append to CSV
    with open("kudos_training.csv", "a") as f:
        writer.writerow(record)
```

**Purpose**:
- Collect data for kudos prediction models
- Analyze which job types are most profitable
- Optimize worker configuration

**File**: `reporting/kudos_training_recorder.py`

### 9. Cleanup (Lines 3350-3400)

**Queue Cleanup**:
```python
# Remove from pending submit queue
jobs_pending_submit.remove(job_info)

# Remove from global job lookup
del jobs_lookup[job_info.job_id]

# Update process state (if needed)
if process_ended:
    replace_process(process_info)
```

**Memory Management**:
- Remove job from all tracking structures
- Free image data (can be large)
- Update process availability

## Performance Characteristics

### Timing

**Typical Submission**:
- **With R2 upload**: 1-5s
  - Image upload: 0.5-3s (depends on size and network)
  - API submit: 0.5-2s
- **Without R2**: 0.5-3s
  - API submit with base64: 0.5-3s (larger payload)

**Batch Job (4 images)**:
- **With R2**: 2-10s (parallel upload possible)
- **Without R2**: 1-5s (larger base64 payload)

### Throughput

**Submit Rate**:
- Process 1 job per 0.01s (minimum delay)
- Effective rate: ~100 jobs/second (theoretical max)
- Practical rate: 10-30 jobs/minute (network limited)

**Concurrent Submits**:
- Loop processes sequentially (one at a time)
- Could be parallelized for higher throughput
- Current design is simpler and sufficient

### Network Usage

**Upload Bandwidth**:
- **With R2**: 0.5-2 MB/job (upload to R2) + 1-5 KB/job (submit to API)
- **Without R2**: 1-10 MB/job (submit to API with base64)

**R2 Benefits**:
- Reduced API bandwidth (just URLs)
- Faster image delivery to end users
- Lower API server load

## Configuration Options

**bridgeData.yaml**:
```yaml
# R2 upload settings
r2_upload: true
r2_account_id: "your_account_id"
r2_access_key: "your_access_key"
r2_secret_key: "your_secret_key"
r2_bucket_name: "horde-images"

# Submit settings
submit_timeout: 30             # API submit timeout (seconds)
max_submit_retries: 3          # Retry count for 5xx errors
submit_interval: 0.01          # Delay between submits (seconds)

# Kudos tracking
kudos_training_mode: false     # Enable training data collection
```

## Key Variables

**Process Manager** (`process_manager.py`):
- `jobs_pending_submit`: `List[HordeJobInfo]`
- `jobs_lookup`: `dict[str, HordeJobInfo]`
- `total_kudos_earned`: `float`
- `session_kudos_earned`: `float`
- `jobs_completed`: `int`

**R2 Client**:
- `s3_client`: boto3 S3 client (configured for R2)
- `r2_bucket_name`: `str`

## Related Flows

**Previous Step**:
- [Safety Check Flow](safety-check-flow.md) → jobs_pending_submit
- [Inference Flow](inference-flow.md) → jobs_pending_submit (if faulted)

**End of Hot Path**:
- Job is complete and removed from all queues
- Worker is ready to accept new jobs

**See Also**:
- [Level 4: R2 Integration](../level-4-components/r2-upload.md)
- [Level 4: Kudos Calculation](../level-4-components/kudos-tracking.md)
- [Level 4: Error Recovery](../level-4-components/error-handling.md)
