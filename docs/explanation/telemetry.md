# Telemetry (OpenTelemetry / Logfire)

The worker and `hordelib` are instrumented with [Pydantic Logfire](https://logfire.pydantic.dev/)
(an OpenTelemetry SDK). This powers optional trace/metric export to a local collector
(Jaeger + Prometheus) or the Logfire cloud, used for performance attribution during development.

## Tracing is OFF by default, and that is deliberate

**`hordelib` creates a span for *every* ComfyUI internal operation** (node execution, model
loads, sampling, VAE, CLIP encode, …); on the order of **hundreds of spans per generated image**.
When the OpenTelemetry SDK is active, those spans are built and processed on background threads.
In CPython those threads contend for the **GIL**, and they do so *while the inference loop is
trying to run*: encoding the result image, building the IPC message, and (critically) launching
CUDA kernels.

### Measured impact

On an sd15, 4-thread / 2-queue soak (6 inference processes, one GPU), with tracing left on but
**no collector running** (the common accidental case):

| Metric | Tracing ON | Tracing OFF | Δ |
|---|--:|--:|--:|
| Throughput (jobs/s) | 0.261 | 0.313 | **+20%** |
| GPU duty-cycle coverage (`busy_fraction`) | 0.86 | 0.93 | **+0.07** |
| Mean GPU utilization | 55.7% | 68.8% | **+13 pp** |
| Per-job result-encode + IPC | ~1.0 s | ~0.01 s | **−99%** |

The ~1 s/job was almost entirely GIL contention from span processing; a trivial ~1 ms base64
encode was being stalled to ~0.5 s because a background OTel thread held the GIL. It is *not* a
hardware limit; it is wasted work.

## Policy: explicit opt-in, hard default-off

Because the cost is paid even with **no collector listening**, and because a developer may carry
ambient `OTEL_*` / `LOGFIRE_*` settings in their shell or system environment, the worker does not
*hope* tracing is off; it **forces it off**:

- `horde_worker_regen.telemetry.enforce_telemetry_default_off()` sets the standard
  `OTEL_SDK_DISABLED=true` kill switch unless tracing is explicitly opted in. It **hard-overrides**
  any ambient `OTEL_SDK_DISABLED=false` / `OTEL_EXPORTER_OTLP_*`.
- It is called as early as possible, before any `hordelib`/`logfire` import, in every launcher
  and worker entry point (`run_worker.py`, the benchmark `level_runner`, and both child entry
  points in `worker_entry_points.py`), so the kill switch is read when the SDK initialises and is
  inherited by every spawned child process.
- `logfire.configure(..., console=False)` additionally disables per-span console printing, which
  is never useful in a worker and is its own (smaller) source of overhead.

## Enabling tracing (development / attribution)

Opt in **only when a collector is actually running** to consume the spans:

```bash
# Start your collector first (e.g. Jaeger on :4318, Prometheus on :9090), then:
export AIWORKER_REGEN_ENABLE_TELEMETRY=1
# Optional explicit endpoints (otherwise hordelib's local OTLP defaults are used):
#   OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
# Or send to the Logfire cloud:
#   export LOGFIRE_TOKEN=...
```

With `AIWORKER_REGEN_ENABLE_TELEMETRY` set to a truthy value (`1`/`true`/`yes`/`on`), the worker
leaves the OTel SDK enabled and `logfire.configure()` governs export as normal.

> **Always run a collector when telemetry is enabled.** Generating spans only to drop them (no
> collector) incurs the full GIL cost for zero benefit.

## Always-on, low-cost metrics are separate

Per-job phase timing used by the benchmark (`hordelib.metrics.MetricsCollector`) and the NVML GPU
duty-cycle sampler are **not** OpenTelemetry and are unaffected by this switch; they remain
available with tracing off.
