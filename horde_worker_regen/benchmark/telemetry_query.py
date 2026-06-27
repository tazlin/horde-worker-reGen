"""Read-only Prometheus + Jaeger query helpers for GPU-duty-cycle idle attribution.

The benchmark's NVML sampler answers *how much* the GPU idled; this module answers *where* the
time went by querying the telemetry the worker and hordelib already emit:

- **Jaeger** (``:16686``) holds the per-span timeline (``comfy.internal.sample``, ``vae_decode``,
  ``load_models_gpu``, etc. and the worker's ``job.*`` spans). From a trace we can compute the
  union of GPU-busy intervals and, by subtraction from the wall-clock, the idle gaps.
- **Prometheus** (``:9090``) holds the histograms/gauges (``comfy_*``, ``job_*``,
  ``gpu_busy_percent``) for aggregate queries over a time range.

Everything degrades gracefully: if an endpoint is unreachable the calls return ``None``/empty and
log at debug, so a run without the telemetry stack (CI, a volunteer machine) is unaffected; the
same posture as ``utils/gpu_monitor.py``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import requests
from loguru import logger

# hordelib's per-span operation names that represent actual GPU work. The union of their wall-clock
# intervals (NOT the naive sum, since sample nests calc_cond_batch) is the GPU-busy time within a trace.
GPU_BUSY_OPERATIONS: frozenset[str] = frozenset(
    {
        "comfy.internal.sample",
        "comfy.internal.calc_cond_batch",
        "comfy.internal.vae_decode",
        "comfy.internal.vae_encode",
        "comfy.internal.load_models_gpu",
    },
)

_DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
_DEFAULT_JAEGER_URL = "http://localhost:16686"
_DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class TelemetryEndpoints:
    """Where to reach Prometheus and Jaeger. Env overrides ease pointing at a remote stack."""

    prometheus_url: str = _DEFAULT_PROMETHEUS_URL
    jaeger_url: str = _DEFAULT_JAEGER_URL
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> TelemetryEndpoints:
        """Build endpoints from ``BENCHMARK_PROMETHEUS_URL`` / ``BENCHMARK_JAEGER_URL`` if set."""
        return cls(
            prometheus_url=os.environ.get("BENCHMARK_PROMETHEUS_URL", _DEFAULT_PROMETHEUS_URL),
            jaeger_url=os.environ.get("BENCHMARK_JAEGER_URL", _DEFAULT_JAEGER_URL),
        )


@dataclasses.dataclass(frozen=True)
class SpanInfo:
    """A single span flattened from a Jaeger trace (times in seconds, epoch-relative)."""

    operation: str
    service: str
    start_seconds: float
    duration_seconds: float

    @property
    def end_seconds(self) -> float:
        """Epoch-relative end time of the span, in seconds."""
        return self.start_seconds + self.duration_seconds


def merge_interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total wall-clock covered by ``[start, end]`` intervals, counting overlaps only once.

    Used to turn (possibly nested/overlapping) GPU-busy spans into the real busy duration.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def gpu_busy_seconds(spans: list[SpanInfo], busy_operations: frozenset[str] = GPU_BUSY_OPERATIONS) -> float:
    """Union of the wall-clock intervals of GPU-busy spans within a trace."""
    intervals = [(s.start_seconds, s.end_seconds) for s in spans if s.operation in busy_operations]
    return merge_interval_union_seconds(intervals)


def trace_wall_seconds(spans: list[SpanInfo]) -> float:
    """Wall-clock span of an entire trace (latest end minus earliest start)."""
    if not spans:
        return 0.0
    return max(s.end_seconds for s in spans) - min(s.start_seconds for s in spans)


def span_derived_duty_cycle(spans: list[SpanInfo]) -> float | None:
    """GPU-busy union ÷ trace wall-clock, the span-derived duty-cycle diagnostic (0..1), or None."""
    wall = trace_wall_seconds(spans)
    if wall <= 0:
        return None
    return gpu_busy_seconds(spans) / wall


def operation_totals_seconds(spans: list[SpanInfo]) -> dict[str, float]:
    """Summed self-reported duration per operation name (diagnostic breakdown; may overlap)."""
    totals: dict[str, float] = {}
    for span in spans:
        totals[span.operation] = totals.get(span.operation, 0.0) + span.duration_seconds
    return totals


class TelemetryQuery:
    """Thin read-only client over Prometheus and Jaeger HTTP APIs (degrades to None/empty)."""

    def __init__(self, endpoints: TelemetryEndpoints | None = None) -> None:
        """Create a query client; defaults to the local Prometheus/Jaeger endpoints."""
        self._endpoints = endpoints or TelemetryEndpoints()
        self._session = requests.Session()

    # -- Prometheus -----------------------------------------------------------------------------

    def prometheus_instant(self, promql: str) -> list[dict] | None:
        """Run an instant query; return the ``result`` list, or None if unreachable/failed."""
        return self._prometheus_request("/api/v1/query", {"query": promql})

    def prometheus_range(
        self,
        promql: str,
        *,
        start_epoch: float,
        end_epoch: float,
        step_seconds: float,
    ) -> list[dict] | None:
        """Run a range query over ``[start_epoch, end_epoch]`` at ``step_seconds`` resolution."""
        return self._prometheus_request(
            "/api/v1/query_range",
            {"query": promql, "start": start_epoch, "end": end_epoch, "step": step_seconds},
        )

    def _prometheus_request(self, path: str, params: dict[str, Any]) -> list[dict] | None:
        url = f"{self._endpoints.prometheus_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self._endpoints.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Prometheus query failed ({url}): {e}")
            return None
        if payload.get("status") != "success":
            logger.debug(f"Prometheus query non-success ({url}): {payload.get('error')}")
            return None
        result = payload.get("data", {}).get("result")
        return result if isinstance(result, list) else None

    # -- Jaeger ---------------------------------------------------------------------------------

    def jaeger_services(self) -> list[str]:
        """List service names known to Jaeger (empty if unreachable)."""
        payload = self._jaeger_request("/api/services", params={})
        if payload is None:
            return []
        data = payload.get("data")
        return [s for s in data if isinstance(s, str)] if isinstance(data, list) else []

    def jaeger_traces(
        self,
        service: str,
        *,
        start_epoch_seconds: float,
        end_epoch_seconds: float,
        operation: str | None = None,
        limit: int = 100,
        tags: dict[str, str] | None = None,
    ) -> list[list[SpanInfo]]:
        """Fetch traces for ``service`` in the window, each flattened to a list of SpanInfo.

        Returns an empty list if Jaeger is unreachable or has no matching traces.
        """
        params: dict[str, object] = {
            "service": service,
            "start": int(start_epoch_seconds * 1_000_000),  # Jaeger wants microseconds
            "end": int(end_epoch_seconds * 1_000_000),
            "limit": limit,
        }
        if operation is not None:
            params["operation"] = operation
        if tags:
            # Jaeger expects a JSON object string for tag filters.
            import json

            params["tags"] = json.dumps(tags)

        payload = self._jaeger_request("/api/traces", params=params)
        if payload is None:
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return [self._flatten_trace(trace) for trace in data]

    @staticmethod
    def _flatten_trace(trace: dict) -> list[SpanInfo]:
        process_services: dict[str, str] = {
            pid: proc.get("serviceName", "")
            for pid, proc in (trace.get("processes") or {}).items()
            if isinstance(proc, dict)
        }
        spans: list[SpanInfo] = []
        for span in trace.get("spans") or []:
            if not isinstance(span, dict):
                continue
            try:
                start_us = float(span["startTime"])
                duration_us = float(span["duration"])
            except (KeyError, TypeError, ValueError):
                continue
            spans.append(
                SpanInfo(
                    operation=span.get("operationName", ""),
                    service=process_services.get(span.get("processID", ""), ""),
                    start_seconds=start_us / 1_000_000,
                    duration_seconds=duration_us / 1_000_000,
                ),
            )
        return spans

    def _jaeger_request(self, path: str, params: dict[str, Any]) -> dict | None:
        url = f"{self._endpoints.jaeger_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self._endpoints.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Jaeger query failed ({url}): {e}")
            return None
        return payload if isinstance(payload, dict) else None
