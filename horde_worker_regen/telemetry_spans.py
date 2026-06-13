"""Span helpers and metric definitions for job lifecycle instrumentation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import logfire

inference_duration_histogram = logfire.metric_histogram(
    "inference.duration_seconds",
    unit="s",
    description="Time spent in basic_inference",
)

model_load_duration_histogram = logfire.metric_histogram(
    "model_load.duration_seconds",
    unit="s",
    description="Time spent loading a model",
)

jobs_completed_counter = logfire.metric_counter(
    "jobs.completed_total",
    description="Total successfully completed inference jobs",
)

jobs_faulted_counter = logfire.metric_counter(
    "jobs.faulted_total",
    description="Total faulted inference jobs",
)

queue_depth_counter = logfire.metric_up_down_counter(
    "queue.depth",
    description="Current number of jobs in the queue",
)

job_queue_wait_histogram = logfire.metric_histogram(
    "job.queue_wait_seconds",
    unit="s",
    description="Time from job pop to inference start",
)

job_e2e_histogram = logfire.metric_histogram(
    "job.e2e_seconds",
    unit="s",
    description="Time from job pop to finalized submit",
)

job_safety_histogram = logfire.metric_histogram(
    "job.safety_seconds",
    unit="s",
    description="Time from safety-check queue entry to submit-ready",
)


@contextlib.contextmanager
def span_job_pop(*, models: str) -> Iterator[logfire.LogfireSpan]:
    """Wrap the API job pop call."""
    with logfire.span("job.pop", models=models) as s:
        yield s


@contextlib.contextmanager
def span_preload_model(*, model_name: str, process_id: int) -> Iterator[logfire.LogfireSpan]:
    """Wrap model preload."""
    with logfire.span("job.preload_model", model_name=model_name, process_id=process_id) as s:
        yield s


@contextlib.contextmanager
def span_inference(
    *,
    model: str,
    steps: int,
    width: int,
    height: int,
    **extra: Any,  # type: ignore # noqa
) -> Iterator[logfire.LogfireSpan]:
    """Wrap the basic_inference call in the child process."""
    with logfire.span(
        "job.inference",
        model=model,
        steps=steps,
        resolution=f"{width}x{height}",
        **extra,
    ) as s:
        yield s


@contextlib.contextmanager
def span_safety_check(*, job_id: str) -> Iterator[logfire.LogfireSpan]:
    """Wrap the safety evaluation."""
    with logfire.span("job.safety_check", job_id=job_id) as s:
        yield s


@contextlib.contextmanager
def span_job_submit(*, job_id: str) -> Iterator[logfire.LogfireSpan]:
    """Wrap R2 upload + API submit."""
    with logfire.span("job.submit", job_id=job_id) as s:
        yield s
