"""Unit tests for the read-only Prometheus/Jaeger telemetry query helpers (no network)."""

from __future__ import annotations

import pytest
import requests

from horde_worker_regen.benchmark.telemetry_query import (
    SpanInfo,
    TelemetryQuery,
    gpu_busy_seconds,
    merge_interval_union_seconds,
    operation_totals_seconds,
    span_derived_duty_cycle,
    trace_wall_seconds,
)


def _span(op: str, start: float, dur: float, service: str = "hordelib") -> SpanInfo:
    return SpanInfo(operation=op, service=service, start_seconds=start, duration_seconds=dur)


class TestIntervalUnion:
    """`merge_interval_union_seconds` counts overlapping wall-clock only once."""

    def test_empty(self) -> None:
        """No intervals means no covered time."""
        assert merge_interval_union_seconds([]) == 0.0

    def test_disjoint_sums(self) -> None:
        """Non-overlapping intervals add up."""
        assert merge_interval_union_seconds([(0.0, 1.0), (2.0, 3.5)]) == pytest.approx(2.5)

    def test_overlapping_counted_once(self) -> None:
        """Overlap is not double-counted."""
        assert merge_interval_union_seconds([(0.0, 2.0), (1.0, 3.0)]) == pytest.approx(3.0)

    def test_nested_counted_once(self) -> None:
        """A nested interval (e.g. calc_cond_batch inside sample) adds nothing."""
        assert merge_interval_union_seconds([(0.0, 5.0), (1.0, 2.0)]) == pytest.approx(5.0)

    def test_touching_merge(self) -> None:
        """Abutting intervals merge into one span."""
        assert merge_interval_union_seconds([(0.0, 1.0), (1.0, 2.0)]) == pytest.approx(2.0)


class TestDutyCycleMath:
    """GPU-busy union, trace wall-clock, and the span-derived duty-cycle ratio."""

    def test_gpu_busy_excludes_non_busy_ops(self) -> None:
        """Only GPU-work operations contribute; nested busy spans are unioned, not summed."""
        spans = [
            _span("comfy.internal.sample", 0.0, 4.0),
            _span("comfy.internal.calc_cond_batch", 0.5, 1.0),  # nested in sample
            _span("comfy.internal.vae_decode", 4.0, 1.0),
            _span("job.submit", 5.0, 2.0),  # not GPU work
        ]
        # union of [0,4] ∪ [0.5,1.5] ∪ [4,5] = 5.0
        assert gpu_busy_seconds(spans) == pytest.approx(5.0)

    def test_trace_wall(self) -> None:
        """Trace wall-clock is latest end minus earliest start."""
        spans = [_span("a", 10.0, 2.0), _span("b", 11.0, 5.0)]
        assert trace_wall_seconds(spans) == pytest.approx(6.0)  # 16 - 10

    def test_duty_cycle_ratio(self) -> None:
        """Duty cycle is GPU-busy union divided by wall-clock."""
        spans = [
            _span("comfy.internal.sample", 0.0, 8.0),
            _span("comfy.internal.vae_decode", 8.0, 1.0),
            _span("job.inference", 0.0, 10.0),  # wall = 10
        ]
        assert span_derived_duty_cycle(spans) == pytest.approx(0.9)

    def test_duty_cycle_none_when_zero_wall(self) -> None:
        """An empty trace has no defined duty cycle."""
        assert span_derived_duty_cycle([]) is None

    def test_operation_totals_sum_per_op(self) -> None:
        """Per-operation totals sum each operation's self-reported duration."""
        spans = [_span("x", 0, 1.0), _span("x", 2, 0.5), _span("y", 0, 3.0)]
        totals = operation_totals_seconds(spans)
        assert totals == {"x": pytest.approx(1.5), "y": pytest.approx(3.0)}


class _FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload: object, *, raise_exc: Exception | None = None) -> None:
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        """Raise the configured error, if any."""
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self) -> object:
        """Return the canned payload."""
        return self._payload


class _FakeSession:
    """A requests.Session stand-in that returns a canned response or raises on get()."""

    def __init__(self, response: _FakeResponse | None = None, *, get_raises: Exception | None = None) -> None:
        self._response = response
        self._get_raises = get_raises
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _FakeResponse:
        """Record the call and return the canned response (or raise)."""
        self.calls.append((url, params or {}))
        if self._get_raises is not None:
            raise self._get_raises
        assert self._response is not None
        return self._response


def _query_with_session(session: _FakeSession) -> TelemetryQuery:
    tq = TelemetryQuery()
    tq._session = session  # type: ignore[assignment]
    return tq


class TestPrometheus:
    """Prometheus instant/range parsing and graceful degradation."""

    def test_instant_success_returns_result(self) -> None:
        """A success payload yields its data.result list."""
        payload = {"status": "success", "data": {"result": [{"value": [0, "1"]}]}}
        tq = _query_with_session(_FakeSession(_FakeResponse(payload)))
        assert tq.prometheus_instant("up") == [{"value": [0, "1"]}]

    def test_non_success_returns_none(self) -> None:
        """A non-success status returns None rather than raising."""
        payload = {"status": "error", "error": "bad query"}
        tq = _query_with_session(_FakeSession(_FakeResponse(payload)))
        assert tq.prometheus_instant("up{") is None

    def test_unreachable_returns_none(self) -> None:
        """A connection error degrades to None for both query shapes."""
        tq = _query_with_session(_FakeSession(get_raises=requests.ConnectionError("refused")))
        assert tq.prometheus_instant("up") is None
        assert tq.prometheus_range("up", start_epoch=0, end_epoch=1, step_seconds=1) is None


class TestJaeger:
    """Jaeger trace flattening, window units, and graceful degradation."""

    def test_flatten_and_parse_traces(self) -> None:
        """Spans are flattened to seconds with their service; malformed spans are dropped."""
        payload = {
            "data": [
                {
                    "traceID": "t1",
                    "processes": {"p1": {"serviceName": "hordelib"}},
                    "spans": [
                        {
                            "operationName": "comfy.internal.sample",
                            "startTime": 1_000_000_000,  # us -> 1000.0 s
                            "duration": 2_000_000,  # us -> 2.0 s
                            "processID": "p1",
                        },
                        {"operationName": "bad", "startTime": "x", "duration": 1},  # skipped
                    ],
                },
            ],
        }
        tq = _query_with_session(_FakeSession(_FakeResponse(payload)))
        traces = tq.jaeger_traces("hordelib", start_epoch_seconds=0, end_epoch_seconds=2000)
        assert len(traces) == 1
        spans = traces[0]
        assert len(spans) == 1  # malformed span dropped
        assert spans[0].operation == "comfy.internal.sample"
        assert spans[0].service == "hordelib"
        assert spans[0].start_seconds == pytest.approx(1000.0)
        assert spans[0].duration_seconds == pytest.approx(2.0)

    def test_traces_microsecond_window_param(self) -> None:
        """The query window is sent to Jaeger in microseconds."""
        session = _FakeSession(_FakeResponse({"data": []}))
        tq = _query_with_session(session)
        tq.jaeger_traces("hordelib", start_epoch_seconds=1.0, end_epoch_seconds=2.0)
        _url, params = session.calls[-1]
        assert params["start"] == 1_000_000
        assert params["end"] == 2_000_000

    def test_unreachable_returns_empty(self) -> None:
        """A connection error degrades to empty results."""
        tq = _query_with_session(_FakeSession(get_raises=requests.ConnectionError("refused")))
        assert tq.jaeger_traces("hordelib", start_epoch_seconds=0, end_epoch_seconds=1) == []
        assert tq.jaeger_services() == []

    def test_services_list(self) -> None:
        """The services endpoint returns the list of known service names."""
        tq = _query_with_session(_FakeSession(_FakeResponse({"data": ["hordelib", "horde-worker-regen"]})))
        assert tq.jaeger_services() == ["hordelib", "horde-worker-regen"]
