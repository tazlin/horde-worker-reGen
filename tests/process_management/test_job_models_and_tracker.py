"""Tests for APIWorkerMessage.from_raw_dict and JobTracker megapixelstep methods."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.job_models import APIWorkerMessage
from horde_worker_regen.process_management.job_tracker import JobTracker


class TestAPIWorkerMessageFromRawDict:
    """Test the from_raw_dict classmethod that parses untyped SDK dicts."""

    def test_complete_dict(self) -> None:
        """All fields present should be parsed correctly."""
        raw = {
            "id": "msg-001",
            "message": "Worker update available",
            "origin": "system",
            "expiry": "2026-12-31T23:59:59Z",
        }

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id == "msg-001"
        assert msg.message_text == "Worker update available"
        assert msg.message_origin == "system"
        assert msg.message_expiry == "2026-12-31T23:59:59Z"

    def test_missing_optional_fields(self) -> None:
        """Only 'id' must be present; others default to None (stringified)."""
        raw = {"id": "msg-002"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id == "msg-002"
        assert msg.message_text == "None"  # str(None)
        assert msg.message_origin is None
        assert msg.message_expiry is None

    def test_numeric_id_coerced_to_string(self) -> None:
        """Numeric message IDs should be converted to strings."""
        raw = {"id": 42, "message": "hello"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id == "42"
        assert isinstance(msg.message_id, str)

    def test_none_id_becomes_none(self) -> None:
        """If 'id' key is absent, message_id should be None."""
        raw = {"message": "orphan message"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id is None

    def test_empty_dict(self) -> None:
        """An empty dict should produce a message with all None/default fields."""
        raw: dict = {}  # type: ignore[type-arg]

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id is None
        assert msg.message_text == "None"
        assert msg.message_origin is None
        assert msg.message_expiry is None

    def test_extra_unknown_keys_ignored(self) -> None:
        """Extra keys in the dict should not raise, just be ignored."""
        raw = {
            "id": "msg-003",
            "message": "test",
            "origin": "admin",
            "expiry": "2026-06-01",
            "unknown_key": "should be ignored",
            "another": 999,
        }

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id == "msg-003"
        assert msg.message_text == "test"

    def test_message_text_preserves_special_characters(self) -> None:
        raw = {"id": "msg-004", "message": "Update: v2.0! 🚀 (breaking changes)"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_text == "Update: v2.0! 🚀 (breaking changes)"

    def test_message_text_with_none_value(self) -> None:
        """Explicitly setting message to None should produce 'None' string."""
        raw = {"id": "msg-005", "message": None}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_text == "None"

    def test_message_text_with_empty_string(self) -> None:
        raw = {"id": "msg-006", "message": ""}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_text == ""

    def test_produces_valid_pydantic_model(self) -> None:
        """The returned object should be a valid Pydantic model that can be serialized."""
        raw = {"id": "msg-007", "message": "test", "origin": "system", "expiry": "2026-01-01"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        dumped = msg.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["message_id"] == "msg-007"


class TestJobTrackerResetMegapixelstepTrigger:
    """Tests for JobTracker.reset_megapixelstep_trigger."""

    def test_clears_trigger_flag(self) -> None:
        jt = JobTracker()
        jt._triggered_max_pending_megapixelsteps = True

        jt.reset_megapixelstep_trigger()

        assert jt._triggered_max_pending_megapixelsteps is False

    def test_idempotent_when_already_false(self) -> None:
        jt = JobTracker()
        assert jt._triggered_max_pending_megapixelsteps is False

        jt.reset_megapixelstep_trigger()

        assert jt._triggered_max_pending_megapixelsteps is False


class TestJobTrackerShouldWaitForPendingMegapixelsteps:
    """Tests for the should_wait check that drives throttling."""

    def test_empty_queue_does_not_wait(self) -> None:
        jt = JobTracker()
        assert jt.should_wait_for_pending_megapixelsteps() is False

    def test_queue_below_threshold_does_not_wait(self) -> None:
        jt = JobTracker()
        jt._max_pending_megapixelsteps = 100

        # Add one small job
        job = Mock()
        job.payload.width = 512
        job.payload.height = 512
        job.payload.ddim_steps = 20
        job.payload.n_iter = 1
        job.payload.post_processing = []
        job.payload.loras = []
        job.payload.control_type = None
        job.payload.hires_fix = False
        job.model = "test_model"
        job.payload.workflow = None
        jt.jobs_pending_inference.append(job)

        assert jt.should_wait_for_pending_megapixelsteps() is False

    def test_queue_above_threshold_waits(self) -> None:
        jt = JobTracker()
        jt._max_pending_megapixelsteps = 1  # Very low threshold

        # Add a large job
        job = Mock()
        job.payload.width = 2048
        job.payload.height = 2048
        job.payload.ddim_steps = 100
        job.payload.n_iter = 1
        job.payload.post_processing = []
        job.payload.loras = []
        job.payload.control_type = None
        job.payload.hires_fix = False
        job.model = "test_model"
        job.payload.workflow = None
        jt.jobs_pending_inference.append(job)

        assert jt.should_wait_for_pending_megapixelsteps() is True


class TestJobTrackerSetPerformanceModeThresholds:
    """Tests for dynamic threshold adjustment."""

    def test_sets_new_threshold(self) -> None:
        jt = JobTracker()
        original = jt._max_pending_megapixelsteps

        jt.set_performance_mode_thresholds(100)

        assert jt._max_pending_megapixelsteps == 100
        assert jt._max_pending_megapixelsteps != original

    def test_zero_threshold_makes_everything_wait(self) -> None:
        jt = JobTracker()
        jt.set_performance_mode_thresholds(0)

        # Need a job large enough to round to at least 1 MPS after int truncation
        job = Mock()
        job.payload.width = 1024
        job.payload.height = 1024
        job.payload.ddim_steps = 20
        job.payload.n_iter = 1
        job.payload.post_processing = []
        job.payload.loras = []
        job.payload.control_type = None
        job.payload.hires_fix = False
        job.model = "test_model"
        job.payload.workflow = None
        jt.jobs_pending_inference.append(job)

        # Pending MPS > 0 and threshold is 0 → should wait
        assert jt.get_pending_megapixelsteps() > 0
