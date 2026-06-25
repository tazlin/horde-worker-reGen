"""Tests for APIWorkerMessage.from_raw_dict and JobTracker megapixelstep methods."""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_models import APIWorkerMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from tests.process_management.conftest import make_mock_job, track_popped_job_async


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

    def test_none_id_assigned_random_id(self) -> None:
        """If 'id' key is absent, message_id should be assigned a random ID."""
        raw = {"message": "orphan message"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id is not None

    def test_empty_dict(self) -> None:
        """An empty dict should produce a message with all None/default fields."""
        raw: dict = {}  # type: ignore[type-arg]

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_id is not None
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
        """Message text should preserve special characters and emojis without alteration."""
        raw = {"id": "msg-004", "message": "Update: v2.0! 🚀 (breaking changes)"}

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_text == "Update: v2.0! 🚀 (breaking changes)"

    def test_message_text_with_none_value(self) -> None:
        """Explicitly setting message to None should produce 'None' string."""
        raw = {
            "id": "msg-005",
            "message": None,  # pyrefly: ignore - we want to ensure that None is handled gracefully and converted to the string "None"
        }

        msg = APIWorkerMessage.from_raw_dict(raw)

        assert msg.message_text == "None"

    def test_message_text_with_empty_string(self) -> None:
        """Explicitly setting message to an empty string should produce an empty string."""
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
        """After calling reset_megapixelstep_trigger, _triggered_max_pending_megapixelsteps should be False."""
        job_tracker = JobTracker()
        job_tracker._triggered_max_pending_megapixelsteps = True

        job_tracker.reset_megapixelstep_trigger()

        assert job_tracker._triggered_max_pending_megapixelsteps is False

    def test_idempotent_when_already_false(self) -> None:
        """Calling reset_megapixelstep_trigger when the flag is already False should keep it False."""
        job_tracker = JobTracker()
        assert job_tracker._triggered_max_pending_megapixelsteps is False

        job_tracker.reset_megapixelstep_trigger()

        assert job_tracker._triggered_max_pending_megapixelsteps is False


class TestJobTrackerShouldWaitForPendingMegapixelsteps:
    """Tests for the should_wait check that drives throttling."""

    def test_empty_queue_does_not_wait(self) -> None:
        """If there are no jobs pending inference, should_wait_for_pending_megapixelsteps should return False."""
        job_tracker = JobTracker()
        assert job_tracker.should_wait_for_pending_megapixelsteps() is False

    async def test_queue_below_threshold_does_not_wait(self) -> None:
        """If pending megapixelsteps is below threshold, should_wait_for_pending_megapixelsteps should return False."""
        job_tracker = JobTracker()
        job_tracker.set_performance_mode_thresholds(100)

        # Add one small job
        job = make_mock_job(model="test_model", width=512, height=512, ddim_steps=20)
        await track_popped_job_async(job_tracker, job)

        assert job_tracker.should_wait_for_pending_megapixelsteps() is False

    async def test_queue_above_threshold_waits(self) -> None:
        """If pending megapixelsteps exceeds threshold, should_wait_for_pending_megapixelsteps should return True."""
        job_tracker = JobTracker()
        job_tracker.set_performance_mode_thresholds(1)  # Very low threshold

        # Add a large job
        job = make_mock_job(model="test_model", width=2048, height=2048, ddim_steps=100)
        await track_popped_job_async(job_tracker, job)

        assert job_tracker.should_wait_for_pending_megapixelsteps() is True


class TestJobTrackerSetPerformanceModeThresholds:
    """Tests for dynamic threshold adjustment."""

    def test_sets_new_threshold(self) -> None:
        """Setting a new threshold should update the internal max_pending_megapixelsteps value."""
        job_tracker = JobTracker()
        original = job_tracker._max_pending_megapixelsteps

        job_tracker.set_performance_mode_thresholds(100)

        assert job_tracker._max_pending_megapixelsteps == 100
        assert job_tracker._max_pending_megapixelsteps != original

    async def test_zero_threshold_makes_everything_wait(self) -> None:
        """Setting threshold to 0 should cause all jobs to trigger the wait condition."""
        job_tracker = JobTracker()
        job_tracker.set_performance_mode_thresholds(0)

        # Need a job large enough to round to at least 1 MPS after int truncation
        job = make_mock_job(model="test_model", width=1024, height=1024, ddim_steps=20)
        await track_popped_job_async(job_tracker, job)

        # Pending MPS > 0 and threshold is 0 → should wait
        assert job_tracker.get_pending_megapixelsteps() > 0
