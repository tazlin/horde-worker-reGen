"""Tests for projecting active alchemy forms into the dashboard work-ledger and queue tables.

An alchemy form is the alchemist worker's unit of work, so it should appear in the same tables an image
job does: the form shown where a model name goes, and the source-image resolution shown as the size.
"""

from __future__ import annotations

import io
import time
from collections import deque
from unittest.mock import Mock

import PIL.Image
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec, HordeAlchemyResultMessage
from horde_worker_regen.process_management.ipc.supervisor_channel import WorkLedgerStage
from horde_worker_regen.process_management.jobs.alchemy_popper import AlchemyCoordinator
from horde_worker_regen.process_management.jobs.job_models import PendingAlchemySubmitJob
from tests.process_management.conftest import make_testable_process_manager


def _submit_job(form_id: str, form: str, *, popped_secs_ago: float = 2.0) -> PendingAlchemySubmitJob:
    return PendingAlchemySubmitJob(
        result_message=HordeAlchemyResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            form_id=form_id,
            form=form,
            state=GENERATION_STATE.ok,
        ),
        r2_upload=None,
        time_popped=time.time() - popped_secs_ago,
    )


def test_record_form_metrics_reports_form_timing_and_resolution() -> None:
    """The coordinator reports a finished form's name, pop->submit e2e, outcome, and size to run metrics."""
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._run_metrics = Mock()
    coordinator._form_resolution = {"s1": (512, 512)}

    coordinator._record_form_metrics(_submit_job("s1", "caption"), faulted=False)

    coordinator._run_metrics.record_alchemy_form.assert_called_once()
    kwargs = coordinator._run_metrics.record_alchemy_form.call_args.kwargs
    assert kwargs["form_id"] == "s1"
    assert kwargs["form"] == "caption"
    assert kwargs["faulted"] is False
    assert (kwargs["width"], kwargs["height"]) == (512, 512)
    assert kwargs["e2e_seconds"] >= 2.0


def test_record_form_metrics_is_noop_without_run_metrics() -> None:
    """A coordinator with no run-metrics aggregator (unit tests) records nothing and does not raise."""
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._run_metrics = None
    coordinator._form_resolution = {}

    coordinator._record_form_metrics(_submit_job("x", "nsfw"), faulted=True)  # must not raise


def _png_bytes(width: int, height: int) -> bytes:
    """Encoded PNG bytes of the given size, for the resolution-decode path."""
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _bare_coordinator() -> AlchemyCoordinator:
    """An AlchemyCoordinator with only the fields ``active_form_statuses`` reads (no I/O collaborators)."""
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._pending_forms = deque()
    coordinator._in_flight = {}
    coordinator._in_flight_owner = {}
    coordinator._pending_submits = deque()
    coordinator._form_resolution = {}
    return coordinator


def test_decode_image_resolution_reads_dimensions() -> None:
    """A valid image yields its (width, height); junk yields None rather than raising."""
    assert AlchemyCoordinator._decode_image_resolution(_png_bytes(640, 480)) == (640, 480)
    assert AlchemyCoordinator._decode_image_resolution(b"not-an-image") is None


def test_active_form_statuses_projects_each_stage_with_resolution() -> None:
    """Pending, in-flight, and awaiting-submit forms each project with their form, stage, size, and owner."""
    coordinator = _bare_coordinator()

    coordinator._pending_forms.append(AlchemyFormSpec(form_id="p1", form="caption", source_image_bytes=b"x"))
    coordinator._form_resolution["p1"] = (640, 480)

    coordinator._in_flight["f1"] = AlchemyFormSpec(form_id="f1", form="RealESRGAN_x4plus", source_image_bytes=b"x")
    coordinator._in_flight_owner["f1"] = (3, 0)
    coordinator._form_resolution["f1"] = (1024, 768)

    coordinator._pending_submits.append(
        PendingAlchemySubmitJob(
            result_message=HordeAlchemyResultMessage(
                process_id=3,
                process_launch_identifier=0,
                info="",
                form_id="s1",
                form="nsfw",
                state=GENERATION_STATE.ok,
            ),
            r2_upload=None,
            time_popped=0.0,
        ),
    )
    coordinator._form_resolution["s1"] = (512, 512)
    coordinator._form_resolution["stale"] = (1, 1)  # no live form; should be pruned

    statuses = {status.form_id: status for status in coordinator.active_form_statuses()}

    assert statuses["p1"].stage == "pending"
    assert (statuses["p1"].width, statuses["p1"].height) == (640, 480)
    assert statuses["p1"].process_id is None

    assert statuses["f1"].stage == "in_flight"
    assert statuses["f1"].form == "RealESRGAN_x4plus"
    assert (statuses["f1"].width, statuses["f1"].height) == (1024, 768)
    assert statuses["f1"].process_id == 3

    assert statuses["s1"].stage == "awaiting_submit"
    assert statuses["s1"].form == "nsfw"
    assert (statuses["s1"].width, statuses["s1"].height) == (512, 512)

    # The resolution cache is pruned to the live forms.
    assert "stale" not in coordinator._form_resolution


def test_work_ledger_and_queue_include_alchemy_forms() -> None:
    """The process manager surfaces alchemy forms into the work ledger and queue with form-as-model + size."""
    manager = make_testable_process_manager()
    coordinator = manager._alchemy_coordinator
    coordinator._pending_forms.append(AlchemyFormSpec(form_id="p1", form="caption", source_image_bytes=b"x"))
    coordinator._form_resolution["p1"] = (640, 480)
    coordinator._in_flight["f1"] = AlchemyFormSpec(form_id="f1", form="RealESRGAN_x4plus", source_image_bytes=b"x")
    coordinator._in_flight_owner["f1"] = (2, 0)
    coordinator._form_resolution["f1"] = (1024, 768)

    queue = {entry.job_id: entry for entry in manager._build_pending_jobs_list()}
    ledger = {entry.job_id: entry for entry in manager._build_work_ledger([])}

    # Only the pending (not-yet-dispatched) form is in the queue, shown as form-as-model with its resolution.
    assert queue["p1"].model == "⚗ caption"
    assert (queue["p1"].width, queue["p1"].height) == (640, 480)
    assert "f1" not in queue

    # Both forms appear in the work ledger at their mapped stages, with the source resolution as the size.
    assert ledger["p1"].stage is WorkLedgerStage.QUEUED
    assert ledger["f1"].stage is WorkLedgerStage.INFERENCE
    assert ledger["f1"].model == "⚗ RealESRGAN_x4plus"
    assert ledger["f1"].process_id == 2
    assert (ledger["f1"].width, ledger["f1"].height) == (1024, 768)
