"""The download process narrows its queue until the worker holds what it needs to start serving.

Critical mass is the safety models plus, when image models are configured, one of them on disk. These tests
drive the process's own bookkeeping (what is present, what is desired) and read the answer off the
scheduler, so they state what an operator sees: which downloads run first, and when the rest are let in.
"""

from __future__ import annotations

import queue
from unittest.mock import Mock

from horde_worker_regen.alchemy_forms import AuxiliaryFetchNeeds
from horde_worker_regen.process_management.models.download_scheduler import DownloadPriorityPolicy
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID, HordeDownloadProcess


def _process(policy: DownloadPriorityPolicy = DownloadPriorityPolicy.SERVE_FIRST) -> HordeDownloadProcess:
    return HordeDownloadProcess(
        process_id=DOWNLOAD_PROCESS_ID,
        process_message_queue=queue.Queue(),  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        process_launch_identifier=0,
        fetch_needs=AuxiliaryFetchNeeds(),
        priority_policy=policy,
    )


def test_focus_holds_while_the_safety_models_are_missing() -> None:
    """Nothing but the safety models (and one checkpoint) runs until the safety models are on disk."""
    process = _process()
    process._safety_present = False
    process._present = ["model_a"]
    process._desired_image_models = {"model_a"}

    process._update_startup_focus()

    assert process._scheduler.startup_focus is True


def test_focus_holds_until_one_configured_image_model_is_on_disk() -> None:
    """With the safety models present, the queue stays narrow until a configured checkpoint lands."""
    process = _process()
    process._safety_present = True
    process._desired_image_models = {"model_a", "model_b"}
    process._present = ["unrelated_leftover"]

    process._update_startup_focus()
    assert process._scheduler.startup_focus is True

    process._present = ["unrelated_leftover", "model_b"]
    process._update_startup_focus()
    assert process._scheduler.startup_focus is False


def test_an_alchemist_only_worker_leaves_focus_once_the_safety_models_land() -> None:
    """A worker with no image models configured needs only the safety models to start serving."""
    process = _process()
    process._safety_present = True
    process._desired_image_models = set()
    process._present = []

    process._update_startup_focus()

    assert process._scheduler.startup_focus is False


def test_focus_never_applies_under_the_parallel_policy() -> None:
    """An operator who chose ``parallel`` gets the whole queue from the first tick, serving or not."""
    process = _process(DownloadPriorityPolicy.PARALLEL)
    process._safety_present = False
    process._desired_image_models = {"model_a"}
    process._present = []

    process._update_startup_focus()

    assert process._scheduler.startup_focus is False


def test_switching_policy_live_applies_the_focus_rule_of_the_new_policy() -> None:
    """Switching to ``serve_first`` mid-session narrows the queue if the worker still cannot serve."""
    from horde_worker_regen.process_management.ipc.messages import HordeDownloadControlMessage

    process = _process(DownloadPriorityPolicy.PARALLEL)
    process._safety_present = False
    process._desired_image_models = {"model_a"}
    process._present = []

    process._handle_control_message(
        HordeDownloadControlMessage(set_priority_policy=DownloadPriorityPolicy.SERVE_FIRST),
    )

    assert process._scheduler.policy is DownloadPriorityPolicy.SERVE_FIRST
    assert process._scheduler.startup_focus is True
