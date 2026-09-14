"""The post-processing lane never downloads: a missing model is a named fault, and a reload makes a landed one visible.

The lane is built without its constructor (no hordelib, no pipes) and handed a fake shared model manager, so
these tests state what a job sees: a present model runs, a missing one faults with the model named, a name
no manager owns is left for hordelib to reject, and a reload re-reads the references.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeControlMessage
from horde_worker_regen.process_management.workers.post_process_process import (
    HordePostProcessProcess,
    PostProcessorNotOnDiskError,
)


def _lane(*, managers: list[SimpleNamespace] | None = None, dry_run: bool = False) -> HordePostProcessProcess:
    lane = HordePostProcessProcess.__new__(HordePostProcessProcess)
    lane._dry_run_skip_post_processing = dry_run
    if managers is None:
        lane._shared_model_manager = None  # type: ignore[assignment]
    else:
        reloads: list[str] = []
        manager = SimpleNamespace(
            get_model_manager_instances=lambda _categories: managers,
            reload_database=lambda: reloads.append("reloaded"),
            reloads=reloads,
        )
        lane._shared_model_manager = SimpleNamespace(manager=manager)  # type: ignore[assignment]
    return lane


def _manager(*, known: dict[str, bool]) -> SimpleNamespace:
    """A fake post-processor manager: reference keys mapped to whether their weight is on disk."""
    return SimpleNamespace(model_reference=dict.fromkeys(known), is_model_available=lambda name: known[name])


def test_a_present_model_is_allowed_to_run() -> None:
    """A model whose weight is on disk passes the check and reaches hordelib."""
    lane = _lane(managers=[_manager(known={"GFPGAN": True})])

    lane._require_post_processor_on_disk("GFPGAN")


def test_a_missing_model_faults_with_its_name_instead_of_downloading() -> None:
    """A model the reference knows but the disk lacks faults the job, naming the model."""
    lane = _lane(managers=[_manager(known={"RealESRGAN_x4plus": False})])

    with pytest.raises(PostProcessorNotOnDiskError, match="RealESRGAN_x4plus") as raised:
        lane._require_post_processor_on_disk("RealESRGAN_x4plus")
    assert raised.value.model_name == "RealESRGAN_x4plus"


def test_the_owning_manager_decides_even_when_it_is_not_first() -> None:
    """The check searches every post-processor manager, so a face fixer under codeformer is judged there."""
    lane = _lane(managers=[_manager(known={"GFPGAN": True}), _manager(known={"CodeFormers": False})])

    with pytest.raises(PostProcessorNotOnDiskError):
        lane._require_post_processor_on_disk("CodeFormers")


def test_a_name_no_manager_owns_is_left_to_hordelib() -> None:
    """An unknown name is not this check's to reject; hordelib reports it as unknown."""
    lane = _lane(managers=[_manager(known={"GFPGAN": True})])

    lane._require_post_processor_on_disk("not_a_model")


def test_a_dry_run_lane_consults_nothing() -> None:
    """A dry-run lane has no managers and never fails a job on presence."""
    lane = _lane(dry_run=True)

    lane._require_post_processor_on_disk("GFPGAN")


def test_a_reload_request_re_reads_the_references() -> None:
    """A reload sent by the parent makes a model the download process just placed resolvable here."""
    lane = _lane(managers=[])

    lane._receive_and_handle_control_message(
        HordeControlMessage(control_flag=HordeControlFlag.RELOAD_MODEL_DATABASE),
    )

    assert lane._shared_model_manager.manager.reloads == ["reloaded"]
