"""The model map's residency predicate: the one truth admission credits and the footprint store learns from."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import (
    PRELOAD_FIRST_REPORT_GRACE_SECONDS,
    HordeModelMap,
    StaleEntryReason,
)
from tests.process_management.conftest import make_mock_process_info

_MODEL = "stable_diffusion"


def _map_with(state: ModelLoadState, *, process_id: int = 1) -> HordeModelMap:
    return HordeModelMap(
        root={_MODEL: ModelInfo(horde_model_name=_MODEL, horde_model_load_state=state, process_id=process_id)},
    )


def test_vram_resident_on_the_matching_process_is_credited() -> None:
    """The ordinary case: the map says the weights are on the card, on this slot."""
    assert _map_with(ModelLoadState.LOADED_IN_VRAM).weights_resident_on_process(_MODEL, make_mock_process_info(1))


def test_a_model_staged_in_system_ram_is_not_credited() -> None:
    """A checkpoint the child holds in system RAM occupies no VRAM, however the slot names it."""
    assert not _map_with(ModelLoadState.LOADED_IN_RAM).weights_resident_on_process(_MODEL, make_mock_process_info(1))


def test_a_primed_lane_is_not_credited_even_when_the_map_says_in_use() -> None:
    """The child marks its job's model IN_USE on taking the job; under the lease the weights are still in RAM."""
    primed = make_mock_process_info(1, state=HordeProcessState.INFERENCE_PRIMED)
    assert not _map_with(ModelLoadState.IN_USE).weights_resident_on_process(_MODEL, primed)


def test_a_retained_model_is_credited_whatever_the_map_or_lane_state_says() -> None:
    """Retention is the only record that covers weights held between jobs, and it outranks the map."""
    primed = make_mock_process_info(1, state=HordeProcessState.INFERENCE_PRIMED)
    primed.retained_resident_model = _MODEL
    assert _map_with(ModelLoadState.LOADED_IN_RAM).weights_resident_on_process(_MODEL, primed)


def test_a_lagging_process_pointer_still_credits_the_reporting_slot() -> None:
    """When the map's process pointer lags, the slot that itself reports the model loaded is credited."""
    lagging = _map_with(ModelLoadState.LOADED_IN_VRAM, process_id=2)
    assert lagging.weights_resident_on_process(_MODEL, make_mock_process_info(1, model_name=_MODEL))
    assert not lagging.weights_resident_on_process(_MODEL, make_mock_process_info(1, model_name="other"))


def test_unknown_inputs_are_never_credited() -> None:
    """No model, no process, or an unmapped model is never resident."""
    model_map = _map_with(ModelLoadState.LOADED_IN_VRAM)
    assert not model_map.weights_resident_on_process(None, make_mock_process_info(1))
    assert not model_map.weights_resident_on_process(_MODEL, None)
    assert not model_map.weights_resident_on_process("absent", make_mock_process_info(1))


_NOW = 10_000.0


def _map(**entries: tuple[ModelLoadState, int]) -> HordeModelMap:
    model_map = HordeModelMap(root={})
    for model, (state, process_id) in entries.items():
        model_map.update_entry(horde_model_name=model, load_state=state, process_id=process_id)
    return model_map


class TestExpireStaleEntries:
    """Each stale shape is removed with its reason; a truthful entry survives."""

    def test_a_gone_owner_expires_the_entry(self) -> None:
        """An entry naming a process the pool no longer holds records nothing."""
        model_map = _map(sd=(ModelLoadState.LOADED_IN_VRAM, 7))

        expired = model_map.expire_stale_entries(ProcessMap({}), now=_NOW)

        assert [(e.model, e.reason, e.process_state) for e in expired] == [("sd", StaleEntryReason.PROCESS_GONE, None)]
        assert "sd" not in model_map.root

    def test_an_abandoned_load_expires_past_the_first_report_grace(self) -> None:
        """A LOADING entry whose idle owner sent no recent preload is stale; inside the grace it is not."""
        owner = make_mock_process_info(1, model_name="sd", state=HordeProcessState.WAITING_FOR_JOB)
        owner.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        owner.last_preload_requested_at = _NOW - 1.0
        model_map = _map(sd=(ModelLoadState.LOADING, 1))

        assert model_map.expire_stale_entries(ProcessMap({1: owner}), now=_NOW) == []

        owner.last_preload_requested_at = _NOW - PRELOAD_FIRST_REPORT_GRACE_SECONDS - 1.0
        expired = model_map.expire_stale_entries(ProcessMap({1: owner}), now=_NOW)
        assert [(e.model, e.reason, e.process_state) for e in expired] == [
            ("sd", StaleEntryReason.LOADING_ABANDONED, HordeProcessState.WAITING_FOR_JOB),
        ]

    def test_a_loading_owner_keeps_its_entry(self) -> None:
        """A LOADING entry on a slot that is preloading is the truth."""
        owner = make_mock_process_info(1, model_name="sd", state=HordeProcessState.PRELOADING_MODEL)
        model_map = _map(sd=(ModelLoadState.LOADING, 1))

        assert model_map.expire_stale_entries(ProcessMap({1: owner}), now=_NOW) == []
        assert "sd" in model_map.root

    def test_a_displaced_entry_names_the_model_that_took_the_slot(self) -> None:
        """A slot that now holds another model cannot still hold this one."""
        owner = make_mock_process_info(1, model_name="xl", state=HordeProcessState.WAITING_FOR_JOB)
        model_map = _map(sd=(ModelLoadState.LOADED_IN_VRAM, 1), xl=(ModelLoadState.LOADED_IN_VRAM, 1))

        expired = model_map.expire_stale_entries(ProcessMap({1: owner}), now=_NOW)

        assert [(e.model, e.reason, e.holder_model) for e in expired] == [("sd", StaleEntryReason.DISPLACED, "xl")]
        assert set(model_map.root) == {"xl"}
