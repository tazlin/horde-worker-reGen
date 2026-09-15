"""Wiring a managed text backend into the worker: the dedicated card rule and the configuration guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.process_manager import _exclude_text_backend_device
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap


def _device_map(*indices: int) -> TorchDeviceMap:
    return TorchDeviceMap(
        root={
            index: TorchDeviceInfo(device_name=f"card-{index}", device_index=index, total_memory=16 * 1024**3)
            for index in indices
        },
    )


def test_a_named_text_card_leaves_the_image_device_map_on_a_multi_card_host() -> None:
    """Naming a card dedicates it to text: image generation drives the others."""
    remaining = _exclude_text_backend_device(_device_map(0, 1, 2), 1)

    assert sorted(remaining.root) == [0, 2]


def test_a_single_card_host_keeps_its_card_shared() -> None:
    """Removing the only card would leave image generation with nothing, so the card stays shared."""
    only_card = _device_map(0)

    assert _exclude_text_backend_device(only_card, 0) is only_card


def test_an_unset_or_undriven_text_card_changes_nothing() -> None:
    """No configured card, or a card that is not driven, leaves the image device map alone."""
    two_cards = _device_map(0, 1)

    assert _exclude_text_backend_device(two_cards, None) is two_cards
    assert _exclude_text_backend_device(two_cards, 7) is two_cards


def test_a_managed_text_backend_requires_a_model_path() -> None:
    """A scribe that launches its own backend cannot start without a model, so the config is refused."""
    with pytest.raises(ValueError, match="text_model_path"):
        reGenBridgeData(api_key="0000000000", scribe=True)


def test_an_attached_text_backend_needs_no_model_path() -> None:
    """A scribe attaching to an operator-run backend leaves the model to that backend."""
    bridge_data = reGenBridgeData(api_key="0000000000", scribe=True, text_backend_managed=False)

    assert bridge_data.text_model_path is None
    assert bridge_data.text_backend_kind is TEXT_BACKENDS.koboldcpp


def test_a_managed_text_backend_with_a_model_path_is_accepted(tmp_path: Path) -> None:
    """The managed default is usable as soon as a model is named."""
    bridge_data = reGenBridgeData(api_key="0000000000", scribe=True, text_model_path=tmp_path / "model.gguf")

    assert bridge_data.text_backend_managed is True
    assert bridge_data.text_backend_port == 5001


def test_the_backend_kind_is_configured_in_the_horde_vocabulary() -> None:
    """The backend choice is one field, spelled as the horde spells backends."""
    bridge_data = reGenBridgeData(
        api_key="0000000000",
        scribe=True,
        text_backend_managed=False,
        text_backend_kind="aphrodite",
    )

    assert bridge_data.text_backend_kind is TEXT_BACKENDS.aphrodite
