"""Wiring the text flow into the worker: the dedicated card rule, the config guard and the credential."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.process_manager import _exclude_text_backend_device
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.text_backends import KoboldApiTextBackend
from tests.process_management.conftest import make_testable_process_manager


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
    with pytest.raises(ValueError, match="text_model"):
        reGenBridgeData(api_key="0000000000", scribe=True)


def test_an_attached_text_backend_needs_no_model_path() -> None:
    """A scribe attaching to an operator-run backend leaves the model to that backend."""
    bridge_data = reGenBridgeData(api_key="0000000000", scribe=True, text_backend_managed=False)

    assert bridge_data.text_model is None
    assert bridge_data.text_backend_kind is TEXT_BACKENDS.koboldcpp


def test_a_managed_text_backend_with_a_model_path_is_accepted(tmp_path: Path) -> None:
    """The managed default is usable as soon as a model is named."""
    bridge_data = reGenBridgeData(api_key="0000000000", scribe=True, text_model=str(tmp_path / "model.gguf"))

    assert bridge_data.text_backend_managed is True
    assert bridge_data.text_backend_port == 5001


def test_the_text_flow_is_registered_whatever_the_role_says_at_start_up() -> None:
    """Registered like the image and alchemy flows, so enabling `scribe` by hot reload needs no restart.

    The flow self-gates on the live configuration, which is what makes registering it on every worker
    harmless: it builds no backend and pops nothing while the role is off.
    """
    dreamer_only = make_testable_process_manager()
    scribe = make_testable_process_manager(scribe=True)

    assert WorkloadKind.TEXT_GENERATION in dreamer_only._flows
    assert WorkloadKind.TEXT_GENERATION in scribe._flows
    assert dreamer_only._text_coordinator.backend_ready is False
    assert dreamer_only._text_coordinator.backend_not_ready_since is None
    assert scribe._text_coordinator.backend_not_ready_since is not None


@pytest.mark.asyncio
async def test_the_supervisor_task_ends_at_shutdown_on_a_worker_that_never_serves_text() -> None:
    """The task is scheduled on every worker, so with the role off it must not outlive the main loop."""
    manager = make_testable_process_manager()
    manager._state.shut_down = True

    await manager._run_text_backend_supervisor()

    assert manager._text_backend_supervisor is None


@pytest.mark.asyncio
async def test_the_supervisor_task_follows_the_role_being_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling `scribe` by hot reload ends the wait, so a managed backend launches without a restart."""
    manager = make_testable_process_manager(text_model=None)
    answers = iter([False, False, True])
    asked = Mock(side_effect=lambda: next(answers))
    monkeypatch.setattr(manager, "_text_backend_is_managed", asked)

    async def _no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("horde_worker_regen.process_management.process_manager.asyncio.sleep", _no_wait)

    # With no `text_model` the launch is refused right after the wait, which is what shows the wait ended.
    await manager._run_text_backend_supervisor()

    assert asked.call_count == 3
    assert manager._text_backend_supervisor is None


def test_an_attached_backends_password_reaches_the_driver() -> None:
    """The operator's `text_backend_password` is the credential their own backend was started with."""
    manager = make_testable_process_manager(
        scribe=True,
        text_backend_managed=False,
        text_backend_password="a-launch-password",
    )
    manager._api_sessions.set_aiohttp_session(Mock())

    driver = manager._text_backend_for(TEXT_BACKENDS.koboldcpp)

    assert isinstance(driver, KoboldApiTextBackend)
    assert driver._password == "a-launch-password"


def test_a_managed_backend_is_given_no_password() -> None:
    """The worker starts its own backend without one, so a credential sent there would be refused."""
    manager = make_testable_process_manager(
        scribe=True,
        text_backend_managed=True,
        text_backend_password="a-launch-password",
    )
    manager._api_sessions.set_aiohttp_session(Mock())

    driver = manager._text_backend_for(TEXT_BACKENDS.koboldcpp)

    assert isinstance(driver, KoboldApiTextBackend)
    assert driver._password is None


def test_the_backend_kind_is_configured_in_the_horde_vocabulary() -> None:
    """The backend choice is one field, spelled as the horde spells backends."""
    bridge_data = reGenBridgeData(
        api_key="0000000000",
        scribe=True,
        text_backend_managed=False,
        text_backend_kind="aphrodite",
    )

    assert bridge_data.text_backend_kind is TEXT_BACKENDS.aphrodite
