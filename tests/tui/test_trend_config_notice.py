"""The Overview's trend notice shows when the worker's capacity or workload changes.

The settings come from the worker as one fingerprint, the same value the card earning store restarts its window
on. LoRA work suspended by downloads or a full disk also shows the notice, without being part of the
fingerprint.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.tui.app import HordeWorkerTUI


def _snapshot(*, fingerprint: str, effective_allow_lora: bool, dreamer_name: str = "Notice") -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(
            dreamer_name=dreamer_name,
            worker_version="0.0.0",
            allow_lora=True,
            effective_allow_lora=effective_allow_lora,
            config_fingerprint=fingerprint,
        )
    )


def _observe(app: Mock, overview: Mock, snapshot: WorkerStateSnapshot) -> None:
    HordeWorkerTUI._maybe_mark_trend_config_change(app, overview, snapshot)


def _stub_app() -> Mock:
    app = Mock()
    app._last_trend_config_state = None
    return app


def test_the_first_snapshot_does_not_show_the_notice() -> None:
    """The first reading is the baseline."""
    app, overview = _stub_app(), Mock()

    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=True))

    overview.note_config_changed.assert_not_called()


def test_a_changed_fingerprint_shows_the_notice() -> None:
    """A new settings fingerprint marks the trends as stabilizing."""
    app, overview = _stub_app(), Mock()
    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=True))

    _observe(app, overview, _snapshot(fingerprint="settings-b", effective_allow_lora=True))

    overview.note_config_changed.assert_called_once()


def test_suspended_lora_work_shows_the_notice() -> None:
    """LoRA work suspended under unchanged settings still marks the trends as stabilizing."""
    app, overview = _stub_app(), Mock()
    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=True))

    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=False))

    overview.note_config_changed.assert_called_once()


def test_an_unchanged_state_does_not_show_the_notice() -> None:
    """A field outside the settings and an unchanged LoRA state leave the trends alone."""
    app, overview = _stub_app(), Mock()
    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=True))

    _observe(app, overview, _snapshot(fingerprint="settings-a", effective_allow_lora=True, dreamer_name="Renamed"))

    overview.note_config_changed.assert_not_called()
