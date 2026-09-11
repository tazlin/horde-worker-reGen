"""The run root: one directory for a run's sentinel, logs and state, announced when it is set."""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from horde_worker_regen import run_root as run_root_module
from horde_worker_regen.run_root import (
    ABORT_SENTINEL_NAME,
    LOGS_DIR_NAME,
    RUN_ROOT_ENV_VAR,
    abort_sentinel_path,
    describe_run_root,
    logs_dir,
    run_root,
)


@pytest.fixture
def announcements() -> list[str]:
    """Every log line the resolver emits during the test, with the dedup state reset first."""
    lines: list[str] = []
    run_root_module._announced = None
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    yield lines
    logger.remove(sink)


def test_unset_means_the_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unset, the root is the working directory and the paths hang off it."""
    monkeypatch.delenv(RUN_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    assert run_root() == tmp_path.resolve()
    assert abort_sentinel_path() == tmp_path.resolve() / ABORT_SENTINEL_NAME
    assert logs_dir() == tmp_path.resolve() / LOGS_DIR_NAME


def test_the_variable_names_the_root_regardless_of_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A set variable wins over the working directory."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(elsewhere))
    monkeypatch.chdir(tmp_path)

    assert run_root() == elsewhere.resolve()
    assert abort_sentinel_path().parent == elsewhere.resolve()


def test_logs_dir_is_created_on_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The logs directory is created only when a caller asks for it."""
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path / "run"))

    assert not logs_dir().exists()
    created = logs_dir(create=True)
    assert created.is_dir()
    assert created == (tmp_path / "run" / LOGS_DIR_NAME).resolve()


def test_a_set_variable_is_announced_once_at_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, announcements: list[str]
) -> None:
    """One announcement per process for a given value, however often it is resolved."""
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path))

    run_root()
    run_root()
    abort_sentinel_path()

    matching = [line for line in announcements if RUN_ROOT_ENV_VAR in line]
    assert len(matching) == 1, announcements
    assert str(tmp_path.resolve()) in matching[0]


def test_a_missing_directory_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, announcements: list[str]
) -> None:
    """A value naming a directory that does not exist is called out."""
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path / "not-yet"))

    run_root()

    assert any("does not exist" in line for line in announcements), announcements


def test_a_changed_value_is_announced_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, announcements: list[str]
) -> None:
    """A different value is a new announcement, so a mid-run change is visible."""
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path / "a"))
    run_root()
    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path / "b"))
    run_root()

    matching = [line for line in announcements if RUN_ROOT_ENV_VAR in line and "Run root from" in line]
    assert len(matching) == 2, announcements


def test_the_description_names_the_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The banner line says whether the root came from the variable or the cwd."""
    monkeypatch.delenv(RUN_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert describe_run_root().endswith("(working directory)")

    monkeypatch.setenv(RUN_ROOT_ENV_VAR, str(tmp_path))
    assert RUN_ROOT_ENV_VAR in describe_run_root()
