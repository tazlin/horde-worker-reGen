"""A weight an older library fetched into a record's own folder is removed once the routed copy is in place.

Every face fixer opens its helper weights from ``gfpgan/``. A library without that routing fetched a
codeformer record's helpers into ``codeformer/``, where nothing reads them. These tests state what happens
to such a copy on the next startup scan: gone when the routed copy is present and the same size, kept when
the routed copy is missing or differs, and never touched when the record's file lives in its own folder.
"""

from __future__ import annotations

import importlib
import queue
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from horde_worker_regen.alchemy_forms import AuxiliaryFetchNeeds
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID, HordeDownloadProcess

HELPER = "detection_Resnet50_Final.pth"


def _process() -> HordeDownloadProcess:
    return HordeDownloadProcess(
        process_id=DOWNLOAD_PROCESS_ID,
        process_message_queue=queue.Queue(),  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        process_launch_identifier=0,
        fetch_needs=AuxiliaryFetchNeeds(post_processing=True),
    )


def _codeformer_manager(root: Path) -> SimpleNamespace:
    """A fake codeformer manager whose record declares its weight plus one helper routed to gfpgan/."""
    return SimpleNamespace(
        model_folder_path=str(root / "codeformer"),
        model_reference={"CodeFormers": object()},
        get_model_filenames=lambda _name: [
            {"file_path": Path("CodeFormers.pth")},
            {"file_path": Path("..") / "gfpgan" / HELPER, "file_type": "face_restore_helper"},
        ],
    )


def _inject(monkeypatch: pytest.MonkeyPatch, codeformer: SimpleNamespace) -> None:
    manager = SimpleNamespace(
        gfpgan=None,
        esrgan=None,
        codeformer=codeformer,
        miscellaneous=None,
        controlnet=None,
        controlnet_annotator=None,
    )
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = SimpleNamespace(manager=manager)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)
    monkeypatch.setattr(importlib.import_module("hordelib"), "api", fake_api, raising=False)


def _place(root: Path, folder: str, name: str, payload: bytes) -> Path:
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_a_stray_helper_copy_is_removed_once_the_routed_copy_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The copy in codeformer/ goes, with its checksum sidecar; the gfpgan/ copy and the weight itself stay."""
    routed = _place(tmp_path, "gfpgan", HELPER, b"helper-bytes")
    stray = _place(tmp_path, "codeformer", HELPER, b"helper-bytes")
    sidecar = _place(tmp_path, "codeformer", "detection_Resnet50_Final.sha256", b"abc")
    weight = _place(tmp_path, "codeformer", "CodeFormers.pth", b"weight")
    _inject(monkeypatch, _codeformer_manager(tmp_path))

    _process()._remove_stray_routed_copies()

    assert not stray.exists()
    assert not sidecar.exists()
    assert routed.read_bytes() == b"helper-bytes"
    assert weight.exists()


def test_a_stray_copy_is_kept_while_the_routed_copy_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing is deleted until the file exists where the restorer reads it."""
    stray = _place(tmp_path, "codeformer", HELPER, b"helper-bytes")
    _inject(monkeypatch, _codeformer_manager(tmp_path))

    _process()._remove_stray_routed_copies()

    assert stray.exists()


def test_a_copy_of_a_different_size_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A same-named file that is not the same download is not this tidy's to judge."""
    _place(tmp_path, "gfpgan", HELPER, b"helper-bytes")
    stray = _place(tmp_path, "codeformer", HELPER, b"something-else-entirely")
    _inject(monkeypatch, _codeformer_manager(tmp_path))

    _process()._remove_stray_routed_copies()

    assert stray.exists()


def test_same_size_but_different_bytes_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only a byte-for-byte duplicate is removed; a coincidence of name and size is not enough."""
    _place(tmp_path, "gfpgan", HELPER, b"helper-bytes")
    stray = _place(tmp_path, "codeformer", HELPER, b"helper-BYTES")
    _inject(monkeypatch, _codeformer_manager(tmp_path))

    _process()._remove_stray_routed_copies()

    assert stray.exists()


def test_a_record_whose_routed_folder_is_its_own_is_never_touched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GFPGAN routes its helpers into gfpgan/, its own folder: the one copy there is the real one and stays."""
    helper = _place(tmp_path, "gfpgan", HELPER, b"helper-bytes")
    gfpgan = SimpleNamespace(
        model_folder_path=str(tmp_path / "gfpgan"),
        model_reference={"GFPGAN": object()},
        get_model_filenames=lambda _name: [
            {"file_path": Path("GFPGANv1.4.pth")},
            {"file_path": Path("..") / "gfpgan" / HELPER, "file_type": "face_restore_helper"},
        ],
    )
    _inject(monkeypatch, gfpgan)

    _process()._remove_stray_routed_copies()

    assert helper.read_bytes() == b"helper-bytes"


def test_a_stray_that_is_a_symlink_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A link an operator placed is theirs; the tidy deletes only regular files it can prove are duplicates."""
    routed = _place(tmp_path, "gfpgan", HELPER, b"helper-bytes")
    (tmp_path / "codeformer").mkdir()
    link = tmp_path / "codeformer" / HELPER
    try:
        link.symlink_to(routed)
    except OSError:
        pytest.skip("symlinks need a privilege this account lacks")
    _inject(monkeypatch, _codeformer_manager(tmp_path))

    _process()._remove_stray_routed_copies()

    assert link.is_symlink()


def test_an_unexpected_path_shape_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only the exact ../<folder>/<name> shape the layout module produces is a candidate."""
    stray = _place(tmp_path, "codeformer", HELPER, b"helper-bytes")
    _place(tmp_path, "gfpgan/sub", HELPER, b"helper-bytes")
    odd = SimpleNamespace(
        model_folder_path=str(tmp_path / "codeformer"),
        model_reference={"CodeFormers": object()},
        get_model_filenames=lambda _name: [{"file_path": Path("..") / "gfpgan" / "sub" / HELPER}],
    )
    _inject(monkeypatch, odd)

    _process()._remove_stray_routed_copies()

    assert stray.exists()
