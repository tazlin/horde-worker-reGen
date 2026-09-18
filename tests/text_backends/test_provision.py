"""Provisioning what a text backend needs: its executable by backend name, and its model file by record."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from horde_model_reference.download_engine import DownloadOutcome
from horde_model_reference.model_reference_records import TextGenerationModelRecord
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.compute_mode import TextBackendAccelerator
from horde_worker_regen.text_backends import provision
from horde_worker_regen.text_backends.model_catalogue import UNKNOWN_FILE_URL, ResolvedTextModel
from horde_worker_regen.text_backends.provision import (
    EXECUTABLE_PROVISIONERS,
    TextBackendProvisionError,
    effective_text_backend_accelerator,
    ensure_text_model_file,
    provision_executable,
)

_DIGEST = "a" * 64
_FILE_NAME = "Measured-Model-Q4_K_M.gguf"


@dataclasses.dataclass
class _FetchCall:
    """One recorded call into the reference package's downloader."""

    origin_url: str
    destination: Path
    sha256: str | None


def _resolved(*, models_dir: Path, file_url: str, size_bytes: int) -> ResolvedTextModel:
    """Return a resolution against a one-file record, the shape a catalogued name resolves to."""
    record = TextGenerationModelRecord.model_validate(
        {
            "name": "author/Measured-Model-Q4_K_M",
            "parameters": 1_000_000_000,
            "size_on_disk_bytes": size_bytes,
            "config": {
                "download": [
                    {
                        "file_name": _FILE_NAME,
                        "file_url": file_url,
                        "sha256sum": _DIGEST,
                        "size_bytes": size_bytes,
                    },
                ],
            },
        },
    )
    return ResolvedTextModel(path=models_dir / _FILE_NAME, canonical_name=record.name, record=record)


def _recording_downloader(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[_FetchCall],
    *,
    written_bytes: bytes | None,
    success: bool = True,
) -> None:
    """Replace the reference downloader with one that records its arguments and writes *written_bytes*."""

    def fake_download_addressed_file(
        origin_url: str,
        destination: Path,
        *,
        sha256: str | None,
        **_kwargs: object,
    ) -> DownloadOutcome:
        calls.append(_FetchCall(origin_url=origin_url, destination=destination, sha256=sha256))
        if written_bytes is not None:
            destination.write_bytes(written_bytes)
        return DownloadOutcome(
            success=success,
            final_path=destination,
            bytes_written=len(written_bytes) if written_bytes is not None else 0,
            sha256=sha256 if success else None,
        )

    monkeypatch.setattr(provision, "download_addressed_file", fake_download_addressed_file)


def test_koboldcpp_has_a_provisioner_keyed_by_horde_name() -> None:
    """The registry uses the reference package's vocabulary and covers the first supported backend."""
    assert all(isinstance(kind, TEXT_BACKENDS) for kind in EXECUTABLE_PROVISIONERS)
    assert TEXT_BACKENDS.koboldcpp in EXECUTABLE_PROVISIONERS


def test_a_backend_without_a_provisioner_is_refused_by_name() -> None:
    """A backend the worker cannot fetch is named in the error along with the supported set and the remedy."""
    unsupported = next(kind for kind in TEXT_BACKENDS if kind not in EXECUTABLE_PROVISIONERS)

    with pytest.raises(TextBackendProvisionError) as raised:
        provision_executable(unsupported, accelerator=TextBackendAccelerator.CUDA)

    assert str(unsupported) in str(raised.value)
    assert str(TEXT_BACKENDS.koboldcpp) in str(raised.value)
    assert "text_backend_executable" in str(raised.value)


def test_provisioning_dispatches_to_the_registered_provisioner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registered callable receives the compute path and its returned path is untouched."""
    expected = tmp_path / "backend.exe"
    asked: list[TextBackendAccelerator] = []

    def record(accelerator: TextBackendAccelerator) -> Path:
        asked.append(accelerator)
        return expected

    monkeypatch.setitem(provision.EXECUTABLE_PROVISIONERS, TEXT_BACKENDS.koboldcpp, record)

    assert provision_executable(TEXT_BACKENDS.koboldcpp, accelerator=TextBackendAccelerator.VULKAN) == expected
    assert asked == [TextBackendAccelerator.VULKAN]


def _refuse_bootstrap_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``worker_bootstrap`` import fail, as it does on a wheel-only install."""
    import builtins

    real_import = builtins.__import__

    def refuse_bootstrap(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("worker_bootstrap"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse_bootstrap)


def test_a_missing_bootstrap_package_is_a_named_provision_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without worker_bootstrap importable, koboldcpp provisioning tells the operator to point at a binary."""
    _refuse_bootstrap_imports(monkeypatch)

    with pytest.raises(TextBackendProvisionError) as raised:
        provision_executable(TEXT_BACKENDS.koboldcpp, accelerator=TextBackendAccelerator.CUDA)

    assert "text_backend_executable" in str(raised.value)


class TestAcceleratorSelection:
    """What the install declares decides the compute path; only an install that declares nothing is probed."""

    def _backend_file(self, tmp_path: Path, token: str | None) -> Path:
        """Return a ``bin/backend`` path holding *token*, or one that does not exist when it is None."""
        backend_file = tmp_path / "bin" / "backend"
        if token is not None:
            backend_file.parent.mkdir(parents=True, exist_ok=True)
            backend_file.write_text(token, encoding="utf-8")
        return backend_file

    @pytest.fixture(autouse=True)
    def _clear_backend_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure a developer's HORDE_WORKER_BACKEND never leaks into these tests."""
        monkeypatch.delenv("HORDE_WORKER_BACKEND", raising=False)

    def test_a_declared_install_is_not_probed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The hardware probe answers a different question and must not override what was declared."""

        def refuse_detection() -> TextBackendAccelerator:
            raise AssertionError("a declared install must not be probed")

        monkeypatch.setattr(provision, "detect_text_backend_accelerator", refuse_detection)

        resolved = effective_text_backend_accelerator(
            TextBackendAccelerator.AUTO,
            backend_file=self._backend_file(tmp_path, "rocm-windows"),
        )

        assert resolved is TextBackendAccelerator.VULKAN

    def test_an_undeclared_install_falls_back_to_the_hardware(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A hand-rolled install records no token, so the only thing left to ask is the hardware."""
        monkeypatch.setattr(
            provision,
            "detect_text_backend_accelerator",
            lambda: TextBackendAccelerator.VULKAN,
        )

        resolved = effective_text_backend_accelerator(
            TextBackendAccelerator.AUTO,
            backend_file=self._backend_file(tmp_path, None),
        )

        assert resolved is TextBackendAccelerator.VULKAN

    @pytest.mark.parametrize(
        ("detected", "expected"),
        [
            ("cuda", TextBackendAccelerator.CUDA),
            ("vulkan", TextBackendAccelerator.VULKAN),
            ("cpu", TextBackendAccelerator.CPU),
        ],
    )
    def test_every_detected_path_has_a_worker_side_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        detected: str,
        expected: TextBackendAccelerator,
    ) -> None:
        """The bootstrap spells the paths itself, so each answer it can give has to land on a member here."""
        from worker_bootstrap import koboldcpp_bin

        monkeypatch.setattr(koboldcpp_bin, "detected_accelerator", lambda: detected)

        assert provision.detect_text_backend_accelerator() is expected

    def test_an_install_that_cannot_probe_keeps_the_previous_command_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wheel-only install carries no probe, and assuming CPU would strip a working card."""
        _refuse_bootstrap_imports(monkeypatch)

        assert provision.detect_text_backend_accelerator() is TextBackendAccelerator.CUDA


class TestModelFileProvisioning:
    """A model file is fetched once, verified by the downloader, and never re-fetched while it is whole."""

    def test_a_present_file_of_the_declared_size_is_not_fetched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Every launch would otherwise re-transfer gigabytes that are already on disk."""
        resolved = _resolved(models_dir=tmp_path, file_url="https://example.invalid/model.gguf", size_bytes=4)
        resolved.path.write_bytes(b"gguf")
        calls: list[_FetchCall] = []
        _recording_downloader(monkeypatch, calls, written_bytes=b"gguf")

        assert ensure_text_model_file(resolved) == resolved.path
        assert calls == []

    def test_an_absent_file_is_fetched_to_its_declared_path_with_its_declared_digest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The digest goes to the downloader, which is what makes an interrupted transfer safe to resume."""
        models_dir = tmp_path / "text_models"
        resolved = _resolved(models_dir=models_dir, file_url="https://example.invalid/model.gguf", size_bytes=4)
        calls: list[_FetchCall] = []
        _recording_downloader(monkeypatch, calls, written_bytes=b"gguf")

        returned = ensure_text_model_file(resolved)

        assert returned == models_dir / _FILE_NAME
        assert len(calls) == 1
        assert calls[0].origin_url == "https://example.invalid/model.gguf"
        assert calls[0].destination == models_dir / _FILE_NAME
        assert calls[0].sha256 == _DIGEST

    def test_a_short_file_is_fetched_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A file left short by an interrupted copy is present and unusable, so presence alone is not enough."""
        resolved = _resolved(models_dir=tmp_path, file_url="https://example.invalid/model.gguf", size_bytes=4)
        resolved.path.write_bytes(b"gg")
        calls: list[_FetchCall] = []
        _recording_downloader(monkeypatch, calls, written_bytes=b"gguf")

        ensure_text_model_file(resolved)

        assert len(calls) == 1

    def test_an_absent_file_without_an_origin_names_the_path_it_was_expected_at(self, tmp_path: Path) -> None:
        """No origin is on record, so the only thing the operator can act on is where the file should be."""
        models_dir = tmp_path / "text_models"
        resolved = _resolved(models_dir=models_dir, file_url=UNKNOWN_FILE_URL, size_bytes=4)

        with pytest.raises(TextBackendProvisionError) as raised:
            ensure_text_model_file(resolved)

        assert str(models_dir / _FILE_NAME) in str(raised.value)
        assert "text_models_dir" in str(raised.value)

    def test_a_short_file_without_an_origin_reports_the_size_it_found(self, tmp_path: Path) -> None:
        """The file is there and wrong, which is a different remedy from the file not being there at all."""
        resolved = _resolved(models_dir=tmp_path, file_url=UNKNOWN_FILE_URL, size_bytes=4)
        resolved.path.write_bytes(b"gg")

        with pytest.raises(TextBackendProvisionError) as raised:
            ensure_text_model_file(resolved)

        assert "2 bytes where 4 is expected" in str(raised.value)

    def test_a_failed_fetch_is_a_provision_error_naming_the_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The downloader fails a digest mismatch, and the supervisor's row needs to say what failed."""
        resolved = _resolved(models_dir=tmp_path, file_url="https://example.invalid/model.gguf", size_bytes=4)
        calls: list[_FetchCall] = []
        _recording_downloader(monkeypatch, calls, written_bytes=None, success=False)

        with pytest.raises(TextBackendProvisionError) as raised:
            ensure_text_model_file(resolved)

        assert "https://example.invalid/model.gguf" in str(raised.value)
        assert _DIGEST in str(raised.value)

    def test_an_operator_supplied_file_that_vanished_is_reported_rather_than_fetched(self, tmp_path: Path) -> None:
        """A path resolution carries no record, so a file removed after resolution has no origin to fall back on."""
        missing = tmp_path / "gone.gguf"
        resolved = ResolvedTextModel(path=missing, canonical_name="gone", record=None)

        with pytest.raises(TextBackendProvisionError) as raised:
            ensure_text_model_file(resolved)

        assert str(missing) in str(raised.value)
