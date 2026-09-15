"""Executable provisioning is keyed by horde backend name and reports what it cannot obtain by name."""

from __future__ import annotations

from pathlib import Path

import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.text_backends import provision
from horde_worker_regen.text_backends.provision import (
    EXECUTABLE_PROVISIONERS,
    TextBackendProvisionError,
    provision_executable,
)


def test_koboldcpp_has_a_provisioner_keyed_by_horde_name() -> None:
    """The registry uses the reference package's vocabulary and covers the first supported backend."""
    assert all(isinstance(kind, TEXT_BACKENDS) for kind in EXECUTABLE_PROVISIONERS)
    assert TEXT_BACKENDS.koboldcpp in EXECUTABLE_PROVISIONERS


def test_a_backend_without_a_provisioner_is_refused_by_name() -> None:
    """A backend the worker cannot fetch is named in the error along with the supported set and the remedy."""
    unsupported = next(kind for kind in TEXT_BACKENDS if kind not in EXECUTABLE_PROVISIONERS)

    with pytest.raises(TextBackendProvisionError) as raised:
        provision_executable(unsupported)

    assert str(unsupported) in str(raised.value)
    assert str(TEXT_BACKENDS.koboldcpp) in str(raised.value)
    assert "text_backend_executable" in str(raised.value)


def test_provisioning_dispatches_to_the_registered_provisioner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registered callable's path is returned untouched."""
    expected = tmp_path / "backend.exe"
    monkeypatch.setitem(provision.EXECUTABLE_PROVISIONERS, TEXT_BACKENDS.koboldcpp, lambda: expected)

    assert provision_executable(TEXT_BACKENDS.koboldcpp) == expected


def test_a_missing_bootstrap_package_is_a_named_provision_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without worker_bootstrap importable, koboldcpp provisioning tells the operator to point at a binary."""
    import builtins

    real_import = builtins.__import__

    def refuse_bootstrap(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("worker_bootstrap"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse_bootstrap)

    with pytest.raises(TextBackendProvisionError) as raised:
        provision_executable(TEXT_BACKENDS.koboldcpp)

    assert "text_backend_executable" in str(raised.value)
