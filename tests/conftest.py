"""Configures pytest and creates fixtures."""

# import hordelib
from pathlib import Path

import pytest
from loguru import logger

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _cuda_device_present() -> bool:
    """Return whether a CUDA accelerator is available, importing torch only when asked (never raising).

    Used to auto-skip ``@pytest.mark.gpu`` tests so CI (and any GPU-less dev box) stays green without them,
    while a real GPU box runs them. Imported lazily inside the collection hook so a GPU-free run never pays
    the torch import just to decide to skip.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch / no driver / probe error all mean "no usable GPU here"
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``gpu``-marked tests when no CUDA device is present (keeps CI and GPU-less boxes green)."""
    if not any(item.get_closest_marker("gpu") for item in items):
        return
    if _cuda_device_present():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available (run on a GPU box to exercise @pytest.mark.gpu)")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def init_hordelib() -> None:
    """Initialise hordelib for the tests."""
    # hordelib.initialise() # FIXME
    logger.warning("hordelib.initialise() not called")


PRECOMMIT_FILE_PATH = Path(__file__).parent.parent / ".pre-commit-config.yaml"
PYPROJECT_FILE_PATH = Path(__file__).parent.parent / "pyproject.toml"

TRACKED_DEPENDENCIES = [
    "horde_sdk",
    "horde_engine",
    "horde_model_reference",
    "horde_safety",
    "torch",
    "pydantic",
]


@pytest.fixture(scope="session")
def tracked_dependencies() -> list[str]:
    """Get the tracked dependencies."""
    return TRACKED_DEPENDENCIES


def _parse_version(spec: str) -> str:
    """Extract version from a PEP 508 dependency specifier."""
    for op in ("~=", "==", ">="):
        if op in spec:
            version = spec.split(op)[1].strip()
            # Strip extras after comma, semicolons (env markers), or +
            for sep in (",", ";", "+"):
                version = version.split(sep)[0].strip()
            return version
    raise ValueError(f"Unsupported version pin: {spec}")


def get_dependency_versions_from_pyproject() -> dict[str, str]:
    """Get the versions of tracked dependencies from pyproject.toml."""
    with open(PYPROJECT_FILE_PATH, "rb") as f:
        data = tomllib.load(f)

    deps = data["project"]["dependencies"]
    versions: dict[str, str] = {}

    for dep_str in deps:
        dep_name = dep_str.split("[")[0].split(">")[0].split("~")[0].split("=")[0].split("<")[0].strip()
        normalised = dep_name.replace("-", "_").lower()
        for tracked in TRACKED_DEPENDENCIES:
            if normalised == tracked.replace("-", "_").lower():
                versions[tracked] = _parse_version(dep_str)

    return versions


@pytest.fixture(scope="session")
def horde_dependency_versions() -> dict[str, str]:
    """Get the versions of horde dependencies from pyproject.toml."""
    return get_dependency_versions_from_pyproject()
