"""Configures pytest and creates fixtures."""

# import hordelib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from loguru import logger

# Mark the process as running under test before any HordeWorkerProcessManager can be constructed. Its
# startup otherwise reaps orphaned child pids recorded in the shared .horde_worker_regen/owned_pids.json
# (killing any still-alive match) and writes an action-ledger file into the working directory; both are
# gated on AI_HORDE_TESTING (see process_manager.py). CI sets it, but a bare local `pytest` would not, so a
# test that builds a real manager could reach across and terminate a live worker's inference/safety
# children sharing this directory. setdefault so an explicitly-provided value still wins.
os.environ.setdefault("AI_HORDE_TESTING", "True")

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_PROBE_TIMINGS: list[tuple[str, str]] = []
"""(capability slug, one-line timing summary) recorded by capability-probe tests for the end-of-run table.

Each cold capability probe boots its own worker, so a probe's wall-clock is dominated by startup and the
one-time model load, not inference. Collecting each probe's breakdown and printing it as a session-end
table makes that warmup-versus-inference split visible across a whole run, where the per-probe loguru
line is otherwise lost once the harness tears down the console sink."""


@pytest.fixture
def record_probe_timing() -> Callable[[str, str], None]:
    """Return a callback a probe test calls with its capability slug and ``ProbeTiming.summary()``."""

    def _record(slug: str, summary: str) -> None:
        _PROBE_TIMINGS.append((slug, summary))

    return _record


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Print the collected capability-probe timing breakdowns (warmup vs inference) after the run."""
    if not _PROBE_TIMINGS:
        return
    terminalreporter.section("capability probe timing (warmup vs inference)")
    for slug, summary in _PROBE_TIMINGS:
        terminalreporter.write_line(f"{slug}: {summary}")


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


@dataclass(frozen=True)
class _OrderPhase:
    """One bucket in the fast-to-slow run order, matched by marker, ``tests/`` sub-package, or module name.

    The three matcher kinds are kept distinct so a coincidental name overlap can never pull a test into the
    wrong bucket: ``packages`` matches a *directory* under ``tests/`` (the namespace), while ``module_prefixes``
    matches a top-level test module by filename stem. That separation is why ``benchmark`` (the namespace) runs
    last without dragging along a unit test merely named ``..._benchmark_...`` that lives elsewhere.
    """

    name: str
    markers: frozenset[str] = field(default_factory=frozenset)
    packages: frozenset[str] = field(default_factory=frozenset)
    module_prefixes: tuple[str, ...] = ()


# Phases that run first, fastest at the top. Cheap, high-signal tests go here so an obvious break surfaces
# without waiting on the slow namespaces below.
_ORDER_PHASES_FIRST: tuple[_OrderPhase, ...] = (
    _OrderPhase("bootstrap", packages=frozenset({"bootstrap"})),
    _OrderPhase("utils", module_prefixes=("test_utils_",)),
)

# Phases that run last, in this order. Marker-driven phases (``e2e``, ``gpu``) match wherever the marker is
# applied; add a new slow namespace by appending an _OrderPhase, no other change required.
_ORDER_PHASES_LAST: tuple[_OrderPhase, ...] = (
    _OrderPhase("process_management", packages=frozenset({"process_management"})),
    _OrderPhase("tui", packages=frozenset({"tui"})),
    _OrderPhase("bridge_data", module_prefixes=("test_bridge_data",)),
    _OrderPhase("analysis", packages=frozenset({"analysis"})),
    _OrderPhase("benchmark", packages=frozenset({"benchmark"})),
    _OrderPhase("e2e", markers=frozenset({"e2e"}), packages=frozenset({"e2e"})),
    _OrderPhase("slow", markers=frozenset({"slow"})),
    _OrderPhase("gpu", markers=frozenset({"gpu"})),
)

_TESTS_ROOT = Path(__file__).parent


def _item_package_and_stem(item: pytest.Item) -> tuple[frozenset[str], str]:
    """Return an item's ``tests/`` directory components and its module filename stem.

    Matching against real path components (not the nodeid string) is what keeps the buckets robust: a test
    named ``test_is_benchmark_stale`` in ``tests/test_app_state.py`` reports stem ``test_app_state`` and no
    ``benchmark`` directory, so the ``benchmark`` namespace phase cannot claim it.
    """
    try:
        rel = item.path.relative_to(_TESTS_ROOT)
    except ValueError:
        return frozenset(), ""
    return frozenset(rel.parts[:-1]), rel.stem


def _phase_matches(phase: _OrderPhase, marks: frozenset[str], packages: frozenset[str], stem: str) -> bool:
    """Whether ``item`` (described by its markers/packages/stem) belongs to ``phase``."""
    if marks & phase.markers:
        return True
    if packages & phase.packages:
        return True
    return any(stem.startswith(prefix) for prefix in phase.module_prefixes)


def _run_order_rank(item: pytest.Item) -> int:
    """Sort key placing first-phase items below 0, unmatched at 0, and last-phase items above 0.

    Among the last phases the *latest* match wins, so an absolute-last marker (e.g. ``gpu``) sorts after the
    namespace it physically lives in. A last-phase match always outranks a first-phase one.
    """
    marks = frozenset(marker.name for marker in item.iter_markers())
    packages, stem = _item_package_and_stem(item)

    last_offsets = [
        offset for offset, phase in enumerate(_ORDER_PHASES_LAST) if _phase_matches(phase, marks, packages, stem)
    ]
    if last_offsets:
        return 1 + max(last_offsets)
    for offset, phase in enumerate(_ORDER_PHASES_FIRST):
        if _phase_matches(phase, marks, packages, stem):
            return offset - len(_ORDER_PHASES_FIRST)
    return 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``gpu``- and ``slow``-marked tests unless explicitly requested, and sort tests fastest-first.

    GPU tests are opt-in on every box: each boots real worker processes with multi-minute warm-ups, so a bare
    ``pytest`` run must never pay for them implicitly just because a CUDA device happens to be present. They
    run only when the ``-m`` expression names ``gpu``, and even then a box with no CUDA device skips them
    rather than failing at device initialisation.

    ``slow`` tests follow the same opt-in contract for the same reason: each spawns real OS subprocesses (or
    runs a multi-second workload), which on Windows pays a full per-child spawn and package import. A default
    ``pytest`` sweep therefore skips them and finishes in minutes; ``-m slow`` runs them (and ``-m "slow or
    gpu"`` runs both bands). Naming a marker in the ``-m`` expression, even negated as in ``-m "not slow"``,
    leaves that band to pytest's own marker filtering rather than this blanket skip.

    The run order is defined by ``_ORDER_PHASES_FIRST`` / ``_ORDER_PHASES_LAST``; the sort is stable, so tests
    sharing a phase keep their collection order (siblings in a namespace stay together).
    """
    m_expression = config.getoption("-m") or ""

    if "gpu" not in m_expression:
        skip_gpu = pytest.mark.skip(
            reason="gpu tests are opt-in: request them with -m gpu (real worker boots, minutes each)",
        )
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip_gpu)
    elif not _cuda_device_present():
        skip_gpu = pytest.mark.skip(reason="no CUDA device available (run on a GPU box to exercise @pytest.mark.gpu)")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip_gpu)

    if "slow" not in m_expression:
        skip_slow = pytest.mark.skip(
            reason="slow tests are opt-in: request them with -m slow (real subprocess spawns / multi-second work)",
        )
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)

    items.sort(key=_run_order_rank)


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
