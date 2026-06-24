"""The long-lived orchestrator process must stay torch-free.

The worker spawns inference/safety children that legitimately load torch (~500MB RSS) for their core
function. The parent orchestrator does not run inference: it schedules, tracks, and talks to the horde
API, and must never pay that footprint. The two ways torch sneaks in are (1) a module-level
``from hordelib.api import ...`` in a parent module (the facade drags torch even for pure-Python helpers),
and (2) a *runtime* device query (``enumerate_accelerators`` / ``get_torch_*_vram_mb``) called in-process.

These tests are the tripwires for both. They run in subprocesses so ``sys.modules`` is clean (an earlier
test in the session may already have imported torch in-process). See ``AGENTS.md`` for the convention.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Parent-process and torch-free-by-contract planning modules. Importing any of these must not pull torch:
# they run in the orchestrator (or the no-boot benchmark ``plan`` preview), which is torch-free.
_TORCH_FREE_IMPORT_MODULES = [
    "horde_worker_regen.process_management.process_manager",
    "horde_worker_regen.process_management.inference_scheduler",
    "horde_worker_regen.process_management.desired_state",
    "horde_worker_regen.process_management.feature_readiness",
    "horde_worker_regen.process_management.resource_budget",
    "horde_worker_regen.process_management.job_popper",
    "horde_worker_regen.process_management.process_map",
    "horde_worker_regen.capabilities",
    "horde_worker_regen.utils.gpu_monitor",
    "horde_worker_regen.utils.accelerator_probe",
]


def _run_torch_free_snippet(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter that asserts torch is absent, returning the completed process.

    The child exits non-zero (and prints the offending ``torch.*`` modules) if torch was imported, so the
    caller can assert on ``returncode`` with a useful failure message.
    """
    checker = (
        f"{body}\n"
        "import sys\n"
        "torch_mods = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "if torch_mods:\n"
        "    print('TORCH_LOADED:' + ','.join(torch_mods))\n"
        "    sys.exit(7)\n"
    )
    env = {**os.environ, "AI_HORDE_TESTING": "True"}
    return subprocess.run(
        [sys.executable, "-c", checker],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("module_name", _TORCH_FREE_IMPORT_MODULES)
def test_orchestrator_module_import_is_torch_free(module_name: str) -> None:
    """Importing a parent/planning module must not drag torch in (guards against a stray ``hordelib.api``)."""
    result = _run_torch_free_snippet(f"import {module_name}")
    assert result.returncode == 0, (
        f"importing {module_name} pulled in torch:\n{result.stdout}\n{result.stderr}\n"
        "Import pure-Python helpers from their torch-free hordelib submodule (feature_impact / "
        "feature_requirements / metrics / utils.torch_memory / utils.logger), never from hordelib.api."
    )


def test_system_resources_detect_does_not_load_torch_in_caller() -> None:
    """Calling ``SystemResources.detect()`` enumerates devices out-of-process, so the caller stays torch-free."""
    result = _run_torch_free_snippet(
        "from horde_worker_regen.process_management.process_manager import SystemResources\nSystemResources.detect()",
    )
    assert result.returncode == 0, (
        f"SystemResources.detect() loaded torch into the orchestrator:\n{result.stdout}\n{result.stderr}\n"
        "Device enumeration must go through accelerator_probe.probe_accelerators (a subprocess), not an "
        "in-process enumerate_accelerators() call."
    )


def test_probe_accelerators_does_not_load_torch_in_caller() -> None:
    """``probe_accelerators`` runs the torch-loading enumeration in a subprocess; the caller stays clean."""
    result = _run_torch_free_snippet(
        "from horde_worker_regen.utils.accelerator_probe import probe_accelerators\nprobe_accelerators()",
    )
    assert result.returncode == 0, (
        f"probe_accelerators() loaded torch into the caller:\n{result.stdout}\n{result.stderr}"
    )


def test_building_the_gpu_sampler_reader_does_not_load_torch() -> None:
    """The GPU duty-cycle sampler runs in the orchestrator, so building its reader must not pull torch.

    Regression tripwire: the reader once delegated to ``get_accelerator_utilization_percent``, which gates on
    the active torch backend and so ``import torch`` -- pulling torch into the parent and tripping a
    partial-init circular import at worker startup. Import-time guards miss it because torch only entered when
    the reader was *built* at ``start()``. NVML utilization is torch-free, so this must stay clean.
    """
    result = _run_torch_free_snippet(
        "from horde_worker_regen.utils.gpu_monitor import _make_utilization_reader\n_make_utilization_reader(0)",
    )
    assert result.returncode == 0, (
        f"building the GPU sampler reader loaded torch into the orchestrator:\n{result.stdout}\n{result.stderr}\n"
        "Read utilization via the torch-free hordelib.utils.nvml helper, never get_accelerator_utilization_percent."
    )
