"""Lock-time tripwire: the installed torch wheel must still match worker_bootstrap.detect's arch table.

The build-selection logic in ``worker_bootstrap.detect`` predicts, from a hardcoded table, which GPU
architectures each torch CUDA build carries kernels for (``_CU126_MAX_COMPUTE_CAP`` and
``_CUDA13_MIN_COMPUTE_CAP``). That prediction is made *before* torch is installed, so it cannot consult
the wheel itself and will silently rot when PyTorch changes its build matrix (a new CUDA line, a dropped
or added architecture). These tests inspect the *actually installed* torch and assert the table still
holds, so the day a maintainer bumps the locked torch the suite fails loudly if the assumptions drifted,
forcing a deliberate update of the constants instead of a stale snapshot.

Skipped when torch is absent or is a CPU/ROCm build (no ``sm_`` tags), since there is nothing to verify.
"""

from __future__ import annotations

import pytest

from worker_bootstrap import detect

torch = pytest.importorskip("torch", reason="torch not installed; nothing to verify against")


def _sm_caps(arch_list: list[str]) -> list[tuple[int, int]]:
    """Extract the binary ``sm_<major><minor>`` capabilities from a torch arch list (ignoring PTX)."""
    caps: list[tuple[int, int]] = []
    for entry in arch_list:
        kind, _, ver = entry.partition("_")
        if kind == "sm" and ver.isdigit() and len(ver) >= 2:
            caps.append((int(ver[:-1]), int(ver[-1])))
    return caps


@pytest.fixture(scope="module")
def installed_arch() -> tuple[list[str], list[tuple[int, int]], int]:
    """Return the installed wheel's (arch_list, sm_caps, cuda_major), skipping when not a CUDA build."""
    arch_list = list(torch.cuda.get_arch_list())
    caps = _sm_caps(arch_list)
    if not caps:
        pytest.skip(f"installed torch is not a CUDA build (arch list: {arch_list or 'empty'})")
    cuda_version = torch.version.cuda  # e.g. "12.6" or "13.0"; None on a CPU/ROCm build
    if not cuda_version:
        pytest.skip("installed torch reports no CUDA toolkit version")
    return arch_list, caps, int(cuda_version.split(".")[0])


def test_installed_wheel_matches_detect_table(installed_arch: tuple[list[str], list[tuple[int, int]], int]) -> None:
    """The installed wheel's real arch window must match the boundary constants detect.py predicts.

    Each assertion guards the specific selection rule that depends on it, so a failure points straight at
    the constant (and the matching literal in detect-backend.ps1) that PyTorch has outgrown.
    """
    arch_list, caps, cuda_major = installed_arch
    highest = max(caps)
    lowest = min(caps)

    if cuda_major <= 12:
        # The cu126 floor rule ("Blackwell+ has no kernels here, lift to cu130") rests on this ceiling.
        assert highest == detect._CU126_MAX_COMPUTE_CAP, (
            f"cu12x wheel now tops out at sm_{highest[0]}{highest[1]}, not "
            f"sm_{detect._CU126_MAX_COMPUTE_CAP[0]}{detect._CU126_MAX_COMPUTE_CAP[1]}; update "
            f"_CU126_MAX_COMPUTE_CAP (and detect-backend.ps1) and revisit the Blackwell floor. "
            f"arch list: {arch_list}"
        )
    else:
        # The cu13x ceiling rule ("pre-Turing was dropped, hold at cu126") rests on this floor.
        assert lowest == detect._CUDA13_MIN_COMPUTE_CAP, (
            f"cu13x wheel now starts at sm_{lowest[0]}{lowest[1]}, not "
            f"sm_{detect._CUDA13_MIN_COMPUTE_CAP[0]}{detect._CUDA13_MIN_COMPUTE_CAP[1]}; update "
            f"_CUDA13_MIN_COMPUTE_CAP (and detect-backend.ps1) and revisit the pre-Turing ceiling. "
            f"arch list: {arch_list}"
        )
        # The Blackwell floor sends those cards to cu130, which is only valid while cu13x actually carries
        # Blackwell kernels. If a future CUDA line drops them, the floor target is wrong.
        assert highest >= (12, 0), (
            f"cu13x wheel no longer carries Blackwell kernels (tops out at sm_{highest[0]}{highest[1]}); "
            f"the cu130 floor target for Blackwell is no longer valid. arch list: {arch_list}"
        )


def test_detect_predicts_installed_wheel_for_its_own_archs(
    installed_arch: tuple[list[str], list[tuple[int, int]], int],
) -> None:
    """detect.gpu_arch_supported (the post-install check) must accept every arch the wheel ships kernels for.

    A self-consistency guard: the predicate the post-sync self-check and the worker runtime both rely on
    has to agree that the installed wheel can run each architecture it was compiled for.
    """
    arch_list, caps, _ = installed_arch
    for cap in caps:
        assert detect.gpu_arch_supported(arch_list, cap), (
            f"gpu_arch_supported rejected sm_{cap[0]}{cap[1]} though the wheel ships that cubin: {arch_list}"
        )
