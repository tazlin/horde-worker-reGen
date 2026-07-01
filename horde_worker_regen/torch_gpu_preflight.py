"""Fail-fast preflight that verifies the baked torch build can run the host GPU (container entrypoint use).

A Docker image bakes one CUDA torch build at build time and, by design, never re-syncs, so it cannot
self-heal a wrong build the way the ``worker_bootstrap`` launch path does. On a card outside the baked
build's architecture window (a Blackwell ``sm_120`` RTX 50-series card on a ``cu126`` image, say) every
GPU model load raises ``cudaErrorNoKernelImageForDevice``, which surfaces only as a cryptic traceback
deep inside the first load (``open_clip``'s ``convert_weights_to_lp``, or a checkpoint load). Running this
before any model is loaded converts that into one clear instruction: rebuild the image with a matching
``TORCH_BACKEND``.

Best-effort and self-contained: it reads the installed wheel's own compiled architecture list (a static
property, no kernel launch) and the device's compute capability, and only fails when a real,
readable mismatch is found. Any probe error returns success so a working container is never blocked on a
flaky probe. The architecture predicate is a deliberate duplicate of the one in
:mod:`worker_bootstrap.detect` (not importable in every runtime layout); a guard test pins them together.
"""

from __future__ import annotations

import sys

# Above this compute capability, the cu126 wheel carries no kernel image (Blackwell sm_100/sm_120), so a
# rebuild must move to a CUDA 13 build. Mirrors worker_bootstrap.detect._CU126_MAX_COMPUTE_CAP.
_CU126_MAX_COMPUTE_CAP = (9, 0)


def _gpu_arch_supported(arch_list: list[str], capability: tuple[int, int]) -> bool:
    """Whether a CUDA torch build compiled for ``arch_list`` has a usable kernel/PTX for ``capability``.

    Mirrors CUDA's compatibility rules: a binary ``sm_<n>`` cubin runs a device of the same major whose
    minor is >= the cubin's (forward-compatible only within a major), and any ``compute_<n>`` PTX
    JIT-forwards to any device at or above its ``(major, minor)``. Kept byte-for-byte in step with the
    copies in :mod:`worker_bootstrap.detect` and the inference process (a guard test enforces it).
    """
    dev_major, dev_minor = capability
    for entry in arch_list:
        kind, _, ver = entry.partition("_")
        if not ver.isdigit() or len(ver) < 2:
            continue
        major, minor = int(ver[:-1]), int(ver[-1])
        if kind == "sm" and major == dev_major and minor <= dev_minor:
            return True
        if kind == "compute" and (major, minor) <= (dev_major, dev_minor):
            return True
    return False


def recommended_backend(capability: tuple[int, int]) -> str:
    """The locked ``TORCH_BACKEND`` extra to rebuild the image with for a card of ``capability``.

    A card above the cu126 architecture ceiling (Blackwell and newer) needs a CUDA 13 build; anything
    else that reached here (a pre-Turing card on a CUDA 13 image) is served by cu126.
    """
    return "cu130" if capability > _CU126_MAX_COMPUTE_CAP else "cu126"


def incompatibility_message(
    device_name: str,
    capability: tuple[int, int],
    arch_list: list[str],
    torch_version: str,
    torch_build: str | None,
) -> str:
    """Build the operator-facing fatal message naming the mismatch and the exact rebuild command."""
    cap_tag = f"sm_{capability[0]}{capability[1]}"
    build_desc = f"{torch_version}, CUDA {torch_build} build" if torch_build else torch_version
    backend = recommended_backend(capability)
    return (
        f"FATAL: this image's PyTorch ({build_desc}) has no CUDA kernels for {device_name} "
        f"(compute capability {capability[0]}.{capability[1]}, {cap_tag}); the wheel was built for "
        f"{' '.join(arch_list)}. A Docker image is immutable and cannot switch torch builds at runtime, so "
        f"every GPU model load would crash (typically deep inside the first CLIP/checkpoint load). Rebuild "
        f"the image for this GPU, for example:\n"
        f"  docker build -f Dockerfiles/Dockerfile.cuda --build-arg TORCH_BACKEND={backend} "
        f"--build-arg CUDA_VERSION=13.0.1 -t horde-worker-regen:cuda .\n"
        f"RTX 50-series / Blackwell cards need {backend} on a CUDA 13 base and a CUDA-13-capable NVIDIA "
        f"driver on the host. See Dockerfiles/README.md."
    )


def main() -> int:
    """Return 0 when the baked torch can run the host GPU (or the answer is unknown), 1 on a real mismatch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        arch_list = list(torch.cuda.get_arch_list())
        # Only CUDA builds tag architectures as sm_/compute_; a ROCm build reports gfx*, so skip it.
        if not any(arch.startswith("sm_") for arch in arch_list):
            return 0
        capability = torch.cuda.get_device_capability(0)
        if _gpu_arch_supported(arch_list, capability):
            return 0
        device_name = torch.cuda.get_device_name(0)
        torch_version = getattr(torch, "__version__", "?")
        torch_build = getattr(torch.version, "cuda", None)
    except Exception as error:  # noqa: BLE001 - a probe failure must never block a working container
        print(f"GPU preflight skipped ({type(error).__name__}: {error}).", file=sys.stderr)
        return 0
    print(incompatibility_message(device_name, capability, arch_list, torch_version, torch_build), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
