"""Tests for the safety process's device resolution.

The parent decides ``cpu_only`` from config plus the torch-free install sentinel, which can disagree
with the actual torch build (a manually installed CPU torch whose ``bin/backend`` sentinel was never
written reports ``cpu_only=False``). The torch-bearing safety child must still load its models on CPU
when CUDA is genuinely unavailable, or horde_safety raises during deserialization.
"""

from __future__ import annotations

from horde_worker_regen.process_management.workers.safety_process import (
    HordeSafetyProcess,
    _OnDemandDeepDanbooruModel,
    resolve_safety_device,
)


def test_cpu_only_always_resolves_to_cpu() -> None:
    """An explicit cpu_only request uses CPU whether or not CUDA is present."""
    assert resolve_safety_device(cpu_only=True, cuda_available=True) == "cpu"
    assert resolve_safety_device(cpu_only=True, cuda_available=False) == "cpu"


def test_falls_back_to_cpu_when_cuda_unavailable() -> None:
    """The regression: cpu_only=False but no real CUDA must not try to load on 'cuda'."""
    assert resolve_safety_device(cpu_only=False, cuda_available=False) == "cpu"


def test_uses_cuda_when_requested_and_available() -> None:
    """A normal GPU worker is unaffected: CUDA is used when requested and present."""
    assert resolve_safety_device(cpu_only=False, cuda_available=True) == "cuda"


class _FakeDeepDanbooruModel:
    """Record model placement while standing in for the torch module."""

    def __init__(self) -> None:
        self._initial_device = "cpu"
        self.moves: list[str] = []
        self.tags = ["safe", "questionable"]

    def to(self, device: str) -> _FakeDeepDanbooruModel:
        self.moves.append(device)
        return self

    def evaluate_tensor(self, tensor: object) -> tuple[object, str]:
        return tensor, self._initial_device


def test_deep_danbooru_uses_gpu_only_during_its_conditional_evaluation() -> None:
    """The optional anime classifier returns to CPU while CLIP remains the safety lane's fixed GPU model."""
    model = _FakeDeepDanbooruModel()
    on_demand = _OnDemandDeepDanbooruModel(model, execution_device="cuda")  # type: ignore[arg-type]

    result = on_demand.evaluate_tensor("image")

    assert result == ("image", "cuda")
    assert model.moves == ["cuda"]
    on_demand.offload()
    assert model.moves == ["cuda", "cpu"]
    assert model._initial_device == "cpu"
    assert on_demand.tags == ["safe", "questionable"]


class _MovableModel:
    """Minimal model exposing the ``to`` contract used by transient safety companions."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> _MovableModel:
        self.moves.append(device)
        return self


def test_idle_safety_offloads_lazy_companions_without_moving_clip() -> None:
    """BLIP and the aesthetic head leave VRAM before WAITING; the hot CLIP model is untouched."""
    clip = _MovableModel()
    caption = _MovableModel()
    aesthetic = _MovableModel()
    process = HordeSafetyProcess.__new__(HordeSafetyProcess)
    process._safety_device = "cuda"
    process._caption_model_loaded = True
    process._interrogator = type(
        "FakeInterrogator",
        (),
        {"clip_model": clip, "caption_model": caption, "caption_offloaded": False},
    )()
    process._aesthetic_scorer = aesthetic  # type: ignore[assignment]

    process._offload_transient_models()

    assert clip.moves == []
    assert caption.moves == ["cpu"]
    assert aesthetic.moves == ["cpu"]
    assert process._interrogator.caption_offloaded is True  # type: ignore[attr-defined]
