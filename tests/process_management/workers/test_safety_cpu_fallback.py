"""An on-GPU safety child reruns a check on the CPU when the device runs out of memory, and stays up.

The safety child shares its card with samplers it does not control, so an evaluation can find the device
full through no fault of its own. Ending the process for that costs a restart and strands the job in
flight. The child instead returns its device memory, runs the same evaluation on the CPU in the same
process, reports the verdict and goes back to being an on-GPU child. Every fake here stands in for a
horde_safety object, so no weights load and no device is touched.
"""

from __future__ import annotations

import queue
import uuid

import pytest
from horde_sdk.ai_horde_api.apimodels.base import GenerationID
from loguru import logger

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes
from horde_worker_regen.process_management.workers import safety_process
from horde_worker_regen.process_management.workers.safety_process import (
    HordeSafetyProcess,
    _ClipResidency,
    _OnDemandDeepDanbooruModel,
)
from tests.process_management.workers.test_safety_device import _FakeDeepDanbooruModel
from tests.process_management.workers.test_safety_orchestration import _RecordingQueue

_DEVICE_OOM_MESSAGE = (
    "CUDA out of memory. Tried to allocate 72.00 MiB. GPU 0 has a total capacity of 23.99 GiB of which "
    "0 bytes is free. Of the allocated memory 2.89 GiB is allocated by PyTorch."
)
"""The allocator text of a full device, recognised by fingerprint so the tests need no torch import."""

_HALF = "float16"
_SINGLE = "float32"


class _FakeTensor:
    """A parameter or buffer reduced to the two things the precision switch reads: ``dtype`` and ``data``."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    @property
    def data(self) -> _FakeTensor:
        return self

    @data.setter
    def data(self, value: _FakeTensor) -> None:
        self.dtype = value.dtype

    def to(self, *, dtype: str) -> _FakeTensor:
        return _FakeTensor(dtype)


class _FakeClipModel:
    """CLIP as loaded on CUDA: half-precision projections beside single-precision normalisation weights."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.device = "cuda"
        self.device_moves_to_fail = 0
        self.parameters_by_name = {"visual.proj": _FakeTensor(_HALF), "ln_final.weight": _FakeTensor(_SINGLE)}
        self.buffers_by_name = {"attn_mask": _FakeTensor(_SINGLE)}

    def to(self, device: str) -> _FakeClipModel:
        if device != "cpu" and self.device_moves_to_fail > 0:
            self.device_moves_to_fail -= 1
            raise RuntimeError(_DEVICE_OOM_MESSAGE)
        self.device = device
        self._events.append(f"clip_to:{device}")
        return self

    def float(self) -> _FakeClipModel:
        for tensor in (*self.parameters_by_name.values(), *self.buffers_by_name.values()):
            tensor.dtype = _SINGLE
        return self

    def named_parameters(self) -> list[tuple[str, _FakeTensor]]:
        return list(self.parameters_by_name.items())

    def named_buffers(self) -> list[tuple[str, _FakeTensor]]:
        return list(self.buffers_by_name.items())

    def dtypes(self) -> dict[str, str]:
        return {name: tensor.dtype for name, tensor in (*self.named_parameters(), *self.named_buffers())}


class _FakeInterrogator:
    def __init__(self, events: list[str]) -> None:
        self.clip_model = _FakeClipModel(events)
        self.clip_offloaded = False
        self.device = "cuda"


class _Verdict:
    """A verdict that records where it was computed: the CPU's answer is NSFW and the device's is clear."""

    def __init__(self, *, is_nsfw: bool) -> None:
        self.is_nsfw = is_nsfw
        self.is_csam = False


class _ScriptedChecker:
    """Stands in for ``NSFWChecker``: records where each check ran and raises whatever the test queued."""

    def __init__(self, interrogator: _FakeInterrogator, events: list[str], text_cache: dict[str, object]) -> None:
        self._interrogator = interrogator
        self._events = events
        self._text_cache = text_cache
        self.errors: list[BaseException | None] = []
        self.cache_seen_per_check: list[dict[str, object]] = []
        self.clip_dtypes_per_check: list[dict[str, str]] = []

    def check_for_nsfw(self, **_kwargs: object) -> _Verdict:
        device = self._interrogator.device
        self._events.append(f"check:{device}:clip_on_{self._interrogator.clip_model.device}")
        self.cache_seen_per_check.append(dict(self._text_cache))
        self.clip_dtypes_per_check.append(self._interrogator.clip_model.dtypes())
        self._text_cache[f"embedding computed on {device}"] = device
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error
        return _Verdict(is_nsfw=device == "cpu")


class _Child:
    """A safety child assembled for one test, with the handles the assertions read."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, safety_device: str = "cuda") -> None:
        self.events: list[str] = []
        self.queue = _RecordingQueue()
        self.interrogator = _FakeInterrogator(self.events)
        self.text_features: dict[str, object] = {"embedding computed on cuda": "cuda"}
        self.text_array_features: dict[tuple[str, ...], object] = {}
        self.tokens: list[tuple[object, object]] = []
        self.checker = _ScriptedChecker(self.interrogator, self.events, self.text_features)
        self.deep_danbooru = _FakeDeepDanbooruModel()

        process = HordeSafetyProcess.__new__(HordeSafetyProcess)
        process.process_id = 0
        process.process_launch_identifier = 0
        process.process_message_queue = self.queue  # type: ignore[assignment]
        process._control_inbox = queue.SimpleQueue()
        process._dry_run_skip_safety = False
        process._safety_device = safety_device
        process._interrogator = self.interrogator  # type: ignore[assignment]
        process._clip_residency = _ClipResidency(self.interrogator, execution_device=safety_device)
        process._deep_danbooru_model = _OnDemandDeepDanbooruModel(
            self.deep_danbooru,  # type: ignore[arg-type]
            execution_device=safety_device,
        )
        process._nsfw_checker = self.checker  # type: ignore[assignment]
        self.process = process

        def _report(include_vram: bool = False, **_kwargs: object) -> bool:
            self.events.append(f"memory_report:vram={include_vram}")
            return True

        monkeypatch.setattr(process, "send_memory_report_message", _report)
        monkeypatch.setattr(
            safety_process,
            "_clear_safety_accelerator_cache",
            lambda: self.events.append("device_cache_cleared"),
        )
        monkeypatch.setattr(
            safety_process,
            "_safety_text_feature_caches",
            lambda: (self.tokens, self.text_array_features, self.text_features),
            raising=False,
        )

    @property
    def residency(self) -> _ClipResidency:
        residency = self.process._clip_residency
        assert residency is not None
        return residency

    def demote(self) -> None:
        """Put the child in the demoted rest state: CLIP in host RAM, staged per evaluation."""
        self.residency.resident = False
        self.interrogator.clip_model.device = "cpu"
        self.interrogator.clip_offloaded = True

    def evaluate(self) -> GenerationID:
        """Hand the child one single-image job through its control loop and return the job id."""
        job_id = GenerationID(root=uuid.uuid4())
        self.send(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=job_id,
                prompt="a test prompt",
                censor_nsfw=False,
                sfw_worker=False,
                images_bytes=[make_dummy_png_bytes()],
                horde_model_info=None,
            ),
        )
        return job_id

    def send(self, message: HordeControlMessage) -> None:
        """Deliver a control message the way the reader thread does, then run the child's loop over it."""
        self.process._control_inbox.put(message)
        self.process.receive_and_handle_control_messages()

    def results(self) -> list[HordeSafetyResultMessage]:
        return [m for m in self.queue.messages if isinstance(m, HordeSafetyResultMessage)]

    def checks(self) -> list[str]:
        return [event for event in self.events if event.startswith("check:")]


def test_a_check_that_runs_out_of_device_memory_is_rerun_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job gets the CPU's verdict, the device is released first, and the child is on-GPU again after."""
    child = _Child(monkeypatch)
    child.checker.errors = [RuntimeError(_DEVICE_OOM_MESSAGE)]
    dtypes_at_rest = child.interrogator.clip_model.dtypes()

    job_id = child.evaluate()

    assert child.process._end_process is False
    results = child.results()
    assert len(results) == 1
    assert results[0].job_id == job_id
    assert results[0].safety_evaluations[0].is_nsfw is True, "the verdict is the CPU rerun's"
    assert child.queue.state_changes()[-1] == HordeProcessState.WAITING_FOR_JOB

    assert child.checks() == ["check:cuda:clip_on_cuda", "check:cpu:clip_on_cpu"]
    rerun_position = child.events.index("check:cpu:clip_on_cpu")
    before_rerun = child.events[:rerun_position]
    assert "clip_to:cpu" in before_rerun
    assert before_rerun.index("clip_to:cpu") < before_rerun.index("device_cache_cleared")
    assert "memory_report:vram=True" in before_rerun[before_rerun.index("device_cache_cleared") :]

    # The rerun computes in single precision against embeddings it computed itself, as a CPU child does.
    assert set(child.checker.clip_dtypes_per_check[1].values()) == {_SINGLE}
    assert child.checker.cache_seen_per_check[1] == {}

    assert child.residency.resident is True
    assert child.interrogator.device == "cuda"
    assert child.interrogator.clip_model.dtypes() == dtypes_at_rest
    assert child.text_features == {"embedding computed on cuda": "cuda"}
    assert child.events[-1] == "memory_report:vram=True"


def test_the_next_job_after_a_fallback_is_attempted_on_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fallback is for one evaluation. The following job stages CLIP and runs on the device."""
    child = _Child(monkeypatch)
    child.checker.errors = [RuntimeError(_DEVICE_OOM_MESSAGE)]
    child.evaluate()

    child.evaluate()

    assert child.process._end_process is False
    assert child.checks()[-1] == "check:cuda:clip_on_cuda"
    assert child.results()[-1].safety_evaluations[0].is_nsfw is False
    assert child.residency.resident is True
    assert child.checker.cache_seen_per_check[-1] == {"embedding computed on cuda": "cuda"}


def test_a_demoted_child_that_cannot_stage_clip_checks_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Staging is the first device allocation of a demoted child's evaluation, and it falls back too."""
    child = _Child(monkeypatch)
    child.demote()
    child.interrogator.clip_model.device_moves_to_fail = 1

    job_id = child.evaluate()

    assert child.process._end_process is False
    results = child.results()
    assert [result.job_id for result in results] == [job_id]
    assert results[0].safety_evaluations[0].is_nsfw is True
    assert child.checks() == ["check:cpu:clip_on_cpu"]
    assert child.residency.resident is False
    assert child.interrogator.clip_offloaded is True
    assert child.interrogator.clip_model.device == "cpu"
    assert child.interrogator.device == "cuda"


def test_an_aesthetic_score_that_runs_out_of_device_memory_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aesthetic pass is part of the evaluation, so its failure reruns the whole job on the CPU."""
    child = _Child(monkeypatch)
    devices_scored_on: list[str] = []

    def _score(_image: object) -> float:
        devices_scored_on.append(child.interrogator.device)
        if len(devices_scored_on) == 1:
            raise RuntimeError(_DEVICE_OOM_MESSAGE)
        return 5.5

    monkeypatch.setattr(child.process, "_score_aesthetic", _score)
    job_id = GenerationID(root=uuid.uuid4())

    child.send(
        HordeSafetyControlMessage(
            control_flag=HordeControlFlag.EVALUATE_SAFETY,
            job_id=job_id,
            prompt="a test prompt",
            censor_nsfw=False,
            sfw_worker=False,
            images_bytes=[make_dummy_png_bytes()],
            horde_model_info=None,
            include_aesthetic_score=True,
        ),
    )

    assert child.process._end_process is False
    assert devices_scored_on == ["cuda", "cpu"]
    evaluation = child.results()[0].safety_evaluations[0]
    assert evaluation.aesthetic_score == 5.5
    assert evaluation.is_nsfw is True


def test_a_fallback_is_logged_once_with_its_job_stage_free_figure_and_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One warning per fallback carries what an operator needs to size the problem."""
    child = _Child(monkeypatch)
    child.checker.errors = [RuntimeError(_DEVICE_OOM_MESSAGE)]
    warnings: list[str] = []
    sink_id = logger.add(lambda record: warnings.append(str(record)), level="WARNING", format="{message}")
    try:
        job_id = child.evaluate()
    finally:
        logger.remove(sink_id)

    fallback_lines = [line for line in warnings if "on the CPU" in line]
    assert len(fallback_lines) == 1
    assert str(job_id) in fallback_lines[0]
    assert "NSFW check" in fallback_lines[0]
    assert "0 bytes free" in fallback_lines[0]
    assert "seconds" in fallback_lines[0]


def test_a_failure_that_is_not_out_of_memory_still_ends_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a full device earns a rerun. Any other failure leaves the process state unknown, as before."""
    child = _Child(monkeypatch)
    child.checker.errors = [ValueError("the checker rejected its input")]

    child.evaluate()

    assert child.process._end_process is True
    assert child.results() == []
    assert child.checks() == ["check:cuda:clip_on_cuda"]


def test_a_failing_cpu_rerun_ends_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is one fallback tier: a rerun that fails for any reason ends the process."""
    child = _Child(monkeypatch)
    child.checker.errors = [RuntimeError(_DEVICE_OOM_MESSAGE), RuntimeError("the CPU rerun failed")]

    child.evaluate()

    assert child.process._end_process is True
    assert child.results() == []
    assert child.checks() == ["check:cuda:clip_on_cuda", "check:cpu:clip_on_cpu"]


def test_a_cpu_child_has_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child already on the CPU has nowhere to fall back to, so the failure ends it as before."""
    child = _Child(monkeypatch, safety_device="cpu")
    child.interrogator.device = "cpu"
    child.interrogator.clip_model.device = "cpu"
    child.checker.errors = [RuntimeError(_DEVICE_OOM_MESSAGE)]

    child.evaluate()

    assert child.process._end_process is True
    assert child.results() == []


def test_a_promotion_that_runs_out_of_device_memory_leaves_the_child_demoted_and_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promotion that cannot stage CLIP settles in the demoted state and reports memory as a demotion does."""
    child = _Child(monkeypatch)
    child.demote()
    child.interrogator.clip_model.device_moves_to_fail = 1

    child.send(HordeControlMessage(control_flag=HordeControlFlag.PROMOTE_SAFETY_WEIGHTS))

    assert child.process._end_process is False
    assert child.residency.resident is False
    assert child.interrogator.clip_offloaded is True
    assert child.interrogator.clip_model.device == "cpu"
    assert child.events[-2:] == ["device_cache_cleared", "memory_report:vram=True"]

    job_id = child.evaluate()
    assert [result.job_id for result in child.results()] == [job_id]
    assert child.checks() == ["check:cuda:clip_on_cuda"]


def test_a_demotion_that_runs_out_of_device_memory_leaves_the_child_demoted_and_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A demotion whose release fails still ends demoted, with the process up and a memory report sent."""
    child = _Child(monkeypatch)

    def _failing_clear() -> None:
        raise RuntimeError(_DEVICE_OOM_MESSAGE)

    monkeypatch.setattr(safety_process, "_clear_safety_accelerator_cache", _failing_clear)

    child.send(HordeControlMessage(control_flag=HordeControlFlag.DEMOTE_SAFETY_WEIGHTS))

    assert child.process._end_process is False
    assert child.residency.resident is False
    assert child.interrogator.clip_offloaded is True
    assert child.interrogator.clip_model.device == "cpu"
    assert child.events[-1] == "memory_report:vram=True"
