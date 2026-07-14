"""Tests for the A3 spawn-pinning follow-ups: on-GPU safety pinning and per-card DirectML indices.

These assert the *spawn kwargs* the lifecycle manager hands each child (via the injected context's
``Process`` factory), so they verify the parent-side wiring without launching a real process or torch.
The single-GPU shape stays byte-identical: an unmasked card passes ``accelerator_kind=None`` and the
global DirectML value through unchanged.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_test_card_runtimes, make_test_runtime_config


def _make_plm(
    *,
    card_runtimes: dict[int, CardRuntime],
    safety_on_gpu: bool = False,
    directml: int | None = None,
) -> tuple[ProcessLifecycleManager, Mock]:
    """Build a PLM whose child spawns are captured by a fake context's ``Process`` mock.

    Returns the manager and the fake context so a test can read ``fake_ctx.Process.call_args``.
    """
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.safety_on_gpu = safety_on_gpu
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False
    bridge_data.dry_run_skip_safety = False
    bridge_data.dry_run_skip_inference = False
    bridge_data.dry_run_inference_delay = 1.0

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 4321

    plm = ProcessLifecycleManager(
        ctx=fake_ctx,  # type: ignore[arg-type]
        process_map=ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=Mock(),
        process_message_queue=Mock(),
        card_runtimes=card_runtimes,
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=directml,
        abort_callback=Mock(),
        state=WorkerState(),
    )
    return plm, fake_ctx


def _spawn_kwargs(fake_ctx: Mock) -> dict[str, object]:
    """The ``kwargs`` dict the lifecycle passed to the most recent child spawn."""
    return fake_ctx.Process.call_args.kwargs["kwargs"]


class TestSafetyOnGpuPinning:
    """The on-GPU safety process is pinned to the first configured card, but only when masked."""

    def test_single_gpu_safety_on_gpu_is_unmasked(self) -> None:
        """A default single-GPU host passes accelerator_kind None, so no pin is applied (byte-identical)."""
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(device_indices=(0,), mask_kind=None),
            safety_on_gpu=True,
        )
        plm.start_safety_processes()
        kwargs = _spawn_kwargs(fake_ctx)
        assert kwargs["device_index"] == 0
        assert kwargs["accelerator_kind"] is None

    def test_multi_gpu_safety_pins_to_first_card(self) -> None:
        """On a masked multi-GPU host the safety model lands on the lowest-indexed (first) card."""
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda"),
            safety_on_gpu=True,
        )
        plm.start_safety_processes()
        kwargs = _spawn_kwargs(fake_ctx)
        assert kwargs["device_index"] == 0
        assert kwargs["accelerator_kind"] == "cuda"

    def test_safety_off_gpu_still_carries_first_card_but_entry_point_skips_pin(self) -> None:
        """cpu_only safety carries the first card's identity; the pin is gated off by cpu_only in the child.

        The lifecycle always passes the first card's device/kind; the entry point only acts on it when the
        safety process is on-GPU. This keeps the spawn uniform while leaving cpu_only safety unpinned.
        """
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda"),
            safety_on_gpu=False,
        )
        plm.start_safety_processes()
        kwargs = _spawn_kwargs(fake_ctx)
        # cpu_only is positional (the 6th arg); the device identity still rides along in kwargs.
        assert fake_ctx.Process.call_args.kwargs["args"][5] is True  # cpu_only
        assert kwargs["accelerator_kind"] == "cuda"


class TestPerCardDirectml:
    """DirectML, lacking an env-var mask, is pinned per card via the --directml comfy arg."""

    def test_multi_card_directml_derives_per_card_index(self) -> None:
        """Without an explicit flag, each DirectML card's process targets its own adapter index."""
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(
                device_indices=(0, 1),
                kind="directml",
                mask_kind="directml",
            ),
            directml=None,
        )
        plm._start_inference_process(0, device_index=0)
        assert _spawn_kwargs(fake_ctx)["directml"] == 0
        plm._start_inference_process(1, device_index=1)
        assert _spawn_kwargs(fake_ctx)["directml"] == 1

    def test_explicit_directml_flag_is_authoritative_for_every_card(self) -> None:
        """An explicit --directml=N selects one adapter, so all cards' processes target N (legacy contract)."""
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(
                device_indices=(0, 1),
                kind="directml",
                mask_kind="directml",
            ),
            directml=5,
        )
        plm._start_inference_process(0, device_index=0)
        assert _spawn_kwargs(fake_ctx)["directml"] == 5
        plm._start_inference_process(1, device_index=1)
        assert _spawn_kwargs(fake_ctx)["directml"] == 5

    def test_non_directml_card_passes_global_directml_through(self) -> None:
        """A CUDA card never derives a DirectML index; it passes the (None) global value unchanged."""
        plm, fake_ctx = _make_plm(
            card_runtimes=make_test_card_runtimes(device_indices=(0,), kind="cuda", mask_kind=None),
            directml=None,
        )
        plm._start_inference_process(0, device_index=0)
        assert _spawn_kwargs(fake_ctx)["directml"] is None
