"""Unit tests for cross-platform GPU/torch-build detection."""

from __future__ import annotations

import pytest

from worker_bootstrap import detect


@pytest.mark.parametrize(
    ("smi_output", "expected"),
    [
        ("NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.8", (12, 8)),
        ("CUDA Version: 13.0", (13, 0)),
        ("CUDA Version: 13.2", (13, 2)),
        ("CUDA Version:13.2", (13, 2)),
        # Drivers that renamed the header field to "CUDA UMD Version" and added "KMD Version".
        ("| NVIDIA-SMI 610.88  KMD Version: 610.88  CUDA UMD Version: 13.3 |", (13, 3)),
        (
            "| NVIDIA-SMI 610.88                 KMD Version: 610.88        CUDA UMD Version: 13.3     |",
            (13, 3),
        ),
        # The `nvidia-smi -q` spellings of both fields.
        ("    CUDA Version                          : 13.3 [Deprecated; will be removed in CUDA 14.0.", (13, 3)),
        ("    CUDA UMD Version                      : 13.3", (13, 3)),
        ("KMD Version: 610.88", (0, 0)),
        ("CUDA Version: unknown", (0, 0)),
        ("no cuda header here", (0, 0)),
        ("", (0, 0)),
    ],
)
def test_parse_cuda_version(smi_output: str, expected: tuple[int, int]) -> None:
    """The (major, minor) CUDA ceiling is parsed from either header spelling ((0, 0) when absent)."""
    assert detect.parse_cuda_version(smi_output) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((12, 6), detect.CU126),
        ((12, 8), detect.CU126),
        ((13, 0), detect.CU130),
        ((13, 1), detect.CU130),  # 13.1 cannot run a 13.2-built wheel, so it stays on cu130
        ((13, 2), detect.CU132),
        ((13, 5), detect.CU132),
        ((14, 0), detect.CU132),  # a newer major still covers the 13.2 build via backward-compat
        ((0, 0), detect.CU126),  # version unreadable -> safe 12.6 build
    ],
)
def test_cuda_build(version: tuple[int, int], expected: str) -> None:
    """_cuda_build picks the newest locked CUDA build the driver's max CUDA version can run."""
    assert detect._cuda_build(version) == expected


@pytest.mark.parametrize(
    ("smi_output", "expected"),
    [
        ("8.6", ((8, 6),)),
        ("12.0\n", ((12, 0),)),
        ("8.9\n12.0\n", ((8, 9), (12, 0))),  # every card, in nvidia-smi's order
        ("12.0\n6.1\n", ((12, 0), (6, 1))),
        ("8.9, 610.88\n12.0, 610.88\n", ((8, 9), (12, 0))),  # the driver version as a second field
        ("compute_cap\n7.5", ((7, 5),)),
        ("", ()),
        ("no caps here", ()),
        ("Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 580.178\n", ()),
        ("8.6\r\n", ((8, 6),)),
    ],
)
def test_parse_compute_cap(smi_output: str, expected: tuple[tuple[int, int], ...]) -> None:
    """parse_compute_cap returns every whole-line compute capability in the query output."""
    assert detect.parse_compute_cap(smi_output) == expected


def test_parse_gpu_query_reads_driver_version() -> None:
    """The driver version comes from the second CSV field, and is None for a capability-only reading."""
    assert detect.parse_gpu_query("8.9, 610.88\n") == detect.GpuQuery(((8, 9),), "610.88")
    assert detect.parse_gpu_query("8.9\n") == detect.GpuQuery(((8, 9),), None)


@pytest.mark.parametrize(
    ("version", "compute_cap", "expected"),
    [
        # Floor: Blackwell (sm_120 / sm_100) does not carry a kernel image in cu126, so lift to cu130.
        ((12, 8), (12, 0), detect.CU130),  # old CUDA 12.x driver, known Blackwell card
        ((0, 0), (12, 0), detect.CU130),  # driver CUDA unreadable but the card is known Blackwell
        ((10, 0), (10, 0), detect.CU130),  # sm_100 datacenter Blackwell, same floor
        ((13, 0), (12, 0), detect.CU130),  # driver already covers Blackwell -> floor is a no-op
        ((13, 2), (12, 0), detect.CU132),  # newest driver-supported build still wins for Blackwell
        # Ceiling: pre-Turing (Maxwell/Pascal/Volta) was dropped from the CUDA 13 wheels, so hold cu126
        # even when the driver is new enough for cu130/cu132 (cu126 still runs on the newer driver).
        ((13, 2), (6, 1), detect.CU126),  # Pascal GTX 10-series on a CUDA 13.2 driver
        ((13, 0), (5, 2), detect.CU126),  # Maxwell on a CUDA 13.0 driver
        ((13, 0), (7, 0), detect.CU126),  # Volta sm_70 is pre-Turing and dropped too
        # In-window cards (Turing..Hopper) take the highest build the driver supports, unchanged.
        ((13, 2), (7, 5), detect.CU132),  # Turing sm_75 is the first arch the CUDA 13 wheels keep
        ((12, 8), (9, 0), detect.CU126),  # Hopper on a CUDA 12.x driver
        ((12, 8), (8, 6), detect.CU126),  # Ampere on a CUDA 12.x driver
        ((13, 2), (8, 6), detect.CU132),  # Ampere on a CUDA 13.2 driver
        # Arch unreadable -> keep the driver-only pick (prior behaviour).
        ((0, 0), (0, 0), detect.CU126),
        ((13, 2), (0, 0), detect.CU132),
    ],
)
def test_cuda_build_architecture_window(
    version: tuple[int, int],
    compute_cap: tuple[int, int],
    expected: str,
) -> None:
    """The driver-based build is clamped into the GPU's valid arch window (cu126 floor and ceiling)."""
    assert detect._cuda_build(version, (compute_cap,)) == expected


def _gpus(*capabilities: tuple[int, int], driver_version: str | None = "610.88") -> detect.GpuQuery:
    """A live GPU reading of the given cards."""
    return detect.GpuQuery(capabilities=capabilities, driver_version=driver_version)


@pytest.mark.parametrize(
    ("driver", "capabilities", "policy", "expected"),
    [
        # The all policy takes the intersection of the cards' windows.
        ((13, 3), ((6, 1), (7, 5)), detect.MixedCardPolicy.ALL, detect.CU126),
        ((12, 8), ((8, 9), (12, 0)), detect.MixedCardPolicy.ALL, detect.CU130),
        # A single build cannot run a pre-Turing card beside a Blackwell one, so all falls back to the highest card.
        ((13, 3), ((6, 1), (12, 0)), detect.MixedCardPolicy.ALL, detect.CU132),
        ((13, 3), ((6, 1), (12, 0)), detect.MixedCardPolicy.HIGHEST, detect.CU132),
        ((13, 3), ((6, 1), (12, 0)), detect.MixedCardPolicy.LOWEST, detect.CU126),
        ((13, 3), ((6, 1), (7, 5)), detect.MixedCardPolicy.HIGHEST, detect.CU132),
        ((12, 8), ((8, 9), (12, 0)), detect.MixedCardPolicy.LOWEST, detect.CU126),
        # None is the non-interactive default, all.
        ((13, 3), ((7, 5), (6, 1)), None, detect.CU126),
        ((13, 3), ((6, 1), (12, 0)), None, detect.CU132),
    ],
)
def test_cuda_build_mixed_card_policy(
    driver: tuple[int, int],
    capabilities: tuple[tuple[int, int], ...],
    policy: detect.MixedCardPolicy | None,
    expected: str,
) -> None:
    """The policy sets which cards the build must carry kernels for."""
    assert detect._cuda_build(driver, capabilities, policy) == expected


def _nvidia_detection(
    monkeypatch: pytest.MonkeyPatch,
    driver: tuple[int, int],
    gpus: detect.GpuQuery,
) -> None:
    """Stub the NVIDIA probes with one driver ceiling and one GPU reading."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: True)
    monkeypatch.setattr(detect, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(detect, "_nvidia_cuda_version", lambda: driver)
    monkeypatch.setattr(detect, "_nvidia_gpu_query", lambda: gpus)


def test_detection_records_every_card_and_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-Turing card beside a Blackwell one records both cards, the policy and the card left without kernels."""
    _nvidia_detection(monkeypatch, (13, 3), _gpus((6, 1), (12, 0)))
    decision = detect.describe_backend_selection()
    assert decision.final_token == detect.CU132
    assert decision.mixed_card_policy is detect.MixedCardPolicy.HIGHEST
    assert decision.to_dict()["compute_capabilities"] == ["6.1", "12.0"]
    assert decision.to_dict()["mixed_card_policy"] == "highest"
    assert "A single build cannot run every card" in decision.reason
    assert "6.1" in decision.reason


@pytest.mark.parametrize(
    "capabilities",
    [((8, 6),), ((8, 6), (8, 6)), ((8, 6), (8, 9))],
)
def test_uniform_box_does_not_record_a_policy(monkeypatch: pytest.MonkeyPatch, capabilities: tuple) -> None:
    """A policy is not recorded when every policy gives the cards the same build."""
    _nvidia_detection(monkeypatch, (13, 3), _gpus(*capabilities))
    decision = detect.describe_backend_selection()
    assert decision.final_token == detect.CU132
    assert decision.mixed_card_policy is None


def test_apply_mixed_card_policy_does_not_probe_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chosen policy is applied to the recorded inputs without a second nvidia-smi run."""
    _nvidia_detection(monkeypatch, (13, 3), _gpus((6, 1), (12, 0)))
    decision = detect.describe_backend_selection()

    def _boom() -> detect.GpuQuery:
        raise AssertionError("probed nvidia-smi again")

    monkeypatch.setattr(detect, "_nvidia_gpu_query", _boom)
    lowest = detect.apply_mixed_card_policy(decision, detect.MixedCardPolicy.LOWEST)
    assert lowest.final_token == detect.CU126
    assert lowest.mixed_card_policy is detect.MixedCardPolicy.LOWEST


def test_cards_disagree_only_when_the_policies_pick_different_builds() -> None:
    """A Blackwell card beside an Ada card disagrees only when the driver ceiling allows cu126 alone."""
    assert detect.cards_disagree((12, 8), ((8, 9), (12, 0)))
    assert not detect.cards_disagree((13, 3), ((8, 9), (12, 0)))
    assert detect.cards_disagree((13, 3), ((6, 1), (7, 5)))
    assert not detect.cards_disagree((13, 3), ((8, 6),))


def _live(driver: tuple[int, int], *capabilities: tuple[int, int]) -> detect.BackendDecision:
    """A live NVIDIA detection under the default policy."""
    return detect._nvidia_decision(smi_path="nvidia-smi", driver=driver, capabilities=capabilities, policy=None)


@pytest.mark.parametrize(
    ("persisted", "live", "expected"),
    [
        (detect.CU126, _live((13, 3), (8, 9)), detect.CU132),  # the driver ceiling now allows cu132
        (detect.CU126, _live((13, 0), (8, 9)), detect.CU130),
        (detect.CU130, _live((12, 8), (8, 9)), detect.CU126),  # the driver no longer supports cu130
        (detect.CU130, _live((0, 0), (8, 9)), detect.CU130),  # an unreadable ceiling does not move a token
        (detect.CU130, _live((0, 0)), detect.CU130),
        (detect.CU130, _live((12, 8), (12, 0)), detect.CU130),  # the arch floor still applies
        (detect.CU132, _live((13, 3), (6, 1)), detect.CU126),  # and the arch ceiling
        (detect.CPU, _live((13, 3), (8, 9)), detect.CPU),  # a non-CUDA token is not re-picked
        (detect.CU126, detect.BackendDecision(stage="detect", final_token=detect.CPU, reason="no GPU"), detect.CU126),
    ],
)
def test_repick_detected_build(persisted: str, live: detect.BackendDecision, expected: str) -> None:
    """A detection-derived CUDA token follows the live pick unless the driver ceiling is unreadable."""
    assert detect.repick_detected_build(persisted, live) == expected


def test_explain_repick_reports_the_direction() -> None:
    """The re-pick explanation states whether the ceiling allows a newer build or no longer supports the old one."""
    assert "now allows cu132" in detect.explain_repick(detect.CU126, _live((13, 3), (8, 9)))
    assert "no longer supports cu130" in detect.explain_repick(detect.CU130, _live((12, 8), (8, 9)))


@pytest.mark.parametrize(
    ("token", "compute_cap", "expected"),
    [
        # A stale cu126 token on a Blackwell card is lifted to cu130 (cu126 has no sm_120 kernel image).
        (detect.CU126, (12, 0), detect.CU130),
        (detect.CU126, (10, 0), detect.CU130),  # datacenter Blackwell sm_100
        # In-window and newer tokens are left alone for a Blackwell card.
        (detect.CU130, (12, 0), detect.CU130),
        (detect.CU132, (12, 0), detect.CU132),
        # A pre-Turing card is held at cu126 even if a CUDA 13 token was persisted.
        (detect.CU132, (6, 1), detect.CU126),
        (detect.CU130, (5, 2), detect.CU126),
        # In-window cards keep whatever runnable token they were given.
        (detect.CU126, (8, 6), detect.CU126),
        (detect.CU132, (8, 6), detect.CU132),
        # Unreadable capability -> leave the token untouched (no second-guessing).
        (detect.CU126, (0, 0), detect.CU126),
    ],
)
def test_reconcile_backend_for_gpu(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    compute_cap: tuple[int, int],
    expected: str,
) -> None:
    """A resolved CUDA token is re-clamped to the live GPU's arch window so an unrunnable build self-heals."""
    monkeypatch.setattr(detect, "_nvidia_gpu_query", lambda: _gpus(compute_cap))
    assert detect.reconcile_backend_for_gpu(token) == expected


@pytest.mark.parametrize(
    ("token", "policy", "expected"),
    [
        (detect.CU132, None, detect.CU126),  # the all policy keeps cu126 for the Pascal card
        (detect.CU132, detect.MixedCardPolicy.HIGHEST, detect.CU132),
        (detect.CU126, detect.MixedCardPolicy.HIGHEST, detect.CU126),
        (detect.CU132, detect.MixedCardPolicy.LOWEST, detect.CU126),
    ],
)
def test_reconcile_applies_the_mixed_card_policy(
    token: str,
    policy: detect.MixedCardPolicy | None,
    expected: str,
) -> None:
    """The reconcile clamps against the cards the policy covers, so it does not undo the policy's pick."""
    assert detect.reconcile_backend_for_gpu(token, policy=policy, gpus=_gpus((6, 1), (7, 5))) == expected


def test_describe_backend_selection_records_the_arch_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Blackwell card on an old driver yields a decision that names the ceiling, the clamp, and the reason."""
    _nvidia_detection(monkeypatch, (12, 8), _gpus((12, 0)))
    decision = detect.describe_backend_selection()
    assert decision.stage == "detect"
    assert decision.final_token == detect.CU130
    assert decision.driver_ceiling_build == detect.CU126  # driver ceiling before the arch clamp
    assert decision.clamp_action == detect.CLAMP_FLOORED
    assert decision.to_dict()["driver_cuda_version"] == "12.8"
    assert decision.to_dict()["compute_capabilities"] == ["12.0"]


def test_describe_backend_selection_reports_unreadable_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable compute capability keeps the driver-only pick and records the skipped clamp."""
    _nvidia_detection(monkeypatch, (13, 2), _gpus())
    decision = detect.describe_backend_selection()
    assert decision.final_token == detect.CU132
    assert decision.clamp_action == detect.CLAMP_SKIPPED


def test_describe_backend_selection_non_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU the decision resolves to cpu and records the absence."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: False)
    monkeypatch.setattr(detect, "_amd_present", lambda: False)
    decision = detect.describe_backend_selection()
    assert decision.final_token == detect.CPU
    assert decision.nvidia_present is False
    assert decision.amd_present is False


@pytest.mark.parametrize(
    ("token", "compute_cap", "expected_token", "expected_action"),
    [
        (detect.CU126, (12, 0), detect.CU130, detect.CLAMP_FLOORED),
        (detect.CU132, (6, 1), detect.CU126, detect.CLAMP_CEILED),
        (detect.CU130, (8, 6), detect.CU130, detect.CLAMP_NONE),
        (detect.CU126, (0, 0), detect.CU126, detect.CLAMP_SKIPPED),
        (detect.CPU, (12, 0), detect.CPU, detect.CLAMP_NOT_CUDA),
    ],
)
def test_describe_reconcile_records_clamp_action(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    compute_cap: tuple[int, int],
    expected_token: str,
    expected_action: str,
) -> None:
    """A reconcile decision records the input token, the resolved token, and what the clamp did."""
    monkeypatch.setattr(detect, "_nvidia_gpu_query", lambda: _gpus(compute_cap))
    decision = detect.describe_reconcile(token)
    assert decision.stage == "reconcile"
    assert decision.input_token == token
    assert decision.final_token == expected_token
    assert decision.clamp_action == expected_action


@pytest.mark.parametrize("token", [detect.CPU, detect.ROCM, detect.ROCM_WINDOWS, detect.AMD_UNSUPPORTED])
def test_reconcile_backend_leaves_non_cuda_tokens_untouched(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    """Non-CUDA tokens pass through unchanged and never trigger an nvidia-smi probe."""

    def _boom() -> detect.GpuQuery:
        raise AssertionError("probed nvidia-smi for a non-CUDA token")

    monkeypatch.setattr(detect, "_nvidia_gpu_query", _boom)
    assert detect.reconcile_backend_for_gpu(token) == token


@pytest.mark.parametrize(
    ("nvidia", "cuda_version", "compute_cap", "amd", "rocm", "expected"),
    [
        (True, (12, 8), (8, 6), False, False, detect.CU126),
        (True, (13, 0), (8, 6), False, False, detect.CU130),
        (True, (13, 2), (8, 6), False, False, detect.CU132),
        (True, (0, 0), (0, 0), False, False, detect.CU126),  # nvidia present but unreadable -> safe 12.6 build
        # Blackwell card on an old CUDA 12.x driver is floored to cu130 by the architecture check.
        (True, (12, 8), (12, 0), False, False, detect.CU130),
        (False, (0, 0), (0, 0), True, False, detect.AMD_UNSUPPORTED),
        (False, (0, 0), (0, 0), True, True, detect.ROCM),
        (False, (0, 0), (0, 0), False, False, detect.CPU),
    ],
)
def test_detect_backend(
    monkeypatch: pytest.MonkeyPatch,
    nvidia: bool,
    cuda_version: tuple[int, int],
    compute_cap: tuple[int, int],
    amd: bool,
    rocm: bool,
    expected: str,
) -> None:
    """detect_backend maps hardware presence + CUDA version + GPU arch onto the right build token."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: nvidia)
    monkeypatch.setattr(detect, "_nvidia_cuda_version", lambda: cuda_version)
    monkeypatch.setattr(detect, "_nvidia_gpu_query", lambda: _gpus(compute_cap))
    monkeypatch.setattr(detect, "_amd_present", lambda: amd)
    monkeypatch.setattr(detect, "_rocm_runtime_present", lambda: rocm)
    monkeypatch.setattr(detect, "_windows_amd_rocm_backend", lambda: None)
    assert detect.detect_backend() == expected


@pytest.mark.parametrize(
    ("adapter_name", "expected"),
    [
        ("AMD Radeon RX 7900 XTX", detect.ROCM_WINDOWS),
        ("AMD Radeon RX 7700 XT", detect.ROCM_WINDOWS),
        ("AMD Radeon PRO W7900", detect.ROCM_WINDOWS),
        ("AMD Radeon RX 9070 XT", detect.ROCM_WINDOWS),
        ("AMD Radeon AI PRO R9700", detect.ROCM_WINDOWS),
        ("AMD Radeon 8060S Graphics", detect.ROCM_WINDOWS),
        ("AMD Ryzen AI 9 HX 370 w/ Radeon Graphics", detect.ROCM_WINDOWS),
        ("AMD Ryzen AI Max+ 395 w/ Radeon Graphics", detect.ROCM_WINDOWS),
    ],
)
def test_windows_amd_rocm_backend_supported_profiles(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
    expected: str,
) -> None:
    """Supported AMD Windows adapter families map to the ROCm Windows profile."""
    monkeypatch.setattr(detect, "_is_windows", lambda: True)
    monkeypatch.setattr(detect, "_windows_display_adapters", lambda: [adapter_name])
    assert detect._windows_amd_rocm_backend() == expected


def test_windows_amd_rocm_backend_unknown_card_is_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown AMD Windows cards remain unsupported instead of getting an unsafe ROCm wheel family."""
    monkeypatch.setattr(detect, "_is_windows", lambda: True)
    monkeypatch.setattr(detect, "_windows_display_adapters", lambda: ["AMD Radeon RX 6800 XT"])
    assert detect._windows_amd_rocm_backend() is None


def test_detect_backend_uses_windows_amd_rocm_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """An AMD Windows profile returned by the matcher wins over the generic unsupported path."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: False)
    monkeypatch.setattr(detect, "_amd_present", lambda: True)
    monkeypatch.setattr(detect, "_windows_amd_rocm_backend", lambda: detect.ROCM_WINDOWS)
    monkeypatch.setattr(detect, "_rocm_runtime_present", lambda: False)
    assert detect.detect_backend() == detect.ROCM_WINDOWS
