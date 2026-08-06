"""Configuration-space contracts for generated scheduling and recovery tests.

The stateful scenario sweep samples a covering array, while this module exhaustively verifies the cheap
configuration-resolution layer that defines which process topologies are valid. Boundary representatives
cover every public range and every cross-field rule that changes lane or semaphore structure.
"""

from __future__ import annotations

import itertools
from typing import Literal, cast

import pytest

from horde_worker_regen.bridge_data.data_model import cap_queue_size, reGenBridgeData
from horde_worker_regen.process_management.process_manager import resolve_card_concurrency

_THREAD_BOUNDARIES = (1, 2, 3, 16)
_QUEUE_BOUNDARIES = (0, 1, 4)
_MODEL_COUNTS = (1, 4)
_SERVES_IMAGE_GENERATION = (False, True)
_LEASE_ENABLED = (False, True)
_LEASE_SLOTS = (None, 1, 16)
_TAIL_OVERLAP = (False, True)

_FEATURE_AXES: dict[str, tuple[object, ...]] = {
    "max_inference_attempts": (1, 2, 5),
    "unload_models_from_vram_often": (False, True),
    "safety_on_gpu": (False, True),
    "post_processing": ("off", "auto", "on"),
    "performance": ("normal", "moderate", "high"),
    "enable_vram_budget": (False, True),
    "exit_on_unhandled_faults": (False, True),
    "sampling_lease": ("off", "default_slots", "single_slot"),
}

type _FeatureRow = dict[str, object]
type _FeaturePair = tuple[str, object, str, object]


def _row_pairs(row: _FeatureRow) -> set[_FeaturePair]:
    """Return every named two-axis projection in one feature row."""
    return {(first, row[first], second, row[second]) for first, second in itertools.combinations(_FEATURE_AXES, 2)}


def _feature_covering_array() -> tuple[_FeatureRow, ...]:
    """Build a deterministic greedy pairwise array over valid runtime feature choices."""
    candidates = [
        dict(zip(_FEATURE_AXES, values, strict=True)) for values in itertools.product(*_FEATURE_AXES.values())
    ]
    uncovered = set().union(*(_row_pairs(row) for row in candidates))
    selected: list[_FeatureRow] = []
    while uncovered:
        best = max(candidates, key=lambda row: (len(_row_pairs(row) & uncovered), repr(row)))
        covered = _row_pairs(best) & uncovered
        if not covered:
            raise AssertionError(f"feature vocabulary contains unreachable pairs: {sorted(uncovered, key=repr)}")
        selected.append(best)
        uncovered -= covered
        candidates.remove(best)
    return tuple(selected)


_FEATURE_COVERING_ARRAY = _feature_covering_array()


@pytest.mark.parametrize(
    (
        "max_threads",
        "requested_queue_size",
        "num_models_to_load",
        "serves_image_generation",
        "lease_enabled",
        "lease_slots",
        "tail_overlap",
    ),
    itertools.product(
        _THREAD_BOUNDARIES,
        _QUEUE_BOUNDARIES,
        _MODEL_COUNTS,
        _SERVES_IMAGE_GENERATION,
        _LEASE_ENABLED,
        _LEASE_SLOTS,
        _TAIL_OVERLAP,
    ),
)
def test_every_boundary_configuration_resolves_to_a_consistent_process_topology(
    max_threads: int,
    requested_queue_size: int,
    num_models_to_load: int,
    serves_image_generation: bool,
    lease_enabled: bool,
    lease_slots: int | None,
    tail_overlap: bool,
) -> None:
    """Every semantic boundary combination preserves the worker's lane and semaphore invariants."""
    queue_size = cap_queue_size(
        max_threads=max_threads,
        queue_size=requested_queue_size,
        log=False,
    )
    resolved = resolve_card_concurrency(
        max_threads=max_threads,
        queue_size=queue_size,
        num_models_to_load=num_models_to_load,
        gpu_sampling_lease_enabled=lease_enabled,
        gpu_sampling_lease_slots=lease_slots,
        gpu_sampling_lease_tail_overlap=tail_overlap,
        max_threads_ceiling=max_threads,
        serves_image_generation=serves_image_generation,
    )

    expected_processes = max_threads + queue_size
    if not serves_image_generation or (num_models_to_load == 1 and max_threads == 1):
        expected_processes = 1

    assert resolved.target_process_count == expected_processes
    assert resolved.max_concurrent_inference == max_threads
    assert resolved.vae_decode_semaphore_size == 1
    assert resolved.gpu_sampling_lease_tail_overlap is (lease_enabled and tail_overlap)

    expected_slots = max_threads if lease_slots is None else lease_slots
    expected_slots = min(max(1, expected_slots), expected_processes)
    assert resolved.gpu_sampling_lease_slots == expected_slots
    if lease_enabled:
        assert resolved.inference_semaphore_size == expected_processes
    else:
        assert resolved.inference_semaphore_size == max_threads


def test_configuration_boundary_vocabulary_covers_each_effective_queue_partition() -> None:
    """The boundary set reaches zero, ordinary, capped, and uncapped maximum queue behavior."""
    effective = {
        (threads, requested): cap_queue_size(max_threads=threads, queue_size=requested, log=False)
        for threads in _THREAD_BOUNDARIES
        for requested in _QUEUE_BOUNDARIES
    }

    assert effective[(1, 0)] == 0
    assert effective[(1, 4)] == 4
    assert effective[(2, 4)] == 3
    assert effective[(16, 4)] == 3


@pytest.mark.parametrize(
    "row", _FEATURE_COVERING_ARRAY, ids=lambda row: "-".join(str(value) for value in row.values())
)
def test_pairwise_runtime_feature_configuration_is_valid_and_preserved(row: _FeatureRow) -> None:
    """Every pair of recovery, placement, memory, and throughput choices survives configuration validation."""
    post_processing = cast(Literal["auto", "off", "on"], row["post_processing"])
    performance = str(row["performance"])
    sampling_lease = str(row["sampling_lease"])
    max_inference_attempts = cast(int, row["max_inference_attempts"])
    config = reGenBridgeData(
        max_threads=3,
        queue_size=1,
        max_inference_attempts=max_inference_attempts,
        unload_models_from_vram_often=bool(row["unload_models_from_vram_often"]),
        safety_on_gpu=bool(row["safety_on_gpu"]),
        allow_post_processing=post_processing == "auto",
        dedicated_post_processing=post_processing,
        high_performance_mode=performance == "high",
        moderate_performance_mode=performance == "moderate",
        enable_vram_budget=bool(row["enable_vram_budget"]),
        exit_on_unhandled_faults=bool(row["exit_on_unhandled_faults"]),
        gpu_sampling_lease_enabled=sampling_lease != "off",
        gpu_sampling_lease_slots=1 if sampling_lease == "single_slot" else None,
    )

    assert config.max_inference_attempts == row["max_inference_attempts"]
    assert config.unload_models_from_vram_often is row["unload_models_from_vram_often"]
    assert config.safety_on_gpu is row["safety_on_gpu"]
    assert config.enable_vram_budget is row["enable_vram_budget"]
    assert config.exit_on_unhandled_faults is row["exit_on_unhandled_faults"]
    assert config.post_processing_lane_enabled is (post_processing != "off")
    assert config.high_performance_mode is (performance == "high")
    assert config.moderate_performance_mode is (performance == "moderate")
    assert config.gpu_sampling_lease_enabled is (sampling_lease != "off")
    assert config.gpu_sampling_lease_slots == (1 if sampling_lease == "single_slot" else None)


def test_runtime_feature_array_covers_every_valid_pair() -> None:
    """The compact validation corpus is a covering array, not an unmeasured collection of examples."""
    expected = {
        (first, first_value, second, second_value)
        for first, second in itertools.combinations(_FEATURE_AXES, 2)
        for first_value in _FEATURE_AXES[first]
        for second_value in _FEATURE_AXES[second]
    }
    actual = set().union(*(_row_pairs(row) for row in _FEATURE_COVERING_ARRAY))

    assert actual == expected
