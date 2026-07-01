"""Tests for the config editor's process-count preview helper and optional-int coercion.

The preview mirrors the worker's authoritative process-count computation (reGenBridgeData validators +
resolve_card_concurrency); these tests pin the mirrored rules so the two cannot drift silently.
"""

from __future__ import annotations

from horde_worker_regen.tui.config_form import (
    CONFIG_FIELDS,
    coerce_value,
    describe_process_plan,
    estimate_inference_processes_per_card,
)

_BY_KEY = {field.key: field for field in CONFIG_FIELDS}


def _estimate(**kwargs: object) -> int:
    base = {
        "max_threads": 1,
        "queue_size": 1,
        "load_entries": ["top 5"],
        "serves_image_generation": True,
        "extra_slow_worker": False,
    }
    base.update(kwargs)
    return estimate_inference_processes_per_card(**base)  # type: ignore[arg-type]


def test_single_concrete_model_collapses_to_one() -> None:
    """One concrete model at max_threads 1 runs a single inference process."""
    assert _estimate(max_threads=1, queue_size=0, load_entries=["Deliberate"]) == 1


def test_meta_command_does_not_collapse() -> None:
    """A meta command (top/bottom/all) counts as many models, so the collapse does not apply."""
    assert _estimate(max_threads=1, queue_size=1, load_entries=["top 5"]) == 2
    # queue_size 1 + threads 1 = 2; a single *concrete* model would collapse to 1, a meta command stays 2.
    assert _estimate(max_threads=1, queue_size=1, load_entries=["all sdxl"]) == 2
    assert _estimate(max_threads=1, queue_size=1, load_entries=["Deliberate"]) == 1


def test_queue_capped_when_threads_two_or_more() -> None:
    """queue_size is capped to 3 once max_threads >= 2, matching the worker's cap_queue_size."""
    assert _estimate(max_threads=2, queue_size=4) == 5  # min(4, 3) + 2
    assert _estimate(max_threads=2, queue_size=1) == 3  # 1 + 2, no cap needed


def test_extra_slow_forces_single_process() -> None:
    """extra_slow_worker clamps threads to 1 and queue to 0, so a single process results."""
    assert _estimate(max_threads=4, queue_size=3, extra_slow_worker=True) == 1


def test_alchemist_only_is_one_process() -> None:
    """A worker not serving image generation runs a single inference process regardless of concurrency."""
    assert _estimate(max_threads=4, queue_size=4, serves_image_generation=False) == 1


def test_describe_single_card_mentions_safety_and_scaling() -> None:
    """The single/auto-card preview names the safety process and the per-card scaling caveat."""
    text = describe_process_plan(
        max_threads=2,
        queue_size=1,
        load_entries=["top 5"],
        serves_image_generation=True,
        extra_slow_worker=False,
        device_indices=[],
    )
    assert "safety process" in text
    assert "per card" in text


def test_describe_multi_card_shows_total_and_cap_note() -> None:
    """With pinned cards the preview shows the summed upper bound and the queue-cap interlock note."""
    text = describe_process_plan(
        max_threads=2,
        queue_size=4,
        load_entries=["top 5"],
        serves_image_generation=True,
        extra_slow_worker=False,
        device_indices=[0, 1],
    )
    assert "up to 10" in text  # (min(4,3) + 2) * 2
    assert "queue capped to 3" in text


def test_describe_extra_slow_notes_the_clamp() -> None:
    """The extra-slow clamp is surfaced in the preview so the interaction is visible."""
    text = describe_process_plan(
        max_threads=4,
        queue_size=3,
        load_entries=["top 5"],
        serves_image_generation=True,
        extra_slow_worker=True,
        device_indices=[],
    )
    assert "extra-slow clamps" in text


def test_optional_int_field_accepts_blank_as_none() -> None:
    """An optional numeric field (gpu_sampling_lease_slots) coerces blank/None to None, not an error."""
    field = _BY_KEY["gpu_sampling_lease_slots"]
    assert field.optional is True
    assert coerce_value(field, "") is None
    assert coerce_value(field, "None") is None
    assert coerce_value(field, "2") == 2
    # Its own default must survive coercion (the operator-trap guard).
    assert coerce_value(field, str(field.default())) is None
