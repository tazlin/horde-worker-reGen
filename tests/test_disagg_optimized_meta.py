"""End-to-end resolution of the ``disagg_optimized N`` meta command through ``BridgeDataLoader``.

Drives ``_resolve_meta_instructions`` with the reference records, popularity order, and local-hash lookup
stubbed, so the worker-local command is exercised exactly as the config load runs it: the literal is stripped
from ``image_models_to_load`` and replaced with the ranked concrete names, coexisting literals are preserved,
and the largest N wins when more than one command is present.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.bridge_data import load_config
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader


def _record(name: str, *, vae_hash: str | None = None) -> ImageGenerationModelRecord:
    """A downloadable SDXL record, optionally declaring an embedded VAE hash for clustering."""
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        nsfw=False,
        inpainting=False,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=f"{name}.safetensors", file_url=f"http://example/{name}")],
            embedded_component_hashes={"vae": vae_hash} if vae_hash is not None else None,
        ),
    )


def _make_bridge(image_models_to_load: list[str]) -> SimpleNamespace:
    """The subset of the config surface ``_resolve_meta_instructions`` reads (see test_beta_models_filter)."""
    return SimpleNamespace(
        meta_load_instructions=None,
        meta_skip_instructions=None,
        image_models_to_load=image_models_to_load,
        image_models_to_skip=[],
        only_models_on_disk=False,
        load_large_models=False,
    )


def _resolve(
    bridge: SimpleNamespace,
    reference_records: dict[str, ImageGenerationModelRecord],
    *,
    popularity_order: list[str],
    local_hashes: dict | None = None,
) -> list[str]:
    """Run ``_resolve_meta_instructions`` with the reference, popularity, and sidecar lookup stubbed."""
    manager = Mock()
    manager.get_all_model_references.return_value = {}
    with (
        patch.object(load_config, "beta_aware_image_records", return_value=reference_records),
        patch.object(load_config, "_make_image_model_load_resolver", return_value=Mock()),
        patch.object(load_config, "AIHordeAPIManualClient", Mock()),
        patch.object(BridgeDataLoader, "_image_popularity_order", return_value=popularity_order),
        patch.object(BridgeDataLoader, "_local_disagg_component_hashes", return_value=local_hashes or {}),
    ):
        return BridgeDataLoader._resolve_meta_instructions(bridge, manager)


def test_disagg_optimized_expands_to_ranked_models() -> None:
    """``disagg_optimized 2`` resolves to the two highest-ranked eligible models and drops its literal."""
    records = {
        "a": _record("a", vae_hash="V"),
        "b": _record("b", vae_hash="V"),
        "c": _record("c", vae_hash="V"),
        "d": _record("d", vae_hash="D"),
    }
    bridge = _make_bridge(["disagg_optimized 2"])

    # d is most popular but unclustered; the shared-VAE trio leads.
    result = _resolve(bridge, records, popularity_order=["d", "a", "b", "c"])

    assert set(result) == {"a", "b"}
    assert "disagg_optimized 2" not in result
    assert "d" not in result


def test_coexisting_literal_is_preserved() -> None:
    """A plain model name listed beside the command survives alongside the resolved set."""
    records = {name: _record(name) for name in ("a", "b", "c", "keep")}
    bridge = _make_bridge(["disagg_optimized 2", "keep"])

    result = _resolve(bridge, records, popularity_order=["a", "b", "c", "keep"])

    assert "keep" in result
    assert "a" in result and "b" in result
    assert "disagg_optimized 2" not in result


def test_largest_n_wins_when_multiple_commands() -> None:
    """Two commands resolve once, to the largest N."""
    records = {name: _record(name) for name in ("a", "b", "c", "d")}
    bridge = _make_bridge(["disagg_optimized 1", "disagg_optimized 3"])

    result = _resolve(bridge, records, popularity_order=["a", "b", "c", "d"])

    assert len(result) == 3
    assert set(result) == {"a", "b", "c"}


def test_degrade_to_popularity_when_no_hash_data() -> None:
    """With no cluster hashes the command resolves to the popularity-ranked eligible top N."""
    records = {name: _record(name) for name in ("a", "b", "c")}
    bridge = _make_bridge(["disagg_optimized 2"])

    result = _resolve(bridge, records, popularity_order=["c", "a", "b"])

    assert set(result) == {"c", "a"}


def test_no_command_is_a_no_op() -> None:
    """Without the command the resolver behaves exactly as before (no disagg expansion)."""
    records = {"a": _record("a")}
    bridge = _make_bridge(["a"])

    result = _resolve(bridge, records, popularity_order=["a"])

    assert result == ["a"]
