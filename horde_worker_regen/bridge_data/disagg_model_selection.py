"""Pure selection logic for the ``disagg_optimized N`` model-load meta command.

The command asks the worker to serve the ``N`` image models that benefit most from the disaggregated stage
pipeline. Two signals order the candidate pool:

1. Shared-VAE cluster membership. Disaggregated jobs decode on a shared VAE lane, so models that carry
   byte-identical VAE weights let one materialised VAE serve many models from a warm in-RAM copy. Models in a
   larger shared-VAE cluster are preferred, largest cluster first. Cluster membership is derived by
   :func:`horde_model_reference.canonical_components.derive_canonical_registry` over each model's component
   content hashes.
2. Popularity, the same usage-ranking signal the SDK's ``top N`` command uses. The caller supplies the
   popularity order (the SDK resolves it from the horde image-stats API); within a cluster tier, a more
   popular model is preferred.

Component hashes come from two independent sources merged per model, with the local source winning on
conflict: the reference record's own hashes (:attr:`DownloadRecord.content_hash` and
:attr:`GenericModelRecordConfig.embedded_component_hashes`) and, when present, a checkpoint's local
component-identity sidecar. Neither source is required: with no hash data at all the ranking degrades to pure
popularity order (identical to ``top N`` filtered to the disaggregation-eligible pool).

This module is torch-free and holds no dependency on the config object: it takes reference records, an
optional local-hash lookup, and ``N``, and returns the ordered names. The ``load_config`` glue resolves the
popularity order and reads the local sidecars, then calls in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from horde_model_reference.canonical_components import derive_canonical_registry
from horde_model_reference.component_hash import ComponentKind, component_kind_for_purpose
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord, ImageGenerationModelRecord

__all__ = [
    "DisaggModelSelection",
    "is_disagg_optimized_candidate",
    "select_disagg_optimized_models",
]


# Mirrors ``process_manager._DISAGGREGATION_V1_BASELINES`` (the SD1.5/SDXL families the disaggregated sample
# path is v1-validated for). Duplicated rather than imported so the torch-free config-load path does not drag
# the process manager in for a two-element constant; keep the two in step.
_DISAGG_OPTIMIZED_BASELINES: frozenset[str] = frozenset(
    {
        str(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
        str(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl),
    },
)

_MIN_SHARED_MODELS = 2
"""Cluster-promotion threshold: a VAE shared by fewer than two candidates is not a cluster (size 1)."""

_UNCLUSTERED_SIZE = 1
"""The shared-VAE cluster size charged to a candidate whose VAE no other candidate shares (itself only)."""


@dataclass(frozen=True)
class DisaggModelSelection:
    """The outcome of a ``disagg_optimized N`` resolution, carrying the rationale for a single log line."""

    selected: list[str]
    """The chosen model names in ranked order (at most ``N``)."""
    candidate_count: int
    """How many models passed the static disaggregation-eligibility and known-downloadable filters."""
    hash_data_available: bool
    """Whether any candidate contributed component-hash data (record or local); False means popularity-only."""
    cluster_sizes: dict[str, int] = field(default_factory=dict)
    """Shared-VAE cluster size for each selected model (1 when its VAE is not shared), for the rationale log."""


def is_disagg_optimized_candidate(record: ImageGenerationModelRecord) -> bool:
    """Whether ``record`` is in the static disaggregation-optimized candidate pool.

    The static half of the disaggregation-eligibility predicate plus a downloadability check: an SD1.5 or
    SDXL-family baseline, not an inpainting-variant checkpoint (whose UNet takes the masked-image input the
    staged txt2img sample graph does not supply), and at least one declared download file so the worker can
    obtain it. Deliberately excludes the per-job dynamic conditions (source processing, control type,
    transparency, monolithic re-route) and all process-liveness state, which are not knowable at config time.
    """
    baseline = record.baseline
    if baseline is None or str(baseline) not in _DISAGG_OPTIMIZED_BASELINES:
        return False
    if bool(record.inpainting):
        return False
    return record.download_count > 0


def _record_component_hashes(record: GenericModelRecord) -> dict[ComponentKind, str]:
    """Return the component content hashes ``record`` itself declares, keyed by kind.

    Split-file components carry their hash on the download entry; monolithic checkpoints carry theirs in
    ``embedded_component_hashes``. The embedded map wins over a same-kind download entry (it is the
    checkpoint's own embedded identity).
    """
    hashes: dict[ComponentKind, str] = {}
    config = record.config
    for download in config.download:
        kind = component_kind_for_purpose(download.file_purpose)
        if kind is not None and download.content_hash is not None:
            hashes[kind] = download.content_hash
    for purpose, content_hash in (config.embedded_component_hashes or {}).items():
        kind = component_kind_for_purpose(purpose)
        if kind is not None:
            hashes[kind] = content_hash
    return hashes


def _merged_component_hashes(
    record: GenericModelRecord,
    local: Mapping[ComponentKind, str] | None,
) -> dict[ComponentKind, str]:
    """Merge a record's declared hashes with a local sidecar's, letting the local source win per kind."""
    merged = _record_component_hashes(record)
    if local:
        merged.update(local)
    return merged


def _vae_cluster_sizes(
    candidate_records: Mapping[str, ImageGenerationModelRecord],
    local_component_hashes: Mapping[str, Mapping[ComponentKind, str]] | None,
) -> tuple[dict[str, int], bool]:
    """Derive each candidate's shared-VAE cluster size over the merged hashes.

    Feeds :func:`derive_canonical_registry` a per-candidate view whose embedded hashes are the merged
    (record-plus-local) identities, so the grouping authority the worker already ships decides which VAEs are
    shared rather than a second implementation. Returns the per-model VAE cluster size (only models in a
    cluster of two or more appear) and whether any candidate contributed hash data at all.
    """
    synthetic: dict[str, ImageGenerationModelRecord] = {}
    hash_data_available = False
    for name, record in candidate_records.items():
        merged = _merged_component_hashes(record, (local_component_hashes or {}).get(name))
        if merged:
            hash_data_available = True
        embedded_map = {kind.value: content_hash for kind, content_hash in merged.items()}
        new_config = record.config.model_copy(
            update={"download": [], "embedded_component_hashes": embedded_map or None},
        )
        synthetic[name] = record.model_copy(update={"config": new_config})

    registry = derive_canonical_registry(synthetic, min_shared_models=_MIN_SHARED_MODELS)

    sizes: dict[str, int] = {}
    for component in registry.for_kind(ComponentKind.VAE):
        for source in component.sources:
            sizes[source.model_name] = max(sizes.get(source.model_name, 0), component.shared_by_model_count)
    return sizes, hash_data_available


def select_disagg_optimized_models(
    records: Mapping[str, ImageGenerationModelRecord],
    n: int,
    *,
    popularity_order: Sequence[str] = (),
    local_component_hashes: Mapping[str, Mapping[ComponentKind, str]] | None = None,
) -> DisaggModelSelection:
    """Select the ``N`` disaggregation-optimized models from ``records``.

    The candidate pool is every record passing :func:`is_disagg_optimized_candidate`. Candidates are ranked
    by shared-VAE cluster size (largest first), then by popularity (their position in ``popularity_order``,
    unranked models last), then by name for a deterministic tie-break, and the first ``N`` are returned.

    Args:
        records: The image-generation reference, keyed by model name.
        n: How many models to select. A value of zero or below selects nothing.
        popularity_order: Model names ordered most-popular first (the SDK ``top N`` usage ranking). Names
            absent from it are treated as least popular. When empty, ranking is by cluster size then name.
        local_component_hashes: Optional per-model local component hashes (from on-disk sidecars) that win
            over the record's declared hashes on conflict.

    Returns:
        The ordered selection plus the rationale (candidate count, whether hash data informed the ranking,
        and the shared-VAE cluster size of each selected model).
    """
    if n <= 0:
        return DisaggModelSelection(selected=[], candidate_count=0, hash_data_available=False)

    candidate_records = {name: record for name, record in records.items() if is_disagg_optimized_candidate(record)}
    if not candidate_records:
        return DisaggModelSelection(selected=[], candidate_count=0, hash_data_available=False)

    cluster_sizes, hash_data_available = _vae_cluster_sizes(candidate_records, local_component_hashes)

    popularity_rank = {name: index for index, name in enumerate(popularity_order)}
    unranked = len(popularity_rank)

    ordered = sorted(
        candidate_records,
        key=lambda name: (
            -cluster_sizes.get(name, _UNCLUSTERED_SIZE),
            popularity_rank.get(name, unranked),
            name,
        ),
    )
    selected = ordered[:n]
    return DisaggModelSelection(
        selected=selected,
        candidate_count=len(candidate_records),
        hash_data_available=hash_data_available,
        cluster_sizes={name: cluster_sizes.get(name, _UNCLUSTERED_SIZE) for name in selected},
    )
