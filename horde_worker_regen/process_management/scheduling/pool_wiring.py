"""Wiring glue between operator config and the pure fixed-pool engine and demand ranker.

The fixed-pool engine (:mod:`horde_worker_regen.process_management.scheduling.model_pool`) and the demand
ranker (:mod:`horde_worker_regen.process_management.scheduling.pool_ranker`) are deliberately unaware of the
config object and the performance model. This module is the small, torch-free adapter layer that turns
operator :class:`~horde_worker_regen.bridge_data.data_model.ModelPoolConfig` into the engine's
:class:`~horde_worker_regen.process_management.scheduling.model_pool.PoolParams`, and turns the performance
model into the ``name -> expected speed`` callable the ranker consumes.

Two pieces of policy live here so the pure modules stay policy-free:

- ``seats: 0`` in config resolves to one seat per inference process (``max_inference_processes``).
- Turning a bare model name into an expected earning rate needs a job signature and a price (resolution,
  steps, baseline, kudos pricing). The ranker refuses to pick those; this adapter does, using a canonical
  per-baseline signature (native large-model baselines at 1024x1024 and smaller baselines at 512x512, both at
  the benchmark baseline step count) and a per-baseline canonical job price. The value handed to the ranker
  is kudos per wall second: canonical job price divided by expected sampling seconds plus a fixed per-job
  overhead, so deep queues of cheap fast jobs (whose wall time is dominated by the overhead every job pays)
  rank below well-paying queues this card clears efficiently. The prices are calibrated against the server
  kudos model at the canonical signatures and are deliberately coarse: seat selection needs the ratio
  between baselines, not exact pricing.

Public surface: :func:`build_pool_params` and :func:`build_expected_value_adapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from horde_worker_regen.process_management.scheduling.model_pool import PinnedModel, PoolParams
from horde_worker_regen.process_management.scheduling.performance_model import baseline_signature

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from horde_worker_regen.bridge_data.data_model import ModelPoolConfig
    from horde_worker_regen.process_management.scheduling.performance_model import JobSignature

__all__ = [
    "build_expected_value_adapter",
    "build_pool_params",
]

_SDXL_BASELINE = "stable_diffusion_xl"

_NATIVE_1024_BASELINES = frozenset(
    {
        _SDXL_BASELINE,
        "stable_cascade",
        "flux_1",
        "qwen_image",
        "z_image_turbo",
    },
)
"""Baselines whose benchmark and normal operating signature uses a 1024-pixel native square."""

_NATIVE_1024_SIGNATURE_RESOLUTION = 1024
"""Canonical square resolution for large-model baselines measured by the benchmark at 1024x1024."""

_DEFAULT_SIGNATURE_RESOLUTION = 512
"""Canonical square resolution for smaller or unknown baselines."""


def build_pool_params(
    config: ModelPoolConfig,
    max_inference_processes: int,
    *,
    eligible_models: Collection[str] | None = None,
) -> PoolParams:
    """Build engine parameters, resolving auto seats and filtering pins against worker eligibility.

    A configured ``seats`` of 0 resolves to one seat per inference process; any positive value passes through.
    Manual pins preserve their affinity. When ``eligible_models`` is supplied, a pin outside that set is
    omitted so the engine cannot seat a model the worker's load/skip rules prevent the popper from advertising.
    The ranker gate and the rotation, dwell, and rescue tunables pass through unchanged. Engine-only tunables
    the config does not surface keep their :class:`PoolParams` defaults.

    Args:
        config: The operator's fixed-pool configuration.
        max_inference_processes: The running inference-process count, used to resolve an auto seat count.
        eligible_models: The worker's resolved advertised model set. None preserves every configured pin for
            import-light callers that do not own eligibility.

    Returns:
        The frozen :class:`PoolParams` the engine reconciles against.
    """
    seat_count = config.seats if config.seats > 0 else max_inference_processes
    eligible = frozenset(eligible_models) if eligible_models is not None else None
    pinned = tuple(
        PinnedModel(name=entry.name, affinity=entry.affinity)
        for entry in config.pinned
        if eligible is None or entry.name in eligible
    )
    return PoolParams(
        seat_count=seat_count,
        pinned=pinned,
        ranker_enabled=config.ranker_enabled,
        rotation_minutes=config.rotation_minutes,
        min_dwell_minutes=config.min_dwell_minutes,
        rescue_enabled=config.rescue_enabled,
        rescue_eta_seconds=config.rescue_eta_seconds,
        rescue_window_minutes=config.rescue_window_minutes,
    )


_CANONICAL_JOB_KUDOS_BY_BASELINE: dict[str, float] = {
    "stable_diffusion_1": 7.84,
    "stable_diffusion_2_512": 7.84,
    "stable_diffusion_2_768": 7.84,
    _SDXL_BASELINE: 20.74,
    "stable_cascade": 20.74,
    "flux_1": 35.0,
}
"""Server kudos paid for one canonical-signature job of each baseline.

The SD1.5 and SDXL figures are exact server-parity prices (the analysis scorer at 512x512 and 1024x1024,
benchmark step count); the rest are coarse standings in the same currency. Only the ratios matter to seat
selection, so a missing or drifted price degrades ranking quality, never correctness."""

_DEFAULT_CANONICAL_JOB_KUDOS = 7.84
"""Price assumed for a baseline absent from the table: the cheap-model figure, so an unknown baseline is
never over-favoured."""

_PER_JOB_OVERHEAD_SECONDS = 5.0
"""Fixed wall-clock cost every job pays outside sampling (pop, decode, safety, upload, submit). This is what
makes many small cheap jobs earn less per hour than fewer well-paying ones even at equal sampling throughput,
so it must be charged when converting a job price into an earning rate."""


def build_expected_value_adapter(
    *,
    expected_its: Callable[[JobSignature], float | None],
    baseline_resolver: Callable[[str | None], str | None],
) -> Callable[[str], float | None]:
    """Create the ranker's ``name -> expected earning rate`` callable (kudos per wall second on this card).

    The returned callable resolves a model's baseline, builds the same native-resolution signature used by the
    benchmark (1024x1024 for SDXL and large-model baselines; 512x512 otherwise), and computes the earning rate:
    the baseline's canonical job price divided by expected sampling seconds (signature steps over the performance
    model's expected it/s) plus the fixed per-job overhead. A model whose baseline is unknown (or which the
    performance model has no rate for) returns ``None``, which the ranker treats as neutral rather than excluding
    the model.

    Args:
        expected_its: The performance model's expected-it/s lookup for a job signature.
        baseline_resolver: Maps a model name to its baseline string, or ``None`` when unknown.

    Returns:
        A callable returning a model's expected kudos per wall second, or ``None`` when it cannot be estimated.
    """

    def expected_value(model_name: str) -> float | None:
        baseline = baseline_resolver(model_name)
        if baseline is None:
            return None
        resolution = (
            _NATIVE_1024_SIGNATURE_RESOLUTION if baseline in _NATIVE_1024_BASELINES else _DEFAULT_SIGNATURE_RESOLUTION
        )
        signature = baseline_signature(baseline=baseline, resolution=resolution)
        sampling_its = expected_its(signature)
        if sampling_its is None or sampling_its <= 0:
            return None
        sampling_seconds = signature.total_sampling_iterations / sampling_its
        job_kudos = _CANONICAL_JOB_KUDOS_BY_BASELINE.get(baseline, _DEFAULT_CANONICAL_JOB_KUDOS)
        return job_kudos / (sampling_seconds + _PER_JOB_OVERHEAD_SECONDS)

    return expected_value
