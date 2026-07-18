"""Named payload mixes for the disaggregation-optimization measurement gate.

The gate compares two worker configurations under identical work (see
:mod:`horde_worker_regen.benchmark.gate_driver`). These builders mint the identical work: a
:class:`~horde_worker_regen.benchmark.scenarios.Scenario` whose shape is chosen to put the
disaggregated pipeline's between-job reload cost on the critical path, so an A/B run measures a
scheduling change against the exact traffic that stresses it.

Two axes of pressure are expressed:

- **Component churn** (:attr:`DisaggGateMix.CHURN_DETERMINISTIC`,
  :attr:`DisaggGateMix.CHURN_SEEDED_RANDOM`) interleaves distinct base models so a component (text
  encoder, VAE, sampler weights) is reloaded between jobs. The deterministic variant is a fixed,
  strict round-robin job list (no two consecutive jobs share a model, so every job pays a switch);
  it is the A/B work-parity mix, byte-identical for identical inputs. The seeded-random variant is a
  duration-paced stream that lets consecutive same-model runs occur by chance, a softer, more
  production-like churn profile.
- **VAE sharing** (:attr:`DisaggGateMix.CLUSTER_SHARED_VAE`,
  :attr:`DisaggGateMix.CLUSTER_DISTINCT_VAE`) holds the churn shape fixed but swaps the model pool
  for one whose checkpoints either share a VAE (so the VAE lane can retain a resident autoencoder
  across the switch) or each carry a distinct VAE (so the VAE reload is unavoidable). The pair
  isolates the VAE-residency component of the switch cost from the rest.

The default cluster pools are real measured clusters derived from the horde-model-reference canonical
components backfill (the per-checkpoint VAE content-hash grouping) as of 2026-07-17. They are carried
here as operator-supplied *data* (model names, not hashes): the pools are parameters with these
defaults, so a re-derivation only updates the constants. Every model named is disaggregation-class
(SD1.5/SDXL, txt2img/img2img, no controlnet); the builders do not re-derive that eligibility, since
the pools are curated inputs, not a filter over the whole catalog.

The img2img share is expressed through each job's ``source_processing`` field. It shapes the mix and
the pipeline path a job takes; it does not attach a source image, so a real-mode VAE-encode lane is
exercised only when the worker supplies a start latent for a ``source_processing`` job.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum, auto

from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario
from horde_worker_regen.benchmark.soak import _LORA_STORM_SHARED_REFERENCES


class DisaggGateMix(StrEnum):
    """A named payload mix the disagg gate can run through the harness."""

    CHURN_DETERMINISTIC = auto()
    """Fixed, strict round-robin over the model pool: the byte-identical A/B work-parity mix."""
    CHURN_SEEDED_RANDOM = auto()
    """Duration-paced seeded-random churn stream: a softer, production-like reload profile."""
    CLUSTER_SHARED_VAE = auto()
    """Deterministic interleave over a pool whose checkpoints share one VAE (retainable across a switch)."""
    CLUSTER_DISTINCT_VAE = auto()
    """Deterministic interleave over a pool whose checkpoints each carry a distinct VAE (reload unavoidable)."""


SHARED_VAE_MODEL_POOL: tuple[str, ...] = (
    "Nova Anime XL",
    "Nova Furry Pony",
    "CyberRealistic Pony",
    "AlbedoBase XL 3.1",
)
"""SDXL checkpoints that resolve to one shared VAE content hash (hmr canonical-components backfill, 2026-07-17).

Interleaving these lets the VAE lane keep a resident autoencoder across a base-model switch, so an A/B
run isolates what the shared VAE buys against :data:`DISTINCT_VAE_MODEL_POOL`."""

DISTINCT_VAE_MODEL_POOL: tuple[str, ...] = (
    "AlbedoBase XL (SDXL)",
    "Nova Anime XL",
    "Juggernaut XL",
    "White Pony Diffusion 4",
)
"""SDXL checkpoints each from a different VAE cluster (hmr canonical-components backfill, 2026-07-17).

Every base-model switch also switches the VAE, so the VAE reload cost is unavoidable: the control pool
against which :data:`SHARED_VAE_MODEL_POOL` measures the shared-VAE saving."""

DEFAULT_CHURN_MODEL_POOL: tuple[str, ...] = (
    "AlbedoBase XL (SDXL)",
    "Juggernaut XL",
    "CyberRealistic Pony",
    "Deliberate",
)
"""A mixed SD1.5/SDXL disagg-class pool for the churn mixes when the operator supplies none.

Spanning both baselines makes the interleave pay a genuine cross-baseline component reload on most
switches rather than a same-family swap the scheduler can partly amortise."""

DEFAULT_GATE_LORA_POOL: tuple[str, ...] = _LORA_STORM_SHARED_REFERENCES
"""The LoRA references the gate mixes draw from (reusing the soak's small shared-reference pool).

Synthetic names, meaningful only in a fake-mode run; a real-mode gate must pass a resolvable pool, the
same contract the soak mixes hold."""

DEFAULT_JOBS_PER_MINUTE: float = 12.0
"""Sizing estimate for the fixed-list mixes: how many jobs a rung of ``rung_seconds`` is expected to run.

Only sets the length of a deterministic job list (``rung_seconds/60 * this``); the actual completion rate
is measured, not assumed. The seeded-random mix ignores it (its stream is duration-paced by the harness)."""

_GATE_LATENT_WIDTH = 1024
_GATE_LATENT_HEIGHT = 1024
_GATE_STEPS = 30
_SEEDED_RANDOM_WEIGHT_SCALE = 1000
"""Resolution the per-model feature-bucket weights are scaled to before rounding to integer draw weights."""


def _default_pool_for(mix: DisaggGateMix) -> tuple[str, ...]:
    """Return the built-in model pool for a mix when the caller supplies none."""
    if mix is DisaggGateMix.CLUSTER_SHARED_VAE:
        return SHARED_VAE_MODEL_POOL
    if mix is DisaggGateMix.CLUSTER_DISTINCT_VAE:
        return DISTINCT_VAE_MODEL_POOL
    return DEFAULT_CHURN_MODEL_POOL


def _share_selected(index: int, total: int, fraction: float) -> bool:
    """Whether job ``index`` of ``total`` carries a feature present on ``fraction`` of the stream.

    Uses the floor-crossing rule, which selects exactly ``floor(total * fraction)`` indices spread as
    evenly as possible, so a fixed list reproduces the target share deterministically without an RNG.
    """
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    return math.floor((index + 1) * fraction) > math.floor(index * fraction)


def _job_seed(seed: int, index: int) -> str:
    """A per-job pinned generation seed derived from the mix seed and job index (byte-stable)."""
    return str(seed * 1_000_003 + index)


def _job_prompt(mix: DisaggGateMix, index: int) -> str:
    """A per-job pinned prompt so a job list is fully reproducible for A/B parity."""
    return f"disagg gate {mix.value} job {index}"


def _build_deterministic_interleave(
    mix: DisaggGateMix,
    *,
    model_pool: Sequence[str],
    lora_pool: Sequence[str],
    lora_pool_is_version: bool,
    rung_seconds: float,
    seed: int,
    img2img_fraction: float,
    lora_fraction: float,
    jobs_per_minute_estimate: float,
    width: int,
    height: int,
    steps: int,
) -> list[CannedImageJobSpec]:
    """Build the fixed, strict round-robin job list shared by the deterministic mixes.

    Model assignment is ``pool[index % len(pool)]`` so no two consecutive jobs share a model (the pool
    has at least two entries): every job lands on a component reload, which is the point. The img2img and
    LoRA shares are placed by :func:`_share_selected` from opposite ends of the list so they decorrelate
    rather than always co-occurring. Every knob is a pure function of ``(seed, index)``, so identical
    inputs yield an equal spec list (generation ids aside, which are minted at expansion time).
    """
    job_count = max(len(model_pool), round(rung_seconds / 60.0 * jobs_per_minute_estimate))
    specs: list[CannedImageJobSpec] = []
    for index in range(job_count):
        model = model_pool[index % len(model_pool)]
        is_img2img = _share_selected(index, job_count, img2img_fraction)
        # LoRA share is placed from the tail so it does not land on the same jobs as the img2img share.
        has_lora = _share_selected(job_count - 1 - index, job_count, lora_fraction)
        specs.append(
            CannedImageJobSpec(
                model=model,
                width=width,
                height=height,
                steps=steps,
                seed=_job_seed(seed, index),
                prompt=_job_prompt(mix, index),
                source_processing="img2img" if is_img2img else None,
                lora_names=[lora_pool[index % len(lora_pool)]] if has_lora else [],
                lora_is_version=lora_pool_is_version,
                count=1,
            ),
        )
    return specs


def _build_seeded_random_templates(
    mix: DisaggGateMix,
    *,
    model_pool: Sequence[str],
    lora_pool: Sequence[str],
    lora_pool_is_version: bool,
    seed: int,
    img2img_fraction: float,
    lora_fraction: float,
    width: int,
    height: int,
    steps: int,
) -> list[CannedImageJobSpec]:
    """Build the weighted templates the seeded-random churn stream generates from.

    Each model gets up to four feature buckets (plain, img2img-only, LoRA-only, both) whose weights are
    the product of the marginal shares, so across the pool the generated stream reproduces both the
    img2img and LoRA target fractions independently. ``count`` carries the draw weight (the soak path
    treats it as a relative weight, not a job count). The LoRA reference each bucket carries is rotated by
    ``seed`` so different seeds give different but reproducible template sets.
    """
    bucket_shares = (
        (False, False, (1.0 - img2img_fraction) * (1.0 - lora_fraction)),
        (True, False, img2img_fraction * (1.0 - lora_fraction)),
        (False, True, (1.0 - img2img_fraction) * lora_fraction),
        (True, True, img2img_fraction * lora_fraction),
    )
    specs: list[CannedImageJobSpec] = []
    for model_index, model in enumerate(model_pool):
        lora_ref = lora_pool[(seed + model_index) % len(lora_pool)]
        for is_img2img, has_lora, share in bucket_shares:
            weight = round(share * _SEEDED_RANDOM_WEIGHT_SCALE)
            if weight <= 0:
                continue
            specs.append(
                CannedImageJobSpec(
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    prompt=_job_prompt(mix, model_index),
                    source_processing="img2img" if is_img2img else None,
                    lora_names=[lora_ref] if has_lora else [],
                    lora_is_version=lora_pool_is_version,
                    count=weight,
                ),
            )
    return specs


def build_disagg_gate_scenario(
    mix: DisaggGateMix,
    *,
    rung_seconds: float,
    seed: int,
    model_pool: Sequence[str] | None = None,
    img2img_fraction: float = 0.2,
    lora_fraction: float = 0.35,
    jobs_per_minute_estimate: float = DEFAULT_JOBS_PER_MINUTE,
    lora_pool: Sequence[str] = DEFAULT_GATE_LORA_POOL,
    lora_pool_is_version: bool = False,
    width: int = _GATE_LATENT_WIDTH,
    height: int = _GATE_LATENT_HEIGHT,
    steps: int = _GATE_STEPS,
) -> Scenario:
    """Build the harness scenario for one gate mix at one rung.

    The deterministic and cluster mixes expand to a fixed round-robin job list (released all at once and
    metered by the worker's queue), so an A/B pair runs byte-identical work. The seeded-random mix returns
    a soak scenario whose ``soak_seconds`` is the rung, streaming generated jobs for the whole rung.

    Args:
        mix: Which named mix to build.
        rung_seconds: The rung duration; sizes the fixed list and sets the soak duration.
        seed: Seeds the deterministic per-job pins (fixed mixes) and the template construction (soak mix).
        model_pool: The disagg-class base models to interleave; defaults to the mix's built-in pool.
        img2img_fraction: Share of the stream carrying ``source_processing=img2img``.
        lora_fraction: Share of the stream carrying a LoRA reference.
        jobs_per_minute_estimate: Expected completion rate used only to size the fixed list.
        lora_pool: The LoRA references LoRA-bearing jobs draw from (rotated by index/seed).
        lora_pool_is_version: Whether the pool entries are CivitAI version ids (exact, cache-resolvable)
            rather than model names; a real-mode run should pass version ids of LoRAs already on disk.
        width: Per-job latent width.
        height: Per-job latent height.
        steps: Per-job sampling steps.

    Returns:
        A scenario ready for :meth:`HarnessConfig.from_scenario`.

    Raises:
        ValueError: If the resolved model pool has fewer than two distinct models (a strict no-consecutive
            interleave is impossible), or a share fraction is outside ``[0, 1]``.
    """
    pool = list(model_pool) if model_pool is not None else list(_default_pool_for(mix))
    if len(pool) < 2:
        raise ValueError("a disagg gate mix needs at least two models so consecutive jobs never share one")
    for name, fraction in (("img2img_fraction", img2img_fraction), ("lora_fraction", lora_fraction)):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{name} must be within [0, 1] (got {fraction})")

    scenario_name = f"disagg_gate_{mix.value}"
    if mix is DisaggGateMix.CHURN_SEEDED_RANDOM:
        templates = _build_seeded_random_templates(
            mix,
            model_pool=pool,
            lora_pool=lora_pool,
            lora_pool_is_version=lora_pool_is_version,
            seed=seed,
            img2img_fraction=img2img_fraction,
            lora_fraction=lora_fraction,
            width=width,
            height=height,
            steps=steps,
        )
        return Scenario(name=scenario_name, image_jobs=templates, soak_seconds=rung_seconds)

    specs = _build_deterministic_interleave(
        mix,
        model_pool=pool,
        lora_pool=lora_pool,
        lora_pool_is_version=lora_pool_is_version,
        rung_seconds=rung_seconds,
        seed=seed,
        img2img_fraction=img2img_fraction,
        lora_fraction=lora_fraction,
        jobs_per_minute_estimate=jobs_per_minute_estimate,
        width=width,
        height=height,
        steps=steps,
    )
    return Scenario(name=scenario_name, image_jobs=specs)


__all__ = [
    "DEFAULT_CHURN_MODEL_POOL",
    "DEFAULT_GATE_LORA_POOL",
    "DEFAULT_JOBS_PER_MINUTE",
    "DISTINCT_VAE_MODEL_POOL",
    "SHARED_VAE_MODEL_POOL",
    "DisaggGateMix",
    "build_disagg_gate_scenario",
]
