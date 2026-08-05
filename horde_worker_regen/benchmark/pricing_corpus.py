"""The ``pricing-corpus`` measurement workload: a cost-attribution corpus over the payload axes.

A kudos model can only price an axis it has seen vary. This module mints a workload whose job list
sweeps every axis that plausibly moves inference cost (steps, resolution, batch, cfg, sampler,
schedule, source processing, post-processing, controlnet, hires fix, LoRA cache state, textual
inversions, and base-model cold loads) while holding the rest of each job fixed against a per-model
anchor, so a fit against the resulting records can attribute cost to one axis at a time.

Three properties make the corpus usable as training data rather than merely as load:

- **Deterministic expansion.** Cells (a fixed payload shape) expand to jobs (cell x replicate) and
  then through a fixed-seed constrained shuffle. The same tier yields a byte-identical definition
  artifact on every call and on every machine, which is what lets two hosts' records be pooled.
- **Ordering that isolates state-dependent costs.** Cold-load, LoRA cache-miss, and post-processing
  queueing costs are properties of a job's *position*, not of its payload, so the order is
  constrained rather than free: see :func:`_order_units`.
- **A definition artifact.** :class:`PricingCorpusDefinition` is emitted alongside the run and is the
  assembler's key for labeling each stats record with the cell (and therefore the axis values) that
  produced it.

The ``census`` tier answers the other half of the question. Where the standard tier moves each axis a
few steps to read a marginal cost off it, the census covers the *vocabulary*: every value of every
categorical axis the kudos feature manifest encodes, swept against a fixed anchor and then sampled
again under jointly varying conditions. Its vocabularies come from the manifest itself rather than
from lists restated here, so its coverage is the encoding contract by construction; any value it
cannot run is named, with its reason, in the definition artifact's :class:`CensusSummary`.

The cell groups are carried here as code constants; the design record naming them (G1..G11 for the
standard tier, C1..C10 for the census) lives with the corpus specification.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario

PricingCorpusTier = Literal["smoke", "standard", "census"]
"""How much of the corpus to run.

``standard`` is the marginal-cost fit set, ``smoke`` a fast shape check, and ``census`` the coverage
tier: every value of every categorical axis the kudos feature manifest encodes, exercised against a
fixed anchor and again under jointly varying conditions."""

SCENARIO_NAME = "pricing-corpus"
SCENARIO_REVISION = "1"
"""Bumped whenever a cell shape, the replicate count, or the ordering changes.

Records only pool with records produced under the same revision, so this travels into every session's
``session_start`` record through :attr:`Scenario.revision`."""

SDXL_A = "AlbedoBase XL (SDXL)"
"""The SDXL anchor model: every SDXL cell varies one axis away from this model's anchor shape."""
SDXL_B = "WAI-NSFW-illustrious-SDXL"
"""The second SDXL model, present so a base-model switch between two same-architecture checkpoints is
measured separately from a switch that also changes architecture."""
SD15_A = "Deliberate"
"""The SD1.5 anchor model."""

_MODEL_TAGS: dict[str, str] = {SDXL_A: "sdxl", SDXL_B: "sdxlb", SD15_A: "sd15"}
"""Short, stable per-model tags used in cell ids (the ids are read by humans and by the assembler)."""

REPLICATES = 3
"""Measurements per cell in the standard tier: enough to reject a single outlier per cell."""

PERMUTATION_SEEDS: tuple[str, ...] = ("pc1-a", "pc1-b")
"""Shuffle seeds for the standard tier's two permutation blocks, run consecutively in one session.

Two permutations with the replicates split across them keep a cell's measurements from sharing one
neighbourhood: a cost that is really an artifact of position averages out instead of loading onto the
cell's axis values."""

PERMUTATION_REPLICATES: dict[str, tuple[int, ...]] = {"pc1-a": (0, 1), "pc1-b": (2,)}
"""Which replicate indices each standard-tier permutation carries."""

PINNED_PROMPTS: tuple[str, str, str] = (
    "a red apple",
    "a weathered lighthouse on a rocky headland at dusk, long shadows, distant fishing boats",
    "an overgrown greenhouse interior at golden hour, cracked glass panes catching the light, ferns and "
    "hanging vines spilling over rusted iron benches, terracotta pots stacked in one corner, a watering "
    "can left on the floor beside a coiled hose, dust motes drifting through the warm air, shot on a wide "
    "lens with soft natural light and gentle depth of field",
)
"""Short, medium, and long prompts rotated by replicate index.

Prompt length is a cost axis of its own (conditioning is tokenized and encoded per job), so a corpus
that pinned a single prompt could not tell a prompt-length cost from a per-job constant. Rotating by
replicate keeps every cell exposed to all three lengths."""

PRICING_CORPUS_LORA_VERSION_IDS: tuple[str, str, str, str, str] = (
    "2028738",
    "2104350",
    "1625769",
    "2364091",
    "2443582",
)
"""CivitAI *version* ids the G8 cells reference: five distinct SDXL-family LoRAs near the 400MB ad-hoc
size cap (~1.9GB aggregate), which is the worst legitimate per-job LoRA load a request can ask for.

The G8 cells measure the cost of a LoRA cache miss against a cache hit, which only means anything when
the reference resolves to a real file. Version ids (not names) so resolution is exact and a pre-seeded
cache is reused. Run preparation must ensure these are cache-absent for the miss measurement to be
real: evict each through the LoRA model manager's ``delete_lora`` (which also drops its reference-db
entry) rather than by touching cache files directly."""

PRICING_CORPUS_TI_NAME = "1063311"
"""Textual-inversion version id for the G9 cell.

A cached embedding is acceptable here: a TI is hundreds of kilobytes, so its cost sits in conditioning
(the apply), not in the fetch the G8 cells exist to isolate."""

_PAIRWISE_SEED = "pc1-pairwise"
"""Fixed seed for the greedy strength-2 covering array, so the categorical rows never move."""

_PAIRWISE_CANDIDATE_ROWS = 24
"""Candidate rows evaluated per emitted row in the greedy covering-array construction (AETG-style).

More candidates buy a slightly smaller array at linear cost; the array is built once per call."""

_COLD_TARGET_SPAN = 0.7
"""Fraction of the permutation over which cold-load cells' target positions are drawn.

Cold cells are placed at the first eligible position at or after their target. Drawing targets from the
leading part of the block keeps a cold cell from landing in the tail, where the remaining pool is
dominated by one model and no preceding model switch may be available."""

_SAME_MODEL_MIN_DISTANCE = 2
"""Desired minimum position distance between two jobs of the same model (outside the stacked cells)."""

_POST_PROCESSING_MIN_DISTANCE = 3
"""Desired minimum position distance between two post-processing jobs outside the queueing-stack cell."""

_SAMPLERS: tuple[str, ...] = (
    "k_euler",
    "k_euler_a",
    "k_dpmpp_2m",
    "k_dpmpp_sde",
    "k_heun",
    "lcm",
    "uni_pc",
    "dpmpp_3m_sde",
)
"""Samplers in the categorical array.

``k_dpm_adaptive`` is excluded by design: it chooses its own step count and would break the steps
marginal, since its records would carry a step count the work never obeyed."""

_SCHEDULERS: tuple[str, ...] = ("karras", "normal", "exponential", "sgm_uniform")

_POST_PROCESSING_CHAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ppnone", ()),
    ("ppesrgan", ("RealESRGAN_x4plus",)),
    ("ppfaces", ("GFPGAN", "CodeFormers")),
    ("ppanimestrip", ("RealESRGAN_x4plus_anime_6B", "strip_background")),
)
"""Post-processing chains in the categorical array, paired with the tag used in cell ids."""

_SOURCE_PROCESSING: tuple[str, ...] = ("txt2img", "img2img", "inpainting")

_IMG2IMG_DENOISE = 0.75
"""Denoising strength for the corpus's img2img cells.

Denoise scales the sampled step count on an img2img job, so leaving it at the factory default would
make an img2img job cost the same as its txt2img twin and hide the source-processing marginal."""

_PP_STACK_CHAIN: tuple[str, ...] = ("RealESRGAN_x4plus", "CodeFormers")
"""The chain carried by the deliberately unspaced queueing-stack cell (G11)."""

_SDXL_SHAPES: tuple[tuple[int, int], ...] = (
    (832, 1216),
    (1216, 832),
    (896, 1152),
    (1152, 896),
    (1024, 1024),
    (1280, 1280),
    (1536, 1024),
)
"""SDXL-family resolutions the corpus draws from, at and around the architecture's native pixel count."""

_SD15_SHAPES: tuple[tuple[int, int], ...] = ((512, 512), (512, 768), (768, 512), (640, 640), (768, 768))
"""SD1.5 resolutions the corpus draws from."""

CENSUS_JOB_BUDGET = 950
"""Jobs the census tier may spend, warmup included.

At the corpus's observed job median this is about four hours of wall time, which is the sitting a
census is expected to fit in. The sweeps take what coverage requires and the conflation block is sized
to whatever remains, so a vocabulary that grows shrinks the conflation block rather than the run."""

CENSUS_SECONDS_PER_JOB = 15.0
"""Observed median seconds per corpus job, used only for the artifact's projected runtime."""

CENSUS_AXES: tuple[str, ...] = (
    "sampler_name",
    "scheduler",
    "control_type",
    "source_processing",
    "post_processing",
    "baseline",
)
"""Manifest features whose vocabularies the census must cover, in coverage-report order."""

_CENSUS_CONFLATION_SEED = "pc1-census-conflation"
"""Fixed seed for the conflation block's draws, so its cells never move between builds."""

_CENSUS_ANCHOR_SAMPLER = "k_euler"
"""Sampler the schedule sweep holds fixed, and the value every non-sampler census cell carries."""

_CENSUS_ANCHOR_SCHEDULER = "karras"
"""Schedule the sampler sweep holds fixed."""

_CENSUS_EXCLUDED_SAMPLERS: dict[str, str] = {
    "k_dpm_adaptive": (
        "chooses its own step count, so its records would carry a step count the work never obeyed and "
        "would corrupt the steps marginal"
    ),
}
"""Sampler vocabulary values the census refuses to run, each with the reason it is absent."""

_CENSUS_EXCLUDED_SAMPLER_SCHEDULE_PAIRS: dict[tuple[str, str], str] = {
    ("dpmpp_3m_sde", "normal"): (
        "the solver diverges to colour noise on this schedule, and hordelib substitutes its "
        "schedule-sensitive fallback, so the record's scheduler would not name the schedule that ran"
    ),
}
"""Sampler/schedule pairs the census refuses to run.

Mirrors hordelib's ``SCHEDULE_SENSITIVE_SAMPLERS`` x ``DIVERGENT_SCHEDULES`` coercion. Carried here as
a named table rather than read from ``hordelib.pipeline.constants`` because the orchestrator process
must not import the inference stack; the pair is emitted into the definition artifact so its absence
is a visible decision."""

_CENSUS_MODEL_BASELINES: dict[str, str] = {
    SDXL_A: "stable_diffusion_xl",
    SDXL_B: "stable_diffusion_xl",
    SD15_A: "stable_diffusion_1",
}
"""The manifest ``baseline`` value each corpus model encodes as."""

_CENSUS_NON_UPSCALING_POST_PROCESSORS: frozenset[str] = frozenset({"CodeFormers", "GFPGAN", "strip_background"})
"""Post-processors that leave the image size alone; every other vocabulary entry enlarges it.

A chain may carry at most one enlarging processor: two 4x upscalers compound to a 16x output whose
activation peak is a VRAM failure rather than a measurement."""

_CENSUS_CONFLATION_BATCHES: tuple[int, ...] = (1, 1, 1, 2, 4)
"""Batch sizes the conflation block draws from, weighted by repetition toward the common single job."""

_CENSUS_SINGLE_IMAGE_PIXELS = 1_100_000
"""Pixel count above which a conflation cell is held to one image, keeping its activation peak servable."""

_CENSUS_MODEL_WEIGHTS: tuple[tuple[str, float], ...] = ((SDXL_A, 0.45), (SD15_A, 0.45), (SDXL_B, 0.10))
"""Model draw weights for the conflation block: the two anchors carry it, the second SDXL adds switches."""

_CENSUS_CONTROL_SHARE = 0.25
"""Fraction of the conflation block's SD1.5 cells that also carry a control type.

Controlnet weights exist for the SD1.5 baseline only, so a control type can only conflate with the
other axes on that model."""


class PricingCorpusError(RuntimeError):
    """Raised when a pricing corpus cannot be built as specified."""


class PricingCorpusOrderingError(PricingCorpusError):
    """Raised when the ordering constraints cannot be satisfied, naming the constraint that failed."""


class CorpusCell(BaseModel):
    """One measurement cell: a fixed payload shape, replicated to produce jobs.

    Every field except :attr:`cell_id`, :attr:`group`, and the ordering flags is an axis value the
    assembler reads back when it labels the records this cell produced.
    """

    cell_id: str
    """Stable, human-readable identity (e.g. ``g3.batch4.sdxl``); the assembler's join key."""
    group: str
    """The cell group this cell belongs to (``g1``..``g11``, or ``warmup``)."""
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    n_iter: int
    sampler_name: str
    scheduler: str
    source_processing: str
    denoising_strength: float | None = None
    hires_fix: bool = False
    post_processing: list[str] = Field(default_factory=list)
    control_type: str | None = None
    lora_version_ids: list[str] = Field(default_factory=list)
    ti_names: list[str] = Field(default_factory=list)
    lora_role: Literal["miss", "hit"] | None = None
    """Whether this cell's LoRA references are measured on their first use (``miss``) or a later one."""
    replicates: int = REPLICATES
    requires_model_switch: bool = False
    """Whether the ordering must put a job of a different model immediately before every job of this cell."""
    keeps_replicates_adjacent: bool = False
    """Whether this cell's replicates must occupy consecutive positions (the queueing-stack cell).

    Such a cell is also exempt from the post-processing spacing rule: the point of the cell is the
    queueing its own back-to-back jobs cause."""


class PricingCorpusJob(BaseModel):
    """One ordered job in the corpus, resolved to the cell and replicate it measures."""

    position: int
    """Zero-based position in the scenario's job list, including the warmup block."""
    cell_id: str
    group: str
    permutation: str
    """``warmup`` or the shuffle seed of the permutation block this job belongs to."""
    replicate: int
    seed: str
    prompt_index: int
    model: str


class CorpusExclusion(BaseModel):
    """One vocabulary value, or one combination of them, the corpus deliberately does not run.

    A census claims to cover an encoding contract, so a value it skips has to be named: an unnamed
    absence is indistinguishable from an oversight, and a fit would silently price the missing value
    from whatever the manifest collapses it onto.
    """

    kind: Literal["value", "pair"]
    """Whether a single value is excluded, or only the combination of two."""
    axis: str
    """The manifest feature the exclusion applies to; ``a+b`` for a pair spanning two features."""
    value: str
    """The excluded value; ``a+b`` for a pair, in the same order as :attr:`axis`."""
    reason: str
    """Why the value cannot be exercised, in terms a reader of the artifact can act on."""


class CensusSummary(BaseModel):
    """The census tier's coverage claim: what it exercised, what it skipped, and what it costs."""

    manifest_version: str
    """Revision of the kudos feature manifest the vocabularies were derived from."""
    job_budget: int
    projected_job_count: int
    projected_runtime_seconds: float
    seconds_per_job: float
    sweep_cell_count: int
    """Cells in the main-effects sweeps, where one axis moves against a fixed anchor."""
    conflation_cell_count: int
    """Cells in the conflation block, where the axes vary jointly."""
    exclusions: list[CorpusExclusion]
    coverage: dict[str, dict[str, int]]
    """Axis to vocabulary value to how many measured (non-warmup) jobs carry it."""


class PricingCorpusDefinition(BaseModel):
    """The machine-readable description of one built corpus: what ran, in what order, measuring what.

    Emitted next to the session's stats stream so an assembler can pair records to cells by position
    within the session and by payload-field match, and so a later reader can tell which corpus
    definition a set of records came from.
    """

    scenario_name: str
    scenario_revision: str
    tier: PricingCorpusTier
    warmup_job_count: int
    """Leading jobs the assembler drops: they measure cold caches, not their cells' axis values."""
    shuffle_seeds: list[str]
    prompts: list[str]
    cells: list[CorpusCell]
    jobs: list[PricingCorpusJob]
    same_model_adjacencies: int
    """Adjacent same-model job pairs the mix made unavoidable (see :func:`_audit_order`)."""
    post_processing_proximities: int
    """Post-processing job pairs closer than the spacing rule wanted, likewise unavoidable."""
    census: CensusSummary | None = None
    """The coverage claim, on the census tier only.

    Absent (and omitted from the rendered artifact) for the other tiers, whose bytes predate this field
    and must keep comparing equal across the revision."""


@dataclass(frozen=True)
class _PlacementUnit:
    """One atom of the ordering: a cell's replicate, or its whole adjacent replicate block."""

    cell: CorpusCell
    replicates: tuple[int, ...]
    permutation: str

    @property
    def size(self) -> int:
        """How many jobs this unit places."""
        return len(self.replicates)

    @property
    def spaced_post_processing(self) -> bool:
        """Whether this unit's post-processing is subject to the spacing rule."""
        return bool(self.cell.post_processing) and not self.cell.keeps_replicates_adjacent


@dataclass
class _OrderState:
    """Mutable ordering state carried across a session's permutation blocks.

    The blocks run consecutively in one worker session, so the model resident at the end of one block
    is the model resident at the start of the next, and a LoRA fetched in one block is cached for the
    next. Carrying the state makes the constraints hold across the block boundary too.
    """

    last_model: str | None = None
    last_post_processing_position: int = -(10**6)
    """Position of the last spacing-governed post-processing job; far negative before the first one."""
    position: int = 0
    seen_lora_ids: set[str] = field(default_factory=set)


def _variant(anchor: CorpusCell, *, cell_id: str, group: str, **updates: object) -> CorpusCell:
    """Derive a cell from an anchor by overriding the axes this cell varies."""
    return anchor.model_copy(update={"cell_id": cell_id, "group": group, **updates})


ANCHOR_SDXL = CorpusCell(
    cell_id="anchor.sdxl",
    group="anchor",
    model=SDXL_A,
    width=1024,
    height=1024,
    steps=30,
    cfg_scale=7.5,
    n_iter=1,
    sampler_name="k_euler",
    scheduler="karras",
    source_processing="txt2img",
)
"""The SDXL anchor shape (A1): every SDXL cell is this with exactly one axis moved."""

ANCHOR_SD15 = _variant(
    ANCHOR_SDXL,
    cell_id="anchor.sd15",
    group="anchor",
    model=SD15_A,
    width=512,
    height=512,
)
"""The SD1.5 anchor shape (A2)."""

ANCHOR_SDXL_B = _variant(ANCHOR_SDXL, cell_id="anchor.sdxlb", group="anchor", model=SDXL_B)
"""The second SDXL model at the SDXL anchor shape (used by the warmup and cold-load cells)."""

_ANCHORS: tuple[CorpusCell, ...] = (ANCHOR_SDXL, ANCHOR_SDXL_B, ANCHOR_SD15)
"""Anchors in warmup order: each model is loaded once, in a fixed order."""


def _tag(model: str) -> str:
    """Return the cell-id tag for a model."""
    return _MODEL_TAGS[model]


def _cfg_tag(cfg_scale: float) -> str:
    """Render a cfg value for a cell id (``7.5`` -> ``cfg7p5``), keeping ids free of a second separator."""
    return f"cfg{cfg_scale:g}".replace(".", "p")


def _steps_cells() -> list[CorpusCell]:
    """G1: the steps sweep from both anchors (the dominant cost axis)."""
    return [
        _variant(anchor, cell_id=f"g1.steps{steps}.{_tag(anchor.model)}", group="g1", steps=steps)
        for anchor in (ANCHOR_SDXL, ANCHOR_SD15)
        for steps in (8, 15, 22, 30, 40, 50)
    ]


def _resolution_cells() -> list[CorpusCell]:
    """G2: the resolution sweep, including the aspect ratios that share a pixel count."""
    sdxl_shapes = _SDXL_SHAPES
    sd15_shapes = _SD15_SHAPES
    return [
        _variant(
            anchor,
            cell_id=f"g2.res{width}x{height}.{_tag(anchor.model)}",
            group="g2",
            width=width,
            height=height,
        )
        for anchor, shapes in ((ANCHOR_SDXL, sdxl_shapes), (ANCHOR_SD15, sd15_shapes))
        for width, height in shapes
    ]


def _batch_cells() -> list[CorpusCell]:
    """G3: the batch sweep, which prices the per-image saving a batched job buys over N single jobs.

    SDXL at batch 8 does not fit a 16GB card at the anchor resolution, so that one cell drops to
    832x1216 and records the substitution in its id: a cell that cannot run measures nothing, and a
    silent resolution change would be attributed to the batch axis.
    """
    cells: list[CorpusCell] = []
    for anchor in (ANCHOR_SDXL, ANCHOR_SD15):
        tag = _tag(anchor.model)
        for n_iter in (1, 2, 4, 8):
            if anchor.model == SDXL_A and n_iter == 8:
                cells.append(
                    _variant(
                        anchor,
                        cell_id=f"g3.batch8.{tag}.832x1216",
                        group="g3",
                        n_iter=8,
                        width=832,
                        height=1216,
                    ),
                )
                continue
            cells.append(_variant(anchor, cell_id=f"g3.batch{n_iter}.{tag}", group="g3", n_iter=n_iter))
    return cells


def _cfg_cells() -> list[CorpusCell]:
    """G4: the cfg sweep, which separates a real cfg cost from a per-step constant."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"g4.{_cfg_tag(cfg_scale)}.{_tag(SDXL_A)}",
            group="g4",
            cfg_scale=cfg_scale,
        )
        for cfg_scale in (4.0, 7.5, 12.0)
    ]


def _covering_array(factor_sizes: tuple[int, ...], *, seed: str) -> list[tuple[int, ...]]:
    """Build a greedy strength-2 covering array over factors of the given value counts.

    Every pair of values drawn from two different factors appears in at least one returned row, at a
    fraction of the full cross product's size. The construction is AETG-style: each emitted row is the
    best of several candidates, and each candidate is seeded with a still-uncovered pair and filled
    greedily. Value *indices* (never strings) drive every iteration and random draw, so the result does
    not depend on string hashing and is identical across processes.

    Args:
        factor_sizes: Number of values each factor offers, in factor order.
        seed: Fixed seed; the array moves only when this or the factor sizes move.

    Returns:
        Rows of value indices, one entry per factor.
    """
    rng = random.Random(seed)
    uncovered: set[tuple[int, int, int, int]] = {
        (left, left_value, right, right_value)
        for left in range(len(factor_sizes))
        for right in range(left + 1, len(factor_sizes))
        for left_value in range(factor_sizes[left])
        for right_value in range(factor_sizes[right])
    }

    def _pair_key(left: int, left_value: int, right: int, right_value: int) -> tuple[int, int, int, int]:
        if left < right:
            return (left, left_value, right, right_value)
        return (right, right_value, left, left_value)

    def _newly_covered(row: tuple[int, ...]) -> int:
        return sum(
            1
            for left in range(len(factor_sizes))
            for right in range(left + 1, len(factor_sizes))
            if (left, row[left], right, row[right]) in uncovered
        )

    def _candidate_row() -> tuple[int, ...]:
        seed_pair = rng.choice(sorted(uncovered))
        row: list[int | None] = [None] * len(factor_sizes)
        row[seed_pair[0]] = seed_pair[1]
        row[seed_pair[2]] = seed_pair[3]
        for factor in rng.sample(range(len(factor_sizes)), len(factor_sizes)):
            if row[factor] is not None:
                continue
            best_value = 0
            best_gain = -1
            for value in range(factor_sizes[factor]):
                gain = sum(
                    1
                    for other, other_value in enumerate(row)
                    if other_value is not None and _pair_key(factor, value, other, other_value) in uncovered
                )
                if gain > best_gain:
                    best_value, best_gain = value, gain
            row[factor] = best_value
        return tuple(value for value in row if value is not None)

    rows: list[tuple[int, ...]] = []
    while uncovered:
        best_row: tuple[int, ...] = ()
        best_gain = -1
        for _ in range(_PAIRWISE_CANDIDATE_ROWS):
            candidate = _candidate_row()
            gain = _newly_covered(candidate)
            if gain > best_gain:
                best_row, best_gain = candidate, gain
        rows.append(best_row)
        for left in range(len(factor_sizes)):
            for right in range(left + 1, len(factor_sizes)):
                uncovered.discard((left, best_row[left], right, best_row[right]))
    return rows


def _categorical_cells() -> list[CorpusCell]:
    """G5: the pairwise categorical array over sampler, schedule, post-processing chain, and source mode.

    The full cross product of these four factors is an order of magnitude too large to run; a
    strength-2 array keeps every two-factor interaction represented, which is the level a per-axis cost
    model can actually consume.
    """
    factor_sizes = (len(_SAMPLERS), len(_SCHEDULERS), len(_POST_PROCESSING_CHAINS), len(_SOURCE_PROCESSING))
    cells: list[CorpusCell] = []
    for index, row in enumerate(_covering_array(factor_sizes, seed=_PAIRWISE_SEED)):
        sampler = _SAMPLERS[row[0]]
        scheduler = _SCHEDULERS[row[1]]
        pp_tag, pp_chain = _POST_PROCESSING_CHAINS[row[2]]
        source_processing = _SOURCE_PROCESSING[row[3]]
        cells.append(
            _variant(
                ANCHOR_SDXL,
                cell_id=f"g5.row{index:02d}.{sampler}.{scheduler}.{pp_tag}.{source_processing}",
                group="g5",
                sampler_name=sampler,
                scheduler=scheduler,
                post_processing=list(pp_chain),
                source_processing=source_processing,
                denoising_strength=_IMG2IMG_DENOISE if source_processing == "img2img" else None,
            ),
        )
    return cells


def _controlnet_cells() -> list[CorpusCell]:
    """G6: controlnet from the SD1.5 anchor, where control-type support is broadest."""
    return [
        _variant(
            ANCHOR_SD15,
            cell_id=f"g6.{control_type}.{sampler}.{_tag(SD15_A)}",
            group="g6",
            control_type=control_type,
            sampler_name=sampler,
            source_processing="img2img",
            denoising_strength=_IMG2IMG_DENOISE,
        )
        for control_type in ("canny", "depth", "openpose")
        for sampler in ("k_euler", "k_dpmpp_2m")
    ]


def _hires_fix_cells() -> list[CorpusCell]:
    """G7: hires fix, whose second pass is a cost the step count alone does not explain."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"g7.hires.{width}x{height}.{_tag(SDXL_A)}",
            group="g7",
            width=width,
            height=height,
            hires_fix=True,
        )
        for width, height in ((832, 1216), (1024, 1024))
    ]


def _lora_cells(lora_version_ids: tuple[str, ...]) -> list[CorpusCell]:
    """G8: LoRA count crossed with cache state.

    A miss cell's first replicate pays the fetch and the load; a hit cell reuses the same references
    after they are cached. The pair is what separates the per-LoRA apply cost from the one-off fetch,
    which a pricing model must not charge twice.
    """
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"g8.lora{count}.{role}.{_tag(SDXL_A)}",
            group="g8",
            lora_version_ids=list(lora_version_ids[:count]),
            lora_role=role,
        )
        for count in (1, 5)
        for role in ("miss", "hit")
    ]


def _textual_inversion_cells(ti_name: str) -> list[CorpusCell]:
    """G9: one textual inversion, whose cost sits in conditioning rather than in sampling."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"g9.ti1.{_tag(SDXL_A)}",
            group="g9",
            ti_names=[ti_name],
        ),
    ]


def _cold_load_cells() -> list[CorpusCell]:
    """G10: one cell per model whose every job is preceded by a job of a different model.

    A base-model switch is paid by the job that follows it, so the cost is a property of position. The
    ordering enforces the switch; the cell only declares that it needs one.
    """
    return [
        _variant(
            anchor,
            cell_id=f"g10.cold.{_tag(anchor.model)}",
            group="g10",
            requires_model_switch=True,
        )
        for anchor in _ANCHORS
    ]


def _post_processing_stack_cell() -> CorpusCell:
    """G11: three adjacent post-processing jobs, deliberately unspaced.

    Every other post-processing job is spaced so its cost is measured with an idle post-processing
    lane. This cell measures the other regime: what a job pays when the lane is already busy.
    """
    return _variant(
        ANCHOR_SDXL,
        cell_id=f"g11.ppstack.{_tag(SDXL_A)}",
        group="g11",
        post_processing=list(_PP_STACK_CHAIN),
        keeps_replicates_adjacent=True,
    )


def _warmup_cells(*, per_model: int) -> list[CorpusCell]:
    """Build the warmup cells: each model at its anchor shape, loaded once in a fixed order.

    The warmup block absorbs the costs a session pays once (process spawn, first model load, engine
    warm-up) so they are not attributed to whichever cell happened to be scheduled first.
    """
    return [
        _variant(anchor, cell_id=f"warmup.{_tag(anchor.model)}", group="warmup", replicates=per_model)
        for anchor in _ANCHORS
    ]


def _standard_cells(lora_version_ids: tuple[str, ...], ti_name: str) -> list[CorpusCell]:
    """Build every measurement cell of the standard tier, in group order."""
    return [
        *_steps_cells(),
        *_resolution_cells(),
        *_batch_cells(),
        *_cfg_cells(),
        *_categorical_cells(),
        *_controlnet_cells(),
        *_hires_fix_cells(),
        *_lora_cells(lora_version_ids),
        *_textual_inversion_cells(ti_name),
        *_cold_load_cells(),
        _post_processing_stack_cell(),
    ]


def _smoke_cells(lora_version_ids: tuple[str, ...]) -> list[CorpusCell]:
    """Build the smoke tier's subset: one replicate over a spanning sample of the groups.

    The smoke tier exists to prove the corpus still expands, orders, and runs end to end. It keeps one
    cell from each mechanism that can break (a steps sweep, a batch sweep, a categorical row, a LoRA
    miss, a cold load) and drops the rest.
    """
    keep_steps = {8, 30, 50}
    keep_batches = {1, 4}
    cells = [cell for cell in _steps_cells() if cell.steps in keep_steps]
    cells += [cell for cell in _batch_cells() if cell.n_iter in keep_batches]
    cells.append(_categorical_cells()[0])
    cells.append(_lora_cells(lora_version_ids)[0])
    # The second SDXL model appears nowhere else in the smoke tier, so its cold cell is also the check
    # that a third model can be resolved and switched to at all.
    cells += [cell for cell in _cold_load_cells() if cell.model == SDXL_B]
    return [cell.model_copy(update={"replicates": 1}) for cell in cells]


_NO_CONTROL_TYPE = "None"
"""The control_type vocabulary entry meaning "no controlnet", spelled as the manifest spells it."""

_CENSUS_ENLARGING_PP_SHARE = 0.4
"""Fraction of conflation cells that carry an enlarging post-processor."""

_CENSUS_PLAIN_PP_SHARE = 0.3
"""Fraction of conflation cells that carry a size-preserving post-processor (face fix, background strip)."""


@dataclass(frozen=True)
class _CensusPlan:
    """What the census built, carried out of cell construction so the definition can report it."""

    manifest_version: str
    vocabularies: dict[str, tuple[str, ...]]
    """The full manifest vocabulary per axis, exclusions included."""
    runnable: dict[str, tuple[str, ...]]
    """The values the census actually exercises, per axis."""
    exclusions: tuple[CorpusExclusion, ...]
    sweep_cell_count: int
    conflation_cell_count: int


def _manifest_vocabularies() -> tuple[str, dict[str, tuple[str, ...]]]:
    """Return the kudos feature manifest's revision and the vocabulary of each census axis.

    The census exists to cover the encoding contract, so its vocabularies are read from that contract
    rather than restated here: a value the manifest gains is a value the census must exercise, and a
    hand-copied list would drift into claiming coverage it does not have.

    Returns:
        The manifest revision identifier and the vocabulary tuple of each axis in :data:`CENSUS_AXES`.

    Raises:
        PricingCorpusError: If the installed hordelib ships no manifest, or the manifest carries no
            vocabulary for one of the census axes.
    """
    try:
        from hordelib.kudos_training.manifest import default_manifest
    except ImportError as error:
        raise PricingCorpusError(
            "the census tier derives its vocabularies from the kudos feature manifest, which the "
            "installed hordelib does not ship (hordelib.kudos_training.manifest); install a hordelib "
            "carrying the manifest and retry",
        ) from error

    manifest = default_manifest()
    vocabularies = {
        feature.name: tuple(vocabulary)
        for feature in manifest.features
        if feature.name in CENSUS_AXES and (vocabulary := getattr(feature, "vocabulary", None)) is not None
    }
    missing = sorted(set(CENSUS_AXES) - set(vocabularies))
    if missing:
        raise PricingCorpusError(
            f"the kudos feature manifest ({manifest.manifest_version}) carries no vocabulary for "
            f"{', '.join(missing)}, so the census cannot claim to cover it",
        )
    return manifest.manifest_version, vocabularies


def _census_validity() -> tuple[str, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], list[CorpusExclusion]]:
    """Resolve which vocabulary values the census can run, and name every one it cannot.

    Returns:
        The manifest revision, the full vocabularies, the runnable values per axis, and the exclusions.
    """
    version, vocabularies = _manifest_vocabularies()
    exclusions: list[CorpusExclusion] = []

    for value, reason in sorted(_CENSUS_EXCLUDED_SAMPLERS.items()):
        if value in vocabularies["sampler_name"]:
            exclusions.append(CorpusExclusion(kind="value", axis="sampler_name", value=value, reason=reason))

    covered_baselines = set(_CENSUS_MODEL_BASELINES.values())
    for value in vocabularies["baseline"]:
        if value not in covered_baselines:
            exclusions.append(
                CorpusExclusion(
                    kind="value",
                    axis="baseline",
                    value=value,
                    reason=("no model of this baseline is in the corpus's pinned model set, so no job can carry it"),
                ),
            )

    runnable = {
        "sampler_name": tuple(
            value for value in vocabularies["sampler_name"] if value not in _CENSUS_EXCLUDED_SAMPLERS
        ),
        "scheduler": vocabularies["scheduler"],
        # The "no controlnet" entry is not swept: it is what every cell naming no control type carries,
        # which is most of the corpus, so a sweep cell for it would measure the anchor a second time.
        "control_type": tuple(value for value in vocabularies["control_type"] if value != _NO_CONTROL_TYPE),
        "source_processing": vocabularies["source_processing"],
        "post_processing": vocabularies["post_processing"],
        "baseline": tuple(value for value in vocabularies["baseline"] if value in covered_baselines),
    }

    for (sampler, scheduler), reason in sorted(_CENSUS_EXCLUDED_SAMPLER_SCHEDULE_PAIRS.items()):
        if sampler in runnable["sampler_name"] and scheduler in runnable["scheduler"]:
            exclusions.append(
                CorpusExclusion(
                    kind="pair",
                    axis="sampler_name+scheduler",
                    value=f"{sampler}+{scheduler}",
                    reason=reason,
                ),
            )
    return version, vocabularies, runnable, exclusions


def _census_anchor(model: str) -> CorpusCell:
    """Return the anchor cell a census cell of *model* varies away from."""
    return ANCHOR_SD15 if model == SD15_A else ANCHOR_SDXL


def _census_sampler_cells(samplers: tuple[str, ...]) -> list[CorpusCell]:
    """C1: every runnable sampler against the anchor schedule.

    ``plms`` and ``dpmsolver`` stay in: hordelib maps them onto another solver's implementation, but the
    manifest prices them as their own vocabulary entries, so the census has to measure what they cost.
    """
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c1.sampler.{sampler}.{_tag(SDXL_A)}",
            group="c1",
            sampler_name=sampler,
            scheduler=_CENSUS_ANCHOR_SCHEDULER,
        )
        for sampler in samplers
    ]


def _census_scheduler_cells(schedulers: tuple[str, ...]) -> list[CorpusCell]:
    """C2: every schedule against the anchor sampler."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c2.schedule.{scheduler}.{_tag(SDXL_A)}",
            group="c2",
            sampler_name=_CENSUS_ANCHOR_SAMPLER,
            scheduler=scheduler,
        )
        for scheduler in schedulers
    ]


def _census_post_processing_cells(post_processors: tuple[str, ...]) -> list[CorpusCell]:
    """C3: every post-processor on its own, so a chain's cost decomposes into its members."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c3.pp.{post_processor}.{_tag(SDXL_A)}",
            group="c3",
            post_processing=[post_processor],
        )
        for post_processor in post_processors
    ]


def _census_control_cells(control_types: tuple[str, ...]) -> list[CorpusCell]:
    """C4: every control type, on SD1.5 because the controlnet weights exist for that baseline only.

    Each type runs its own annotator, whose cost differs by an order of magnitude across the set (a
    pure-cv2 edge detector against a depth network), which is exactly why one control cell cannot price
    the rest.
    """
    return [
        _variant(
            ANCHOR_SD15,
            cell_id=f"c4.control.{control_type}.{_tag(SD15_A)}",
            group="c4",
            control_type=control_type,
            source_processing="img2img",
            denoising_strength=_IMG2IMG_DENOISE,
        )
        for control_type in control_types
    ]


def _census_source_processing_cells(source_modes: tuple[str, ...]) -> list[CorpusCell]:
    """C5: every source-processing mode other than the anchor's txt2img."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c5.source.{source_processing}.{_tag(SDXL_A)}",
            group="c5",
            source_processing=source_processing,
            denoising_strength=_IMG2IMG_DENOISE if source_processing == "img2img" else None,
        )
        for source_processing in source_modes
        if source_processing != "txt2img"
    ]


def _census_hires_fix_cells() -> list[CorpusCell]:
    """C6: hires fix on and off at one shape, so the second pass is priced against its own control."""
    return [
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c6.hires{int(hires_fix)}.{_tag(SDXL_A)}",
            group="c6",
            hires_fix=hires_fix,
        )
        for hires_fix in (False, True)
    ]


def _census_lora_cells(lora_version_ids: tuple[str, ...]) -> list[CorpusCell]:
    """C7: the LoRA count levels, with the fetch paid once by a leading miss cell.

    The count levels measure the per-LoRA apply cost, which is only separable from the one-off fetch
    when the fetch has already happened: the miss cell carries every reference the hit cells use, and
    the ordering guarantees it runs first.
    """
    cells = [
        _variant(ANCHOR_SDXL, cell_id=f"c7.lora0.{_tag(SDXL_A)}", group="c7"),
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c7.lora{len(lora_version_ids)}.miss.{_tag(SDXL_A)}",
            group="c7",
            lora_version_ids=list(lora_version_ids),
            lora_role="miss",
        ),
    ]
    cells.extend(
        _variant(
            ANCHOR_SDXL,
            cell_id=f"c7.lora{count}.hit.{_tag(SDXL_A)}",
            group="c7",
            lora_version_ids=list(lora_version_ids[:count]),
            lora_role="hit",
        )
        for count in (1, 2, len(lora_version_ids))
    )
    return cells


def _census_textual_inversion_cells(ti_name: str) -> list[CorpusCell]:
    """C8: one and no textual inversions, the two levels a request realistically carries."""
    return [
        _variant(ANCHOR_SDXL, cell_id=f"c8.ti0.{_tag(SDXL_A)}", group="c8"),
        _variant(ANCHOR_SDXL, cell_id=f"c8.ti1.{_tag(SDXL_A)}", group="c8", ti_names=[ti_name]),
    ]


def _census_cold_load_cells() -> list[CorpusCell]:
    """C9: one cold-load cell per model, as in the standard tier.

    A base-model switch is paid by the job that follows it, and the census runs three models, so the
    switch cost has to be measurable or it loads onto whatever cell happened to follow a switch.
    """
    return [
        _variant(
            anchor,
            cell_id=f"c9.cold.{_tag(anchor.model)}",
            group="c9",
            requires_model_switch=True,
        )
        for anchor in _ANCHORS
    ]


def _shuffled_cycle(values: tuple[str, ...], length: int, rng: random.Random) -> list[str]:
    """Return *length* draws from *values* in which every value appears before any value repeats.

    A plain weighted draw over a 40-value vocabulary leaves several values absent from a 200-cell block
    by chance. Cycling a reshuffled copy makes the conflation block's coverage a property of the
    construction rather than of the seed.
    """
    drawn: list[str] = []
    while len(drawn) < length:
        block = list(values)
        rng.shuffle(block)
        drawn.extend(block)
    return drawn[:length]


def _runnable_schedule(sampler: str, preferred: str, schedulers: tuple[str, ...]) -> str:
    """Return *preferred*, or the next schedule in vocabulary order that *sampler* may run with."""
    start = schedulers.index(preferred)
    for offset in range(len(schedulers)):
        candidate = schedulers[(start + offset) % len(schedulers)]
        if (sampler, candidate) not in _CENSUS_EXCLUDED_SAMPLER_SCHEDULE_PAIRS:
            return candidate
    raise PricingCorpusError(f"every schedule is excluded for sampler {sampler!r}")


def _enlarges(post_processing: list[str]) -> bool:
    """Return whether a chain contains a processor that enlarges the image."""
    return any(value not in _CENSUS_NON_UPSCALING_POST_PROCESSORS for value in post_processing)


def _census_post_processing_chain(rng: random.Random, post_processors: tuple[str, ...]) -> list[str]:
    """Draw a conflation cell's post-processing chain.

    A chain carries at most one enlarging processor and at most one size-preserving one, which is the
    shape a real request takes and the only shape whose activation peak stays servable.
    """
    enlarging = tuple(value for value in post_processors if value not in _CENSUS_NON_UPSCALING_POST_PROCESSORS)
    plain = tuple(value for value in post_processors if value in _CENSUS_NON_UPSCALING_POST_PROCESSORS)
    chain: list[str] = []
    if enlarging and rng.random() < _CENSUS_ENLARGING_PP_SHARE:
        chain.append(rng.choice(enlarging))
    if plain and rng.random() < _CENSUS_PLAIN_PP_SHARE:
        chain.append(rng.choice(plain))
    return chain


def _census_conflation_cells(count: int, runnable: dict[str, tuple[str, ...]]) -> list[CorpusCell]:
    """C10: cells whose axes vary jointly, drawn under one fixed seed.

    The sweeps price each axis against a fixed anchor, which is the only way to read a marginal cost
    off them, and is also the regime real traffic is least like. This block covers the other regime: a
    representative sample of the combination space (not a covering array over it, which the census's
    vocabulary sizes put far out of budget) that lets a fit see whether the marginals still add up when
    several axes move at once.

    Sampler and schedule are cycled rather than drawn, so every runnable value of both appears in a
    conflated context; the remaining axes are drawn from the one seeded generator in a fixed order.
    """
    rng = random.Random(_CENSUS_CONFLATION_SEED)
    schedulers = runnable["scheduler"]
    samplers = _shuffled_cycle(runnable["sampler_name"], count, rng)
    preferred_schedules = _shuffled_cycle(schedulers, count, rng)
    models = [model for model, _weight in _CENSUS_MODEL_WEIGHTS]
    weights = [weight for _model, weight in _CENSUS_MODEL_WEIGHTS]

    cells: list[CorpusCell] = []
    for index in range(count):
        sampler = samplers[index]
        scheduler = _runnable_schedule(sampler, preferred_schedules[index], schedulers)
        model = rng.choices(models, weights=weights, k=1)[0]
        post_processing = _census_post_processing_chain(rng, runnable["post_processing"])
        # An enlarging processor multiplies the generated pixel count by its factor squared, so the
        # post-processing lane's activation peak, not the generation's, is what has to fit. The chain is
        # drawn first and then bounds both the shape and the batch, keeping every drawn cell servable:
        # a cell that faults on memory measures nothing and leaves the definition unpaired.
        bounded = _enlarges(post_processing)
        shapes = _SD15_SHAPES if model == SD15_A else _SDXL_SHAPES
        if bounded:
            shapes = tuple(shape for shape in shapes if shape[0] * shape[1] <= _CENSUS_SINGLE_IMAGE_PIXELS)
        width, height = rng.choice(shapes)
        n_iter = rng.choice(_CENSUS_CONFLATION_BATCHES)
        source_processing = rng.choice(runnable["source_processing"])
        control_type: str | None = None
        if model == SD15_A and rng.random() < _CENSUS_CONTROL_SHARE:
            control_type = rng.choice(runnable["control_type"])
            # A control job carries its control image, so it is an img2img-class job whatever else was
            # drawn; leaving the drawn mode would label the cell with a source mode it does not run.
            source_processing = "img2img"
        if bounded or width * height > _CENSUS_SINGLE_IMAGE_PIXELS:
            n_iter = 1
        cells.append(
            _variant(
                _census_anchor(model),
                cell_id=f"c10.mix{index:03d}.{_tag(model)}",
                group="c10",
                model=model,
                width=width,
                height=height,
                n_iter=n_iter,
                sampler_name=sampler,
                scheduler=scheduler,
                post_processing=post_processing,
                source_processing=source_processing,
                denoising_strength=_IMG2IMG_DENOISE if source_processing == "img2img" else None,
                control_type=control_type,
            ),
        )
    return cells


def _census_cells(
    lora_version_ids: tuple[str, ...],
    ti_name: str,
    *,
    warmup_job_count: int,
) -> tuple[list[CorpusCell], _CensusPlan]:
    """Build the census tier's cells and the plan describing what they cover.

    Args:
        lora_version_ids: The pinned LoRA references the count levels use.
        ti_name: The pinned textual-inversion reference.
        warmup_job_count: Jobs the warmup block spends, which come out of the same budget.

    Returns:
        The measurement cells and the coverage plan.

    Raises:
        PricingCorpusError: If the sweeps alone exceed :data:`CENSUS_JOB_BUDGET`.
    """
    version, vocabularies, runnable, exclusions = _census_validity()
    sweeps = [
        *_census_sampler_cells(runnable["sampler_name"]),
        *_census_scheduler_cells(runnable["scheduler"]),
        *_census_post_processing_cells(runnable["post_processing"]),
        *_census_control_cells(runnable["control_type"]),
        *_census_source_processing_cells(runnable["source_processing"]),
        *_census_hires_fix_cells(),
        *_census_lora_cells(lora_version_ids),
        *_census_textual_inversion_cells(ti_name),
        *_census_cold_load_cells(),
    ]
    spent = warmup_job_count + len(sweeps) * REPLICATES
    if spent > CENSUS_JOB_BUDGET:
        raise PricingCorpusError(
            f"the census sweeps need {spent} jobs, above the {CENSUS_JOB_BUDGET}-job budget; the "
            f"manifest ({version}) has outgrown the tier and its budget or its replicate count has to "
            "be revisited",
        )
    conflation_count = (CENSUS_JOB_BUDGET - spent) // REPLICATES
    conflation = _census_conflation_cells(conflation_count, runnable)
    plan = _CensusPlan(
        manifest_version=version,
        vocabularies=vocabularies,
        runnable=runnable,
        exclusions=tuple(exclusions),
        sweep_cell_count=len(sweeps),
        conflation_cell_count=len(conflation),
    )
    return [*sweeps, *conflation], plan


def _census_coverage(
    jobs: list[PricingCorpusJob],
    cells: dict[str, CorpusCell],
    plan: _CensusPlan,
    *,
    warmup_job_count: int,
) -> dict[str, dict[str, int]]:
    """Count how many measured jobs carry each vocabulary value, seeded with a zero for every value.

    Seeding matters more than counting: a value that never ran has to appear as a zero rather than as
    an absent key, so a coverage check reads as a claim about the whole vocabulary.
    """
    excluded = {(exclusion.axis, exclusion.value) for exclusion in plan.exclusions if exclusion.kind == "value"}
    coverage: dict[str, dict[str, int]] = {
        axis: {value: 0 for value in plan.vocabularies[axis] if (axis, value) not in excluded} for axis in CENSUS_AXES
    }

    def _count(axis: str, value: str) -> None:
        if value in coverage[axis]:
            coverage[axis][value] += 1

    for job in jobs[warmup_job_count:]:
        cell = cells[job.cell_id]
        _count("sampler_name", cell.sampler_name)
        _count("scheduler", cell.scheduler)
        _count("source_processing", cell.source_processing)
        _count("control_type", cell.control_type if cell.control_type is not None else _NO_CONTROL_TYPE)
        _count("baseline", _CENSUS_MODEL_BASELINES[cell.model])
        for post_processor in cell.post_processing:
            _count("post_processing", post_processor)
    return coverage


def _units_for_permutation(
    cells: list[CorpusCell],
    permutation: str,
    replicates: tuple[int, ...],
) -> list[_PlacementUnit]:
    """Expand cells into the placement units this permutation block carries."""
    units: list[_PlacementUnit] = []
    for cell in cells:
        wanted = tuple(replicate for replicate in replicates if replicate < cell.replicates)
        if not wanted:
            continue
        if cell.keeps_replicates_adjacent:
            # The stack is one atom: splitting its replicates across permutations would leave a block of
            # one, which measures the idle lane the rest of the corpus already covers.
            if permutation != PERMUTATION_SEEDS[0]:
                continue
            units.append(_PlacementUnit(cell=cell, replicates=tuple(range(cell.replicates)), permutation=permutation))
            continue
        units.extend(
            _PlacementUnit(cell=cell, replicates=(replicate,), permutation=permutation) for replicate in wanted
        )
    return units


def _smooth_weighted_pick(models: list[str], weights: dict[str, int], credit: dict[str, int]) -> str:
    """Pick the next model by smooth weighted round-robin over the remaining job counts.

    Preferring "any model but the last one" would exhaust the minority models in the first half of the
    block and leave a long single-model tail, concentrating every model-switch measurement at the start.
    Smooth weighted round-robin instead interleaves each model at its own share's spacing, which spreads
    the switches (and so the unavoidable same-model adjacencies) evenly across the block.
    """
    total = sum(weights[model] for model in models)
    for model in models:
        credit[model] = credit.get(model, 0) + weights[model]
    chosen = max(sorted(models), key=lambda model: credit[model])
    credit[chosen] -= total
    return chosen


def _order_units(units: list[_PlacementUnit], *, seed: str, state: _OrderState) -> list[_PlacementUnit]:
    """Order one permutation block's units under the corpus's ordering constraints.

    The constraints are applied in priority order. Three are hard, because a violated one destroys the
    measurement the cell exists for: a cold-load cell must follow a different model, the stack cell's
    replicates must stay adjacent, and a LoRA reference must first appear in a miss cell. Two are
    preferences, because the mix cannot satisfy them everywhere: with one model holding most of the
    corpus, same-model adjacency is arithmetically unavoidable, so the rule is honored where a choice
    exists and the residue is counted into the definition artifact rather than hidden.

    Args:
        units: The block's units; consumed in a fixed shuffle of this list.
        seed: The block's shuffle seed.
        state: Session-wide ordering state, advanced in place.

    Returns:
        The block's units in placement order.

    Raises:
        PricingCorpusOrderingError: If a hard constraint leaves no placeable unit.
    """
    rng = random.Random(seed)
    pool = list(units)
    rng.shuffle(pool)

    cold_targets: dict[int, int] = {}
    span = max(1, int(len(pool) * _COLD_TARGET_SPAN))
    for index, unit in enumerate(pool):
        if unit.cell.requires_model_switch:
            cold_targets[index] = rng.randrange(span)

    remaining = list(range(len(pool)))
    remaining_jobs: dict[str, int] = {}
    for unit in pool:
        remaining_jobs[unit.cell.model] = remaining_jobs.get(unit.cell.model, 0) + unit.size
    credit: dict[str, int] = {}
    ordered: list[_PlacementUnit] = []
    placed = 0

    def _is_eligible(index: int) -> bool:
        unit = pool[index]
        if unit.cell.requires_model_switch and state.last_model == unit.cell.model:
            return False
        if unit.cell.lora_role == "hit":
            return all(reference in state.seen_lora_ids for reference in unit.cell.lora_version_ids)
        return True

    def _spacing_ok(index: int) -> bool:
        unit = pool[index]
        if not unit.spaced_post_processing:
            return True
        return state.position - state.last_post_processing_position >= _POST_PROCESSING_MIN_DISTANCE

    while remaining:
        eligible = [index for index in remaining if _is_eligible(index)]
        if not eligible:
            blocked = sorted({pool[index].cell.cell_id for index in remaining})
            raise PricingCorpusOrderingError(
                "no unit can be placed without violating a hard ordering constraint (cold-load model "
                f"switch or LoRA miss-before-hit); blocked cells: {blocked}",
            )

        due_cold = [index for index in eligible if index in cold_targets and placed >= cold_targets[index]]
        if due_cold:
            chosen = due_cold[0]
        else:
            # A cold unit whose turn has come but whose model is currently resident can only be unblocked
            # by placing a different model next, so its model steps out of the running for this position.
            deferred_models = {
                pool[index].cell.model
                for index in remaining
                if index in cold_targets and placed >= cold_targets[index] and index not in eligible
            }
            candidate_models = sorted({pool[index].cell.model for index in eligible} - deferred_models)
            if not candidate_models:
                candidate_models = sorted({pool[index].cell.model for index in eligible})
            model = _smooth_weighted_pick(candidate_models, remaining_jobs, credit)
            candidates = [index for index in eligible if pool[index].cell.model == model]
            spaced = [index for index in candidates if _spacing_ok(index)]
            chosen = (spaced or candidates)[0]

        unit = pool[chosen]
        ordered.append(unit)
        remaining.remove(chosen)
        cold_targets.pop(chosen, None)
        remaining_jobs[unit.cell.model] -= unit.size
        if unit.spaced_post_processing:
            state.last_post_processing_position = state.position + unit.size - 1
        state.last_model = unit.cell.model
        state.position += unit.size
        state.seen_lora_ids.update(unit.cell.lora_version_ids)
        placed += unit.size

    return ordered


def _units_to_jobs(units: list[_PlacementUnit], *, start_position: int) -> list[PricingCorpusJob]:
    """Expand ordered units into the ordered job records that go into the definition artifact."""
    jobs: list[PricingCorpusJob] = []
    position = start_position
    for unit in units:
        for replicate in unit.replicates:
            jobs.append(
                PricingCorpusJob(
                    position=position,
                    cell_id=unit.cell.cell_id,
                    group=unit.cell.group,
                    permutation=unit.permutation,
                    replicate=replicate,
                    seed=f"pc1:{unit.cell.cell_id}:{replicate}",
                    prompt_index=replicate % len(PINNED_PROMPTS),
                    model=unit.cell.model,
                ),
            )
            position += 1
    return jobs


def _audit_order(
    jobs: list[PricingCorpusJob],
    cells: dict[str, CorpusCell],
    *,
    warmup_job_count: int,
) -> tuple[int, int]:
    """Verify the hard ordering constraints and count the residue of the soft ones.

    Args:
        jobs: The ordered job list, warmup block included.
        cells: Cells by id.
        warmup_job_count: How many leading jobs are warmup (exempt from every constraint).

    Returns:
        The number of same-model adjacencies and of under-spaced post-processing pairs.

    Raises:
        PricingCorpusOrderingError: If a hard constraint is violated, naming it.
    """
    for job in jobs[warmup_job_count:]:
        cell = cells[job.cell_id]
        if cell.requires_model_switch and jobs[job.position - 1].model == job.model:
            raise PricingCorpusOrderingError(
                f"cold-load constraint violated: {job.cell_id} at position {job.position} follows its own "
                f"model ({job.model})",
            )

    stack_positions: dict[str, list[int]] = {}
    for job in jobs:
        if cells[job.cell_id].keeps_replicates_adjacent:
            stack_positions.setdefault(job.cell_id, []).append(job.position)
    for cell_id, positions in sorted(stack_positions.items()):
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise PricingCorpusOrderingError(
                f"adjacency constraint violated: {cell_id} occupies non-consecutive positions {positions}",
            )

    first_use: dict[str, PricingCorpusJob] = {}
    for job in jobs:
        for reference in cells[job.cell_id].lora_version_ids:
            first_use.setdefault(reference, job)
    for reference, job in sorted(first_use.items()):
        if cells[job.cell_id].lora_role != "miss":
            raise PricingCorpusOrderingError(
                f"miss-before-hit constraint violated: LoRA {reference} is first used by {job.cell_id} at "
                f"position {job.position}, which is not a miss cell",
            )

    same_model = 0
    for previous, job in zip(jobs[warmup_job_count:-1], jobs[warmup_job_count + 1 :], strict=True):
        if previous.model != job.model:
            continue
        if previous.cell_id == job.cell_id and cells[job.cell_id].keeps_replicates_adjacent:
            continue
        if job.position - previous.position < _SAME_MODEL_MIN_DISTANCE:
            same_model += 1

    pp_positions = [
        job.position
        for job in jobs[warmup_job_count:]
        if cells[job.cell_id].post_processing and not cells[job.cell_id].keeps_replicates_adjacent
    ]
    proximities = sum(
        1
        for previous, position in zip(pp_positions, pp_positions[1:], strict=False)
        if position - previous < _POST_PROCESSING_MIN_DISTANCE
    )
    return same_model, proximities


def _job_spec(cell: CorpusCell, job: PricingCorpusJob) -> CannedImageJobSpec:
    """Build the canned job spec for one corpus job."""
    return CannedImageJobSpec(
        model=cell.model,
        width=cell.width,
        height=cell.height,
        steps=cell.steps,
        cfg_scale=cell.cfg_scale,
        n_iter=cell.n_iter,
        hires_fix=cell.hires_fix,
        sampler_name=cell.sampler_name,
        scheduler=cell.scheduler,
        seed=job.seed,
        prompt=PINNED_PROMPTS[job.prompt_index],
        # txt2img is the canned factory's default; leaving it unset keeps the payload identical to every
        # other txt2img job in the repo rather than differing by an explicitly-set field.
        source_processing=None if cell.source_processing == "txt2img" else cell.source_processing,
        denoising_strength=cell.denoising_strength,
        lora_names=list(cell.lora_version_ids),
        lora_is_version=bool(cell.lora_version_ids),
        ti_names=list(cell.ti_names),
        control_type=cell.control_type,
        post_processing=list(cell.post_processing),
        count=1,
    )


def build_pricing_corpus_scenario(
    tier: PricingCorpusTier,
    *,
    lora_version_ids: tuple[str, ...] = PRICING_CORPUS_LORA_VERSION_IDS,
    ti_name: str = PRICING_CORPUS_TI_NAME,
) -> tuple[Scenario, PricingCorpusDefinition]:
    """Build the pricing corpus for a tier: the scenario to run and the definition describing it.

    Args:
        tier: ``standard`` for the marginal-cost fit set, ``census`` for full vocabulary coverage,
            ``smoke`` for the fast shape check.
        lora_version_ids: CivitAI version ids for the LoRA cells, five of them. Defaults to the pinned
            references in :data:`PRICING_CORPUS_LORA_VERSION_IDS`.
        ti_name: The textual-inversion reference for the G9 cell. Defaults to
            :data:`PRICING_CORPUS_TI_NAME`.

    Returns:
        The scenario (its job list already in final order) and its definition artifact.

    Raises:
        PricingCorpusError: If the tier is unknown, too few LoRA ids are supplied, or the census tier
            cannot read the kudos feature manifest its vocabularies come from.
        PricingCorpusOrderingError: If the ordering constraints cannot be satisfied.
    """
    if tier not in ("smoke", "standard", "census"):
        raise PricingCorpusError(f"unknown pricing-corpus tier: {tier!r}")
    if len(lora_version_ids) < len(PRICING_CORPUS_LORA_VERSION_IDS):
        raise PricingCorpusError(
            f"the LoRA cells need {len(PRICING_CORPUS_LORA_VERSION_IDS)} version ids, got {len(lora_version_ids)}",
        )

    warmup_per_model = 1 if tier == "smoke" else 2
    warmup_cells = _warmup_cells(per_model=warmup_per_model)
    warmup_units = [
        _PlacementUnit(cell=cell, replicates=tuple(range(cell.replicates)), permutation="warmup")
        for cell in warmup_cells
    ]
    warmup_jobs = _units_to_jobs(warmup_units, start_position=0)

    plan: _CensusPlan | None = None
    if tier == "standard":
        cells = _standard_cells(lora_version_ids, ti_name)
    elif tier == "census":
        cells, plan = _census_cells(lora_version_ids, ti_name, warmup_job_count=len(warmup_jobs))
    else:
        cells = _smoke_cells(lora_version_ids)

    state = _OrderState(last_model=warmup_units[-1].cell.model)
    state.position = len(warmup_jobs)

    seeds = [PERMUTATION_SEEDS[0]] if tier == "smoke" else list(PERMUTATION_SEEDS)
    replicates_by_seed = {PERMUTATION_SEEDS[0]: (0,)} if tier == "smoke" else PERMUTATION_REPLICATES

    jobs = list(warmup_jobs)
    for seed in seeds:
        units = _units_for_permutation(cells, seed, replicates_by_seed[seed])
        ordered = _order_units(units, seed=seed, state=state)
        jobs.extend(_units_to_jobs(ordered, start_position=len(jobs)))

    all_cells = [*warmup_cells, *cells]
    cells_by_id = {cell.cell_id: cell for cell in all_cells}
    same_model, proximities = _audit_order(jobs, cells_by_id, warmup_job_count=len(warmup_jobs))

    scenario = Scenario(
        name=SCENARIO_NAME,
        revision=SCENARIO_REVISION,
        image_jobs=[_job_spec(cells_by_id[job.cell_id], job) for job in jobs],
    )
    definition = PricingCorpusDefinition(
        scenario_name=SCENARIO_NAME,
        scenario_revision=SCENARIO_REVISION,
        tier=tier,
        warmup_job_count=len(warmup_jobs),
        shuffle_seeds=seeds,
        prompts=list(PINNED_PROMPTS),
        cells=all_cells,
        jobs=jobs,
        same_model_adjacencies=same_model,
        post_processing_proximities=proximities,
        census=(
            None
            if plan is None
            else CensusSummary(
                manifest_version=plan.manifest_version,
                job_budget=CENSUS_JOB_BUDGET,
                projected_job_count=len(jobs),
                projected_runtime_seconds=len(jobs) * CENSUS_SECONDS_PER_JOB,
                seconds_per_job=CENSUS_SECONDS_PER_JOB,
                sweep_cell_count=plan.sweep_cell_count,
                conflation_cell_count=plan.conflation_cell_count,
                exclusions=list(plan.exclusions),
                coverage=_census_coverage(jobs, cells_by_id, plan, warmup_job_count=len(warmup_jobs)),
            )
        ),
    )
    return scenario, definition


_TIER_SPECIFIC_FIELDS: tuple[str, ...] = ("census",)
"""Definition fields only some tiers populate; omitted from the artifact when unset."""


def _canonical(value: object) -> object:
    """Round floats to a fixed precision so the artifact's bytes do not depend on float repr drift."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def definition_json(definition: PricingCorpusDefinition) -> str:
    """Render a definition as the canonical JSON text written to disk.

    Keys are sorted and floats are rounded to a fixed precision, so two hosts building the same tier
    produce byte-identical text and a diff of two artifacts shows only real differences.

    A tier-specific field carries no key at all when the tier does not use it, which is what keeps a
    tier's bytes comparable across a revision that added a field for a different tier.
    """
    exclude = {field_name for field_name in _TIER_SPECIFIC_FIELDS if getattr(definition, field_name) is None}
    payload = _canonical(definition.model_dump(mode="json", exclude=exclude))
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_definition_artifact(definition: PricingCorpusDefinition, path: Path) -> Path:
    """Write a definition artifact to ``path``, creating its directory, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(definition_json(definition), encoding="utf-8")
    return path
