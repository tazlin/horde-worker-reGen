"""Tests for the pricing-corpus generator.

The corpus is training data for a cost model, so its properties are the test subject: the expansion is
byte-reproducible, the categorical array covers every two-factor interaction, and the ordering delivers
the positional conditions (cold model, LoRA cache miss, post-processing queueing) the state-dependent
cells exist to measure. None of this needs a worker or a GPU.
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import pytest

from horde_worker_regen.benchmark.pricing_corpus import (
    _CENSUS_EXCLUDED_SAMPLER_SCHEDULE_PAIRS,
    _CENSUS_EXCLUDED_SAMPLERS,
    _CENSUS_NON_UPSCALING_POST_PROCESSORS,
    _CENSUS_SAMPLER_TRAJECTORY_STEPS,
    _POST_PROCESSING_CHAINS,
    _POST_PROCESSING_MIN_DISTANCE,
    _SAMPLERS,
    _SCHEDULERS,
    _SOURCE_PROCESSING,
    CENSUS_AXES,
    CENSUS_JOB_BUDGET,
    CENSUS_SECONDS_PER_JOB,
    PRICING_CORPUS_LORA_VERSION_IDS,
    PRICING_CORPUS_TI_NAME,
    REPLICATES,
    SCENARIO_NAME,
    SCENARIO_REVISION,
    SD15_A,
    SDXL_A,
    SDXL_B,
    CorpusCell,
    PricingCorpusDefinition,
    PricingCorpusError,
    PricingCorpusJob,
    PricingCorpusOrderingError,
    _audit_order,
    _categorical_cells,
    build_pricing_corpus_scenario,
    definition_json,
    write_definition_artifact,
)
from horde_worker_regen.benchmark.scenarios import Scenario

_REAL_LORA_IDS = ("111111", "222222", "333333", "444444", "555555")
"""Stand-ins for operator-supplied CivitAI version ids: only their difference from the sentinels matters."""


def _manifest_is_available() -> bool:
    """Return whether the installed hordelib ships the kudos feature manifest the census reads."""
    try:
        import hordelib.kudos_training.manifest  # noqa: F401
    except ImportError:
        return False
    return True


_REQUIRES_MANIFEST = pytest.mark.skipif(
    not _manifest_is_available(),
    reason="the census derives its vocabularies from hordelib's kudos feature manifest, which this "
    "hordelib does not ship",
)


@pytest.fixture(scope="module")
def standard() -> tuple[Scenario, PricingCorpusDefinition]:
    """The full corpus, built once (its construction is pure, so sharing it across tests is safe)."""
    return build_pricing_corpus_scenario("standard")


@pytest.fixture(scope="module")
def smoke() -> tuple[Scenario, PricingCorpusDefinition]:
    """The smoke-tier corpus."""
    return build_pricing_corpus_scenario("smoke")


@pytest.fixture(scope="module")
def census() -> tuple[Scenario, PricingCorpusDefinition]:
    """The census-tier corpus, built once."""
    if not _manifest_is_available():
        pytest.skip("hordelib ships no kudos feature manifest")
    return build_pricing_corpus_scenario("census")


def _cells_by_id(definition: PricingCorpusDefinition) -> dict[str, CorpusCell]:
    return {cell.cell_id: cell for cell in definition.cells}


def _measured_jobs(definition: PricingCorpusDefinition) -> list[PricingCorpusJob]:
    """The jobs the assembler keeps: everything after the warmup block."""
    return definition.jobs[definition.warmup_job_count :]


class TestDeterminism:
    """Two builds of a tier must be indistinguishable, or records cannot be pooled across hosts."""

    @pytest.mark.parametrize("tier", ["smoke", "standard"])
    def test_repeated_builds_are_byte_identical(self, tier: str) -> None:
        """Two builds of the same tier render identical artifact text."""
        first = build_pricing_corpus_scenario(tier)[1]  # type: ignore[arg-type]
        second = build_pricing_corpus_scenario(tier)[1]  # type: ignore[arg-type]
        assert definition_json(first) == definition_json(second)

    def test_written_artifact_is_byte_identical(self, tmp_path: Path) -> None:
        """Two builds written to different paths leave identical bytes."""
        first = build_pricing_corpus_scenario("standard")[1]
        second = build_pricing_corpus_scenario("standard")[1]
        first_path = write_definition_artifact(first, tmp_path / "nested" / "first.json")
        second_path = write_definition_artifact(second, tmp_path / "second.json")
        assert first_path.read_bytes() == second_path.read_bytes()

    def test_artifact_round_trips(self, tmp_path: Path, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """A written artifact parses back into an equal definition."""
        path = write_definition_artifact(standard[1], tmp_path / "definition.json")
        reloaded = PricingCorpusDefinition.model_validate_json(path.read_text(encoding="utf-8"))
        assert definition_json(reloaded) == definition_json(standard[1])

    def test_scenario_identity_travels(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """The scenario carries the corpus name and revision, and one spec per job."""
        scenario, definition = standard
        assert (scenario.name, scenario.revision) == (SCENARIO_NAME, SCENARIO_REVISION)
        assert scenario.total_image_jobs == len(definition.jobs)


class TestCategoricalArray:
    """The strength-2 array is the reason the categorical axes are affordable at all."""

    def test_every_pair_of_values_co_occurs(self) -> None:
        """Every pair of values from two different factors appears in some row."""
        cells = _categorical_cells()
        chain_by_tuple = {tag: list(chain) for tag, chain in _POST_PROCESSING_CHAINS}
        rows = [
            (
                cell.sampler_name,
                cell.scheduler,
                next(tag for tag, chain in chain_by_tuple.items() if chain == cell.post_processing),
                cell.source_processing,
            )
            for cell in cells
        ]
        factors = [
            list(_SAMPLERS),
            list(_SCHEDULERS),
            [tag for tag, _chain in _POST_PROCESSING_CHAINS],
            list(_SOURCE_PROCESSING),
        ]
        for left, right in itertools.combinations(range(len(factors)), 2):
            realized = {(row[left], row[right]) for row in rows}
            missing = {
                (left_value, right_value) for left_value in factors[left] for right_value in factors[right]
            } - realized
            assert not missing, f"factors {left}x{right} miss {sorted(missing)}"

    def test_array_is_far_smaller_than_the_cross_product(self) -> None:
        """The array costs a fraction of the full cross product."""
        cross_product = len(_SAMPLERS) * len(_SCHEDULERS) * len(_POST_PROCESSING_CHAINS) * len(_SOURCE_PROCESSING)
        assert len(_categorical_cells()) < cross_product // 4

    def test_adaptive_sampler_is_excluded(self) -> None:
        """The step-count-ignoring sampler stays out of the array."""
        # It picks its own step count, so its records would carry a step count the work never obeyed.
        assert "k_dpm_adaptive" not in _SAMPLERS

    def test_img2img_rows_carry_a_partial_denoise(self) -> None:
        """img2img rows sample at partial strength, not the full default."""
        img2img = [cell for cell in _categorical_cells() if cell.source_processing == "img2img"]
        assert img2img
        assert all(cell.denoising_strength == pytest.approx(0.75) for cell in img2img)

    def test_inpainting_rows_expand_with_a_source_image_and_mask(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Every expanded inpainting job carries both inputs; without the mask its cost is not an inpaint's."""
        scenario, _definition = standard
        jobs = scenario.expand_image_jobs()
        inpainting = [job for job in jobs if str(job.source_processing) == "inpainting"]
        assert inpainting, "the covering array must exercise inpainting rows"
        assert all(job.source_image for job in inpainting)
        assert all(job.source_mask for job in inpainting)


class TestWarmupBlock:
    """The warmup block absorbs the once-per-session costs; it must cover every model, in a fixed order."""

    def test_standard_warmup_is_two_jobs_per_model_at_anchor_shape(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The standard warmup loads each model twice, at its anchor shape."""
        scenario, definition = standard
        assert definition.warmup_job_count == 6
        warmup = definition.jobs[: definition.warmup_job_count]
        assert [job.model for job in warmup] == [SDXL_A, SDXL_A, SDXL_B, SDXL_B, SD15_A, SD15_A]
        assert all(job.permutation == "warmup" for job in warmup)
        cells = _cells_by_id(definition)
        for job in warmup:
            cell = cells[job.cell_id]
            assert cell.group == "warmup"
            assert (cell.width, cell.height) == ((512, 512) if cell.model == SD15_A else (1024, 1024))
            assert not cell.post_processing and cell.control_type is None and not cell.lora_version_ids
        assert [spec.model for spec in scenario.image_jobs[:6]] == [job.model for job in warmup]

    def test_smoke_warmup_is_one_job_per_model(self, smoke: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """The smoke warmup loads each model once."""
        _scenario, definition = smoke
        assert definition.warmup_job_count == 3
        assert [job.model for job in definition.jobs[:3]] == [SDXL_A, SDXL_B, SD15_A]

    def test_warmup_cells_are_never_measured_cells(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """No measured cell shares an id with a warmup cell."""
        warmup_ids = {job.cell_id for job in standard[1].jobs[: standard[1].warmup_job_count]}
        assert not warmup_ids & {job.cell_id for job in _measured_jobs(standard[1])}


class TestOrdering:
    """The positional constraints are the corpus's only handle on state-dependent costs."""

    def test_cold_cells_follow_a_different_model(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """Every cold-load job is preceded by a job of another model."""
        _scenario, definition = standard
        cells = _cells_by_id(definition)
        cold = [job for job in _measured_jobs(definition) if cells[job.cell_id].requires_model_switch]
        assert {cells[job.cell_id].model for job in cold} == {SDXL_A, SDXL_B, SD15_A}
        for job in cold:
            assert definition.jobs[job.position - 1].model != job.model

    def test_stack_cell_replicates_are_consecutive(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """The queueing-stack cell's three jobs are adjacent."""
        _scenario, definition = standard
        cells = _cells_by_id(definition)
        positions = [job.position for job in definition.jobs if cells[job.cell_id].keeps_replicates_adjacent]
        assert len(positions) == 3
        assert positions == list(range(positions[0], positions[0] + 3))

    def test_every_lora_reference_is_first_used_by_a_miss_cell(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """A LoRA reference's first appearance is in a miss cell."""
        _scenario, definition = standard
        cells = _cells_by_id(definition)
        first_use: dict[str, PricingCorpusJob] = {}
        for job in definition.jobs:
            for reference in cells[job.cell_id].lora_version_ids:
                first_use.setdefault(reference, job)
        assert set(first_use) == set(PRICING_CORPUS_LORA_VERSION_IDS)
        for job in first_use.values():
            assert cells[job.cell_id].lora_role == "miss"

    def test_same_model_adjacency_never_exceeds_what_the_mix_forces(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Same-model adjacency stays at the arithmetic floor."""
        # One model holds most of the corpus, so the min-distance rule cannot hold everywhere: n jobs of
        # the majority model cannot be separated by fewer than n-1 jobs of other models. What is testable
        # is that the ordering adds nothing on top of that arithmetic floor.
        _scenario, definition = standard
        counts = Counter(job.model for job in _measured_jobs(definition))
        majority = max(counts.values())
        others = sum(counts.values()) - majority
        assert majority > others + 1, "the floor below is only meaningful while one model dominates"
        assert definition.same_model_adjacencies <= majority - (others + 1)

    def test_minority_models_are_spread_and_runs_stay_short(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """No model is delivered in one long uninterrupted block."""
        # "Churn present but not pathological": the majority model must not be delivered in one long block
        # with the minority models exhausted up front.
        _scenario, definition = standard
        runs = [
            len(list(group)) for _model, group in itertools.groupby(job.model for job in _measured_jobs(definition))
        ]
        assert max(runs) <= 6

    def test_post_processing_jobs_are_spaced_outside_the_stack_cell(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Post-processing jobs are spaced, and the residue is reported."""
        _scenario, definition = standard
        cells = _cells_by_id(definition)
        positions = [
            job.position
            for job in _measured_jobs(definition)
            if cells[job.cell_id].post_processing and not cells[job.cell_id].keeps_replicates_adjacent
        ]
        gaps = [right - left for left, right in itertools.pairwise(positions)]
        short = [gap for gap in gaps if gap < _POST_PROCESSING_MIN_DISTANCE]
        assert len(short) == definition.post_processing_proximities
        # The rule is the lowest-priority one, so a residue is expected; a quarter of the gaps is the point
        # past which the spaced measurement stops being the common case.
        assert len(short) <= len(gaps) // 4

    def test_permutations_split_the_replicates(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """Each cell's three replicates are split across the two permutations."""
        _scenario, definition = standard
        assert definition.shuffle_seeds == ["pc1-a", "pc1-b"]
        by_permutation = Counter(job.permutation for job in _measured_jobs(definition))
        assert by_permutation["pc1-a"] > by_permutation["pc1-b"] > 0
        cells = _cells_by_id(definition)
        for cell in definition.cells:
            if cell.group in ("warmup", "g11"):
                continue
            replicates = sorted(job.replicate for job in definition.jobs if job.cell_id == cell.cell_id)
            assert replicates == [0, 1, 2]
        # Each replicate is measured once, in one place.
        seeds = [job.seed for job in definition.jobs]
        assert len(seeds) == len(set(seeds))
        assert all(job.seed == f"pc1:{job.cell_id}:{job.replicate}" for job in definition.jobs)
        assert all(job.group == cells[job.cell_id].group for job in definition.jobs)

    def test_a_violated_hard_constraint_is_raised_by_name(self) -> None:
        """An order that breaks a hard constraint raises, naming it."""
        cell = CorpusCell(
            cell_id="g10.cold.sdxl",
            group="g10",
            model=SDXL_A,
            width=1024,
            height=1024,
            steps=30,
            cfg_scale=7.5,
            n_iter=1,
            sampler_name="k_euler",
            scheduler="karras",
            source_processing="txt2img",
            requires_model_switch=True,
        )
        jobs = [
            PricingCorpusJob(
                position=index,
                cell_id=cell.cell_id,
                group=cell.group,
                permutation="pc1-a",
                replicate=index,
                seed=f"pc1:{cell.cell_id}:{index}",
                prompt_index=0,
                model=SDXL_A,
            )
            for index in range(2)
        ]
        with pytest.raises(PricingCorpusOrderingError, match="cold-load constraint violated"):
            _audit_order(jobs, {cell.cell_id: cell}, warmup_job_count=0)


class TestTiers:
    """Each tier's realized shape, so a silent change in cell count cannot pass unnoticed."""

    def test_standard_cell_and_job_counts(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """The standard tier's realized cell groups and job count."""
        _scenario, definition = standard
        measured = [cell for cell in definition.cells if cell.group != "warmup"]
        assert len(measured) == 84
        assert Counter(cell.group for cell in measured) == {
            "g1": 12,
            "g2": 12,
            "g3": 8,
            "g4": 3,
            "g5": 32,
            "g6": 6,
            "g7": 2,
            "g8": 4,
            "g9": 1,
            "g10": 3,
            "g11": 1,
        }
        assert len(definition.jobs) == 6 + sum(cell.replicates for cell in measured)
        assert all(cell.replicates == 3 for cell in measured)

    def test_smoke_is_a_single_replicate_subset(self, smoke: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """The smoke tier keeps one replicate of a spanning subset."""
        scenario, definition = smoke
        measured = [cell for cell in definition.cells if cell.group != "warmup"]
        assert Counter(cell.group for cell in measured) == {"g1": 6, "g3": 4, "g5": 1, "g8": 1, "g10": 1}
        assert all(cell.replicates == 1 for cell in measured)
        assert len(definition.jobs) == 3 + len(measured)
        assert {cell.steps for cell in measured if cell.group == "g1"} == {8, 30, 50}
        assert {cell.n_iter for cell in measured if cell.group == "g3"} == {1, 4}
        assert definition.shuffle_seeds == ["pc1-a"]
        assert scenario.models_referenced() == sorted({SDXL_A, SDXL_B, SD15_A})

    def test_unknown_tier_is_rejected(self) -> None:
        """An unknown tier is refused rather than silently defaulted."""
        with pytest.raises(PricingCorpusError, match="unknown pricing-corpus tier"):
            build_pricing_corpus_scenario("deep")  # type: ignore[arg-type]


class TestPinnedReferences:
    """The LoRA and TI cells reference real, pinned CivitAI version ids so their fetch cost is real."""

    def test_operator_overrides_replace_the_pinned_references(self) -> None:
        """References supplied at build time reach every LoRA cell."""
        _scenario, definition = build_pricing_corpus_scenario(
            "standard",
            lora_version_ids=_REAL_LORA_IDS,
            ti_name="real-ti",
        )
        cells = _cells_by_id(definition)
        assert {reference for cell in cells.values() for reference in cell.lora_version_ids} == set(_REAL_LORA_IDS)

    def test_too_few_lora_ids_is_rejected(self) -> None:
        """Fewer references than the LoRA cells need is refused."""
        with pytest.raises(PricingCorpusError, match="version ids"):
            build_pricing_corpus_scenario("standard", lora_version_ids=("1", "2"))

    def test_pinned_references_are_real_version_ids(self) -> None:
        """The defaults are five distinct numeric version ids (plus a numeric TI reference).

        A non-numeric reference would fail resolution and its cells would measure the failure path
        rather than the LoRA cost, so the pins must always look like what CivitAI issues.
        """
        assert len(set(PRICING_CORPUS_LORA_VERSION_IDS)) == 5
        assert all(reference.isdigit() for reference in PRICING_CORPUS_LORA_VERSION_IDS)
        assert PRICING_CORPUS_TI_NAME.isdigit()


class TestJobExpansion:
    """The scenario the harness runs must carry the axis values the definition claims it does."""

    def test_specs_match_their_cells(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """Every canned spec carries its cell's axis values and its job's seed."""
        scenario, definition = standard
        cells = _cells_by_id(definition)
        for spec, job in zip(scenario.image_jobs, definition.jobs, strict=True):
            cell = cells[job.cell_id]
            assert spec.model == cell.model
            assert (spec.width, spec.height) == (cell.width, cell.height)
            assert spec.steps == cell.steps
            assert spec.n_iter == cell.n_iter
            assert spec.sampler_name == cell.sampler_name
            assert spec.scheduler == cell.scheduler
            assert spec.seed == job.seed
            assert spec.count == 1
            assert spec.lora_names == cell.lora_version_ids
            assert spec.post_processing == cell.post_processing

    def test_prompts_rotate_by_replicate(self, standard: tuple[Scenario, PricingCorpusDefinition]) -> None:
        """Prompts rotate through the pinned set by replicate index."""
        scenario, definition = standard
        prompts = {spec.prompt for spec in scenario.image_jobs}
        assert prompts == set(definition.prompts)
        assert all(job.prompt_index == job.replicate % 3 for job in definition.jobs)

    def test_census_field_is_absent_from_the_other_tiers_artifacts(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
        smoke: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """A tier that mints no coverage claim carries no key for one."""
        # The census summary was added after these tiers' artifacts were first produced, so its key has to
        # stay out of their bytes: records only pool with records built from the same definition text.
        for _scenario, definition in (standard, smoke):
            assert definition.census is None
            assert '"census"' not in definition_json(definition)

    def test_sdxl_batch_eight_drops_resolution_and_says_so(
        self,
        standard: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The batch-8 SDXL cell substitutes a resolution in its id."""
        cell = _cells_by_id(standard[1])["g3.batch8.sdxl.832x1216"]
        assert (cell.width, cell.height, cell.n_iter) == (832, 1216, 8)
        # Every other SDXL batch cell stays at the anchor resolution, so the batch axis is otherwise clean.
        others = [
            cell
            for cell in _cells_by_id(standard[1]).values()
            if cell.group == "g3" and cell.model == SDXL_A and cell.n_iter != 8
        ]
        assert all((cell.width, cell.height) == (1024, 1024) for cell in others)


@_REQUIRES_MANIFEST
class TestCensusCoverage:
    """The census claims to cover the kudos manifest's vocabularies, so the claim is the test subject."""

    def test_repeated_builds_are_byte_identical(self) -> None:
        """Two builds of the census render identical artifact text."""
        first = build_pricing_corpus_scenario("census")[1]
        second = build_pricing_corpus_scenario("census")[1]
        assert definition_json(first) == definition_json(second)

    def test_every_runnable_vocabulary_value_is_measured_at_least_three_times(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Every value the census does not exclude carries at least a cell's worth of measurements."""
        # One measurement of a value is an outlier waiting to be fitted; a replicated cell is the smallest
        # sample from which one bad reading can be rejected, and that is the floor the census promises.
        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        for axis, counts in summary.coverage.items():
            thin = {value: count for value, count in counts.items() if count < REPLICATES}
            assert not thin, f"{axis} values measured fewer than {REPLICATES} times: {thin}"

    def test_coverage_spans_the_manifest_vocabulary_exactly(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The coverage report names every manifest value except the ones excluded by name."""
        # This is the alarm for manifest growth: a vocabulary that gains a value the census does not sweep,
        # and does not exclude with a reason, fails here rather than shipping as a silent absence.
        from hordelib.kudos_training.manifest import default_manifest

        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        manifest = default_manifest()
        assert summary.manifest_version == manifest.manifest_version
        vocabularies = {
            feature.name: set(feature.vocabulary)
            for feature in manifest.features
            if getattr(feature, "vocabulary", None) is not None
        }
        excluded = {(exclusion.axis, exclusion.value) for exclusion in summary.exclusions if exclusion.kind == "value"}
        assert set(summary.coverage) == set(CENSUS_AXES)
        for axis in CENSUS_AXES:
            expected = {value for value in vocabularies[axis] if (axis, value) not in excluded}
            assert set(summary.coverage[axis]) == expected, f"{axis} coverage does not span its vocabulary"

    def test_every_exclusion_is_named_with_a_reason(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The step-ignoring sampler, the divergent pair, and the absent baselines are all declared."""
        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        assert all(exclusion.reason for exclusion in summary.exclusions)
        values = {(exclusion.axis, exclusion.value) for exclusion in summary.exclusions if exclusion.kind == "value"}
        pairs = {exclusion.value for exclusion in summary.exclusions if exclusion.kind == "pair"}
        assert ("sampler_name", "k_dpm_adaptive") in values
        assert ("baseline", "flux_1") in values
        assert "dpmpp_3m_sde+normal" in pairs

    def test_excluded_values_and_pairs_never_reach_a_cell(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """No cell carries an excluded sampler, or an excluded sampler and schedule together."""
        _scenario, definition = census
        for cell in definition.cells:
            assert cell.sampler_name not in _CENSUS_EXCLUDED_SAMPLERS
            assert (cell.sampler_name, cell.scheduler) not in _CENSUS_EXCLUDED_SAMPLER_SCHEDULE_PAIRS

    def test_the_job_count_stays_inside_the_budget(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The census fits its budget, and the artifact projects what running it will cost."""
        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        assert len(definition.jobs) <= CENSUS_JOB_BUDGET
        assert summary.projected_job_count == len(definition.jobs)
        assert summary.projected_runtime_seconds == pytest.approx(len(definition.jobs) * CENSUS_SECONDS_PER_JOB)
        # The conflation block takes what the sweeps leave, so a manifest that grew until nothing remained
        # would still build while quietly measuring no conflated conditions at all.
        assert summary.conflation_cell_count > 0
        measured = [cell for cell in definition.cells if cell.group != "warmup"]
        assert summary.sweep_cell_count + summary.conflation_cell_count == len(measured)

    def test_the_conflation_block_varies_the_axes_jointly(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Every runnable sampler and schedule also appears away from the anchor, alongside other moves."""
        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        conflated = [cell for cell in definition.cells if cell.group == "c10"]
        assert {cell.sampler_name for cell in conflated} == set(summary.coverage["sampler_name"])
        assert {cell.scheduler for cell in conflated} == set(summary.coverage["scheduler"])
        # The block samples the combination space rather than sweeping one axis, so several axes have to
        # sit off their anchor values within it.
        assert len({(cell.width, cell.height) for cell in conflated}) > 1
        assert len({cell.source_processing for cell in conflated}) > 1
        assert len({cell.model for cell in conflated}) > 1
        assert any(cell.post_processing for cell in conflated)
        assert any(cell.control_type for cell in conflated)

    def test_every_conflated_cell_is_servable(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """No drawn cell asks for a combination whose activation peak the card cannot hold."""
        # A cell that faults on memory measures nothing and leaves its records unpaired, so the draw is
        # bounded rather than trusted: one enlarging processor at most, and no batch behind one.
        _scenario, definition = census
        for cell in (cell for cell in definition.cells if cell.group == "c10"):
            enlarging = [value for value in cell.post_processing if value not in _CENSUS_NON_UPSCALING_POST_PROCESSORS]
            assert len(enlarging) <= 1, cell.cell_id
            if enlarging:
                assert cell.n_iter == 1, cell.cell_id
                assert cell.width * cell.height <= 1024 * 1024, cell.cell_id

    def test_control_cells_run_on_the_baseline_that_has_controlnet_weights(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Every control-carrying cell is an SD1.5 img2img job."""
        _scenario, definition = census
        control_cells = [cell for cell in definition.cells if cell.control_type is not None]
        assert control_cells
        for cell in control_cells:
            assert cell.model == SD15_A, cell.cell_id
            assert cell.source_processing == "img2img", cell.cell_id

    def test_the_sweeps_move_one_axis_at_a_time(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Each sweep group holds the anchor and offers a cell per vocabulary value.

        The sampler group offers one per value per trajectory length; the rest offer exactly one.
        """
        _scenario, definition = census
        summary = definition.census
        assert summary is not None
        by_group: dict[str, list[CorpusCell]] = {}
        for cell in definition.cells:
            by_group.setdefault(cell.group, []).append(cell)

        samplers = by_group["c1"]
        assert {cell.sampler_name for cell in samplers} == set(summary.coverage["sampler_name"])
        assert {cell.scheduler for cell in samplers} == {"karras"}
        schedules = by_group["c2"]
        assert {cell.scheduler for cell in schedules} == set(summary.coverage["scheduler"])
        assert {cell.sampler_name for cell in schedules} == {"k_euler"}
        assert {value for cell in by_group["c3"] for value in cell.post_processing} == set(
            summary.coverage["post_processing"],
        )
        assert {cell.control_type for cell in by_group["c4"]} == set(summary.coverage["control_type"]) - {"None"}
        assert {cell.source_processing for cell in by_group["c5"]} == set(summary.coverage["source_processing"]) - {
            "txt2img",
        }
        assert {cell.hires_fix for cell in by_group["c6"]} == {False, True}
        assert {len(cell.ti_names) for cell in by_group["c8"]} == {0, 1}

    def test_every_sampler_is_measured_at_two_trajectory_lengths(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """No sampler the census measures is confined to a single trajectory length."""
        # A sampler's cost is part per-trajectory-step and part per-job. Observed at one trajectory
        # length the two are a single number, and a fit can only price the sampler by borrowing another
        # sampler's slope, so the sweep carries the levels rather than leaving them to the conflation draw.
        _scenario, definition = census
        sweep_levels: dict[str, set[int]] = {}
        measured_levels: dict[str, set[int]] = {}
        for cell in definition.cells:
            if cell.group == "warmup":
                continue
            measured_levels.setdefault(cell.sampler_name, set()).add(cell.steps)
            if cell.group == "c1":
                sweep_levels.setdefault(cell.sampler_name, set()).add(cell.steps)

        assert len(_CENSUS_SAMPLER_TRAJECTORY_STEPS) >= 2
        assert sweep_levels
        assert set(sweep_levels) == set(measured_levels)
        for sampler, levels in sorted(sweep_levels.items()):
            assert levels == set(_CENSUS_SAMPLER_TRAJECTORY_STEPS), sampler
        thin = {sampler: sorted(levels) for sampler, levels in measured_levels.items() if len(levels) < 2}
        assert not thin, f"samplers measured at one trajectory length: {thin}"

    def test_lora_levels_pay_the_fetch_once(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The LoRA count levels are measured after a leading miss cell has cached every reference."""
        _scenario, definition = census
        cells = _cells_by_id(definition)
        levels = {len(cell.lora_version_ids) for cell in definition.cells if cell.group == "c7"}
        assert levels == {0, 1, 2, len(PRICING_CORPUS_LORA_VERSION_IDS)}
        first_use: dict[str, PricingCorpusJob] = {}
        for job in definition.jobs:
            for reference in cells[job.cell_id].lora_version_ids:
                first_use.setdefault(reference, job)
        assert set(first_use) == set(PRICING_CORPUS_LORA_VERSION_IDS)
        assert all(cells[job.cell_id].lora_role == "miss" for job in first_use.values())

    def test_the_ordering_machinery_is_the_standard_tiers(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """The census runs the same warmup, permutation and cold-load machinery as the standard tier."""
        scenario, definition = census
        assert definition.warmup_job_count == 6
        assert definition.shuffle_seeds == ["pc1-a", "pc1-b"]
        cells = _cells_by_id(definition)
        cold = [job for job in _measured_jobs(definition) if cells[job.cell_id].requires_model_switch]
        assert {cells[job.cell_id].model for job in cold} == {SDXL_A, SDXL_B, SD15_A}
        for job in cold:
            assert definition.jobs[job.position - 1].model != job.model
        assert scenario.total_image_jobs == len(definition.jobs)
        for cell in definition.cells:
            if cell.group == "warmup":
                continue
            replicates = sorted(job.replicate for job in definition.jobs if job.cell_id == cell.cell_id)
            assert replicates == [0, 1, 2], cell.cell_id

    def test_outpainting_jobs_expand_with_a_source_image_and_mask(
        self,
        census: tuple[Scenario, PricingCorpusDefinition],
    ) -> None:
        """Outpainting is a masked mode; without the mask its cost is not an outpaint's."""
        scenario, _definition = census
        outpainting = [job for job in scenario.expand_image_jobs() if str(job.source_processing) == "outpainting"]
        assert outpainting
        assert all(job.source_image for job in outpainting)
        assert all(job.source_mask for job in outpainting)
