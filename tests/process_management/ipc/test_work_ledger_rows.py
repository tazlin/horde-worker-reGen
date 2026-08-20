"""Tests for what a work-ledger row states about its job, and the order the rows arrive in.

The ledger is the operator's per-job view. A row therefore has to carry what decides how much work its
job actually is (resolution, steps, batch count, and what the sampler asks the model for per step), and
the rows have to arrive in pop order, which is not the order the tracker holds them in once jobs start
advancing through stages.
"""

from __future__ import annotations

import pytest
from horde_sdk.generation_parameters.image.constraints import SAMPLER_CONSTRAINTS
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SDE = SAMPLER_CONSTRAINTS[KNOWN_IMAGE_SAMPLERS.k_dpmpp_sde]


def test_sampler_summary_states_the_order_and_the_measured_cost() -> None:
    """A fixed-rate sampler reports the evaluations one step costs and its measured ratio to k_euler."""
    summary = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "stable_diffusion_xl")

    assert summary is not None
    assert summary.name == "k_dpmpp_sde"
    assert summary.work_per_step == 2
    assert summary.adaptive is False
    assert summary.cost_ratio == _SDE.measured_cost_ratio_sdxl


def test_sampler_summary_prices_a_512_class_baseline_from_the_512_class_column() -> None:
    """The two published ratios differ for the same sampler, so the baseline decides which one is read."""
    small = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "stable_diffusion_1")
    large = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "flux_1")

    assert small is not None
    assert large is not None
    assert small.cost_ratio == _SDE.measured_cost_ratio_sd15
    assert large.cost_ratio == _SDE.measured_cost_ratio_sdxl
    assert small.cost_ratio != large.cost_ratio


def test_sampler_summary_marks_an_adaptive_sampler_and_claims_no_cost() -> None:
    """An adaptive sampler picks its own iteration count, so neither an order nor a ratio applies."""
    summary = HordeWorkerProcessManager._sampler_summary("k_dpm_adaptive", "stable_diffusion_xl")

    assert summary is not None
    assert summary.adaptive is True
    assert summary.work_per_step is None
    assert summary.cost_ratio is None


def test_sampler_summary_names_a_sampler_the_sdk_does_not_know() -> None:
    """The horde asked for it, so the operator sees it; the cost fields stay unstated rather than guessed."""
    summary = HordeWorkerProcessManager._sampler_summary("k_not_a_real_sampler", "stable_diffusion_xl")

    assert summary is not None
    assert summary.name == "k_not_a_real_sampler"
    assert summary.work_per_step is None
    assert summary.adaptive is False
    assert summary.cost_ratio is None


def test_sampler_summary_is_absent_without_a_sampler() -> None:
    """An alchemy form or a record built without a pop payload names no sampler at all."""
    assert HordeWorkerProcessManager._sampler_summary(None, "stable_diffusion_xl") is None


@pytest.mark.asyncio
async def test_work_ledger_row_carries_batch_count_and_sampler() -> None:
    """A row states the batch and the sampler, so two same-size jobs are not read as the same work."""
    manager = make_testable_process_manager()
    job = make_job_pop_response(width=832, height=1216, ddim_steps=28, n_iter=4, sampler_name="k_heun")
    await track_popped_job_async(manager._job_tracker, job)

    row = manager._build_work_ledger([])[0]

    assert row.batch_size == 4
    assert row.sampler is not None
    assert row.sampler.name == "k_heun"
    assert row.sampler.work_per_step == 2


@pytest.mark.asyncio
async def test_work_ledger_rows_arrive_in_pop_order() -> None:
    """Rows follow pop order even after a later-popped job advances a stage ahead of an earlier one.

    The tracker orders by stage entry, so the job that starts inference first would otherwise jump the
    row it was popped behind, and the ledger would disagree with the pop-order column it carries.
    """
    manager = make_testable_process_manager()
    jobs = [make_job_pop_response(model=f"model_{index}") for index in range(3)]
    for job in jobs:
        await track_popped_job_async(manager._job_tracker, job)
    await mark_job_in_progress_async(manager._job_tracker, jobs[2])

    rows = manager._build_work_ledger([])

    assert [row.queue_order for row in rows] == [1, 2, 3]
    assert [row.job_id for row in rows] == [str(job.id_) for job in jobs]
