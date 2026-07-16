"""The disaggregation-scoped post-processing co-residency margin lever selects the effective second-say margin.

The post-processing/sampling co-residency gate gives the measured device-free reading a second say on a static
miss, admitting only when the free (net of the reserve, sampling peak, and any pending chain reserve) clears a
fixed margin. These pin that ``pp_overlap_margin_mb_disaggregated`` overrides that margin only for a
disaggregation-class-eligible candidate, that a monolithic job always keeps the default, and that an unset
lever leaves today's behavior intact.
"""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.inference_scheduler import _PP_OVERLAP_MEASURED_MARGIN_MB
from tests.process_management.conftest import make_job_pop_response, make_mock_bridge_data
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_OVERRIDE_MB = 256.0


class TestPpOverlapMarginLever:
    """`_pp_overlap_margin_mb` resolves the second-say margin from the candidate's class eligibility and the lever."""

    def test_eligible_job_with_override_uses_the_override(self) -> None:
        """A class-eligible candidate with the lever set prices its second say against the override margin."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(pp_overlap_margin_mb_disaggregated=_OVERRIDE_MB),
        )
        scheduler._is_disaggregation_class_eligible = lambda _job: True
        job = make_job_pop_response("stable_diffusion")

        assert scheduler._pp_overlap_margin_mb(job) == _OVERRIDE_MB

    def test_ineligible_job_ignores_the_override(self) -> None:
        """A monolithic-path candidate keeps the default margin even when the lever is set."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(pp_overlap_margin_mb_disaggregated=_OVERRIDE_MB),
        )
        scheduler._is_disaggregation_class_eligible = lambda _job: False
        job = make_job_pop_response("stable_diffusion")

        assert scheduler._pp_overlap_margin_mb(job) == _PP_OVERLAP_MEASURED_MARGIN_MB

    def test_unset_lever_keeps_the_default_for_eligible_job(self) -> None:
        """An unset lever (read with explicit None handling, never bare truthiness) keeps the default margin."""
        scheduler = _make_inference_scheduler(bridge_data=make_mock_bridge_data())
        scheduler._is_disaggregation_class_eligible = lambda _job: True
        job = make_job_pop_response("stable_diffusion")

        assert scheduler._pp_overlap_margin_mb(job) == _PP_OVERLAP_MEASURED_MARGIN_MB
