"""The disaggregation-scoped post-processing co-residency margin lever selects the effective second-say margin.

The post-processing/sampling co-residency gate gives the measured device-free reading a second say on a static
miss, admitting only when the free (net of the reserve, sampling peak, and any pending chain reserve) clears a
fixed margin. These pin that ``pp_overlap_margin_mb_disaggregated`` overrides that margin only for a
disaggregation-class-eligible candidate, that a monolithic job always keeps the default, and that an unset
lever leaves today's behavior intact.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.device_memory_source import (
    DeviceMemoryReading,
    DeviceMemorySourceKind,
    ReadingProvenance,
)
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


class TestMeasuredOverlapNeedsADeviceWideReading:
    """The measured second say spends room on the reading alone, so it is given only on a device-wide one."""

    _FREE_MB = 20000.0
    _TOTAL_MB = 24576.0

    @classmethod
    def _admits(cls, provenance: ReadingProvenance) -> bool:
        scheduler = _make_inference_scheduler(bridge_data=make_mock_bridge_data())
        reading = DeviceMemoryReading(
            free_mb=cls._FREE_MB,
            total_mb=cls._TOTAL_MB,
            source=DeviceMemorySourceKind.CHILD_REPORT,
            provenance=provenance,
        )
        scheduler.set_device_reading_provider(lambda _device_index: reading)
        return scheduler._pp_overlap_measured_admits(
            sampling_peak_mb=4000.0,
            pending_pp_reserve_mb=0.0,
            device_index=None,
            margin_mb=_PP_OVERLAP_MEASURED_MARGIN_MB,
        )

    def test_a_device_wide_reading_with_room_admits(self) -> None:
        """Ample device-wide free admits the overlap the static gate withheld."""
        assert self._admits(ReadingProvenance.DEVICE_WIDE) is True

    def test_the_same_figure_from_a_process_local_reading_does_not(self) -> None:
        """A process-local figure can hide a sibling's memory, so the static verdict stands."""
        assert self._admits(ReadingProvenance.PROCESS_LOCAL) is False

    def test_the_same_figure_from_a_host_memory_reading_does_not(self) -> None:
        """A host-memory figure is system RAM, so the static verdict stands."""
        assert self._admits(ReadingProvenance.HOST_MEMORY) is False
