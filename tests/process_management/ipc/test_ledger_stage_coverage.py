"""Coverage guard for the job-stage to work-ledger-stage mapping used by the supervisor snapshot.

``_ledger_stage`` is indexed by a dict literal, so any :class:`JobStage` it does not map raises ``KeyError``
when the snapshot is assembled for a job in that stage, dropping the whole supervisor snapshot for that tick.
Iterating every stage member here means a future stage addition fails this test loudly rather than silently
regressing the live view.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkLedgerStage
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


@pytest.mark.parametrize("stage", list(JobStage))
def test_every_job_stage_maps_to_a_ledger_stage(stage: JobStage) -> None:
    """Every JobStage must map to a WorkLedgerStage without raising, so snapshot assembly never fails."""
    assert isinstance(HordeWorkerProcessManager._ledger_stage(stage), WorkLedgerStage)


def test_disaggregation_decoding_reads_as_inference() -> None:
    """A disaggregated job still decoding on the image lane surfaces as INFERENCE, not a dropped snapshot."""
    assert HordeWorkerProcessManager._ledger_stage(JobStage.DISAGGREGATION_DECODING) == WorkLedgerStage.INFERENCE
