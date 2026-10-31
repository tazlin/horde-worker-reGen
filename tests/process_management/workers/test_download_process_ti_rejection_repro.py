"""The download process's ad-hoc textual-inversion (TI) prefetch must reach LoRA-parity on terminal rejection.

A real :class:`HordeDownloadProcess` is driven through its real control/executor methods with a fake TI
manager standing in for hordelib (no network, no torch). The contract under test: when the TI manager can
report a terminal upstream rejection, the download process surfaces it as a non-retryable rejection outcome
carrying the reason, exactly as the LoRA path does, so the parent skips the file and dispatches the job rather
than faulting it. A fetch whose key the requested name does not resolve to is the same kind of rejection. A
companion case documents that one plain failure outcome names every requesting job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    HordeAuxPrefetchControlMessage,
)
from horde_worker_regen.process_management.workers.download_process import _TI_NAME_UNRESOLVED_REASON

from .test_download_process_aux_prefetch import (
    _JOB_A,
    _JOB_B,
    _collect_prefetch_outcomes,
    _drain_scheduler,
    _entry,
    _FakeTiManager,
    _make_process,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.messages import AuxPrefetchOutcome

_TI_TERMINAL_REASON = "embedding index refused the file type"


class _ReasonedRejectionTiManager(_FakeTiManager):
    """A TI fake whose plain fetch never lands but which can report a terminal rejection reason.

    Mirrors the LoRA fake's reason-returning surface: ``fetch_adhoc_ti_with_reason`` pairs the non-landing
    fetch with a terminal reason a permanent upstream refusal would carry. ``reasoned_calls`` records whether
    the download process consumed that variant.
    """

    def __init__(self, *, reason: str) -> None:
        super().__init__(present=set(), fetch_succeeds=False)
        self.reason = reason
        self.reasoned_calls: list[str] = []

    def fetch_adhoc_ti_with_reason(
        self,
        name: str,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[str | None, str | None]:
        self.reasoned_calls.append(name)
        return None, self.reason


def test_ti_terminal_rejection_surfaces_reason_not_bare_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminally-rejected TI reports ok False, not retryable, and carries the rejection reason string.

    This is LoRA parity: the reason tells the parent to skip the embedding and dispatch the job without it
    rather than fault every requesting job on a permanent upstream refusal.
    """
    ti = _ReasonedRejectionTiManager(reason=_TI_TERMINAL_REASON)
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("bad-emb", kind=AuxModelKind.TI, job=_JOB_A)]),
    )
    _drain_scheduler(process)

    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "bad-emb"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].retryable is False
    assert outcomes[0].rejection_reason == _TI_TERMINAL_REASON
    assert ti.reasoned_calls == ["bad-emb"]
    assert [str(j) for j in outcomes[0].requesting_job_ids] == [_JOB_A]


class _WrongKeyTiManager(_FakeTiManager):
    """A TI fake whose fetch lands a different embedding, so the requested name does not resolve."""

    def __init__(self, *, landed_key: str) -> None:
        super().__init__(present=set())
        self.landed_key = landed_key

    def fetch_adhoc_ti(
        self,
        name: str,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        self.fetch_calls.append(name)
        self.present.add(self.landed_key)
        return self.landed_key


def test_ti_fetch_returning_an_unresolvable_key_is_a_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A returned key the requested name does not resolve to is a terminal rejection, not a success.

    The job's readiness probe looks the TI up by the requested name. Reporting success here would leave that
    probe missing and queue the same fetch again.
    """
    ti = _WrongKeyTiManager(landed_key="negative_hand negative embedding")
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[_entry("Negative & Positive Embeddings By Stable Yogi", kind=AuxModelKind.TI, job=_JOB_A)],
        ),
    )
    _drain_scheduler(process)

    outcomes = _collect_prefetch_outcomes(process)
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].retryable is False
    assert outcomes[0].rejection_reason == _TI_NAME_UNRESOLVED_REASON
    assert ti.fetch_calls == ["Negative & Positive Embeddings By Stable Yogi"]


def test_ti_failure_outcome_names_every_requesting_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """One TI name requested by two jobs downloads once, and a single failure outcome names both jobs."""
    ti = _FakeTiManager(present=set(), fetch_succeeds=False)
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[
                _entry("emb-1", kind=AuxModelKind.TI, job=_JOB_A),
                _entry("emb-1", kind=AuxModelKind.TI, job=_JOB_B),
            ],
        ),
    )
    _drain_scheduler(process)

    assert ti.fetch_calls == ["emb-1"]
    outcomes: list[AuxPrefetchOutcome] = [o for o in _collect_prefetch_outcomes(process) if o.name == "emb-1"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert {str(j) for j in outcomes[0].requesting_job_ids} == {_JOB_A, _JOB_B}
