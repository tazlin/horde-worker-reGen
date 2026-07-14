"""The download process's side of the ad-hoc auxiliary (LoRA/TI) prefetch: fetch, short-circuit, dedup, pins.

A real :class:`HordeDownloadProcess` is driven through its real control/executor methods with fake LoRA/TI
managers standing in for hordelib (no network, no torch): an already-present entry short-circuits to an
immediate success outcome, a missing entry is fetched once, two jobs sharing a file dedup to one download yet
both are named in the outcome, a fetch failure reports a failure outcome, and the pin set is applied to both
managers.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from horde_sdk.ai_horde_api.fields import GenerationID
from hordelib.model_manager.lora import LoRaRejectionReason

from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxModelRef,
    AuxPrefetchEntry,
    AuxPrefetchOutcome,
    HordeAuxPrefetchControlMessage,
    HordeAuxPrefetchResultMessage,
)
from horde_worker_regen.process_management.models.download_scheduler import DownloadKind, DownloadTask

if TYPE_CHECKING:
    from horde_worker_regen.process_management.workers.download_process import HordeDownloadProcess


class _FakeLoraManager:
    """A LoRA manager stand-in: an in-memory present-set plus recorded fetch/pin calls."""

    def __init__(
        self,
        present: set[str] | None = None,
        *,
        fetch_succeeds: bool = True,
        reject_reason: LoRaRejectionReason | None = None,
    ) -> None:
        self.present = set(present or set())
        self.fetch_succeeds = fetch_succeeds
        # A terminal ad-hoc rejection (invalid/too-large/NSFW) the fetch reports when the file does not land,
        # so a test can drive the skip-the-LoRA-and-proceed path rather than a plain retryable failure.
        self.reject_reason = reject_reason
        self.fetch_calls: list[tuple[str, bool]] = []
        self.pins: set[str] | None = None

    def is_lora_available(self, name: str, is_version: bool = False) -> bool:
        return name in self.present

    def fetch_adhoc_lora(
        self,
        name: str,
        timeout: int | None = 45,
        is_version: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        self.fetch_calls.append((name, is_version))
        if not self.fetch_succeeds:
            return None
        self.present.add(name)
        return name

    def fetch_adhoc_lora_with_reason(
        self,
        name: str,
        timeout: int | None = 45,
        is_version: bool = False,
        job_context: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[str | None, LoRaRejectionReason | None]:
        # The download process calls this reason-returning variant; delegate to the recorder so subclasses
        # (timeout-then-present, disk-guard) keep their behaviour, and pair a non-landing fetch with the
        # configured terminal rejection (None models a transient failure the parent may still retry).
        key = self.fetch_adhoc_lora(name, timeout=timeout, is_version=is_version, progress_callback=progress_callback)
        if key is not None:
            return key, None
        return None, self.reject_reason

    def fuzzy_find_lora_key(self, name: str) -> str | None:
        return name if name in self.present else None

    def set_eviction_pins(self, keys: set[str]) -> None:
        self.pins = set(keys)


class _FakeTiManager:
    """A textual-inversion manager stand-in mirroring the LoRA fake."""

    def __init__(
        self,
        present: set[str] | None = None,
        *,
        fetch_succeeds: bool = True,
        reject_reason: str | None = None,
    ) -> None:
        self.present = set(present or set())
        self.fetch_succeeds = fetch_succeeds
        # A terminal ad-hoc rejection (permanent upstream refusal/bad metadata) the fetch reports when the file
        # does not land, so a test can drive the skip-the-TI-and-proceed path rather than a plain retryable
        # failure. None models a transient failure the parent may still retry.
        self.reject_reason = reject_reason
        self.fetch_calls: list[str] = []
        self.pins: set[str] | None = None

    def fuzzy_find_ti_key(self, name: str) -> str | None:
        return name if name in self.present else None

    def fetch_adhoc_ti(
        self,
        name: str,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        self.fetch_calls.append(name)
        if not self.fetch_succeeds:
            return None
        self.present.add(name)
        return name

    def fetch_adhoc_ti_with_reason(
        self,
        name: str,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[str | None, str | None]:
        # The download process calls this reason-returning variant; delegate to the recorder so subclasses keep
        # their behaviour, and pair a non-landing fetch with the configured terminal rejection.
        key = self.fetch_adhoc_ti(name, timeout=timeout, progress_callback=progress_callback)
        if key is not None:
            return key, None
        return None, self.reject_reason

    def set_eviction_pins(self, keys: set[str]) -> None:
        self.pins = set(keys)


def _make_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lora: _FakeLoraManager | None,
    ti: _FakeTiManager | None,
) -> HordeDownloadProcess:
    from horde_worker_regen.process_management.workers.download_process import HordeDownloadProcess

    fake_manager = SimpleNamespace(
        compvis=SimpleNamespace(available_models=set()),
        lora=lora,
        ti=ti,
        gfpgan=None,
        esrgan=None,
        codeformer=None,
        miscellaneous=None,
        controlnet=None,
        controlnet_annotator=None,
    )
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = SimpleNamespace(manager=fake_manager)  # type: ignore[attr-defined]
    hordelib_stub = sys.modules.get("hordelib") or types.ModuleType("hordelib")
    hordelib_stub.api = fake_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib", hordelib_stub)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)

    ctx = mp.get_context("spawn")
    _parent_conn, child_conn = ctx.Pipe()
    process = HordeDownloadProcess(
        process_id=9000,
        process_message_queue=ctx.Queue(),
        pipe_connection=child_conn,
        disk_lock=ctx.Lock(),
        download_bandwidth_semaphore=ctx.Semaphore(1),
        process_launch_identifier=1,
    )
    process._safety_present = True
    process._safety_ensured = True
    process._aux_enqueued = True
    return process


def _current_fake_manager() -> SimpleNamespace:
    """Return the monkeypatched SharedModelManager.manager namespace so a test can toggle manager presence."""
    import hordelib.api

    return hordelib.api.SharedModelManager.manager  # type: ignore[attr-defined,no-any-return]


def _drain_scheduler(process: HordeDownloadProcess, *, max_tasks: int = 20) -> None:
    """Run every currently-admissible scheduler task to completion inline (no executor threads)."""
    for _ in range(max_tasks):
        task = process._scheduler.acquire(timeout=0.0)
        if task is None:
            return
        try:
            process._run_task(task)
        finally:
            process._scheduler.release(task)


def _collect_prefetch_outcomes(process: HordeDownloadProcess) -> list[AuxPrefetchOutcome]:
    """Drain the process message queue and return all aux-prefetch outcomes it emitted."""
    outcomes: list[AuxPrefetchOutcome] = []
    while True:
        try:
            message = process.process_message_queue.get(timeout=0.5)
        except queue.Empty:
            break
        if isinstance(message, HordeAuxPrefetchResultMessage):
            outcomes.extend(message.outcomes)
    return outcomes


def _entry(
    name: str, *, kind: AuxModelKind = AuxModelKind.LORA, is_version: bool = False, job: str
) -> AuxPrefetchEntry:
    return AuxPrefetchEntry(kind=kind, name=name, is_version=is_version, requesting_job_id=GenerationID(job))


_JOB_A = "11111111-1111-1111-1111-111111111111"
_JOB_B = "22222222-2222-2222-2222-222222222222"


def test_present_lora_short_circuits_without_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-present LoRA reports an immediate success outcome and never calls the CivitAI fetch."""
    lora = _FakeLoraManager(present={"styleA"})
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("styleA", job=_JOB_A)]),
    )

    assert lora.fetch_calls == []
    assert process._scheduler.has_work() is False
    outcomes = _collect_prefetch_outcomes(process)
    assert len(outcomes) == 1
    assert outcomes[0].ok is True
    assert outcomes[0].name == "styleA"
    assert [str(j) for j in outcomes[0].requesting_job_ids] == [_JOB_A]


def test_missing_lora_is_fetched_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing LoRA is fetched once and reported as a success once it lands."""
    lora = _FakeLoraManager(present=set())
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("styleA", job=_JOB_A)]),
    )
    assert process._scheduler.has_work() is True
    _drain_scheduler(process)

    assert lora.fetch_calls == [("styleA", False)]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "styleA"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is True


def test_two_jobs_sharing_a_lora_dedup_to_one_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two jobs requesting the same missing LoRA download it once, and both are named in the outcome."""
    lora = _FakeLoraManager(present=set())
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[_entry("shared", job=_JOB_A), _entry("shared", job=_JOB_B)],
        ),
    )
    _drain_scheduler(process)

    assert lora.fetch_calls == [("shared", False)]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "shared"]
    assert len(outcomes) == 1
    assert {str(j) for j in outcomes[0].requesting_job_ids} == {_JOB_A, _JOB_B}


def test_failed_fetch_reports_failure_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LoRA that cannot be fetched reports a failure outcome for its requesting job."""
    lora = _FakeLoraManager(present=set(), fetch_succeeds=False)
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("missing", job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert lora.fetch_calls == [("missing", False)]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "missing"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    # A plain (non-rejection) fetch failure stays retryable within the parent's per-job deadline and carries
    # no rejection reason, so the parent faults-and-retries rather than skipping the LoRA.
    assert outcomes[0].retryable is True
    assert outcomes[0].rejection_reason is None
    assert [str(j) for j in outcomes[0].requesting_job_ids] == [_JOB_A]


def test_terminal_rejection_reports_non_retryable_outcome_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminally-rejected LoRA reports ok False, not retryable, and carries the rejection reason string.

    This is what tells the parent to skip the LoRA and dispatch the job without it rather than fault it.
    """
    lora = _FakeLoraManager(present=set(), fetch_succeeds=False, reject_reason=LoRaRejectionReason.INVALID)
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("badlora", job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert lora.fetch_calls == [("badlora", False)]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "badlora"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].retryable is False
    assert outcomes[0].rejection_reason == str(LoRaRejectionReason.INVALID)
    assert [str(j) for j in outcomes[0].requesting_job_ids] == [_JOB_A]


def test_present_ti_short_circuits_without_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-present textual inversion reports success immediately and never fetches."""
    ti = _FakeTiManager(present={"emb-1"})
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("emb-1", kind=AuxModelKind.TI, job=_JOB_A)]),
    )

    assert ti.fetch_calls == []
    outcomes = _collect_prefetch_outcomes(process)
    assert len(outcomes) == 1
    assert outcomes[0].ok is True
    assert outcomes[0].kind is AuxModelKind.TI


def test_missing_ti_is_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing textual inversion is fetched once and reported as a success."""
    ti = _FakeTiManager(present=set())
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("emb-1", kind=AuxModelKind.TI, job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert ti.fetch_calls == ["emb-1"]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "emb-1"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is True


def test_pins_are_applied_to_both_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin set resolves to manager keys and is applied to both the LoRA and TI managers."""
    lora = _FakeLoraManager(present={"pinnedL"})
    ti = _FakeTiManager(present={"pinnedT"})
    process = _make_process(monkeypatch, lora=lora, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[],
            pins=[
                AuxModelRef(kind=AuxModelKind.LORA, name="pinnedL"),
                AuxModelRef(kind=AuxModelKind.TI, name="pinnedT"),
                AuxModelRef(kind=AuxModelKind.LORA, name="not-on-disk"),
            ],
        ),
    )

    assert lora.pins == {"pinnedL"}
    assert ti.pins == {"pinnedT"}


def test_prefetch_before_managers_load_emits_no_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request arriving before the managers load stashes the entry and emits no (failure) outcome.

    Faulting here would self-inflict a job fault plus a LoRA-download pop backoff on the parent for a
    downloader still in its startup window, so the entry must be held, not rejected.
    """
    process = _make_process(monkeypatch, lora=None, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("styleA", job=_JOB_A)]),
    )

    assert _collect_prefetch_outcomes(process) == []
    assert process._scheduler.has_work() is False
    assert [e.name for e in process._pending_aux_entries] == ["styleA"]


def test_replay_after_load_produces_normal_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the managers load, replay stages the stashed entries: present ones ok, missing ones fetched."""
    process = _make_process(monkeypatch, lora=None, ti=None)
    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[_entry("present", job=_JOB_A), _entry("missing", job=_JOB_B)],
        ),
    )
    assert _collect_prefetch_outcomes(process) == []

    lora = _FakeLoraManager(present={"present"})
    _current_fake_manager().lora = lora

    process._replay_stashed_aux_prefetch()
    _drain_scheduler(process)

    assert lora.fetch_calls == [("missing", False)]
    outcomes = {o.name: o for o in _collect_prefetch_outcomes(process)}
    assert outcomes["present"].ok is True
    assert outcomes["missing"].ok is True
    assert process._pending_aux_entries == []


def test_mixed_request_processes_loaded_kind_and_stashes_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one manager loaded and one absent, the loaded kind resolves now and the absent kind is stashed."""
    ti = _FakeTiManager(present={"emb-1"})
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[
                _entry("styleA", kind=AuxModelKind.LORA, job=_JOB_A),
                _entry("emb-1", kind=AuxModelKind.TI, job=_JOB_B),
            ],
        ),
    )

    outcomes = _collect_prefetch_outcomes(process)
    assert len(outcomes) == 1
    assert outcomes[0].kind is AuxModelKind.TI
    assert outcomes[0].ok is True
    assert [e.name for e in process._pending_aux_entries] == ["styleA"]

    lora = _FakeLoraManager(present=set())
    _current_fake_manager().lora = lora
    process._replay_stashed_aux_prefetch()
    _drain_scheduler(process)

    assert lora.fetch_calls == [("styleA", False)]
    replayed = [o for o in _collect_prefetch_outcomes(process) if o.name == "styleA"]
    assert len(replayed) == 1
    assert replayed[0].ok is True


class _FakeLoraManagerWithDiskGuard(_FakeLoraManager):
    """A LoRA fake that also exposes the disk-guard surface :func:`constrain_lora_cache_to_disk` drives.

    Records eviction calls so a test can assert the floor guard ran without evicting when disk is healthy.
    """

    def __init__(self, present: set[str] | None, *, folder: str, fetch_succeeds: bool = True) -> None:
        super().__init__(present, fetch_succeeds=fetch_succeeds)
        self.model_folder_path = folder
        self.max_adhoc_disk = 10_240
        self.adhoc_cache_mb = 0.0
        self.evicted = 0

    def calculate_adhoc_cache(self) -> float:
        return self.adhoc_cache_mb

    def find_oldest_adhoc_entry(self) -> object | None:
        # The installed hordelib's real implementation skips eviction-pinned entries here, so the floor
        # guard never evicts a file another still-pending job needs; returning None models a cache with
        # nothing (unpinned) left to reclaim.
        return None

    def delete_oldest(self) -> None:
        self.evicted += 1

    def save_reference_to_disk(self) -> None:
        return None


def test_adhoc_lora_fetch_proceeds_when_disk_floor_satisfied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    """An ad-hoc LoRA fetch enforces the free-disk floor first, then downloads when the floor is satisfied.

    The floor guard runs against the manager's own room/eviction surface before the fetch. With ample free
    space on the cache volume it evicts nothing and the download proceeds. Eviction (when a low disk forces
    it) goes through ``find_oldest_adhoc_entry``/``delete_oldest``, which honour the eviction pins the
    download process applies via ``set_eviction_pins``, so a pinned file another pending job needs is never
    evicted to make room. That pin-honouring seam lives in the hordelib manager, modelled here by the fake
    returning no unpinned evictable entry.
    """
    lora = _FakeLoraManagerWithDiskGuard(present=set(), folder=str(tmp_path))
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("styleA", job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert lora.fetch_calls == [("styleA", False)]
    assert lora.evicted == 0
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "styleA"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is True


class _TimeoutThenAvailableLoraManager(_FakeLoraManager):
    """A LoRA fake whose fetch raises ``TimeoutError`` after the file has landed at the wait bound.

    Models hordelib's fetch: its internal wait can raise even when the file arrived exactly at the timeout, so
    the process must fall through to the availability re-check rather than report a failure for a present file.
    """

    def fetch_adhoc_lora(
        self,
        name: str,
        timeout: int | None = 45,
        is_version: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        self.fetch_calls.append((name, is_version))
        self.present.add(name)
        raise TimeoutError("fetch wait exceeded its bound")


class _TimeoutThenAvailableTiManager(_FakeTiManager):
    """A textual-inversion fake mirroring the LoRA timeout-then-present manager."""

    def fetch_adhoc_ti(
        self,
        name: str,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        self.fetch_calls.append(name)
        self.present.add(name)
        raise TimeoutError("fetch wait exceeded its bound")


def test_lora_fetch_timeout_with_landed_file_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LoRA fetch that raises TimeoutError but whose file then probes present reports a success outcome."""
    lora = _TimeoutThenAvailableLoraManager(present=set())
    process = _make_process(monkeypatch, lora=lora, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("styleA", job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert lora.fetch_calls == [("styleA", False)]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "styleA"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is True
    assert [str(j) for j in outcomes[0].requesting_job_ids] == [_JOB_A]


def test_ti_fetch_timeout_with_landed_file_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TI fetch that raises TimeoutError but whose embedding then probes present reports a success outcome."""
    ti = _TimeoutThenAvailableTiManager(present=set())
    process = _make_process(monkeypatch, lora=None, ti=ti)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(entries=[_entry("emb-1", kind=AuxModelKind.TI, job=_JOB_A)]),
    )
    _drain_scheduler(process)

    assert ti.fetch_calls == ["emb-1"]
    outcomes = [o for o in _collect_prefetch_outcomes(process) if o.name == "emb-1"]
    assert len(outcomes) == 1
    assert outcomes[0].ok is True


def test_stashed_pins_applied_on_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pin set that arrived before the managers loaded is applied to both managers once they load."""
    process = _make_process(monkeypatch, lora=None, ti=None)

    process._handle_aux_prefetch_request(
        HordeAuxPrefetchControlMessage(
            entries=[],
            pins=[
                AuxModelRef(kind=AuxModelKind.LORA, name="pinnedL"),
                AuxModelRef(kind=AuxModelKind.TI, name="pinnedT"),
            ],
        ),
    )

    lora = _FakeLoraManager(present={"pinnedL"})
    ti = _FakeTiManager(present={"pinnedT"})
    manager = _current_fake_manager()
    manager.lora = lora
    manager.ti = ti

    process._replay_stashed_aux_prefetch()

    assert lora.pins == {"pinnedL"}
    assert ti.pins == {"pinnedT"}


class TestPerFileRetryRequeue:
    """A failed per-file fetch must actually retry.

    The failing run still holds the task's in-flight dedup slot when the retry is recorded, so an
    immediate re-enqueue would be silently deduplicated away.
    """

    @staticmethod
    def _image_task() -> DownloadTask:
        return DownloadTask(
            kind=DownloadKind.IMAGE_MODEL,
            model_name="some model",
            host="example.com",
            feature="image model",
        )

    def test_retry_survives_the_in_flight_dedup_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The retry lands back in the queue after its backoff even though it was recorded mid-run."""
        process = _make_process(monkeypatch, lora=None, ti=None)
        process._scheduler.enqueue(self._image_task())
        claimed = process._scheduler.acquire(timeout=0.5)
        assert claimed is not None

        process._maybe_retry(claimed, "connection reset")
        # Recorded, not enqueued: an enqueue here would be dropped against the still-held in-flight slot.
        assert process._scheduler.pending_snapshot() == []
        process._scheduler.release(claimed)

        # Before the backoff elapses the drain is a no-op.
        process._drain_due_retries()
        assert process._scheduler.pending_snapshot() == []

        with process._lock:
            process._pending_retries = [(0.0, task) for _due, task in process._pending_retries]
        process._drain_due_retries()
        assert [task.model_name for task in process._scheduler.pending_snapshot()] == ["some model"]

    def test_stale_image_retry_dropped_when_model_removed_from_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deferred retry for a model no longer in config is dropped, mirroring the queue prune."""
        process = _make_process(monkeypatch, lora=None, ti=None)
        process._scheduler.enqueue(self._image_task())
        claimed = process._scheduler.acquire(timeout=0.5)
        assert claimed is not None
        process._maybe_retry(claimed, "connection reset")
        process._scheduler.release(claimed)

        with process._lock:
            process._pending_retries = [(0.0, task) for _due, task in process._pending_retries]
            process._desired_image_models = {"a different model"}
        process._drain_due_retries()
        assert process._scheduler.pending_snapshot() == []
