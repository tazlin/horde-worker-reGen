"""Unit + real-download tests for the shared model-download core.

Covers the two responsibilities the core centralizes: the per-chunk pause/rate-limit pacing
(:class:`ChunkPacer`) and the dedup + validate/retry download loop (:func:`ensure_models_present`,
:func:`download_one_model`). The download loop runs against a real loopback server so "downloads happen"
is proven, not mocked.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.model_download_core import (
    ChunkPacer,
    DownloadAborted,
    ModelProgress,
    download_one_model,
    ensure_models_present,
)
from tests.download_test_helpers import FakeModelServer, RealDownloadCompVis, deterministic_bytes


def _record(name: str, file_name: str, base_url: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        description="core test record",
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=file_name, file_url=f"{base_url}/{file_name}")],
        ),
    )


# --- ChunkPacer: pause and rate-limit are enforced live -----------------------------------------------


def test_chunk_pacer_blocks_while_paused_then_resumes() -> None:
    """step() blocks in the pause loop until the pause flag clears, then returns progress."""
    resume = threading.Event()
    pacer = ChunkPacer()
    result: list[ModelProgress] = []

    def run() -> None:
        result.append(
            pacer.step(
                512,
                1024,
                is_paused=lambda: not resume.is_set(),
                rate_limit_kbps=lambda: None,
                should_abort=lambda: False,
            ),
        )

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "step should still be blocked while paused"
    assert not result

    resume.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert result and result[0].downloaded_bytes == 512


def test_chunk_pacer_aborts_when_signalled() -> None:
    """step() raises DownloadAborted when should_abort is true (e.g. shutdown)."""
    pacer = ChunkPacer()
    with pytest.raises(DownloadAborted):
        pacer.step(0, 100, is_paused=lambda: False, rate_limit_kbps=lambda: None, should_abort=lambda: True)


def test_chunk_pacer_rate_limit_paces_a_chunk() -> None:
    """A capped rate sleeps long enough to honour the cap for the chunk's byte delta."""
    pacer = ChunkPacer()
    started = time.time()
    # 10240 bytes at 50 kB/s => ~0.2s of pacing on this first chunk.
    pacer.step(10240, 10240, is_paused=lambda: False, rate_limit_kbps=lambda: 50, should_abort=lambda: False)
    assert time.time() - started >= 0.15


# --- download_one_model: validate + retry-once --------------------------------------------------------


class _FlakyManager:
    """A manager whose first validate fails, forcing exactly one re-download."""

    def __init__(self) -> None:
        self.download_calls = 0
        self._validate_results = [False]  # first validate fails; absent => assume valid

    def is_model_available(self, model_name: str) -> bool:
        return False

    def download_model(self, model_name: str, *, callback: Callable[[int, int], None] | None = None) -> bool:
        self.download_calls += 1
        return True

    def validate_model(self, model_name: str, skip_checksum: bool = False) -> bool:
        return self._validate_results.pop(0) if self._validate_results else True


def test_download_one_model_redownloads_once_on_invalid() -> None:
    """A failed checksum triggers a single re-download; a valid one does not."""
    flaky = _FlakyManager()
    assert download_one_model(flaky, "M") is True
    assert flaky.download_calls == 2  # initial + one forced re-download


# --- ensure_models_present: dedup + real downloads ----------------------------------------------------


def test_ensure_downloads_missing_and_skips_present(tmp_path: Path) -> None:
    """Already-present models are skipped; missing ones are fetched for real, with progress callbacks."""
    server = FakeModelServer()
    server.add("present.safetensors", deterministic_bytes("present", 2048))
    server.add("missing.safetensors", deterministic_bytes("missing", 4096))
    server.start()
    try:
        records = {
            "Present": _record("Present", "present.safetensors", server.base_url),
            "Missing": _record("Missing", "missing.safetensors", server.base_url),
        }
        # Pre-place the present model so it is on disk before the run.
        present_dir = tmp_path / "compvis"
        present_dir.mkdir(parents=True)
        (present_dir / "present.safetensors").write_bytes(deterministic_bytes("present", 2048))

        compvis = RealDownloadCompVis(tmp_path, records)
        started: list[str] = []
        finished: list[tuple[str, bool]] = []
        progressed: list[str] = []

        outcome = ensure_models_present(
            compvis,
            ["Present", "Missing"],
            on_model_start=lambda name, _i, _t: started.append(name),
            on_progress=lambda name, _i, _t, _p: progressed.append(name),
            on_model_finish=lambda name, _i, _t, ok: finished.append((name, ok)),
        )

        assert outcome.present == ["Present"]
        assert outcome.downloaded == 1
        assert outcome.failed == 0
        assert (tmp_path / "compvis" / "missing.safetensors").exists()
        # The present model was never requested; only the missing one's file was fetched.
        assert server.hits["/present.safetensors"] == 0
        assert server.hits["/missing.safetensors"] == 1
        assert started == ["Missing"]
        assert finished == [("Missing", True)]
        assert progressed and set(progressed) == {"Missing"}
    finally:
        server.stop()


def test_ensure_reports_failure_for_unservable_model(tmp_path: Path) -> None:
    """A model the server cannot provide is counted as failed without aborting the others."""
    server = FakeModelServer()
    server.add("ok.safetensors", deterministic_bytes("ok", 1024))
    server.start()
    try:
        records = {
            "Ok": _record("Ok", "ok.safetensors", server.base_url),
            "Gone": _record("Gone", "gone.safetensors", server.base_url),  # 404 on the server
        }
        compvis = RealDownloadCompVis(tmp_path, records)

        # RealDownloadCompVis raises on a 404; wrap it so a fetch failure is a False, not an exception,
        # mirroring how the real compvis.download_model reports a failed fetch.
        original = compvis.download_model

        def _safe_download(model_name: str, *, callback: Callable[[int, int], None] | None = None) -> bool:
            try:
                return original(model_name, callback=callback)
            except Exception:  # noqa: BLE001 - a fetch error is a failed download, not a crash
                return False

        compvis.download_model = _safe_download  # type: ignore[method-assign]

        outcome = ensure_models_present(compvis, ["Ok", "Gone"])
        assert outcome.downloaded == 1
        assert outcome.failed == 1
        assert outcome.failures == ["Gone"]
    finally:
        server.stop()
