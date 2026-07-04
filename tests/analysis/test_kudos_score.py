"""Tests for the server-parity kudos scorer used by session simulations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from horde_worker_regen.analysis.kudos_score import (
    BASIS_PAYLOAD,
    KNOWN_CONTROL_TYPES,
    KNOWN_POST_PROCESSORS,
    KNOWN_SAMPLERS,
    KNOWN_SOURCE_PROCESSING,
    KudosModelScorer,
    KudosPayload,
    payload_from_job_record,
    score_session,
)
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord

_CKPT_ENV = "AI_HORDE_KUDOS_MODEL_CKPT"


def _checkpoint_path() -> Path | None:
    raw = os.environ.get(_CKPT_ENV)
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_file() else None


class TestFeatureEncoding:
    """The feature layout must match the server's ``payload_to_tensor`` exactly."""

    def test_vector_is_47_dims(self) -> None:
        """The checkpoint expects exactly 47 input features."""
        assert len(BASIS_PAYLOAD.to_feature_vector()) == 47

    def test_vocabulary_sizes(self) -> None:
        """The one-hot vocabularies match the server's sizes."""
        assert len(KNOWN_SAMPLERS) == 15
        assert len(KNOWN_CONTROL_TYPES) == 10
        assert len(KNOWN_SOURCE_PROCESSING) == 4
        assert len(KNOWN_POST_PROCESSORS) == 8

    def test_basis_floats(self) -> None:
        """The basis payload's continuous features encode to the server's values."""
        floats = BASIS_PAYLOAD.to_feature_vector()[:10]
        assert floats == [0.5, 0.5, 0.5, 0.25, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    def test_unknown_sampler_falls_back_to_k_euler(self) -> None:
        """Unknown samplers price as k_euler, as on the server."""
        payload = KudosPayload(width=512, height=512, steps=50, sampler_name="not_a_sampler")
        offset = 10 + KNOWN_SAMPLERS.index("k_euler")
        assert payload.to_feature_vector()[offset] == 1.0

    def test_control_type_forces_source_semantics(self) -> None:
        """Control jobs charge control strength and reset denoising."""
        # A control job with a source image charges control_strength and resets denoising to 1.0,
        # mirroring the server's precedence.
        payload = KudosPayload(
            width=512,
            height=512,
            steps=50,
            source_image=True,
            denoising_strength=0.4,
            control_strength=0.7,
            control_type="canny",
            source_processing="img2img",
        )
        floats = payload.to_feature_vector()[:10]
        assert floats[4] == 1.0  # denoising reset
        assert floats[5] == 0.7  # control strength charged

    def test_remix_maps_to_img2img(self) -> None:
        """The server's remix source-processing hack is mirrored."""
        payload = KudosPayload(width=512, height=512, steps=50, source_processing="remix")
        offset = 10 + len(KNOWN_SAMPLERS) + len(KNOWN_CONTROL_TYPES) + KNOWN_SOURCE_PROCESSING.index("img2img")
        assert payload.to_feature_vector()[offset] == 1.0


class TestJobRecordMapping:
    """Job records map onto priceable payloads with the harness constants."""

    def test_control_record_gets_source_image(self) -> None:
        """A control-typed job record implies a source image."""
        record = JobMetricsRecord(job_id="x", width=512, height=512, steps=30, control_type="canny")
        payload = payload_from_job_record(record)
        assert payload.source_image is True
        assert payload.source_processing == "img2img"

    def test_plain_record_is_txt2img(self) -> None:
        """A plain record fills the harness's constant payload fields."""
        record = JobMetricsRecord(job_id="x", width=1024, height=1024, steps=30)
        payload = payload_from_job_record(record)
        assert payload.source_image is False
        assert payload.source_processing == "txt2img"
        assert payload.sampler_name == "k_euler"
        assert payload.cfg_scale == 7.5


@pytest.mark.skipif(_checkpoint_path() is None, reason=f"{_CKPT_ENV} not set or file missing")
class TestServerParity:
    """Pinned against the AI-Horde server's own ``KudosModel.calculate_kudos`` outputs.

    Reference values were produced by running the server's kudos.py directly against
    kudos-v21-206.ckpt with the default ``basis_adjustment=1``.
    """

    @pytest.fixture(scope="class")
    def scorer(self) -> KudosModelScorer:
        """Load the checkpoint once for the class."""
        path = _checkpoint_path()
        assert path is not None
        return KudosModelScorer(path)

    def test_basis_job_prices_at_11(self, scorer: KudosModelScorer) -> None:
        """The basis job prices at 10 plus the default basis adjustment."""
        # 10-kudos basis plus the server's default +1 basis adjustment.
        assert scorer.score_payload(BASIS_PAYLOAD) == pytest.approx(11.0, abs=0.01)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                KudosPayload(
                    width=1024,
                    height=1024,
                    steps=30,
                    post_processing=["RealESRGAN_x4plus", "GFPGAN"],
                ),
                71.07,
            ),
            (KudosPayload(width=512, height=768, steps=30, hires_fix=True), 14.0),
            (
                KudosPayload(
                    width=832,
                    height=1216,
                    steps=25,
                    cfg_scale=5.0,
                    sampler_name="k_dpmpp_2m",
                    post_processing=["CodeFormers"],
                ),
                18.11,
            ),
            (KudosPayload(width=1024, height=1024, steps=20, cfg_scale=1.0, karras=False), 17.11),
        ],
    )
    def test_matches_server_pricing(self, scorer: KudosModelScorer, payload: KudosPayload, expected: float) -> None:
        """Selected payloads price identically to the server implementation."""
        assert scorer.score_payload(payload) == pytest.approx(expected, abs=0.05)

    def test_session_scoring_counts_batches_and_faults(self, scorer: KudosModelScorer) -> None:
        """Batches multiply the price; faults forfeit it."""
        records = [
            JobMetricsRecord(
                job_id="a",
                width=1024,
                height=1024,
                steps=30,
                batch_count=2,
                time_popped=100.0,
                stage_timestamps={"FINALIZED": 160.0},
            ),
            JobMetricsRecord(job_id="b", width=1024, height=1024, steps=30, faulted=True),
        ]
        report = score_session(records, scorer)
        assert report.num_jobs_scored == 1
        assert report.num_jobs_faulted == 1
        assert report.forfeited_kudos > 0
        # Per-image price doubled by the batch count.
        per_image = scorer.score_payload(payload_from_job_record(records[0]))
        assert report.total_kudos == pytest.approx(per_image * 2, abs=0.01)
        assert report.window_seconds == pytest.approx(60.0)
        assert report.kudos_per_hour == pytest.approx(report.total_kudos * 60, rel=0.01)
