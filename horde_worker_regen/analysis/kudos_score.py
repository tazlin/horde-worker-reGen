"""Kudos scoring for harness/simulation sessions using the AI-Horde server's kudos model.

The AI-Horde server prices an image job by running its payload through a small torch MLP (the
"kudos model", ``horde/classes/stable/kudos.py`` in the AI-Horde repository) that predicts the
job's wall time on a reference worker, then scales a 10-kudos basis job by the predicted-time
ratio. A worker's real earning rate is therefore ``sum(predicted-time-priced kudos) / hour``,
which makes the same figure the natural objective for offline session simulations: two
scheduling policies can be compared by replaying the same job mix through the harness and
scoring the finished jobs here.

This module mirrors the server's feature encoding exactly (same one-hot vocabularies, sorted the
same way, same normalisations) and loads the same pickled checkpoint, so a payload scores here
what it would score on the server. Feature values the harness's canned jobs hold constant
(sampler ``k_euler``, cfg 7.5, karras, denoising 1.0) are filled with those constants when
scoring from :class:`~horde_worker_regen.process_management.resources.run_metrics.JobMetricsRecord`
entries, which do not retain them.

torch is imported lazily inside the scoring entry points: the analysis package is imported by the
torch-free orchestrator-side tooling, and scoring is expected to run post-hoc (or via the
``python -m horde_worker_regen.analysis.kudos_score`` CLI in a separate process), never inside a
live worker parent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord

if TYPE_CHECKING:
    pass

KUDOS_BASIS = 10.0
"""The server's anchor: a 50-step 512x512 k_euler txt2img job is worth 10 kudos."""

KUDOS_BASIS_ADJUSTMENT = 1.0
"""The server's default ``basis_adjustment``: every job is priced from a ``10 + 1`` basis before
the time-ratio scaling, so the effective anchor is 11 kudos. Mirrored so scores match the server's
default pricing path exactly."""

# The vocabularies the server one-hot encodes over. The server sorts these in place before
# encoding, so they are kept pre-sorted here; changing their contents or order desynchronises
# the feature layout from the checkpoint.
KNOWN_POST_PROCESSORS: list[str] = sorted(
    [
        "4x_AnimeSharp",
        "CodeFormers",
        "GFPGAN",
        "NMKD_Siax",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus",
        "strip_background",
    ],
)

KNOWN_SAMPLERS: list[str] = sorted(
    [
        "ddim",
        "k_dpm_2_a",
        "k_dpm_2",
        "k_dpm_adaptive",
        "k_dpm_fast",
        "k_dpmpp_2m",
        "k_dpmpp_2s_a",
        "k_dpmpp_sde",
        "k_euler_a",
        "k_euler",
        "k_heun",
        "k_lms",
        "plms",
        "uni_pc_bh2",
        "uni_pc",
    ],
)

KNOWN_CONTROL_TYPES: list[str] = sorted(
    [
        "canny",
        "depth",
        "fakescribbles",
        "hed",
        "hough",
        "None",
        "normal",
        "openpose",
        "scribble",
        "seg",
    ],
)

KNOWN_SOURCE_PROCESSING: list[str] = sorted(
    [
        "img2img",
        "inpainting",
        "outpainting",
        "txt2img",
    ],
)


class KudosPayload(BaseModel):
    """The exact payload view the server's kudos model prices.

    Field semantics and defaults mirror the server's ``payload_to_tensor``: unknown samplers fall
    back to ``k_euler``, ``remix`` source processing is treated as ``img2img``, and the
    denoising/control-strength interplay follows the source-image/control-type presence rules.
    """

    width: int
    height: int
    steps: int
    cfg_scale: float = 7.5
    denoising_strength: float = 1.0
    control_strength: float = 1.0
    karras: bool = True
    hires_fix: bool = False
    source_image: bool = False
    source_mask: bool = False
    source_processing: str = "txt2img"
    sampler_name: str = "k_euler"
    control_type: str | None = None
    post_processing: list[str] = Field(default_factory=list)

    def to_feature_vector(self) -> list[float]:
        """Return the 47-dim feature vector in the checkpoint's layout."""
        denoising = 1.0
        control_strength = 1.0
        if self.source_image:
            denoising = self.denoising_strength
            if self.control_type is not None and self.control_type != "None":
                control_strength = self.control_strength
                denoising = 1.0

        floats = [
            self.height / 1024,
            self.width / 1024,
            self.steps / 100,
            self.cfg_scale / 30,
            denoising,
            control_strength,
            1.0 if self.karras else 0.0,
            1.0 if self.hires_fix else 0.0,
            1.0 if self.source_image else 0.0,
            1.0 if self.source_mask else 0.0,
        ]

        sampler = self.sampler_name if self.sampler_name in KNOWN_SAMPLERS else "k_euler"
        floats += _one_hot(sampler, KNOWN_SAMPLERS)

        control = self.control_type if self.control_type is not None else "None"
        floats += _one_hot(control, KNOWN_CONTROL_TYPES)

        source_processing = self.source_processing
        if source_processing == "remix":
            source_processing = "img2img"
        floats += _one_hot(source_processing, KNOWN_SOURCE_PROCESSING)

        post_processing_hot = [0.0] * len(KNOWN_POST_PROCESSORS)
        for name in self.post_processing:
            if name in KNOWN_POST_PROCESSORS:
                post_processing_hot[KNOWN_POST_PROCESSORS.index(name)] = 1.0
        floats += post_processing_hot

        return floats


BASIS_PAYLOAD = KudosPayload(width=512, height=512, steps=50, karras=True)
"""The server's 10-kudos reference job."""


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    hot = [0.0] * len(vocabulary)
    hot[vocabulary.index(value)] = 1.0
    return hot


class KudosModelScorer:
    """Prices payloads with the server's pickled kudos checkpoint.

    The checkpoint is a plain ``torch.nn.Sequential`` saved in eval mode, so inference here is
    deterministic and matches the server byte-for-byte on the same feature vector.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """Load the checkpoint and compute the basis time.

        Args:
            checkpoint_path: Path to the server's kudos checkpoint (e.g. ``kudos-v21-206.ckpt``).
        """
        import pickle

        with checkpoint_path.open("rb") as checkpoint_file:
            self._model = pickle.load(checkpoint_file)  # noqa: S301  # the server's own artifact
        self._time_basis = self.predict_seconds(BASIS_PAYLOAD)

    @property
    def time_basis_seconds(self) -> float:
        """The predicted wall time of the 10-kudos basis job."""
        return self._time_basis

    def _forward(self, features: list[float]) -> float:
        import torch

        with torch.no_grad():
            output = self._model(torch.tensor(features).float())
        return round(float(output.item()), 2)

    def predict_seconds(self, payload: KudosPayload) -> float:
        """Return the model's predicted wall time in seconds for one image of this payload."""
        return self._forward(payload.to_feature_vector())

    def score_payload(self, payload: KudosPayload) -> float:
        """Return the kudos price of one image of this payload."""
        basis = KUDOS_BASIS + KUDOS_BASIS_ADJUSTMENT
        return round(basis * self.predict_seconds(payload) / self._time_basis, 2)


class JobKudosLine(BaseModel):
    """One scored job in a session report."""

    job_id: str
    model_name: str | None = None
    kudos: float
    """Total kudos for the job (per-image price times the batch count)."""
    batch_count: int = 1
    faulted: bool = False
    """Faulted jobs earn nothing; retained in the report so forfeited kudos is visible."""
    forfeited_kudos: float = 0.0
    """What the job would have earned had it not faulted."""


class SessionKudosReport(BaseModel):
    """The kudos outcome of one harness/simulation session."""

    total_kudos: float
    forfeited_kudos: float
    """Kudos lost to faulted jobs (work the worker accepted but did not deliver)."""
    kudos_per_hour: float
    """Earned kudos over the scored window (first pop to last finalization)."""
    window_seconds: float
    num_jobs_scored: int
    num_jobs_faulted: int
    jobs: list[JobKudosLine] = Field(default_factory=list)


def payload_from_job_record(record: JobMetricsRecord) -> KudosPayload:
    """Build the priceable payload view from a finished harness job record.

    The record does not retain sampler/cfg/karras/denoising; the harness's canned jobs hold those
    constant (``k_euler``/7.5/karras/1.0), so the constants are filled here. A control-typed job
    always carries a source image on the harness path, mirroring ``make_canned_job``.
    """
    has_control = record.control_type is not None and record.control_type != "None"
    return KudosPayload(
        width=record.width if record.width is not None else 512,
        height=record.height if record.height is not None else 512,
        steps=record.steps if record.steps is not None else 30,
        hires_fix=record.hires_fix,
        source_image=has_control,
        source_processing="img2img" if has_control else "txt2img",
        control_type=record.control_type,
        post_processing=list(record.post_processing),
    )


def score_session(
    records: list[JobMetricsRecord],
    scorer: KudosModelScorer,
    *,
    window_seconds: float | None = None,
) -> SessionKudosReport:
    """Score a session's finished image jobs and derive the kudos/hr figure.

    The per-image price is multiplied by the job's batch count, matching how the server pays a
    multi-image job. The rate window defaults to first-pop to last-finalization across the
    records, so a cold boot before the first pop does not dilute the rate; pass
    ``window_seconds`` to override (e.g. with the full soak duration).

    Args:
        records: Finished-job records from a run metrics snapshot (alchemy records are ignored).
        scorer: The loaded kudos checkpoint.
        window_seconds: Optional explicit window for the kudos/hr denominator.

    Returns:
        The session report, including per-job lines and the forfeited kudos of faulted jobs.
    """
    lines: list[JobKudosLine] = []
    total = 0.0
    forfeited = 0.0
    window_start: float | None = None
    window_end: float | None = None

    for record in records:
        if record.is_alchemy:
            continue
        payload = payload_from_job_record(record)
        per_image = scorer.score_payload(payload)
        job_kudos = round(per_image * max(record.batch_count, 1), 2)

        if record.faulted:
            forfeited += job_kudos
            lines.append(
                JobKudosLine(
                    job_id=record.job_id,
                    model_name=record.model_name,
                    kudos=0.0,
                    batch_count=record.batch_count,
                    faulted=True,
                    forfeited_kudos=job_kudos,
                ),
            )
        else:
            total += job_kudos
            lines.append(
                JobKudosLine(
                    job_id=record.job_id,
                    model_name=record.model_name,
                    kudos=job_kudos,
                    batch_count=record.batch_count,
                ),
            )

        if record.time_popped is not None:
            window_start = record.time_popped if window_start is None else min(window_start, record.time_popped)
        finalized = record.stage_timestamps.get("FINALIZED")
        if finalized is not None:
            window_end = finalized if window_end is None else max(window_end, finalized)

    if window_seconds is None:
        window_seconds = (window_end - window_start) if window_start is not None and window_end is not None else 0.0

    kudos_per_hour = (total / window_seconds) * 3600.0 if window_seconds > 0 else 0.0

    return SessionKudosReport(
        total_kudos=round(total, 2),
        forfeited_kudos=round(forfeited, 2),
        kudos_per_hour=round(kudos_per_hour, 1),
        window_seconds=round(window_seconds, 2),
        num_jobs_scored=sum(1 for line in lines if not line.faulted),
        num_jobs_faulted=sum(1 for line in lines if line.faulted),
        jobs=lines,
    )


def _main() -> int:
    """Score a JSONL file of ``JobMetricsRecord`` entries and print the session report as JSON."""
    parser = argparse.ArgumentParser(description="Score harness job records with the AI-Horde kudos model.")
    parser.add_argument("jobs_jsonl", type=Path, help="JSONL file of JobMetricsRecord entries")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the kudos model checkpoint")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Explicit kudos/hr window; defaults to first-pop..last-finalize across the records",
    )
    args = parser.parse_args()

    records: list[JobMetricsRecord] = []
    with args.jobs_jsonl.open("r", encoding="utf-8") as jobs_file:
        for line in jobs_file:
            line = line.strip()
            if not line:
                continue
            data: dict[str, Any] = json.loads(line)
            records.append(JobMetricsRecord(**data))

    scorer = KudosModelScorer(args.ckpt)
    report = score_session(records, scorer, window_seconds=args.window_seconds)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
