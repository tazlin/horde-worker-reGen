"""LAION aesthetic predictor: a small MLP head scoring CLIP ViT-L/14 image embeddings.

The safety process already embeds every image it sees with OpenAI ViT-L/14 (``image_to_features``),
which is exactly the input the LAION "improved-aesthetic-predictor" was trained on. This module loads
that predictor's tiny MLP head so a 0-10 aesthetic score can be produced from an embedding the worker
has already computed, with no extra CLIP load.

The weight (``sac+logos+ava1-l14-linearMSE.pth``) is fetched once from its canonical upstream and
cached on disk rather than vendored, so the worker never redistributes it. torch is imported here, so
this module is loaded only inside the safety process (never the torch-free orchestrator).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import torch
from loguru import logger
from torch import nn

AESTHETIC_EMBEDDING_DIM = 768
"""The OpenAI ViT-L/14 image-embedding dimension the predictor consumes."""

AESTHETIC_WEIGHT_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"
"""The canonical LAION linear-MSE predictor weight (SAC + LAION-Logos + AVA1 datasets, ViT-L/14)."""

AESTHETIC_WEIGHT_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/" + AESTHETIC_WEIGHT_FILENAME
)
"""Canonical upstream source. The worker downloads from here on first use; it does not vendor the file."""

AESTHETIC_WEIGHT_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
"""SHA-256 of the upstream weight, verified after download so a truncated/tampered file is rejected."""


class AestheticPredictor(nn.Module):
    """The LAION predictor's MLP head: a stack of linear layers over a CLIP image embedding.

    The layer shapes (and the interleaved dropouts that carry no parameters) mirror the upstream
    training definition exactly so the published ``state_dict`` loads without remapping. Dropout is
    inert in ``eval`` mode, so inference is the plain linear stack.
    """

    def __init__(self, input_size: int = AESTHETIC_EMBEDDING_DIM) -> None:
        """Build the MLP for the given embedding width (768 for ViT-L/14)."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return the scalar aesthetic score(s) for the given image embedding(s)."""
        return self.layers(embedding)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aesthetic_cache_dir() -> Path:
    """Resolve a writable directory for the cached predictor weight.

    Prefers the worker's model-cache home (``AIWORKER_CACHE_HOME``, set in the process environment) so
    the weight sits alongside other downloaded assets; falls back to a temp dir when it is unset.
    """
    cache_home = os.environ.get("AIWORKER_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path(tempfile.gettempdir())
    return base / "aesthetic"


def ensure_aesthetic_weight() -> Path:
    """Return the path to the cached predictor weight, downloading and verifying it once if absent.

    The on-disk copy is validated by SHA-256 on every call, so a previously-truncated download is
    re-fetched rather than trusted. Raises on a checksum mismatch after download (a tampered or
    corrupt upstream file should fault loudly, not score silently wrong).
    """
    cache_dir = _aesthetic_cache_dir()
    weight_path = cache_dir / AESTHETIC_WEIGHT_FILENAME

    if weight_path.exists() and _sha256_of(weight_path) == AESTHETIC_WEIGHT_SHA256:
        return weight_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading aesthetic predictor weight from {AESTHETIC_WEIGHT_URL}")
    temp_path = weight_path.with_suffix(weight_path.suffix + ".partial")
    urllib.request.urlretrieve(AESTHETIC_WEIGHT_URL, temp_path)  # noqa: S310 (fixed canonical https URL)

    actual = _sha256_of(temp_path)
    if actual != AESTHETIC_WEIGHT_SHA256:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Aesthetic weight checksum mismatch: expected {AESTHETIC_WEIGHT_SHA256}, got {actual}",
        )
    temp_path.replace(weight_path)
    return weight_path


class AestheticScorer:
    """Loads the predictor once and scores CLIP ViT-L/14 image embeddings on a chosen device."""

    def __init__(self, weight_path: Path, device: str = "cpu") -> None:
        """Load the predictor weight onto *device*; the caller passes the safety process's device."""
        self._device = device
        self._model = AestheticPredictor()
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(device)

    @torch.no_grad()
    def score(self, image_features: torch.Tensor) -> float:
        """Return the rounded 0-10 aesthetic score for a (single) CLIP image embedding.

        The embedding is L2-normalised defensively (the upstream predictor expects unit vectors, and
        ``image_to_features`` already normalises, so this is idempotent) and cast to float32 to match
        the linear-MSE weights regardless of the interrogator's autocast dtype.
        """
        features = image_features.to(self._device, dtype=torch.float32)
        features = nn.functional.normalize(features, dim=-1)
        return round(float(self._model(features).squeeze().item()), 4)


def load_aesthetic_scorer(device: str = "cpu") -> AestheticScorer:
    """Ensure the weight is present, then build a scorer on *device*."""
    return AestheticScorer(ensure_aesthetic_weight(), device=device)
