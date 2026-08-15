"""Validate and materialize worker-local image model registrations for hordelib."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from hordelib.pipeline.families.image_gen.baselines import (
    CASCADE_BASELINES,
    UNET_LOADER_BASELINES,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

_SERVICE_FORBIDDEN_CUSTOM_MODEL_BASELINES = frozenset({KNOWN_IMAGE_GENERATION_BASELINE.flux_dev})

CUSTOM_MODEL_BASELINES: tuple[KNOWN_IMAGE_GENERATION_BASELINE, ...] = tuple(
    baseline
    for baseline in KNOWN_IMAGE_GENERATION_BASELINE
    if baseline is not KNOWN_IMAGE_GENERATION_BASELINE.infer
    and baseline not in UNET_LOADER_BASELINES
    and baseline not in CASCADE_BASELINES
    and baseline not in _SERVICE_FORBIDDEN_CUSTOM_MODEL_BASELINES
)
"""Single-checkpoint baselines accepted by the installed HMR/hordelib capability vocabulary.

``custom_models`` currently describes exactly one fused checkpoint. HMR owns the baseline enum;
hordelib owns which baselines use split UNet/component loading and which use Stable Cascade's two-stage
loading. New fused-checkpoint baselines therefore appear here automatically, while incompatible shapes
remain unavailable until the worker config schema can describe all of their files.
"""

_CUSTOM_MODEL_BASELINE_SET = frozenset(CUSTOM_MODEL_BASELINES)


class CustomModelDefinition(BaseModel):
    """One operator-owned checkpoint exposed under a Horde model name."""

    model_config = ConfigDict(extra="forbid")

    name: str
    baseline: KNOWN_IMAGE_GENERATION_BASELINE
    filepath: str

    @field_validator("name", "filepath")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        """Strip and reject blank identity/path fields."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("baseline")
    @classmethod
    def _single_checkpoint_baseline(
        cls,
        value: KNOWN_IMAGE_GENERATION_BASELINE,
    ) -> KNOWN_IMAGE_GENERATION_BASELINE:
        """Reject known baselines that the one-file custom schema cannot safely materialize."""
        if value not in _CUSTOM_MODEL_BASELINE_SET:
            supported = ", ".join(str(baseline) for baseline in CUSTOM_MODEL_BASELINES)
            raise ValueError(f"must be a supported single-checkpoint baseline: {supported}")
        return value


@dataclass(frozen=True)
class CustomModelIssue:
    """Why one configured custom model is unsafe to advertise."""

    model_name: str
    reason: str

    def summary(self) -> str:
        """Return a compact operator-facing description."""
        return f"{self.model_name}: {self.reason}"


@dataclass(frozen=True)
class CustomModelPreparation:
    """The registrations that are safe for parent scheduling and child loading."""

    records: dict[str, ImageGenerationModelRecord]
    issues: tuple[CustomModelIssue, ...]
    registry_path: Path | None

    @property
    def ready_names(self) -> frozenset[str]:
        """Return custom model names whose local registration is usable."""
        return frozenset(self.records)


def _definition_to_legacy_entry(definition: CustomModelDefinition, filepath: Path) -> dict[str, Any]:
    """Build the legacy JSON shape consumed by hordelib's custom overlay."""
    return {
        "name": definition.name,
        "baseline": definition.baseline,
        "config": {"files": [{"path": str(filepath)}]},
    }


def _definition_to_record(definition: CustomModelDefinition, filepath: Path) -> ImageGenerationModelRecord:
    """Build the matching parent-side reference record used for scheduling decisions."""
    return ImageGenerationModelRecord(
        record_type=MODEL_REFERENCE_CATEGORY.image_generation,
        name=definition.name,
        description="Custom model (worker-local)",
        baseline=definition.baseline,
        nsfw=False,
        inpainting=False,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=str(filepath), file_url="")],
        ),
    )


def _model_file_issue(filepath: Path) -> str | None:
    """Return why *filepath* cannot be loaded, or None when it is a readable regular file."""
    if not filepath.exists():
        return f"file does not exist: {filepath}"
    if not filepath.is_file():
        return f"path is not a regular file: {filepath}"
    if not os.access(filepath, os.R_OK):
        return f"file is not readable: {filepath}"
    return None


def _atomic_write_json(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    """Replace *path* atomically with a formatted JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=4)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _external_entry_matches(
    entry: object,
    *,
    definition: CustomModelDefinition,
    filepath: Path,
) -> bool:
    """Return whether an external hordelib registry entry matches the worker configuration."""
    if not isinstance(entry, dict):
        return False
    if entry.get("baseline") != definition.baseline:
        return False
    files = entry.get("config", {}).get("files", []) if isinstance(entry.get("config"), dict) else []
    if not isinstance(files, list) or not files or not isinstance(files[0], dict):
        return False
    external_path = files[0].get("path")
    if not isinstance(external_path, str):
        return False
    return Path(external_path).expanduser().resolve(strict=False) == filepath


def prepare_custom_models(
    definitions: list[CustomModelDefinition],
    *,
    known_model_names: set[str],
    working_directory: Path | None = None,
) -> CustomModelPreparation:
    """Validate custom checkpoints, materialize hordelib's registry, and return ready parent records.

    An explicitly supplied ``HORDELIB_CUSTOM_MODELS`` registry remains operator-owned and is never
    overwritten. The worker verifies that each configured entry agrees with it. Otherwise the worker owns
    ``.horde_worker_regen/custom_models.json`` in its working directory and atomically reconciles it before
    children spawn. Invalid entries remain visible as issues but are omitted from both the registry and
    advertised model set. The distinct durable-state path prevents this writer from clobbering an
    operator-maintained legacy ``custom_models.json`` in the working directory.
    """
    workdir = (working_directory or Path.cwd()).resolve()
    owned_registry_path = workdir / ".horde_worker_regen" / "custom_models.json"
    configured_registry = os.getenv("HORDELIB_CUSTOM_MODELS")
    configured_path = Path(configured_registry).expanduser().resolve(strict=False) if configured_registry else None
    external_registry = configured_path is not None and configured_path != owned_registry_path

    external_entries: dict[str, Any] = {}
    external_error: str | None = None
    if external_registry:
        try:
            parsed = json.loads(configured_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("registry root is not an object")
            external_entries = parsed
        except (OSError, ValueError) as error:
            external_error = f"external HORDELIB_CUSTOM_MODELS registry cannot be read: {error}"

    issues: list[CustomModelIssue] = []
    records: dict[str, ImageGenerationModelRecord] = {}
    legacy_entries: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    normalized_known_names = {name.casefold() for name in known_model_names}

    for definition in definitions:
        normalized_name = definition.name.casefold()
        if normalized_name in seen:
            issues.append(CustomModelIssue(definition.name, "name is configured more than once"))
            continue
        seen.add(normalized_name)
        if normalized_name in normalized_known_names:
            issues.append(CustomModelIssue(definition.name, "name conflicts with the Horde model reference"))
            continue

        filepath = Path(definition.filepath).expanduser().resolve(strict=False)
        file_issue = _model_file_issue(filepath)
        if file_issue is not None:
            issues.append(CustomModelIssue(definition.name, file_issue))
            continue
        if external_error is not None:
            issues.append(CustomModelIssue(definition.name, external_error))
            continue
        if external_registry and not _external_entry_matches(
            external_entries.get(definition.name),
            definition=definition,
            filepath=filepath,
        ):
            issues.append(
                CustomModelIssue(
                    definition.name,
                    "external HORDELIB_CUSTOM_MODELS entry is missing or does not match baseline/filepath",
                ),
            )
            continue

        legacy_entries[definition.name] = _definition_to_legacy_entry(definition, filepath)
        records[definition.name] = _definition_to_record(definition, filepath)

    registry_path = configured_path if external_registry else owned_registry_path
    if not external_registry:
        try:
            _atomic_write_json(owned_registry_path, legacy_entries)
            os.environ["HORDELIB_CUSTOM_MODELS"] = str(owned_registry_path)
        except OSError as error:
            reason = f"could not write hordelib custom-model registry: {error}"
            issues.extend(CustomModelIssue(name, reason) for name in records)
            records = {}
            registry_path = None
            os.environ.pop("HORDELIB_CUSTOM_MODELS", None)

    for issue in issues:
        logger.error(f"Custom model unavailable: {issue.summary()}")
    logger.info(
        f"Custom model readiness: {len(records)}/{len(definitions)} ready"
        + (f" via {registry_path}" if registry_path is not None else ""),
    )
    return CustomModelPreparation(records=records, issues=tuple(issues), registry_path=registry_path)
