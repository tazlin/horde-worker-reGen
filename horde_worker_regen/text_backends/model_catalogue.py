"""The text models this worker has measured, contributed to the model reference as one more source.

koboldcpp needs a quantised file, and the canonical text reference is a name registry: its records carry
the publisher's page and a parameter count, never a file. So the measured models are offered to the
reference through a [`StaticModelProvider`][horde_model_reference.providers.static_provider.StaticModelProvider]
under [`TEXT_CATALOGUE_SOURCE_ID`][horde_worker_regen.text_backends.model_catalogue.TEXT_CATALOGUE_SOURCE_ID],
so a read of the text category with `ANY_SOURCE` sees them beside the canonical records through the same
API every other model category is read through. A canonical record of the same name wins the merge, which
is the end state this is aimed at: once the reference carries these files itself, its records shadow these
and nothing above this module changes.

The file facts live in the fields the reference already has: `config.download` for the file's name, origin
and digest, `size_on_disk_bytes` for its size. The facts that come from running the model have no field
yet, so they ride in `settings` under the keys in
[`TextMeasurementKey`][horde_worker_regen.text_backends.model_catalogue.TextMeasurementKey] and are read
back through [`measured_facts`][horde_worker_regen.text_backends.model_catalogue.measured_facts]. Making
those keys canonical reference fields is an open issue the operator holds; until then they are this
worker's extension and no other consumer of the reference knows them.

A model appears here only once someone has run it: an entry states the card it was measured on, the
context it was measured at, and what it then cost and produced, so a fit decision is made against a
measurement rather than an estimate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from horde_model_reference import ANY_SOURCE, MODEL_REFERENCE_CATEGORY, StaticModelProvider
from horde_model_reference.model_reference_records import DownloadRecord, TextGenerationModelRecord
from horde_model_reference.on_disk_layout import resolve_weights_root
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.text_backends.provision import TextBackendProvisionError

if TYPE_CHECKING:
    from horde_model_reference.model_reference_manager import ModelReferenceManager

TEXT_CATALOGUE_SOURCE_ID = "regen_text"
"""The source id the worker's measured text models are served under."""

TEXT_MODELS_FOLDER_NAME = "text_models"
"""The folder fetched text model files are kept in, under the model cache."""

UNKNOWN_FILE_URL = ""
"""A declared file whose origin is not on record: it can be verified and used, never fetched.

`DownloadRecord.file_url` is required and the reference has no way to say "this file exists and I do not
know where it came from", which is the state of a file measured from an operator's disk. An empty origin
is that state, the way `"FIXME"` is an unknown digest, and every fetch path checks for it.
"""

HUGGINGFACE_MODEL_PAGE_PREFIX = "https://huggingface.co/"
"""Where a text model's `author/Model` name resolves to its publisher's page."""


class TextMeasurementKey:
    """The `settings` keys carrying what running a text model showed, which the reference has no fields for.

    Read through [`measured_facts`][horde_worker_regen.text_backends.model_catalogue.measured_facts] rather
    than by key, so the one place these spellings are known is this class.
    """

    QUANT = "quant"
    BACKEND = "measured_backend"
    CONTEXT = "measured_context"
    FOOTPRINT_MB = "measured_footprint_mb"
    TOKENS_PER_SECOND = "measured_tokens_per_second"
    SOURCE_CARD = "source_card"


@dataclass(frozen=True)
class TextModelMeasurement:
    """Represents what one text model cost and produced on one card, at one context length."""

    quant: str
    """The quantisation of the measured file, as the file name spells it (`Q4_K_M`)."""
    backend: TEXT_BACKENDS
    """The backend the measurement was taken through."""
    context: int
    """The context length the backend was launched at, which sizes the KV cache in the footprint."""
    footprint_mb: int
    """The VRAM the loaded model held, measured as the drop in free VRAM across the load."""
    tokens_per_second: float | None
    """Generation rate, or None when the run recorded no timing."""
    source_card: str
    """The card the measurement was taken on, without which the other figures cannot be compared."""


@dataclass(frozen=True)
class ResolvedTextModel:
    """Represents the file a configured `text_model` names, and what the reference knows about it."""

    path: Path
    """Where the file is, or is expected to be once fetched."""
    canonical_name: str
    """The name to advertise this model under, before the backend prefix is applied."""
    record: TextGenerationModelRecord | None
    """The reference record the name resolved to, or None for a path the operator supplied."""

    @property
    def declared_file(self) -> DownloadRecord | None:
        """Return the record's primary declared file, or None when no record declares one."""
        if self.record is None or not self.record.config.download:
            return None
        return self.record.config.download[0]

    @property
    def declared_size_bytes(self) -> int | None:
        """Return the size the reference declares for this model, or None when it declares none."""
        if self.record is None:
            return None
        return self.record.declared_total_size_bytes

    @property
    def fetch_url(self) -> str | None:
        """Return the origin to fetch this file from, or None when none is on record."""
        declared_file = self.declared_file
        if declared_file is None or declared_file.file_url == UNKNOWN_FILE_URL:
            return None
        return declared_file.file_url


class TextModelResolutionError(TextBackendProvisionError):
    """A configured `text_model` is neither a file on disk nor a model with a file on offer."""


def _measured_gguf_record(
    *,
    plain_name: str,
    quant: str,
    parameters_count: int,
    file_url: str,
    size_bytes: int,
    sha256: str,
    backend: TEXT_BACKENDS,
    measured_context: int,
    measured_footprint_mb: int,
    measured_tokens_per_second: float | None,
    source_card: str,
) -> tuple[str, dict[str, object]]:
    """Create one measured model's `(reference name, raw record)` pair from the facts of its measurement.

    Three spellings are derived from *plain_name* and *quant* rather than restated: the reference name a
    quantised artefact takes (the record's name plus the quant suffix), the publisher's page, and the file
    name. The file name drops the author segment because that is what koboldcpp names a model after, so a
    file named any other way is advertised under a name the horde has no work for.

    The parameter count is the rounded figure the text reference records against the unquantised model, so
    a record here and the canonical record it will one day be shadowed by agree on it.

    Args:
        plain_name: The unquantised model's reference name (`meta-llama/Llama-3.2-3B-Instruct`).
        quant: The quantisation of the measured file (`Q4_K_M`).
        parameters_count: Parameters, rounded as the reference rounds them.
        file_url: Where the file came from, or `UNKNOWN_FILE_URL` when that is not on record.
        size_bytes: The file's size on disk.
        sha256: The file's hex digest.
        backend: The backend the measurement was taken through.
        measured_context: The context the backend was launched at.
        measured_footprint_mb: The VRAM the loaded model held.
        measured_tokens_per_second: The generation rate, or None when the run recorded no timing.
        source_card: The card the measurement was taken on.

    Returns:
        The reference name and the raw record to validate against `TextGenerationModelRecord`.
    """
    reference_name = f"{plain_name}-{quant}"
    file_name = f"{plain_name.rsplit('/', 1)[-1]}-{quant}.gguf"
    settings: dict[str, int | float | str | bool] = {
        TextMeasurementKey.QUANT: quant,
        TextMeasurementKey.BACKEND: backend.value,
        TextMeasurementKey.CONTEXT: measured_context,
        TextMeasurementKey.FOOTPRINT_MB: measured_footprint_mb,
        TextMeasurementKey.SOURCE_CARD: source_card,
    }
    # `settings` values are scalars, so an unrecorded rate is an absent key rather than a null.
    if measured_tokens_per_second is not None:
        settings[TextMeasurementKey.TOKENS_PER_SECOND] = measured_tokens_per_second
    raw_record: dict[str, object] = {
        "parameters": parameters_count,
        "url": f"{HUGGINGFACE_MODEL_PAGE_PREFIX}{plain_name}",
        "size_on_disk_bytes": size_bytes,
        "config": {
            "download": [
                {
                    "file_name": file_name,
                    "file_url": file_url,
                    "sha256sum": sha256,
                    "size_bytes": size_bytes,
                },
            ],
        },
        "settings": settings,
    }
    return reference_name, raw_record


_RTX_4070_TI_SUPER = "NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)"
"""The card every seed measurement was taken on, so the seeds are comparable with each other."""

MEASURED_TEXT_MODELS: dict[str, dict[str, object]] = dict(
    (
        _measured_gguf_record(
            plain_name="meta-llama/Llama-3.2-3B-Instruct",
            quant="Q4_K_M",
            parameters_count=3_000_000_000,
            file_url=UNKNOWN_FILE_URL,
            size_bytes=2_019_377_696,
            sha256="6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
            backend=TEXT_BACKENDS.koboldcpp,
            measured_context=4096,
            measured_footprint_mb=2911,
            measured_tokens_per_second=190.0,
            source_card=_RTX_4070_TI_SUPER,
        ),
        _measured_gguf_record(
            plain_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            quant="Q4_K_M",
            parameters_count=8_000_000_000,
            file_url=UNKNOWN_FILE_URL,
            size_bytes=4_920_739_232,
            sha256="7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
            backend=TEXT_BACKENDS.koboldcpp,
            measured_context=8192,
            measured_footprint_mb=5954,
            measured_tokens_per_second=106.0,
            source_card=_RTX_4070_TI_SUPER,
        ),
        _measured_gguf_record(
            plain_name="mistralai/Mistral-Nemo-Instruct-2407",
            quant="Q4_K_M",
            parameters_count=12_000_000_000,
            file_url=UNKNOWN_FILE_URL,
            size_bytes=7_477_208_192,
            sha256="7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46",
            backend=TEXT_BACKENDS.koboldcpp,
            measured_context=8192,
            measured_footprint_mb=8633,
            measured_tokens_per_second=73.0,
            source_card=_RTX_4070_TI_SUPER,
        ),
    ),
)
"""The measured text models, keyed by the reference name a worker advertises them under.

Each entry is one file that was loaded and generated with. Adding one means measuring it; the figures are
what [`measure_text_model`](../../docs/plans/text-generation/measurements) reports, and a guessed figure
would be priced against as though it were a measurement.
"""


def text_catalogue_provider() -> StaticModelProvider:
    """Create the provider serving the measured text models under `TEXT_CATALOGUE_SOURCE_ID`.

    Raises:
        pydantic.ValidationError: A seed record does not validate as a `TextGenerationModelRecord`.
    """
    return StaticModelProvider.from_raw(
        TEXT_CATALOGUE_SOURCE_ID,
        {MODEL_REFERENCE_CATEGORY.text_generation: MEASURED_TEXT_MODELS},
    )


def register_text_model_catalogue(manager: ModelReferenceManager) -> None:
    """Mutate *manager* so its text category also serves the worker's measured models.

    Idempotent, because the manager is a singleton several entry points build and every one of them has to
    end up with the same text category: a name one context can resolve and another cannot is a model the
    worker advertises and then cannot load.
    """
    if manager.get_provider(TEXT_CATALOGUE_SOURCE_ID) is not None:
        return
    manager.register_provider(text_catalogue_provider(), replace=True)


def text_model_records(manager: ModelReferenceManager) -> dict[str, TextGenerationModelRecord]:
    """Return every text model *manager* knows, canonical records merged with the measured ones by name.

    Registers the catalogue first, so no caller can read a text category that is missing it. A canonical
    record wins its name, which is why a measured record can be shadowed without being removed.
    """
    register_text_model_catalogue(manager)
    records = manager.query(MODEL_REFERENCE_CATEGORY.text_generation, source=ANY_SOURCE).to_list()
    return {record.name: record for record in records}


def measured_facts(record: TextGenerationModelRecord) -> TextModelMeasurement | None:
    """Return what running *record*'s model showed, or None when nobody has measured it.

    A canonical text record carries no measurement, so None is the ordinary answer for most of the text
    reference and means "unknown", never "does not fit".
    """
    settings = record.settings
    if settings is None:
        return None

    quant = settings.get(TextMeasurementKey.QUANT)
    backend = settings.get(TextMeasurementKey.BACKEND)
    context = settings.get(TextMeasurementKey.CONTEXT)
    footprint_mb = settings.get(TextMeasurementKey.FOOTPRINT_MB)
    source_card = settings.get(TextMeasurementKey.SOURCE_CARD)
    if not isinstance(quant, str) or not isinstance(backend, str) or not isinstance(source_card, str):
        return None
    if not isinstance(context, int) or not isinstance(footprint_mb, int):
        return None
    if backend not in {member.value for member in TEXT_BACKENDS}:
        return None

    tokens_per_second = settings.get(TextMeasurementKey.TOKENS_PER_SECOND)
    return TextModelMeasurement(
        quant=quant,
        backend=TEXT_BACKENDS(backend),
        context=context,
        footprint_mb=footprint_mb,
        tokens_per_second=float(tokens_per_second) if isinstance(tokens_per_second, int | float) else None,
        source_card=source_card,
    )


def text_models_that_fit(
    records: Iterable[TextGenerationModelRecord],
    *,
    card_total_mb: int,
    headroom_mb: int,
) -> list[TextGenerationModelRecord]:
    """Return the measured models whose footprint fits a card, largest first.

    Largest first because the largest model that fits is the best one on offer at a given quantisation, so
    a caller picking a default takes the head of the list. A record nobody has measured is left out: its
    footprint is unknown, and offering it would be offering an estimate.

    Args:
        records: The text models to consider (a merged reference read).
        card_total_mb: The card's total VRAM.
        headroom_mb: VRAM to keep free for everything else on the card, image work included.

    Returns:
        The fitting records, ordered by measured footprint descending.
    """
    budget_mb = card_total_mb - headroom_mb
    fitting: list[tuple[int, TextGenerationModelRecord]] = []
    for record in records:
        measurement = measured_facts(record)
        if measurement is None or measurement.footprint_mb > budget_mb:
            continue
        fitting.append((measurement.footprint_mb, record))
    fitting.sort(key=lambda entry: entry[0], reverse=True)
    return [record for _footprint_mb, record in fitting]


def default_text_models_dir() -> Path:
    """Return where fetched text model files go when the operator names no directory.

    The reference publishes an on-disk folder for each category whose files it declares and has none for
    text models, so the worker names one itself. It sits under the resolved model root so a text model
    lands beside the image weights instead of somewhere the operator has to find separately.
    """
    return resolve_weights_root() / TEXT_MODELS_FOLDER_NAME


def resolve_text_model(
    text_model: str,
    records: Mapping[str, TextGenerationModelRecord],
    text_models_dir: Path | None = None,
) -> ResolvedTextModel:
    """Resolve a configured `text_model` to a file, a name to advertise, and the record behind it.

    The one key is either a file the operator supplied or a model the reference knows, and which one is
    decided by the filesystem rather than by shape: an existing path is that file, so a name that happens
    to look like a path cannot shadow a catalogued model that exists. A name resolves to the file its
    record declares, under *text_models_dir*, whether or not it has been fetched yet.

    Only the record's primary declared file is resolved, so a model the reference splits across files is
    not something this can express yet.

    Args:
        text_model: The operator's `text_model` value.
        records: The merged text reference read, from `text_model_records`.
        text_models_dir: Where fetched files are kept, or None for `default_text_models_dir`.

    Returns:
        The resolved file, its canonical name, and its record when it has one.

    Raises:
        TextModelResolutionError: The value is neither an existing path nor a model with a file on offer.
    """
    supplied_path = Path(text_model)
    if supplied_path.is_file():
        return ResolvedTextModel(path=supplied_path, canonical_name=supplied_path.stem, record=None)

    record = records.get(text_model)
    if record is None:
        raise TextModelResolutionError(
            f"`text_model` is set to {text_model!r}, which is neither a file on disk nor a model in the "
            "text model reference. Give the path to a model file, or a reference name such as "
            f"{_example_catalogue_name()!r}.",
        )

    declared_file = record.config.download[0] if record.config.download else None
    if declared_file is None:
        raise TextModelResolutionError(
            f"`text_model` is set to {text_model!r}, which the text model reference knows as a model but "
            "declares no file for, so the worker cannot tell which artefact to load. Give the path to a "
            "model file you have, or a reference name that carries one.",
        )

    directory = default_text_models_dir() if text_models_dir is None else text_models_dir
    return ResolvedTextModel(
        path=directory / declared_file.file_name,
        canonical_name=record.name,
        record=record,
    )


def _example_catalogue_name() -> str:
    """Return one measured model's name, for an error that has to show the operator what one looks like."""
    return next(iter(MEASURED_TEXT_MODELS))


__all__ = [
    "HUGGINGFACE_MODEL_PAGE_PREFIX",
    "MEASURED_TEXT_MODELS",
    "TEXT_CATALOGUE_SOURCE_ID",
    "TEXT_MODELS_FOLDER_NAME",
    "UNKNOWN_FILE_URL",
    "ResolvedTextModel",
    "TextMeasurementKey",
    "TextModelMeasurement",
    "TextModelResolutionError",
    "default_text_models_dir",
    "measured_facts",
    "register_text_model_catalogue",
    "resolve_text_model",
    "text_catalogue_provider",
    "text_model_records",
    "text_models_that_fit",
]
