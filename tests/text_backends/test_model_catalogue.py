"""The measured text models as a reference source: what they declare, how they merge, how a name resolves."""

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

import pytest
from horde_model_reference import (
    ANY_SOURCE,
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    PrefetchStrategy,
    StaticModelProvider,
)
from horde_model_reference.model_reference_records import TextGenerationModelRecord
from horde_model_reference.text_backend_names import TEXT_BACKENDS, get_model_name_variants

from horde_worker_regen.text_backends.model_catalogue import (
    MEASURED_TEXT_MODELS,
    TEXT_CATALOGUE_SOURCE_ID,
    TEXT_MODELS_FOLDER_NAME,
    UNKNOWN_FILE_URL,
    TextMeasurementKey,
    TextModelResolutionError,
    default_text_models_dir,
    measured_facts,
    register_text_model_catalogue,
    resolve_text_model,
    text_catalogue_provider,
    text_model_records,
    text_models_that_fit,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture
def offline_reference_manager() -> Generator[ModelReferenceManager]:
    """Yield an offline reference manager, restoring the singleton the rest of the suite shares."""
    previous = ModelReferenceManager._instance
    ModelReferenceManager._instance = None
    try:
        yield ModelReferenceManager(offline=True, prefetch_strategy=PrefetchStrategy.NONE)
    finally:
        ModelReferenceManager._instance = previous


def _seed_records() -> dict[str, TextGenerationModelRecord]:
    """Return the measured records as the provider validates them, without touching a manager."""
    records = text_catalogue_provider().fetch_category(MODEL_REFERENCE_CATEGORY.text_generation)
    assert records is not None
    return {name: record for name, record in records.items() if isinstance(record, TextGenerationModelRecord)}


def _record_with_file(*, name: str, file_name: str, file_url: str, sha256: str, size_bytes: int) -> dict[str, object]:
    """Return a raw text record declaring one file, for the rows the seeds cannot cover."""
    return {
        "name": name,
        "parameters": 1_000_000_000,
        "size_on_disk_bytes": size_bytes,
        "config": {
            "download": [
                {"file_name": file_name, "file_url": file_url, "sha256sum": sha256, "size_bytes": size_bytes},
            ],
        },
    }


class TestSeedRecords:
    """The committed measurements, which everything downstream prices and advertises from."""

    def test_every_seed_declares_one_file_with_a_real_digest_and_size(self) -> None:
        """A seed exists because a file was measured, so its file, size and digest are all on record."""
        for name, record in _seed_records().items():
            declared_files = record.config.download
            assert len(declared_files) == 1, name
            declared_file = declared_files[0]
            assert declared_file.file_name.endswith(".gguf"), name
            assert _SHA256_PATTERN.match(declared_file.sha256sum), name
            assert declared_file.size_bytes == record.size_on_disk_bytes
            assert record.declared_total_size_bytes == record.size_on_disk_bytes

    def test_names_and_file_names_are_unique(self) -> None:
        """Two entries sharing a name or a file would make one of them unreachable."""
        records = _seed_records()
        file_names = [record.config.download[0].file_name for record in records.values()]

        assert len(records) == len(MEASURED_TEXT_MODELS)
        assert len(set(file_names)) == len(file_names)

    def test_every_seed_carries_a_measurement_with_a_footprint_above_its_file_size(self) -> None:
        """The loaded model holds its weights plus a KV cache, so a footprint under the file size is wrong."""
        for name, record in _seed_records().items():
            measurement = measured_facts(record)
            assert measurement is not None, name
            assert record.size_on_disk_bytes is not None
            assert measurement.footprint_mb * 1024 * 1024 > record.size_on_disk_bytes, name
            assert measurement.context > 0
            assert measurement.source_card

    def test_every_seed_name_has_a_variant_for_the_backend_it_was_measured_through(self) -> None:
        """A name the reference produces no backend variant for cannot be advertised at all."""
        for name, record in _seed_records().items():
            measurement = measured_facts(record)
            assert measurement is not None, name
            prefix = f"{measurement.backend.value}/"
            assert any(variant.startswith(prefix) for variant in get_model_name_variants(name)), name

    def test_the_seeds_origins_are_recorded_as_unknown_rather_than_guessed(self) -> None:
        """No quantiser repository is on record for the measured files, and a guessed one would not verify."""
        for name, record in _seed_records().items():
            assert record.config.download[0].file_url == UNKNOWN_FILE_URL, name

    def test_a_record_with_no_measurement_reads_as_unknown(self) -> None:
        """Most of the canonical text reference carries no measurement, which is not a statement about fit."""
        unmeasured = TextGenerationModelRecord.model_validate(
            _record_with_file(
                name="author/Unmeasured",
                file_name="Unmeasured.gguf",
                file_url="https://example.invalid/Unmeasured.gguf",
                sha256="0" * 64,
                size_bytes=10,
            ),
        )

        assert measured_facts(unmeasured) is None

    def test_a_measurement_missing_a_required_key_reads_as_unknown(self) -> None:
        """A partial measurement is not a measurement: reporting it would price against half a figure."""
        raw = _record_with_file(
            name="author/Partial",
            file_name="Partial.gguf",
            file_url="https://example.invalid/Partial.gguf",
            sha256="0" * 64,
            size_bytes=10,
        )
        raw["settings"] = {TextMeasurementKey.FOOTPRINT_MB: 1000, TextMeasurementKey.QUANT: "Q4_K_M"}

        assert measured_facts(TextGenerationModelRecord.model_validate(raw)) is None


class TestFit:
    """Which measured models a card has room for."""

    def test_the_largest_fitting_model_comes_first(self) -> None:
        """A caller taking a default takes the head of the list, so ordering is the answer."""
        fitting = text_models_that_fit(_seed_records().values(), card_total_mb=16384, headroom_mb=2048)
        footprints = [measured_facts(record).footprint_mb for record in fitting]  # type: ignore[union-attr]

        assert len(fitting) == 3
        assert footprints == sorted(footprints, reverse=True)

    def test_headroom_is_taken_off_the_card_before_anything_fits(self) -> None:
        """The headroom is reserved for everything else on the card, so it is never spent on the model."""
        seeds = _seed_records().values()
        largest = text_models_that_fit(seeds, card_total_mb=16384, headroom_mb=0)[0]
        footprint_mb = measured_facts(largest).footprint_mb  # type: ignore[union-attr]

        assert text_models_that_fit(seeds, card_total_mb=footprint_mb, headroom_mb=0)[0].name == largest.name
        assert all(
            record.name != largest.name
            for record in text_models_that_fit(seeds, card_total_mb=footprint_mb, headroom_mb=1)
        )

    def test_a_card_too_small_for_anything_measured_offers_nothing(self) -> None:
        """An empty list is the honest answer for a card no measured model has been run on."""
        assert text_models_that_fit(_seed_records().values(), card_total_mb=2048, headroom_mb=512) == []

    def test_an_unmeasured_record_never_fits(self) -> None:
        """An unknown footprint is left out rather than estimated, however much room there is."""
        unmeasured = TextGenerationModelRecord.model_validate(
            _record_with_file(
                name="author/Unmeasured",
                file_name="Unmeasured.gguf",
                file_url="https://example.invalid/Unmeasured.gguf",
                sha256="0" * 64,
                size_bytes=10,
            ),
        )

        assert text_models_that_fit([unmeasured], card_total_mb=99999, headroom_mb=0) == []


class TestRegistration:
    """The catalogue reaches consumers as one more source on the reference manager."""

    def test_registration_is_idempotent(self, offline_reference_manager: ModelReferenceManager) -> None:
        """Several entry points build the manager, and each of them registers, so a second call is a no-op."""
        register_text_model_catalogue(offline_reference_manager)
        first = offline_reference_manager.get_provider(TEXT_CATALOGUE_SOURCE_ID)
        register_text_model_catalogue(offline_reference_manager)

        assert first is not None
        assert offline_reference_manager.get_provider(TEXT_CATALOGUE_SOURCE_ID) is first
        assert offline_reference_manager.list_providers().count(TEXT_CATALOGUE_SOURCE_ID) == 1

    def test_the_merged_read_carries_the_seeds_with_their_measurements(
        self,
        offline_reference_manager: ModelReferenceManager,
    ) -> None:
        """A consumer reads text models through the reference and sees the measured ones beside canonical."""
        records = text_model_records(offline_reference_manager)

        for name in MEASURED_TEXT_MODELS:
            assert name in records
            measurement = measured_facts(records[name])
            assert measurement is not None
            assert isinstance(measurement.footprint_mb, int)
            assert isinstance(measurement.context, int)
            assert measurement.backend is TEXT_BACKENDS.koboldcpp

    def test_the_first_source_listed_for_a_colliding_name_is_the_one_that_wins(
        self,
        offline_reference_manager: ModelReferenceManager,
    ) -> None:
        """`ANY_SOURCE` expands canonical first, so a canonical record of a measured name shadows it."""
        register_text_model_catalogue(offline_reference_manager)
        contested_name = next(iter(MEASURED_TEXT_MODELS))
        rival = StaticModelProvider.from_raw(
            "rival_text",
            {
                MODEL_REFERENCE_CATEGORY.text_generation: {
                    contested_name: {"parameters": 1_000_000_000, "description": "the rival"},
                },
            },
        )
        offline_reference_manager.register_provider(rival, replace=True)

        query = offline_reference_manager.query(MODEL_REFERENCE_CATEGORY.text_generation, source=ANY_SOURCE)
        sources_for_name = query.duplicate_names()[contested_name]
        winner = next(record for record in query.to_list() if record.name == contested_name)

        assert sources_for_name.index(TEXT_CATALOGUE_SOURCE_ID) < sources_for_name.index("rival_text")
        assert measured_facts(winner) is not None


class TestResolution:
    """One `text_model` key, resolved to a file, a name, and the record behind it."""

    def test_a_path_to_an_existing_file_resolves_to_itself(self, tmp_path: Path) -> None:
        """A file the operator supplied is used where it sits and is never looked up or fetched."""
        supplied = tmp_path / "MyOwnModel-Q5_K_M.gguf"
        supplied.write_bytes(b"gguf")

        resolved = resolve_text_model(str(supplied), _seed_records(), tmp_path / "unused")

        assert resolved.path == supplied
        assert resolved.canonical_name == "MyOwnModel-Q5_K_M"
        assert resolved.record is None
        assert resolved.declared_file is None
        assert resolved.declared_size_bytes is None
        assert resolved.fetch_url is None

    def test_a_catalogued_name_resolves_under_the_models_directory(self, tmp_path: Path) -> None:
        """A name resolves to where its file goes, whether or not it has been fetched yet."""
        records = _seed_records()
        name, record = next(iter(records.items()))

        resolved = resolve_text_model(name, records, tmp_path)

        assert resolved.path == tmp_path / record.config.download[0].file_name
        assert resolved.canonical_name == name
        assert resolved.record is not None
        assert resolved.declared_size_bytes == record.size_on_disk_bytes

    def test_an_unset_models_directory_falls_back_to_the_worker_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The reference publishes no folder for text models, so the worker's own default has to apply."""
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
        records = _seed_records()
        name = next(iter(records))

        resolved = resolve_text_model(name, records)

        assert default_text_models_dir() == tmp_path / TEXT_MODELS_FOLDER_NAME
        assert resolved.path.parent == tmp_path / TEXT_MODELS_FOLDER_NAME

    def test_a_known_model_with_no_file_on_offer_is_refused_with_that_reason(self, tmp_path: Path) -> None:
        """A canonical text record names a model and declares no file, so there is nothing to load."""
        records = dict(_seed_records())
        records["author/NameOnly"] = TextGenerationModelRecord.model_validate(
            {"name": "author/NameOnly", "parameters": 1_000_000_000},
        )

        with pytest.raises(TextModelResolutionError) as raised:
            resolve_text_model("author/NameOnly", records, tmp_path)

        assert "declares no file" in str(raised.value)

    def test_an_unknown_value_names_both_things_it_could_have_been(self, tmp_path: Path) -> None:
        """Neither a file nor a name, so the error has to say which two things the key accepts."""
        with pytest.raises(TextModelResolutionError) as raised:
            resolve_text_model("not-a-model", _seed_records(), tmp_path)

        assert "file on disk" in str(raised.value)
        assert "text model reference" in str(raised.value)
        assert next(iter(MEASURED_TEXT_MODELS)) in str(raised.value)
