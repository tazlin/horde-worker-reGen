"""The merged canonical-plus-beta reference view and the manager double the beta reader tests share."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from horde_model_reference import HORDE_SOURCE_ID, PENDING_SOURCE_ID, StaticModelProvider
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

import horde_worker_regen.bridge_data.beta_source as beta_source
from horde_worker_regen.bridge_data.beta_source import (
    BETA_MODELS_DIR_ENV_VAR,
    BOUNDED_PENDING_RETRY_MAX_ATTEMPTS,
    BOUNDED_PENDING_TIMEOUT_SECONDS,
    beta_aware_image_records,
    invalidate_merged_records,
    merged_category_records,
    publish_pending_records,
    share_pending_records_with_children,
)

BETA_CATEGORIES_ENV_VAR = "HORDELIB_BETA_MODEL_CATEGORIES"


def image_record(
    name: str,
    *,
    file_name: str | None = None,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
) -> ImageGenerationModelRecord:
    """Return an image record declaring one checkpoint file, named after the model unless given."""
    return ImageGenerationModelRecord(
        name=name,
        baseline=baseline,
        nsfw=False,
        description=f"{name} record",
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name=file_name or f"{name}.safetensors",
                    file_url=f"https://example.invalid/{name}.safetensors",
                ),
            ],
        ),
    )


def lay_checkpoint(cache_home: Path, record: ImageGenerationModelRecord) -> None:
    """Write empty files at every path ``record`` declares under ``cache_home``'s image folder."""
    for download in record.config.download:
        target = cache_home / "compvis" / download.file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


class _QueryDouble:
    """The ``to_list`` / ``to_list_with_source`` surface of a reference query."""

    def __init__(self, sourced: list[tuple[GenericModelRecord, str]]) -> None:
        self._sourced = sourced

    def to_list(self) -> list[GenericModelRecord]:
        return [record for record, _source in self._sourced]

    def to_list_with_source(self) -> list[tuple[GenericModelRecord, str]]:
        return list(self._sourced)


class PendingAwareManagerDouble:
    """A reference manager double serving canonical records and whatever pending provider is registered on it.

    ``query`` reads the sources in the selector's order and keeps the first record of each name, the same
    precedence the real manager applies, so a pending record listed first replaces a canonical one.
    """

    def __init__(self, canonical: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord]]) -> None:
        """Serve ``canonical`` as the canonical source, with no provider registered."""
        self._canonical = {category: dict(records) for category, records in canonical.items()}
        self._providers: dict[str, StaticModelProvider] = {}
        self.registrations = 0
        self.queries = 0

    def get_all_model_references(self) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]:
        """Return a copy of the canonical records by category."""
        return {category: dict(records) for category, records in self._canonical.items()}

    def get_provider(self, source_id: str) -> StaticModelProvider | None:
        """Return the provider registered under ``source_id``, or None."""
        return self._providers.get(source_id)

    def register_provider(self, provider: StaticModelProvider, *, replace: bool = False) -> None:
        """Register ``provider`` under its source id and count the registration."""
        self.registrations += 1
        self._providers[provider.source_id] = provider

    def query(self, category: MODEL_REFERENCE_CATEGORY, *, source: str | list[str] = HORDE_SOURCE_ID) -> _QueryDouble:
        """Return ``category``'s records from the selected sources, the first record of each name winning."""
        self.queries += 1
        source_ids = [source] if isinstance(source, str) else list(source)
        sourced: dict[str, tuple[GenericModelRecord, str]] = {}
        for source_id in source_ids:
            if source_id == HORDE_SOURCE_ID:
                records = self._canonical.get(category, {})
            else:
                records = self._providers[source_id].fetch_category(category) or {}
            for name, record in records.items():
                sourced.setdefault(name, (record, source_id))
        return _QueryDouble(list(sourced.values()))


def static_pending_provider(
    records_by_category: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord]],
) -> StaticModelProvider:
    """Return a pending-source provider serving fixed records, standing in for the PRIMARY's queue."""
    return StaticModelProvider(source_id=PENDING_SOURCE_ID, records_by_category=records_by_category)


def serve_pending(
    monkeypatch: pytest.MonkeyPatch,
    records_by_category: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord]],
    *,
    categories: str = "image_generation",
) -> list[dict[str, object]]:
    """Opt ``categories`` into beta and make hordelib build a static pending provider; return the build calls.

    Each entry in the returned list is the keyword arguments one ``build_pending_provider`` call received.
    """
    import hordelib.beta_models as beta_models

    calls: list[dict[str, object]] = []

    def _build(**kwargs: object) -> StaticModelProvider:
        calls.append(kwargs)
        return static_pending_provider(records_by_category)

    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, categories)
    monkeypatch.setattr(beta_models, "build_pending_provider", _build)
    return calls


@pytest.fixture(autouse=True)
def _fresh_merged_view() -> Iterator[None]:
    """Start and end every test with an empty merged-view cache."""
    invalidate_merged_records()
    yield
    invalidate_merged_records()


_IMAGE = MODEL_REFERENCE_CATEGORY.image_generation


def _canonical_manager() -> PendingAwareManagerDouble:
    return PendingAwareManagerDouble(
        {
            _IMAGE: {
                "Canonical-Only": image_record("Canonical-Only"),
                "Shared": image_record("Shared", file_name="old"),
            }
        },
    )


def test_pending_record_replaces_canonical_record_of_the_same_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pending source is read first, so its record wins a name collision and its names are flagged beta."""
    pending_shared = image_record("Shared", file_name="new")
    serve_pending(monkeypatch, {_IMAGE: {"Shared": pending_shared, "Beta-Only": image_record("Beta-Only")}})

    merged = merged_category_records(_canonical_manager(), _IMAGE)  # type: ignore[arg-type]

    assert set(merged.records) == {"Canonical-Only", "Shared", "Beta-Only"}
    assert merged.records["Shared"] is pending_shared
    assert merged.beta_names == frozenset({"Shared", "Beta-Only"})


def test_image_convenience_returns_a_mutable_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typed image mapping belongs to the caller, so changing it leaves the cached view intact."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()

    first = beta_aware_image_records(manager)  # type: ignore[arg-type]
    first.pop("Beta-Only")

    assert "Beta-Only" in beta_aware_image_records(manager)  # type: ignore[arg-type]


def test_provider_read_error_falls_open_to_the_canonical_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending read that raises returns the canonical records with no beta names."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()

    def _unreachable(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("PRIMARY unreachable")

    monkeypatch.setattr(manager, "query", _unreachable)

    merged = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    assert set(merged.records) == {"Canonical-Only", "Shared"}
    assert merged.beta_names == frozenset()


def test_provider_build_error_falls_open_to_the_canonical_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that cannot be built returns the canonical records."""
    import hordelib.beta_models as beta_models

    def _broken(**_kwargs: object) -> None:
        raise ValueError("malformed PRIMARY URL")

    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setattr(beta_models, "build_pending_provider", _broken)

    merged = merged_category_records(_canonical_manager(), _IMAGE)  # type: ignore[arg-type]

    assert set(merged.records) == {"Canonical-Only", "Shared"}


def test_unconfigured_beta_reads_the_canonical_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider build that returns None, as with no PRIMARY URL or key, leaves the canonical records alone."""
    import hordelib.beta_models as beta_models

    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setattr(beta_models, "build_pending_provider", lambda **_kwargs: None)
    manager = _canonical_manager()

    merged = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    assert set(merged.records) == {"Canonical-Only", "Shared"}
    assert not merged.pending_read_ok
    assert manager.registrations == 0


def test_canonical_read_error_propagates_and_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the pending half fails open. A canonical read failure reaches the caller and the next read retries."""
    monkeypatch.delenv(BETA_CATEGORIES_ENV_VAR, raising=False)
    manager = _canonical_manager()
    working_read = manager.get_all_model_references

    def _missing() -> None:
        raise FileNotFoundError("no reference on disk")

    monkeypatch.setattr(manager, "get_all_model_references", _missing)
    with pytest.raises(FileNotFoundError):
        merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    monkeypatch.setattr(manager, "get_all_model_references", working_read)
    assert set(merged_category_records(manager, _IMAGE).records) == {"Canonical-Only", "Shared"}  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bounded", "expected_build_arguments"),
    [
        (
            True,
            {
                "timeout_seconds": BOUNDED_PENDING_TIMEOUT_SECONDS,
                "retry_max_attempts": BOUNDED_PENDING_RETRY_MAX_ATTEMPTS,
            },
        ),
        (False, {}),
    ],
)
def test_bounded_mode_builds_the_bounded_provider(
    monkeypatch: pytest.MonkeyPatch,
    bounded: bool,  # noqa: FBT001 - parametrized
    expected_build_arguments: dict[str, object],
) -> None:
    """Bounded mode builds a single short attempt, and the default mode keeps the provider's own budget."""
    calls = serve_pending(monkeypatch, {_IMAGE: {}})

    merged_category_records(_canonical_manager(), _IMAGE, bounded=bounded)  # type: ignore[arg-type]

    assert calls == [expected_build_arguments]


def test_provider_is_registered_once_per_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads of several opted-in categories share one registered provider."""
    calls = serve_pending(monkeypatch, {_IMAGE: {}}, categories="image_generation,esrgan")
    manager = _canonical_manager()

    merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]
    merged_category_records(manager, MODEL_REFERENCE_CATEGORY.esrgan)  # type: ignore[arg-type]

    assert manager.registrations == 1
    assert len(calls) == 1


def test_cache_hit_until_invalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated read is served from the cache, and invalidation makes the next read query again."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()

    first = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]
    second = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]
    assert second is first
    assert manager.queries == 1

    invalidate_merged_records()
    third = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    assert third is not first
    assert manager.queries == 2


def _fail_first_query(monkeypatch: pytest.MonkeyPatch, manager: PendingAwareManagerDouble) -> None:
    """Make the manager's first pending-first query raise, and later queries succeed."""
    working_query = manager.query
    failures = [ConnectionError("PRIMARY unreachable")]

    def _flaky(category: MODEL_REFERENCE_CATEGORY, *, source: str | list[str] = HORDE_SOURCE_ID) -> object:
        if failures:
            manager.queries += 1
            raise failures.pop()
        return working_query(category, source=source)

    monkeypatch.setattr(manager, "query", _flaky)


def test_default_mode_fail_open_is_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the default mode a failed pending read is not cached, so the next read recovers the beta models."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()
    _fail_first_query(monkeypatch, manager)

    first = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]
    second = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    assert not first.pending_read_ok
    assert "Beta-Only" not in first.records
    assert second.pending_read_ok
    assert "Beta-Only" in second.records
    assert manager.queries == 2


def test_bounded_mode_fail_open_is_cached_until_invalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """In bounded mode a failed pending read is served from the cache until invalidation."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()
    _fail_first_query(monkeypatch, manager)

    first = merged_category_records(manager, _IMAGE, bounded=True)  # type: ignore[arg-type]
    second = merged_category_records(manager, _IMAGE, bounded=True)  # type: ignore[arg-type]

    assert not first.pending_read_ok
    assert second is first
    assert manager.queries == 1

    invalidate_merged_records()
    third = merged_category_records(manager, _IMAGE, bounded=True)  # type: ignore[arg-type]

    assert third.pending_read_ok
    assert "Beta-Only" in third.records


def test_empty_category_is_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A category with no records is not cached, so a reference that loads later is seen on the next read."""
    monkeypatch.delenv(BETA_CATEGORIES_ENV_VAR, raising=False)
    manager = PendingAwareManagerDouble({})

    assert merged_category_records(manager, _IMAGE).records == {}  # type: ignore[arg-type]

    manager._canonical[_IMAGE] = {"Canonical-Only": image_record("Canonical-Only")}
    assert set(merged_category_records(manager, _IMAGE).records) == {"Canonical-Only"}  # type: ignore[arg-type]


def test_cache_is_per_manager_and_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another manager, or a changed opt-in, reads afresh instead of reusing the cached view."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    manager = _canonical_manager()
    merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    other = _canonical_manager()
    assert "Beta-Only" in merged_category_records(other, _IMAGE).records  # type: ignore[arg-type]

    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "")
    assert "Beta-Only" not in merged_category_records(manager, _IMAGE).records  # type: ignore[arg-type]


def test_categories_come_from_the_environment_opt_in_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A category outside the opt-in reads canonical records, even with a pending provider registered."""
    pending = {_IMAGE: {"Beta-Only": image_record("Beta-Only")}}
    serve_pending(monkeypatch, pending, categories="esrgan")
    manager = _canonical_manager()
    manager.register_provider(static_pending_provider(pending))

    merged = merged_category_records(manager, _IMAGE)  # type: ignore[arg-type]

    assert set(merged.records) == {"Canonical-Only", "Shared"}
    assert merged.beta_names == frozenset()
    assert manager.queries == 0


_ESRGAN = MODEL_REFERENCE_CATEGORY.esrgan


def _read_published(directory: Path, category: MODEL_REFERENCE_CATEGORY) -> dict[str, dict[str, object]]:
    return json.loads((directory / f"{category.value}.json").read_text(encoding="utf-8"))


def test_publish_writes_the_pending_records_of_each_opted_in_category(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each opted-in category gets a file of its pending records alone, in the shape ``from_raw`` reads."""
    pending_shared = image_record("Shared", file_name="new")
    serve_pending(
        monkeypatch,
        {_IMAGE: {"Shared": pending_shared, "Beta-Only": image_record("Beta-Only")}, _ESRGAN: {}},
        categories="image_generation,esrgan",
    )

    written = publish_pending_records(_canonical_manager(), tmp_path)  # type: ignore[arg-type]

    assert written == frozenset({_IMAGE, _ESRGAN})
    image_file = _read_published(tmp_path, _IMAGE)
    assert set(image_file) == {"Shared", "Beta-Only"}
    assert _read_published(tmp_path, _ESRGAN) == {}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["esrgan.json", "image_generation.json"]
    served = StaticModelProvider.from_raw(PENDING_SOURCE_ID, {_IMAGE: image_file}).fetch_category(_IMAGE)
    assert served is not None
    assert served["Shared"] == pending_shared


def test_publish_replaces_the_file_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A write that fails before the replace leaves the previous file and no temporary file."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    previous = tmp_path / "image_generation.json"
    previous.write_text('{"Previous": {}}', encoding="utf-8")

    def _interrupted(_source: object, _destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(beta_source.os, "replace", _interrupted)
    with pytest.raises(OSError, match="disk full"):
        publish_pending_records(_canonical_manager(), tmp_path)  # type: ignore[arg-type]

    assert previous.read_text(encoding="utf-8") == '{"Previous": {}}'
    assert [path.name for path in tmp_path.iterdir()] == ["image_generation.json"]


def test_publish_keeps_the_previous_file_when_the_pending_read_fails_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A provider that reports a failed fetch as None fails the read open, and the published file stays."""
    serve_pending(monkeypatch, {_ESRGAN: {}})
    previous = tmp_path / "image_generation.json"
    previous.write_text('{"Previous": {}}', encoding="utf-8")
    manager = _canonical_manager()

    written = publish_pending_records(manager, tmp_path)  # type: ignore[arg-type]

    assert written == frozenset()
    assert not merged_category_records(manager, _IMAGE).pending_read_ok  # type: ignore[arg-type]
    assert previous.read_text(encoding="utf-8") == '{"Previous": {}}'


def test_share_publishes_and_sets_the_directory_for_spawned_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The children inherit the directory the records were written to."""
    serve_pending(monkeypatch, {_IMAGE: {"Beta-Only": image_record("Beta-Only")}})
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, "unrelated")
    monkeypatch.setattr(beta_source, "pending_records_directory", lambda: tmp_path)

    share_pending_records_with_children(_canonical_manager())  # type: ignore[arg-type]

    assert os.environ[BETA_MODELS_DIR_ENV_VAR] == str(tmp_path)
    assert set(_read_published(tmp_path, _IMAGE)) == {"Beta-Only"}
