"""Guard: the bootstrap's static feature-extra list must agree with hordelib's requirement registry.

``worker_bootstrap.backend.FEATURE_EXTRAS`` is a static list on purpose: the bootstrap runs via
``uv run --script`` before the project venv exists, so it cannot import hordelib (see
``test_stdlib_only`` and the comment on ``FEATURE_EXTRAS``). hordelib's ``feature_requirements`` registry
remains the source of truth for which features are backend-gated and which horde-engine extra each needs.
This test ties the two together so the static mirror cannot silently drift from the registry: it derives
the expected worker feature extras from the registry via the worker-local re-export aliases in
``capabilities`` and asserts they match.
"""

from __future__ import annotations

from hordelib.api import get_feature_requirement_registry

from horde_worker_regen.capabilities import _HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA, _worker_extra_for_feature
from worker_bootstrap import backend


def _expected_worker_feature_extras() -> set[str]:
    """Worker feature extras implied by hordelib's registry, mapped through the re-export aliases."""
    return {
        _HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA.get(requirement.extra, requirement.extra)
        for requirement in get_feature_requirement_registry().values()
    }


def test_feature_extras_match_hordelib_registry() -> None:
    """backend.FEATURE_EXTRAS equals the set hordelib's requirement registry implies (no drift)."""
    assert set(backend.FEATURE_EXTRAS) == _expected_worker_feature_extras()


def test_every_alias_has_a_registry_extra() -> None:
    """Every horde-engine extra we alias is actually a gated feature in hordelib's registry.

    A leftover alias for an extra hordelib no longer gates would be dead config; catch it.
    """
    registry_extras = {requirement.extra for requirement in get_feature_requirement_registry().values()}
    assert set(_HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA) <= registry_extras


def test_runtime_feature_map_targets_known_extras() -> None:
    """Each gated feature resolves to a worker extra that the bootstrap actually installs."""
    for worker_extra in _worker_extra_for_feature().values():
        assert worker_extra in backend.FEATURE_EXTRAS
