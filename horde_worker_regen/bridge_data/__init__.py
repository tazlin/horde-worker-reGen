"""The sub-package for the bridge data."""

AIWORKER_REGEN_PREFIX = "AIWORKER_REGEN_"
"""The documented environment-variable prefix for reGen bridge-data fields (``AIWORKER_REGEN_<FIELD>``)."""

AIWORKER_LEGACY_ENV_PREFIX = "AIWORKER_"
"""The historical prefix accepted for backwards compatibility (``AIWORKER_<FIELD>``)."""

AIWORKER_ENV_PREFIXES: tuple[str, ...] = (AIWORKER_REGEN_PREFIX, AIWORKER_LEGACY_ENV_PREFIX)
"""Accepted env-var prefixes, ordered longest first so a key is matched against the most specific prefix."""
