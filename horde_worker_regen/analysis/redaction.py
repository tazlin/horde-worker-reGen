"""Scrub secrets and personal identifiers out of text before it leaves the operator's machine.

A support bundle is sent to a maintainer (often pasted into a public channel), so anything in it must be
safe to share. Two classes of thing are not: **secrets** (the horde ``api_key`` and the CivitAI token,
which let someone impersonate the worker or the operator's accounts) and **personal identifiers** (the
home-directory path and OS username baked into every absolute path, and the worker name). This module
removes both.

The redaction is *value-based first*: when the config is readable we know the exact secret strings, so we
replace those literal values everywhere they appear: config field, an env-var echo in a subprocess
traceback, anywhere. A *pattern backstop* (``api_key: <something>``) catches a foreign bundle whose
config we do not have, so we can still scrub a key whose value we never learned. It deliberately does
**not** scrub by entropy ("any long hex string"): job ids, sha256 checksums, and model names are long
and high-entropy but are exactly the context a maintainer needs, so blanket scrubbing would gut the
bundle's usefulness.

Pure-stdlib and torch-free so it runs in the orchestrator / CLI without the inference stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_REDACTION_MARK = "<REDACTED>"
_HOME_MARK = "<HOME>"
_USER_MARK = "<USER>"
_WORKER_MARK = "<WORKER_NAME>"

# Backstop for a foreign bundle: redact the value after a known secret key, even when we never learned
# the value itself (so it is not in the value-based set). Matches ``api_key: abc`` and ``API_KEY=abc``.
_SECRET_KEY_NAMES = ("api_key", "civitai_api_token", "civit_api_token", "aihorde_api_key", "civit_api_token")
_SECRET_KEY_VALUE_RE = re.compile(
    r"(?P<key>\b(?:api_key|civitai_api_token|civit_api_token|aihorde_api_key)\b)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<val>[^\s,;'\"]+)",
    re.IGNORECASE,
)

# A secret short enough to be a common word is more likely a false positive than a real key; real horde
# keys are 22 chars and CivitAI tokens are long hex. Only value-scrub strings at least this long.
_MIN_SECRET_LEN = 6
# The published anonymous key is not a secret; never treat it as one.
_ANON_API_KEY = "0000000000"


@dataclass
class Redactor:
    """Replaces known secret values and personal identifiers in text, reporting how much it changed."""

    secret_values: list[str] = field(default_factory=list)
    identifier_rules: list[tuple[str, str]] = field(default_factory=list)
    """(literal, replacement) pairs for personal identifiers (home path, username, worker name)."""
    _compiled: tuple[re.Pattern[str] | None, dict[str, str]] | None = field(default=None, init=False, repr=False)

    def _literal_matcher(self) -> tuple[re.Pattern[str] | None, dict[str, str]]:
        """Compile every literal rule (secrets + identifiers) into one alternation, longest match first.

        A single pass over the (potentially tens-of-MB) text replaces all literals at once, which matters
        because logs can be large and this runs interactively from the TUI. Longest-first ordering means a
        username nested inside the home path does not pre-empt the home-path rule.
        """
        if self._compiled is None:
            rules = [(value, _REDACTION_MARK) for value in self.secret_values if value]
            rules += [(literal, replacement) for literal, replacement in self.identifier_rules if literal]
            rules.sort(key=lambda rule: len(rule[0]), reverse=True)
            if rules:
                mapping = dict(rules)
                pattern = re.compile("|".join(re.escape(literal) for literal, _ in rules))
                self._compiled = (pattern, mapping)
            else:
                self._compiled = (None, {})
        return self._compiled

    def scrub(self, text: str) -> tuple[str, int]:
        """Return ``text`` with secrets and identifiers replaced, and the number of replacements made.

        Idempotent: scrubbing already-scrubbed text makes no further changes (the marks contain none of
        the secret values).
        """
        count = 0
        pattern, mapping = self._literal_matcher()

        if pattern is not None:

            def _replace_literal(match: re.Match[str]) -> str:
                nonlocal count
                count += 1
                return mapping[match.group(0)]

            text = pattern.sub(_replace_literal, text)

        # Pattern backstop: redact the value following a known secret key even if we never had the value.
        # Count only real masks: a no-op (already redacted, or the public anonymous key) is left alone.
        def _mask_key_value(match: re.Match[str]) -> str:
            nonlocal count
            value = match.group("val")
            if value in (_REDACTION_MARK, _ANON_API_KEY):
                return match.group(0)
            count += 1
            return f"{match.group('key')}{match.group('sep')}{_REDACTION_MARK}"

        text = _SECRET_KEY_VALUE_RE.sub(_mask_key_value, text)

        return text, count


def _is_real_secret(value: str | None) -> bool:
    """Whether ``value`` is a non-empty, non-anonymous secret long enough to scrub by value safely."""
    return bool(value) and value != _ANON_API_KEY and len(value) >= _MIN_SECRET_LEN


def build_redactor(
    *,
    secrets: list[str | None] | None = None,
    home_path: str | None = None,
    username: str | None = None,
    worker_name: str | None = None,
    redact_identifiers: bool = True,
) -> Redactor:
    """Assemble a :class:`Redactor` from discovered secret values and personal identifiers.

    Secrets are filtered to real, non-anonymous values. Identifier rules are added only when
    ``redact_identifiers`` is set; the home-path rule is registered with both native and forward-slash
    spellings because logs mix them on Windows.

    Args:
        secrets: Candidate secret values (e.g. the configured ``api_key`` and CivitAI token); None/blank/
            anonymous entries are ignored.
        home_path: The operator's home directory, replaced with ``<HOME>`` (both slash styles).
        username: The OS username, replaced with ``<USER>``.
        worker_name: The worker's registered name, replaced with ``<WORKER_NAME>``.
        redact_identifiers: When False, only secrets are scrubbed (paths/username/worker name kept).
    """
    secret_values = [value for value in (secrets or []) if _is_real_secret(value)]
    # mypy: the comprehension above narrows out None, but be explicit for the type checker.
    secret_values = [value for value in secret_values if value is not None]

    identifier_rules: list[tuple[str, str]] = []
    if redact_identifiers:
        if home_path:
            normalized = home_path.rstrip("\\/")
            identifier_rules.append((normalized, _HOME_MARK))
            if "\\" in normalized:
                identifier_rules.append((normalized.replace("\\", "/"), _HOME_MARK))
        # Username after home path so "C:\Users\<USER>" does not partially mask the home rule first.
        if username:
            identifier_rules.append((username, _USER_MARK))
        if worker_name:
            identifier_rules.append((worker_name, _WORKER_MARK))

    return Redactor(secret_values=secret_values, identifier_rules=identifier_rules)
