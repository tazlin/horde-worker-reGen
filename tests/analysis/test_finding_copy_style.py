"""The copy a finding shows a reader follows ``docs/how-to/write-a-finding.md``.

A finding is read by an operator who may not know the worker's internals and may not read English as a
first language, so its plain layer (the headline and the "Do this" line) is held to short sentences,
plain words and no bare code identifiers. The detail layer may name subsystems but keeps the sentence
cap. This module checks the parts of the guide a test can check; the rest is review.

The spec text is checked from :data:`FINDING_SPECS` directly. The headline is written by the detector
per emit, so it is checked from what the detectors actually produce: every detector's golden fixture in
:mod:`tests.analysis.test_detector_contract` is run and each emitted finding is held to the same rules.

A kind whose copy is being reworked can be listed in :data:`_NOT_YET_REWRITTEN` and skipped until the
rewrite lands; a kind not in the list is held to the guide.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from horde_worker_regen.analysis.finding_kinds import FINDING_SPECS, Finding, FindingKind
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from tests.analysis.test_detector_contract import CONTRACTS
from tests.analysis.test_detectors import _diagnose

_MAX_WORDS_PLAIN = 25
_MAX_WORDS_DETAIL = 30
_MAX_SENTENCES_HEADLINE = 1
_MAX_SENTENCES_ACTION = 3

_NOT_YET_REWRITTEN: frozenset[FindingKind] = frozenset()
"""Kinds whose copy is mid-rewrite and temporarily exempt. Empty when every kind follows the guide."""

_BANNED_PLAIN_WORDS: tuple[str, ...] = (
    # Internal mechanisms with a plain-word equivalent.
    "wedge",
    "wedged",
    "reap",
    "reaped",
    "reaps",
    "lane",
    "lanes",
    "head",
    "watchdog",
    "quarantine",
    "quarantined",
    "punt",
    "punted",
    "tick",
    "ticks",
    "latch",
    "latched",
    "storm",
    "residency",
    "admission",
    "admitted",
    "arbiter",
    "backpressure",
    "breaker",
    "spiral",
    "census",
    "dominance",
    "invariant",
    "telemetry",
    "teardown",
    "governor",
    "orphan",
    "orphaned",
    "dispatch",
    "dispatched",
    "preload",
    "preloaded",
    "preloads",
    "headroom",
    "materialized",
    "reconcile",
    "reconciliation",
    "starvation",
    "starved",
    "liveness",
    "pop",
    "pops",
    "popped",
    "slot",
    "slots",
    "co-resident",
    "co-residency",
    "soft reset",
    "save-our-ship",
    "limp-by",
    "give-up",
    "bail-out",
    "abandon ship",
    # Platform and library names the reader is not expected to know.
    "rlimit",
    "rlimit_nofile",
    "wddm",
    "emfile",
    "errno",
    "ipc",
    "asyncio",
)
_BANNED_PLAIN = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(word) for word in _BANNED_PLAIN_WORDS) + r")(?![\w-])",
    re.IGNORECASE,
)

_PLAIN_PUNCTUATION = re.compile(r"[();—]|\s-\s|--|\be\.g\.|\bi\.e\.")
"""Brackets, semicolons, dashes used as punctuation, and Latin abbreviations: start a new sentence."""

_IDENTIFIER = re.compile(
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"  # snake_case
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"  # CamelCase
    r"|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\("  # a dotted call
    r"|\b[a-z_]+_<\w+>\.\w+\b"  # a file pattern such as bridge_<N>.log
)
_BACKTICKED = re.compile(r"`([^`]+)`")
_QUOTED = re.compile(r'"[^"]*"')
"""Text a detector quotes from the log (an exception name, a model name): data, not the writer's words, so
it is exempt from the identifier, banned-word and punctuation rules. It still counts toward sentence length."""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_CONFIG_KEYS: frozenset[str] = frozenset(reGenBridgeData.model_fields)
"""The only identifiers copy may name: the keys of the operator's config file, as written there."""


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_END.split(text.strip()) if sentence]


def _outside_backticks(text: str) -> str:
    return _QUOTED.sub(" quoted ", _BACKTICKED.sub(" ", text))


def _problems(label: str, text: str, *, max_words: int, max_sentences: int | None, plain: bool) -> Iterator[str]:
    """Every way ``text`` breaks the guide, each as one line naming the field and the rule."""
    sentences = _sentences(text)
    if max_sentences is not None and len(sentences) > max_sentences:
        yield f"{label}: {len(sentences)} sentences, at most {max_sentences} allowed"
    for sentence in sentences:
        words = len(sentence.split())
        if words > max_words:
            yield f"{label}: {words}-word sentence, at most {max_words} allowed: {sentence!r}"
    for span in _BACKTICKED.findall(text):
        if span.startswith("horde-log") or span in _CONFIG_KEYS:
            continue
        yield f"{label}: backticked {span!r} is neither a config key nor a horde-log command"
    bare = _outside_backticks(text)
    for match in _IDENTIFIER.finditer(bare):
        yield f"{label}: bare identifier {match.group(0)!r}, put a config key in backticks or use words"
    if plain:
        for match in _BANNED_PLAIN.finditer(bare):
            yield f"{label}: {match.group(0)!r} is an internal term, use the plain word"
        for match in _PLAIN_PUNCTUATION.finditer(bare):
            yield f"{label}: {match.group(0)!r}, start a new sentence instead"


def _spec_problems(kind: FindingKind) -> list[str]:
    spec = FINDING_SPECS[kind]
    problems = list(
        _problems("title", spec.title, max_words=_MAX_WORDS_PLAIN, max_sentences=_MAX_SENTENCES_HEADLINE, plain=True)
    )
    # An empty spec action is allowed where every emit words its own (the guide's measured-fix case); the
    # emitted-copy test then requires the combined "Do this" line to be present.
    problems.extend(
        _problems(
            "action",
            spec.action,
            max_words=_MAX_WORDS_PLAIN,
            max_sentences=_MAX_SENTENCES_ACTION,
            plain=True,
        )
    )
    problems.extend(_problems("detail", spec.detail, max_words=_MAX_WORDS_DETAIL, max_sentences=None, plain=False))
    return problems


def _emit_problems(finding: Finding) -> list[str]:
    problems = list(
        _problems(
            "headline",
            finding.headline,
            max_words=_MAX_WORDS_PLAIN,
            max_sentences=_MAX_SENTENCES_HEADLINE,
            plain=True,
        )
    )
    if finding.title_override:
        problems.extend(
            _problems(
                "title_override",
                finding.title_override,
                max_words=_MAX_WORDS_PLAIN,
                max_sentences=_MAX_SENTENCES_HEADLINE,
                plain=True,
            )
        )
    if not finding.action:
        problems.append("action: empty, say what to do or that no action is needed")
    problems.extend(
        _problems(
            "action",
            finding.action,
            max_words=_MAX_WORDS_PLAIN,
            max_sentences=_MAX_SENTENCES_ACTION,
            plain=True,
        )
    )
    return problems


def _rewritten(kind: FindingKind) -> None:
    if kind in _NOT_YET_REWRITTEN:
        pytest.skip(f"{kind.value}: copy not yet rewritten to docs/how-to/write-a-finding.md")


@pytest.mark.parametrize("kind", list(FindingKind), ids=lambda kind: kind.value)
def test_the_declared_copy_follows_the_guide(kind: FindingKind) -> None:
    """The kind's title and its constant advice read as the guide asks."""
    _rewritten(kind)
    problems = _spec_problems(kind)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("detector_name", sorted(CONTRACTS), ids=lambda name: name)
def test_the_emitted_copy_follows_the_guide(detector_name: str, tmp_path: Path) -> None:
    """What a detector writes per emit reads as the guide asks, checked on its golden fixture.

    A detector may emit more than one kind; each emitted finding is held to its own kind's status, so
    a rewritten kind is checked even when the fixture belongs to a detector whose primary kind is not.
    """
    contract = CONTRACTS[detector_name]
    findings = _diagnose(tmp_path, contract.bridge, contract.child_logs or None)
    checked = [finding for finding in findings.values() if finding.kind not in _NOT_YET_REWRITTEN]
    if not checked:
        pytest.skip(f"{detector_name}: no emitted kind rewritten yet")
    problems = [f"{finding.id}: {problem}" for finding in checked for problem in _emit_problems(finding)]
    assert not problems, "\n".join(problems)


def test_the_pending_list_names_only_declared_kinds() -> None:
    """The skip list cannot hold a kind that no longer exists, so a retired kind leaves it too."""
    assert frozenset(FindingKind) >= _NOT_YET_REWRITTEN


def test_the_rules_catch_what_they_claim_to() -> None:
    """The checker itself: each rule fires on a one-line example and stays quiet on clean copy."""
    clean = "The worker stopped taking jobs for 4 minutes. Raise `queue_size` if this repeats."
    assert not list(_problems("x", clean, max_words=_MAX_WORDS_PLAIN, max_sentences=None, plain=True))
    quoted = 'Loading "Flux.1-Schnell fp8 (Compact)" crashed with "RuntimeError: head_size mismatch".'
    assert not list(_problems("x", quoted, max_words=_MAX_WORDS_PLAIN, max_sentences=None, plain=True))
    cases = {
        "internal term": "The head of the queue was parked.",
        "bare identifier": "Set max_batch lower.",
        "unknown backtick": "Run `abandon_all_hope`.",
        "punctuation": "It stalled (twice); check the card.",
        "long sentence": " ".join(["word"] * (_MAX_WORDS_PLAIN + 1)) + ".",
    }
    for name, text in cases.items():
        assert list(_problems("x", text, max_words=_MAX_WORDS_PLAIN, max_sentences=None, plain=True)), name
