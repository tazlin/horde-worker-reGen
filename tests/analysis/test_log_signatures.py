"""The literal-pin half of the log-line contract: every registered pattern still matches its sample.

The registry in :mod:`horde_worker_regen.analysis.log_signatures` is the only link between a parser here
and an f-string in ``process_management/``. These tests hold two ends of it: each pattern must match the
literal line recorded beside it, and no module in the analysis package may carry a worker-log pattern
that is not in the registry. The other end (that the worker still emits those lines) is
``test_log_contract_dry_run.py``.
"""

from __future__ import annotations

import ast
import inspect
import re
from types import ModuleType

import pytest

from horde_worker_regen.analysis import detectors, job_lifecycle
from horde_worker_regen.analysis.log_signatures import SIGNATURES, LogSignature, pattern_for

# Patterns detectors.py compiles on its own rather than drawing from the registry, each for a reason the
# fence below accepts: a fragment read out of a message another (registered) pattern already matched, a
# composite that ORs together several already-covered emit sites rather than naming one worker line, an
# OS/third-party error string a worker line merely wraps, or a format string whose live emitter no longer
# exists in this codebase. See the module docstring in ``log_signatures.py`` for what the registry covers.
_ALLOWLISTED_DETECTOR_PATTERNS: dict[str, str] = {
    r"(Queue deadlock detected|Deadlock detected|Save-our-ship)": "composite",
    r"Model: (?P<model>.+?)\. Error:": "fragment",
    r"of which ([\d.]+) MiB is free": "fragment",
    r"Process \d+ has ([\d.]+) GiB memory in use": "fragment",
    r"Too many open files(?! in system)": "external",
    r"Too many open files: '(?P<path>[^']+)'|open file <(?P<file>[^>]+)> in read-only mode": "external",
    r"device_free_vram=(\d+)MB": "fragment",
    (r"GitCommandError|git clone .* failed|Untracked working tree file|unable to checkout working tree"): "external",
    r"untrusted users can only have|maintenance mode|invalid api key|wrong credentials|"
    r"worker .*is not allowed|account .*suspend": "classifier",
}


def _compiled_pattern_literals(module: ModuleType) -> set[str]:
    """The regex source text of every ``re.compile(...)`` call written directly in ``module``'s own file.

    Walks the module's own source rather than its live namespace, so a pattern merely imported from another
    module (e.g. a governor or OOM signature reused here) is not mistaken for one this module owns.
    """
    tree = ast.parse(inspect.getsource(module))
    literals: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            literals.add(node.args[0].value)
    return literals


@pytest.mark.parametrize("signature", list(SIGNATURES.values()), ids=lambda s: s.name)
def test_pattern_matches_its_recorded_sample(signature: LogSignature) -> None:
    """Each registered pattern matches the literal worker line recorded with it."""
    assert signature.pattern.search(signature.sample) is not None, (
        f"{signature.name} no longer matches its sample; the worker emit at {signature.emitter} "
        f"and this pattern have diverged"
    )


@pytest.mark.parametrize("signature", list(SIGNATURES.values()), ids=lambda s: s.name)
def test_signature_names_a_worker_emitter(signature: LogSignature) -> None:
    """Every signature names the ``module:function`` that emits it, so a reword is traceable."""
    assert ":" in signature.emitter
    assert signature.emitter.split(":")[1]


@pytest.mark.parametrize("signature", list(SIGNATURES.values()), ids=lambda s: s.name)
def test_field_signatures_name_a_registered_parent(signature: LogSignature) -> None:
    """A pattern that reads a fragment names the line signature it reads that fragment out of."""
    if signature.field_of is None:
        return
    assert signature.field_of in SIGNATURES


def test_lifecycle_module_owns_no_unregistered_pattern() -> None:
    """The lifecycle parser compiles nothing of its own; every pattern it uses comes from the registry.

    This is the fence that stops a new parser from being added without a sample and an emitter: a bare
    ``re.compile`` in the module would appear here and not in the registry.
    """
    registered = {signature.pattern for signature in SIGNATURES.values()}
    in_module = {value for value in vars(job_lifecycle).values() if isinstance(value, re.Pattern)}
    unregistered = {pattern.pattern for pattern in in_module - registered}
    assert not unregistered, f"unregistered worker-log patterns in job_lifecycle: {sorted(unregistered)}"


def test_detectors_module_owns_no_unregistered_pattern() -> None:
    """detectors.py compiles nothing of its own beyond an explicit, reasoned allowlist.

    Mirrors the ``job_lifecycle`` fence above: every pattern a detector applies to a worker log line must
    come from the registry, so a new parser cannot be added there without a sample and an emitter. The
    allowlist is the few patterns that are not worker-line signatures in their own right (see its comment)
    rather than an escape hatch, so it is checked both ways: nothing outside it is unregistered, and
    nothing in it has quietly gone stale (no longer present in the module at all).
    """
    literals = _compiled_pattern_literals(detectors)
    registered = {signature.pattern.pattern for signature in SIGNATURES.values()}
    unregistered = literals - registered - set(_ALLOWLISTED_DETECTOR_PATTERNS)
    assert not unregistered, f"unregistered worker-log patterns in detectors: {sorted(unregistered)}"
    stale_allowlist = set(_ALLOWLISTED_DETECTOR_PATTERNS) - literals
    assert not stale_allowlist, f"allowlisted patterns no longer present in detectors: {sorted(stale_allowlist)}"


def test_pattern_for_rejects_an_unknown_name() -> None:
    """Asking for a pattern that is not registered fails loudly rather than returning None."""
    with pytest.raises(KeyError):
        pattern_for("not_a_registered_signature")
