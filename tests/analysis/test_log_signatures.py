"""The literal-pin half of the log-line contract: every registered pattern still matches its sample.

The registry in :mod:`horde_worker_regen.analysis.log_signatures` is the only link between a parser here
and an f-string in ``process_management/``. These tests hold two ends of it: each pattern must match the
literal line recorded beside it, and no module in the analysis package may carry a worker-log pattern
that is not in the registry. The other end (that the worker still emits those lines) is
``test_log_contract_dry_run.py``.
"""

from __future__ import annotations

import re

import pytest

from horde_worker_regen.analysis import job_lifecycle
from horde_worker_regen.analysis.log_signatures import SIGNATURES, LogSignature, pattern_for


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


def test_pattern_for_rejects_an_unknown_name() -> None:
    """Asking for a pattern that is not registered fails loudly rather than returning None."""
    with pytest.raises(KeyError):
        pattern_for("not_a_registered_signature")
