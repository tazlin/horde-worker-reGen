"""The configured alchemy ``forms`` decide what an alchemist offers and which models it must hold.

An operator writes form names by hand, in whichever spelling reads naturally to them, and an empty list
means "everything this worker can serve". These tests pin both halves of that promise: the default set is
the complete list of forms the worker serves, and spelling never changes which form is meant.
"""

from __future__ import annotations

import pytest
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS

from horde_worker_regen.alchemy_forms import (
    DEFAULT_ALCHEMY_FORMS,
    configured_alchemy_forms,
    normalise_alchemy_form_name,
)
from horde_worker_regen.consts import WORKER_KNOWN_EXTRA_ALCHEMY_FORMS


def test_empty_forms_offers_every_default_form() -> None:
    """Leaving ``forms`` unset offers every form the worker serves, not none of them."""
    assert configured_alchemy_forms([]) == frozenset(DEFAULT_ALCHEMY_FORMS)


def test_default_forms_cover_every_form_the_worker_serves() -> None:
    """The default set is every form the horde publishes plus the ones only this worker serves."""
    defaults = frozenset(DEFAULT_ALCHEMY_FORMS)

    assert {form.value for form in KNOWN_ALCHEMY_FORMS} <= defaults
    assert defaults >= WORKER_KNOWN_EXTRA_ALCHEMY_FORMS


def test_default_forms_list_each_form_once() -> None:
    """No form is offered twice, so a count of the default set is a count of distinct forms."""
    assert len(DEFAULT_ALCHEMY_FORMS) == len(set(DEFAULT_ALCHEMY_FORMS))


@pytest.mark.parametrize(
    "written_form",
    [
        pytest.param("post-process", id="hyphenated-yaml-spelling"),
        pytest.param("post_process", id="underscored-spelling"),
        pytest.param("Post-Process", id="mixed-case"),
        pytest.param(KNOWN_ALCHEMY_FORMS.post_process, id="enum-member"),
    ],
)
def test_form_spellings_all_mean_the_same_form(written_form: object) -> None:
    """However an operator spells a form, the worker offers the one form it names."""
    assert normalise_alchemy_form_name(written_form) == KNOWN_ALCHEMY_FORMS.post_process.value
    assert configured_alchemy_forms([written_form]) == frozenset({KNOWN_ALCHEMY_FORMS.post_process.value})
