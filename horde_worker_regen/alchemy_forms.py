"""Torch-free helpers for the alchemy ``forms`` setting and the model downloads it implies.

The configured forms decide which alchemy jobs a worker offers, and therefore which auxiliary model
files it must hold on disk before those offers are honest. The parent process (offering) and the
download process (fetching) both read the setting through this module so they cannot drift apart.

Public members:

- :data:`DEFAULT_ALCHEMY_FORMS`: what an alchemist offers when ``forms`` is left empty.
- :func:`configured_alchemy_forms`: the normalised set of configured form names.
- :class:`AuxiliaryFetchNeeds`: which auxiliary model groups a configuration needs fetched.
"""

from __future__ import annotations

from collections.abc import Iterable

from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS
from pydantic import BaseModel, ConfigDict

from horde_worker_regen.consts import WORKER_KNOWN_EXTRA_ALCHEMY_FORMS

DEFAULT_ALCHEMY_FORMS: tuple[str, ...] = tuple(
    dict.fromkeys((*(form.value for form in KNOWN_ALCHEMY_FORMS), *sorted(WORKER_KNOWN_EXTRA_ALCHEMY_FORMS))),
)
"""Forms an alchemist offers when ``bridge_data.forms`` is left unset (an empty list means "all").

Every SDK-known form plus the worker-side extras. The SDK's ``default_forms`` validator does not fire for
the default empty list, so both the dispatch path and the dashboard projection fall back to this set.
"""


def normalise_alchemy_form_name(form: object) -> str:
    """Return a form name in the SDK's canonical spelling (lower case, underscores).

    The same rule the SDK applies when validating ``forms``, so yaml spellings such as ``post-process``
    and enum members compare equal to the canonical value.
    """
    return str(form).lower().replace("-", "_")


def configured_alchemy_forms(forms: Iterable[object]) -> frozenset[str]:
    """Return the configured form names in canonical spelling; empty ``forms`` means every default form."""
    names = [normalise_alchemy_form_name(form) for form in forms]
    return frozenset(names or DEFAULT_ALCHEMY_FORMS)


class AuxiliaryFetchNeeds(BaseModel):
    """Represents which auxiliary model groups a worker configuration needs on disk.

    Derived from bridge data by the parent and carried to the download process unchanged, so the
    download process never has to interpret ``alchemist``, ``forms`` or the lane settings itself.
    """

    model_config = ConfigDict(frozen=True)

    post_processing: bool = False
    """Upscalers and face fixers (the esrgan, gfpgan and codeformer references)."""
    strip_background: bool = False
    """The rembg ``u2net`` weight used by background removal."""
    caption: bool = False
    """The BLIP caption model used by the caption alchemy form."""


DEFAULT_AUXILIARY_FETCH_NEEDS = AuxiliaryFetchNeeds(post_processing=True, strip_background=True)
"""What a download entry point assumes when it is started without bridge data: post-processing on."""

__all__ = [
    "DEFAULT_ALCHEMY_FORMS",
    "DEFAULT_AUXILIARY_FETCH_NEEDS",
    "AuxiliaryFetchNeeds",
    "configured_alchemy_forms",
    "normalise_alchemy_form_name",
]
