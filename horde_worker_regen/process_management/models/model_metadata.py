"""Read-only view over the loaded stable diffusion model reference.

Public members:
    ``ModelMetadata``: query surface for model baseline/category lookups.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


class ModelMetadata:
    """Exposes lookups against the stable-diffusion model reference.

    The reference is loaded at startup (or injected from a pre-loaded value in
    tests) and is effectively immutable thereafter. Components that only need
    metadata queries should depend on this class rather than the full context.
    """

    _reference: dict[str, ImageGenerationModelRecord] | None

    def __init__(self) -> None:
        """Create an empty metadata holder. The reference is set by the owner."""
        self._reference = None

    def set_reference(self, reference: dict[str, ImageGenerationModelRecord] | None) -> None:
        """Install the loaded model reference (or clear it)."""
        self._reference = reference

    @property
    def reference(self) -> dict[str, ImageGenerationModelRecord] | None:
        """Return the currently loaded reference, or ``None`` if not yet loaded."""
        return self._reference

    @property
    def has_reference(self) -> bool:
        """Return True if a reference has been loaded."""
        return self._reference is not None

    def require_reference(self) -> dict[str, ImageGenerationModelRecord]:
        """Return the loaded reference, raising if none is loaded.

        Raises:
            RuntimeError: If the reference has not been loaded yet.
        """
        if self._reference is None:
            raise RuntimeError(
                "stable diffusion reference accessed before it was loaded",
            )
        return self._reference

    def get_baseline(
        self,
        model_name: str,
    ) -> KNOWN_IMAGE_GENERATION_BASELINE | str | None:
        """Return the baseline category for ``model_name``, or ``None`` if unknown."""
        if self._reference is None:
            return None
        record = self._reference.get(model_name)
        if record is None:
            return None
        return record.baseline
