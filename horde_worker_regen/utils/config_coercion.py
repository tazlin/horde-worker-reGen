"""Shared coercion for partially-mocked config values and startup measurements."""

from __future__ import annotations


def config_number(value: object) -> float | None:
    """Return ``value`` as a float when it is a real (non-bool) int/float, else None.

    Config attributes and startup measurements may arrive partially mocked (a ``Mock``, ``None``, or a
    ``bool``, which is an ``int`` subclass) in tests and during a reload. The budget and overhead gates
    treat any such non-numeric reading as "unset" and fall back to their safe default rather than acting
    on it. Callers apply their own threshold (``>= 0``, ``> 0``, ...) to the returned number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
