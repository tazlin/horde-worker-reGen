"""A simple LRU cache for tracking recently used models.

This is implemented as an ordered set — the dictionary values are always
``None`` and only the insertion order of keys matters.
"""

from __future__ import annotations

import collections


class LRUCache:
    """A simple LRU cache (ordered set) for tracking the most recently used models."""

    def __init__(self, capacity: int) -> None:
        """Initializes the LRU cache.

        Args:
            capacity: The maximum number of elements that the cache can hold.
        """
        self.capacity = capacity
        self.cache: collections.OrderedDict[str, None] = collections.OrderedDict()

    def append(self, key: str) -> str | None:
        """Adds an element to the LRU cache, and potentially bumps one from the cache.

        Args:
            key: The key to add to the cache.

        Returns:
            The bumped element, if there was one.
        """
        bumped = None
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            bumped, _ = self.cache.popitem(last=False)
        self.cache[key] = None
        return bumped
