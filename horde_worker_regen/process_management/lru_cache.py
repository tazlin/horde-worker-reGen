"""A simple LRU cache for tracking recently used models."""

from __future__ import annotations

import collections

from horde_worker_regen.process_management.messages import ModelInfo


class LRUCache:
    """A simple LRU cache. This is used to keep track of the most recently used models."""

    def __init__(self, capacity: int) -> None:
        """Initializes the LRU cache.

        Args:
            capacity: The maximum number of elements that the cache can hold.
        """
        self.capacity = capacity
        self.cache: collections.OrderedDict[str, ModelInfo | None] = collections.OrderedDict()

    def append(self, key: str) -> object:
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
