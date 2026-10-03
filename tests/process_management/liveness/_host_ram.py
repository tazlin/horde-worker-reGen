"""Conserved host-RAM physics, including reclaimable checkpoint cache and private working sets."""

from dataclasses import dataclass, field


@dataclass
class HostRamLedger:
    """Represents a host whose available RAM includes reclaimable checkpoint pages.

    Private pages and in-flight transients reduce available RAM. Cached checkpoint pages occupy physical
    memory but remain reclaimable, as with Linux psutil.available. A pressure eviction drops that cache;
    the next load pays a disk-read delay. Each live process has a separate private/RSS reading.
    """

    total_mb: float
    foreign_mb: float
    private_mb: dict[int, float] = field(default_factory=dict)
    checkpoints: dict[int, tuple[str, float]] = field(default_factory=dict)
    transients_mb: dict[int, float] = field(default_factory=dict)
    cache_mb: dict[str, float] = field(default_factory=dict)
    cache_misses: int = 0

    @property
    def available_mb(self) -> float:
        """Return free plus reclaimable pages, after foreign, private and transient charges."""
        return max(
            0.0, self.total_mb - self.foreign_mb - sum(self.private_mb.values()) - sum(self.transients_mb.values())
        )

    @property
    def free_mb(self) -> float:
        """Return unoccupied physical pages, excluding the reclaimable cache."""
        return self.available_mb - sum(self.cache_mb.values())

    def trim_cache(self) -> None:
        """Evict oldest clean pages until private work and the remaining cache fit in physical RAM."""
        excess = max(0.0, sum(self.cache_mb.values()) - self.available_mb)
        for model in list(self.cache_mb):
            released = min(excess, self.cache_mb[model])
            self.cache_mb[model] -= released
            excess -= released
            if self.cache_mb[model] == 0:
                del self.cache_mb[model]
            if excess <= 0:
                break

    def rss_mb(self, process_id: int) -> float:
        """Return the process's private and mapped pages, allowing shared pages in multiple RSS readings."""
        checkpoint = self.checkpoints.get(process_id)
        return self.private_mb.get(process_id, 0.0) + (self.cache_mb.get(checkpoint[0], 0.0) if checkpoint else 0.0)

    def stage(self, process_id: int, model: str, checkpoint_mb: float, private_mb: float) -> float:
        """Book a checkpoint swap and return its page-cache-dependent load delay."""
        missing_mb = max(0.0, checkpoint_mb - self.cache_mb.get(model, 0.0))
        self.cache_misses += int(missing_mb > 0)
        self.checkpoints[process_id] = (model, checkpoint_mb)
        self.private_mb[process_id] = 1100.0 + private_mb
        self.cache_mb[model] = checkpoint_mb
        self.trim_cache()
        return missing_mb / 2000.0

    def evict(self, process_id: int) -> None:
        """Release an idle model's private pages and checkpoint cache; keep the cold interpreter."""
        checkpoint = self.checkpoints.pop(process_id, None)
        if checkpoint is not None:
            self.cache_mb.pop(checkpoint[0], None)
        self.private_mb[process_id] = 1100.0
        self.transients_mb.pop(process_id, None)
