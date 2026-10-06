"""Conserved host-RAM physics, including reclaimable checkpoint cache and private working sets."""

from dataclasses import dataclass, field


@dataclass
class HostRamLedger:
    """Represents a host whose available RAM includes reclaimable checkpoint pages.

    Private pages and in-flight transients reduce available RAM. Cached checkpoint pages occupy physical
    memory but remain reclaimable, as with Linux psutil.available. A pressure eviction drops the least recently
    used cache first; the next load of that checkpoint pays a disk-read delay. Each live process has a separate
    private/RSS reading.
    """

    total_mb: float
    foreign_mb: float
    private_mb: dict[int, float] = field(default_factory=dict)
    checkpoints: dict[int, tuple[str, float]] = field(default_factory=dict)
    transients_mb: dict[int, float] = field(default_factory=dict)
    cache_mb: dict[str, float] = field(default_factory=dict)
    """Cached checkpoint pages per model, least recently used first."""
    cache_misses: int = 0
    read_mb_per_second: float = 2000.0
    """Disk read rate a checkpoint load pays for the part of the checkpoint that is not cached."""
    foreign_mb_by_tick: dict[int, float] = field(default_factory=dict)
    """Scripted foreign RAM, keyed by the tick it takes effect on, so host pressure can move inside one run."""
    commit_limit_mb: float | None = None
    """The host's commit limit (physical RAM plus page files), or None for a row that does not model commit.

    When set, every checkpoint a lane maps charges its whole size against the limit for as long as the lane
    holds it, as a copy-on-write file view does on Windows, whether or not its pages are cached."""

    @property
    def available_mb(self) -> float:
        """Return free plus reclaimable pages, after foreign, private and transient charges."""
        return max(
            0.0, self.total_mb - self.foreign_mb - sum(self.private_mb.values()) - sum(self.transients_mb.values())
        )

    @property
    def commit_charged_mb(self) -> float:
        """Return the commit charged: each lane's mapped checkpoint at full size, plus every private allocation."""
        mapped_mb = sum(checkpoint_mb for _model, checkpoint_mb in self.checkpoints.values())
        private_mb = self.foreign_mb + sum(self.private_mb.values()) + sum(self.transients_mb.values())
        return mapped_mb + private_mb

    @property
    def available_commit_mb(self) -> float | None:
        """Return the commit left under the limit, or None when the row does not model commit."""
        if self.commit_limit_mb is None:
            return None
        return max(0.0, self.commit_limit_mb - self.commit_charged_mb)

    @property
    def free_mb(self) -> float:
        """Return unoccupied physical pages, excluding the reclaimable cache."""
        return self.available_mb - sum(self.cache_mb.values())

    def apply_foreign_script(self, tick: int) -> None:
        """Set foreign RAM to the value scripted for ``tick``, if the script names that tick."""
        scripted_mb = self.foreign_mb_by_tick.get(tick)
        if scripted_mb is not None:
            self.foreign_mb = scripted_mb

    def trim_cache(self) -> None:
        """Evict least recently used clean pages until private work and the remaining cache fit in physical RAM."""
        excess = max(0.0, sum(self.cache_mb.values()) - self.available_mb)
        for model in list(self.cache_mb):
            released = min(excess, self.cache_mb[model])
            self.cache_mb[model] -= released
            excess -= released
            if self.cache_mb[model] == 0:
                del self.cache_mb[model]
            if excess <= 0:
                break

    def touch(self, model: str) -> None:
        """Mark ``model``'s cached pages most recently used, so a pressure eviction takes them last."""
        cached_mb = self.cache_mb.pop(model, None)
        if cached_mb is not None:
            self.cache_mb[model] = cached_mb

    def rss_mb(self, process_id: int) -> float:
        """Return the process's private and mapped pages, allowing shared pages in multiple RSS readings."""
        checkpoint = self.checkpoints.get(process_id)
        return self.private_mb.get(process_id, 0.0) + (self.cache_mb.get(checkpoint[0], 0.0) if checkpoint else 0.0)

    def load_delay(self, model: str, checkpoint_mb: float) -> float:
        """Return the disk-read delay a load of ``model`` would pay now, without staging it."""
        return max(0.0, checkpoint_mb - self.cache_mb.get(model, 0.0)) / self.read_mb_per_second

    def read_into_cache(self, model: str, checkpoint_mb: float) -> float:
        """Read ``model``'s uncached pages back into the cache and return the disk-read delay that cost.

        The read a device load of a RAM-held checkpoint pays when its pages were reclaimed: the process keeps
        its private pages, and only the mapped checkpoint is refetched.
        """
        delay = self.load_delay(model, checkpoint_mb)
        self.cache_misses += int(delay > 0)
        self.cache_mb.pop(model, None)
        self.cache_mb[model] = checkpoint_mb
        self.trim_cache()
        return delay

    def stage(self, process_id: int, model: str, checkpoint_mb: float, private_mb: float) -> float:
        """Book a checkpoint swap and return its page-cache-dependent load delay."""
        missing_mb = max(0.0, checkpoint_mb - self.cache_mb.get(model, 0.0))
        self.cache_misses += int(missing_mb > 0)
        self.checkpoints[process_id] = (model, checkpoint_mb)
        self.private_mb[process_id] = 1100.0 + private_mb
        self.cache_mb.pop(model, None)
        self.cache_mb[model] = checkpoint_mb
        self.trim_cache()
        return missing_mb / self.read_mb_per_second

    def evict(self, process_id: int) -> None:
        """Release an idle model's private pages and checkpoint cache; keep the cold interpreter."""
        checkpoint = self.checkpoints.pop(process_id, None)
        if checkpoint is not None:
            self.cache_mb.pop(checkpoint[0], None)
        self.private_mb[process_id] = 1100.0
        self.transients_mb.pop(process_id, None)
