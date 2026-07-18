"""Unit tests for the parent-side component-residency map.

The map decodes each child's memory-report residency snapshot into a per-process view the parent queries
for the staged-checkpoint set and the RAM-pressure eviction candidates. These tests pin its update, expiry,
stale-launch rejection, and query behaviour.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot
from horde_worker_regen.process_management.models.component_residency_map import ComponentResidencyMap


def _snapshot(kind: str, identity: str, approx_ram_mb: float = 100.0) -> HeldComponentSnapshot:
    return HeldComponentSnapshot(kind=kind, identity=identity, approx_ram_mb=approx_ram_mb)


class TestUpdateAndQuery:
    """A report replaces a process's residency and the query surfaces it."""

    def test_checkpoint_identities_are_the_staged_models(self) -> None:
        """A checkpoint entry's identity is the bare model name, so it is the staged-model set."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA"), _snapshot("vae", "vae@abc")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelA"})

    def test_non_checkpoint_kinds_excluded_from_staged_set(self) -> None:
        """Bare unet/clip/vae components are not staged whole-model checkpoints."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("vae", "vae@abc"), _snapshot("unet", "ModelA:unet")])

        assert residency.checkpoint_models_held_on([1]) == frozenset()

    def test_identities_held_unions_across_processes_and_filters_by_kind(self) -> None:
        """``identities_held`` unions every process, optionally filtered to a single kind."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA"), _snapshot("vae", "vae@abc")])
        residency.update_from_report(2, 5, [_snapshot("checkpoint", "ModelB")])

        assert residency.identities_held() == frozenset({"ModelA", "vae@abc", "ModelB"})
        assert residency.identities_held(kind="checkpoint") == frozenset({"ModelA", "ModelB"})
        assert residency.identities_held(kind="vae") == frozenset({"vae@abc"})

    def test_checkpoint_models_held_on_restricts_to_named_processes(self) -> None:
        """Only the requested processes' checkpoints are returned."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(2, 5, [_snapshot("checkpoint", "ModelB")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelA"})
        assert residency.checkpoint_models_held_on([1, 2]) == frozenset({"ModelA", "ModelB"})
        assert residency.checkpoint_models_held_on([]) == frozenset()

    def test_later_report_replaces_the_prior_snapshot(self) -> None:
        """A fresh report for the same launch replaces the process's residency wholesale."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelB")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelB"})

    def test_empty_list_clears_residency(self) -> None:
        """A real cache-bearing child reports an empty list when its cache empties, which clears the entry."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(1, 5, [])

        assert residency.checkpoint_models_held_on([1]) == frozenset()


class TestStaleLaunchRejection:
    """A report from a replaced generation of a process id must not overwrite the live one."""

    def test_older_launch_report_is_dropped(self) -> None:
        """A launch identifier below the recorded one is a late message from a replaced process."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 7, [_snapshot("checkpoint", "NewModel")])
        residency.update_from_report(1, 6, [_snapshot("checkpoint", "OldModel")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"NewModel"})

    def test_same_launch_report_is_accepted(self) -> None:
        """A report at the recorded launch is the live generation and updates residency."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 7, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(1, 7, [_snapshot("checkpoint", "ModelB")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelB"})

    def test_newer_launch_report_is_accepted(self) -> None:
        """A higher launch identifier is a fresh generation of the slot and replaces residency."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 7, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(1, 8, [_snapshot("checkpoint", "ModelB")])

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelB"})


class TestNoneReportAndExpiry:
    """A None snapshot leaves prior data untouched; expiry forgets a process entirely."""

    def test_none_report_does_not_clear_existing_residency(self) -> None:
        """None means "no data" (an older child or a process with no cache), not "nothing resident"."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(1, 6, None)

        assert residency.checkpoint_models_held_on([1]) == frozenset({"ModelA"})

    def test_none_report_for_unknown_process_creates_no_entry(self) -> None:
        """A None report from a process with no residency yet leaves the map empty."""
        residency = ComponentResidencyMap()
        residency.update_from_report(9, 1, None)

        assert residency.identities_held() == frozenset()

    def test_expire_process_forgets_its_residency(self) -> None:
        """Expiring a dead/recycled process drops only that process's entry."""
        residency = ComponentResidencyMap()
        residency.update_from_report(1, 5, [_snapshot("checkpoint", "ModelA")])
        residency.update_from_report(2, 5, [_snapshot("checkpoint", "ModelB")])

        residency.expire_process(1)

        assert residency.checkpoint_models_held_on([1]) == frozenset()
        assert residency.checkpoint_models_held_on([2]) == frozenset({"ModelB"})

    def test_expire_unknown_process_is_a_noop(self) -> None:
        """Expiring a process never reported is harmless."""
        residency = ComponentResidencyMap()
        residency.expire_process(42)
        assert residency.identities_held() == frozenset()
