"""Tests for the file-descriptor limit/headroom helpers and the low-headroom warning latch.

These exercise the observability-and-hardening layer for descriptor leaks: the metric helpers (which are
cross-platform), the soft-limit raise (which must only ever raise, never lower, and no-op where there is
no such limit), and the rising-edge warning that flags a climb toward ``EMFILE`` while a slot is still
serving. All run on any platform; the POSIX-specific behaviour is asserted behind a platform guard.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from horde_worker_regen.process_management.fd_limits import (
    FD_HEADROOM_WARN_FRACTION,
    descriptor_headroom_fraction,
    descriptor_soft_limit,
    open_descriptor_count,
    raise_open_file_soft_limit,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap

_POSIX = sys.platform != "win32"


class TestDescriptorMetrics:
    """The per-process descriptor count, soft limit, and headroom math."""

    def test_open_descriptor_count_is_positive(self) -> None:
        """The current process holds at least a few open descriptors on any platform."""
        count = open_descriptor_count()
        assert count is not None
        assert count > 0

    def test_headroom_fraction_math(self) -> None:
        """The fraction is usage/limit, and None whenever it cannot be computed."""
        assert descriptor_headroom_fraction(512, 1024) == pytest.approx(0.5)
        assert descriptor_headroom_fraction(None, 1024) is None
        assert descriptor_headroom_fraction(512, None) is None
        assert descriptor_headroom_fraction(512, 0) is None

    def test_soft_limit_matches_platform(self) -> None:
        """A finite soft limit exists on POSIX; Windows has no RLIMIT_NOFILE, so it is None."""
        limit = descriptor_soft_limit()
        if _POSIX:
            assert limit is None or limit > 0
        else:
            assert limit is None


class TestRaiseSoftLimit:
    """The startup hardening: raise the soft ceiling, never lower it, never raise an exception."""

    def test_never_lowers_the_limit(self) -> None:
        """Raising returns a non-decreasing pair (or None when there is nothing to do)."""
        before = descriptor_soft_limit()
        result = raise_open_file_soft_limit()
        after = descriptor_soft_limit()
        if result is None:
            assert before == after
        else:
            old_soft, new_soft = result
            assert new_soft >= old_soft
            assert after is None or after >= (before or 0)

    @pytest.mark.skipif(_POSIX, reason="Windows has no RLIMIT_NOFILE; the raise is a documented no-op there")
    def test_noop_on_windows(self) -> None:
        """On Windows there is no descriptor ceiling, so both the raise and the limit query are None."""
        assert raise_open_file_soft_limit() is None
        assert descriptor_soft_limit() is None


class TestLowHeadroomWarning:
    """The rising-edge warning latch on ``ProcessMap`` that flags a descriptor climb toward EMFILE."""

    def _info(self, *, open_fds: int | None, fd_soft_limit: int | None) -> SimpleNamespace:
        """A minimal stand-in carrying only the attributes the warning reads."""
        return SimpleNamespace(
            process_id=3,
            loaded_horde_model_name="WAI-NSFW-illustrious-SDXL",
            open_fds=open_fds,
            fd_soft_limit=fd_soft_limit,
            fd_headroom_warned=False,
        )

    def test_rising_edge_latches_then_rearms_with_hysteresis(self) -> None:
        """The warning fires once on the climb, stays latched while high, and re-arms only when well clear."""
        process_map = ProcessMap()
        limit = 1024
        info = self._info(open_fds=100, fd_soft_limit=limit)

        # Comfortable headroom: no warning.
        process_map._warn_on_low_descriptor_headroom(info)  # type: ignore[arg-type]
        assert info.fd_headroom_warned is False

        # Crosses the threshold: latches on the rising edge.
        info.open_fds = int(limit * (FD_HEADROOM_WARN_FRACTION + 0.05))
        process_map._warn_on_low_descriptor_headroom(info)  # type: ignore[arg-type]
        assert info.fd_headroom_warned is True

        # Still high: stays latched (no repeat spam).
        info.open_fds = limit
        process_map._warn_on_low_descriptor_headroom(info)  # type: ignore[arg-type]
        assert info.fd_headroom_warned is True

        # Falls well below the threshold (past the hysteresis band): re-arms for a future climb.
        info.open_fds = int(limit * FD_HEADROOM_WARN_FRACTION * 0.8)
        process_map._warn_on_low_descriptor_headroom(info)  # type: ignore[arg-type]
        assert info.fd_headroom_warned is False

    def test_unknown_headroom_never_warns(self) -> None:
        """A platform without a descriptor limit (open_fds/limit None) is never flagged."""
        process_map = ProcessMap()
        info = self._info(open_fds=None, fd_soft_limit=None)
        process_map._warn_on_low_descriptor_headroom(info)  # type: ignore[arg-type]
        assert info.fd_headroom_warned is False
