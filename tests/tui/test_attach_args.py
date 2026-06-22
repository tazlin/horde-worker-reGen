"""Tests for the dashboard's ``--attach`` argument, including the optional-port reattach shorthand."""

from __future__ import annotations

from horde_worker_regen.tui import app
from horde_worker_regen.tui import socket_protocol as sp


def test_attach_bare_defaults_to_host_port() -> None:
    """A bare ``--attach`` reattaches to the default worker-host address so users need not know the port."""
    args = app._parse_args(["--attach"])
    assert args.attach == f"{sp.DEFAULT_HOST_ADDRESS}:{sp.DEFAULT_HOST_PORT}"


def test_attach_with_explicit_value_is_kept() -> None:
    """An explicit ``host:port`` is preserved for attaching to a non-default host."""
    args = app._parse_args(["--attach", "1.2.3.4:9000"])
    assert args.attach == "1.2.3.4:9000"


def test_no_attach_owns_the_worker() -> None:
    """Without ``--attach`` the value is None, so the dashboard owns the worker as before."""
    assert app._parse_args([]).attach is None
