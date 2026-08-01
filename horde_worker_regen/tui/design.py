"""Terminal projection of the Haidra Horde design system.

The web source of truth is ``@haidra-org/horde-design-system`` commit
``33fd5c412e5ba0a9d85867dcd52ee963722aa57d``. Textual cannot reproduce browser blur, so the TUI
projects the same semantic colours and hierarchy into opaque terminal surfaces, borders, and focus.
Keep names semantic: widgets consume theme variables rather than copying palette literals.
"""

from __future__ import annotations

from typing import Any

from textual.app import App
from textual.theme import BUILTIN_THEMES, Theme

DESIGN_SYSTEM_COMMIT = "33fd5c412e5ba0a9d85867dcd52ee963722aa57d"

HORDE_DARK = Theme(
    name="horde-dark",
    primary="#2563eb",
    secondary="#9333ea",
    accent="#06b6d4",
    success="#22c55e",
    warning="#f59e0b",
    error="#ef4444",
    foreground="#f8fafc",
    background="#080c17",
    surface="#111827",
    panel="#172033",
    boost="#26344f",
    dark=True,
    luminosity_spread=0.14,
    variables={
        "horde-brand": "#1d4ed8",
        "horde-brand-hover": "#1e40af",
        "horde-purple": "#9333ea",
        "horde-info": "#06b6d4",
        "horde-edge": "#334155",
        "horde-muted": "#94a3b8",
        "horde-hero": "#172554",
    },
)

HORDE_LIGHT = Theme(
    name="horde-light",
    primary="#1d4ed8",
    secondary="#7e22ce",
    accent="#0e7490",
    success="#15803d",
    warning="#b45309",
    error="#b91c1c",
    foreground="#0f172a",
    background="#f8fafc",
    surface="#ffffff",
    panel="#eef2ff",
    boost="#dbeafe",
    dark=False,
    luminosity_spread=0.12,
    variables={
        "horde-brand": "#1d4ed8",
        "horde-brand-hover": "#1e40af",
        "horde-purple": "#7e22ce",
        "horde-info": "#0e7490",
        "horde-edge": "#cbd5e1",
        "horde-muted": "#475569",
        "horde-hero": "#dbeafe",
    },
)

HORDE_ANSI = Theme(
    name="horde-ansi",
    primary="ansi_blue",
    secondary="ansi_magenta",
    accent="ansi_cyan",
    success="ansi_green",
    warning="ansi_yellow",
    error="ansi_red",
    foreground="ansi_default",
    background="ansi_default",
    surface="ansi_default",
    panel="ansi_default",
    boost="ansi_default",
    dark=True,
    ansi=True,
    variables={
        **BUILTIN_THEMES["ansi-dark"].variables,
        "horde-brand": "ansi_blue",
        "horde-brand-hover": "ansi_bright_blue",
        "horde-purple": "ansi_magenta",
        "horde-info": "ansi_cyan",
        "horde-edge": "ansi_bright_black",
        "horde-muted": "ansi_white",
        "horde-hero": "ansi_default",
    },
)

HORDE_THEMES = (HORDE_DARK, HORDE_LIGHT, HORDE_ANSI)
HORDE_THEME_NAMES = tuple(theme.name for theme in HORDE_THEMES)


def register_horde_themes(app: App[Any]) -> None:
    """Register every Horde theme on ``app``."""
    for theme in HORDE_THEMES:
        app.register_theme(theme)


__all__ = [
    "DESIGN_SYSTEM_COMMIT",
    "HORDE_ANSI",
    "HORDE_DARK",
    "HORDE_LIGHT",
    "HORDE_THEMES",
    "HORDE_THEME_NAMES",
    "register_horde_themes",
]
