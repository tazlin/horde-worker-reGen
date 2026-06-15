"""Textual TUI frontend for the reGen worker.

``horde-worker`` (entry point ``horde_worker_regen.tui.app:main``) launches and supervises the worker
as a child process over a duplex pipe, then renders its live state: an overview, a per-process live
view, the main and subprocess logs, a config editor, and actionable insights. The same app runs in a
terminal or in a browser via ``textual serve``.

The headless ``run_worker`` path is unchanged; the TUI is an optional, attachable frontend.
"""
