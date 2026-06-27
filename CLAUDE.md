# CLAUDE.md: horde-worker-reGen

> [!IMPORTANT]
> **Read `AGENTS.md` before starting any work in this repository.**
> It contains standing architectural contracts, invariants, and cross-repo coordination rules that must be respected.

## Dev Tooling

- **Type checking:** `pyrefly` (not mypy)
- **Formatting:** `ruff format` (not black)
- **Lint + format pipeline:** `ruff format . && ruff check . --fix`
- **Line length:** 119 characters
- **Docstrings:** Google style
