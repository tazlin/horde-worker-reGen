"""Per-concern state the inference scheduler carries across scheduling cycles, with the pure arithmetic beside it.

Each module here owns one concern's clocks, tallies and in-flight records (a ledger) plus the pure functions that
price or select over scheduler inputs. The scheduler gathers the inputs from its collaborators and owns every
actuation; nothing in this package sends a message or starts a process.
"""
