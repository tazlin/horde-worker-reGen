"""The scheduler's admission pipelines as decisions over an immutable per-cycle snapshot.

Preload admission, dispatch admission and clearance each decide over a :class:`SchedulingSnapshot` and return a
plan: a decision, the commands to run, the ledger updates to apply and the records to emit. The scheduler builds
the snapshot at the start of a cycle and executes plans; nothing in this package reads a collaborator or sends a
message. Where a pipeline must act and then decide again against the changed card, the plan says so and the
executor re-snapshots that card, bounded to two phases per pass.
"""
