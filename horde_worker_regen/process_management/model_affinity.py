"""Model->process affinity for the high-throughput (models <= processes) regime.

When the worker serves at least as many inference processes as distinct models, every model
can have a permanent home process and never needs reloading. The scheduler's default preload
target picker (``get_first_available``) prefers empty processes but, when none are free, falls
back to *any* idle process — which can displace a model that is still wanted, forcing a disk
reload of that model when its next job arrives. Under the popper's 2-per-model in-flight cap,
hot models spawn second instances that consume the spare processes, and the fallback then evicts
a cold model's only copy. Measured: a 4-model / 6-process soak did a full disk reload on more than
half its jobs even with VRAM residency on, capping GPU duty cycle by bloating each job's
non-sampling time.

This module computes which processes hold the *last remaining copy* of a still-wanted model, so
the scheduler can mark them off-limits as preload displacement targets. Surplus copies and
processes holding no-longer-wanted models stay displaceable, so spare capacity is still usable.
Pure and table-testable; no scheduler/process imports.
"""

from __future__ import annotations


def compute_protected_processes(
    process_models: dict[int, str | None],
    models_to_load: set[str],
) -> set[int]:
    """Return the process ids that must not be displaced to preserve model affinity.

    A process is protected when it holds the *only* loaded copy of a model that is still in
    ``models_to_load``. Processes holding a surplus (2nd+) copy of a model, or a model no longer
    wanted, are left displaceable so spare slots remain usable for second instances of hot models.

    Args:
        process_models: Inference ``process_id -> loaded model name`` (None if no model loaded).
        models_to_load: The set of models the worker is currently configured to serve.

    Returns:
        The set of process ids to add to a preload target's ``disallowed`` set.
    """
    procs_by_model: dict[str, list[int]] = {}
    for process_id, model in process_models.items():
        if model is None or model not in models_to_load:
            continue
        procs_by_model.setdefault(model, []).append(process_id)

    protected: set[int] = set()
    for procs in procs_by_model.values():
        # Keep exactly one copy pinned (the lowest id, for determinism); extras stay displaceable.
        protected.add(min(procs))
    return protected


def affinity_active(num_models_to_load: int, num_inference_processes: int) -> bool:
    """Whether model->process affinity applies (only when every model can have a home).

    With more models than processes, models must share processes and reloading is unavoidable
    (the existing ``horde_model_stickiness`` path handles that slow-disk regime); pinning would
    only deadlock preloads, so affinity is off there.
    """
    return 0 < num_models_to_load <= num_inference_processes
