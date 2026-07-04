"""Derive a process's 'temperature' from its raw state and the worker's queue.

The raw :class:`HordeProcessState` collapses several materially different idle slots onto one
``WAITING_FOR_JOB`` reading: a slot holding a model in VRAM that a queued job is about to use looks
identical to one holding no model at all. That makes a primed, multi-threaded worker read as mostly idle
in the status line and the TUI process table. Temperature restores the distinction the scheduler already
acts on: which slots are doing GPU work now (hot), which are primed for an upcoming popped job (next),
which hold a ready model with nothing queued for it (warm), which are still loading one (priming), and
which are genuinely empty (cold), without adding another process state to the wire protocol.

This module is pure (state names as strings, no torch, no Rich), so the torch-free orchestrator's status
line and the TUI both classify from the same source of truth.
"""

from __future__ import annotations

from strenum import StrEnum


class ProcessTemperature(StrEnum):
    """How 'hot' an inference slot is: whether and how soon it will do GPU work."""

    HOT = "hot"
    """Actively running a job on the GPU (sampling, post-processing, alchemy, or a safety evaluation)."""
    NEXT = "next"
    """Primed: a model is resident or loading, and a queued job targets that model, so it fires next."""
    WARM = "warm"
    """A model is resident and ready, but no queued job needs it yet (kept hot for affinity/reuse)."""
    PRIMING = "priming"
    """Loading or downloading a model (or still starting up): warming toward ready, not usable yet."""
    COLD = "cold"
    """Alive but holds no model: an empty slot with nothing staged."""
    DOWN = "down"
    """Ended, failed, or shutting down: not a usable slot."""


_HOT_STATES = frozenset(
    {
        "INFERENCE_STARTING",
        "POST_PROCESSING",
        "ALCHEMY_STARTING",
        "EVALUATING_SAFETY",
        "SAFETY_STARTING",
        "JOB_RECEIVED",
    },
)
"""States in which the slot is actively occupying the GPU with a job."""

_PRIMING_STATES = frozenset(
    {
        "PRELOADING_MODEL",
        "DOWNLOADING_MODEL",
        "DOWNLOADING_AUX_MODEL",
        "PROCESS_STARTING",
    },
)
"""States in which the slot is loading toward ready but cannot accept work yet."""

_DOWN_STATES = frozenset(
    {
        "PROCESS_ENDING",
        "PROCESS_ENDED",
        "INFERENCE_FAILED",
        "ALCHEMY_FAILED",
        "SAFETY_FAILED",
    },
)
"""Terminal or failed states: the slot is not usable."""


def classify_process_temperature(
    *,
    state: str,
    loaded_model: str | None,
    pending_models: frozenset[str] | set[str],
) -> ProcessTemperature:
    """Classify a slot's temperature from its raw state, resident model, and the pending-job models.

    ``state`` is the :class:`HordeProcessState` name; ``loaded_model`` is the slot's resident model (None
    when it holds none); ``pending_models`` is the set of models named by jobs queued but not yet in
    progress. A ready slot whose resident model appears in ``pending_models`` is :attr:`ProcessTemperature.NEXT`
    (a queued job will dispatch to it); one with a resident model nothing queued needs is
    :attr:`ProcessTemperature.WARM`; one holding no model is :attr:`ProcessTemperature.COLD`.
    """
    if state in _HOT_STATES:
        return ProcessTemperature.HOT
    if state in _DOWN_STATES:
        return ProcessTemperature.DOWN
    if state in _PRIMING_STATES:
        return ProcessTemperature.PRIMING
    if loaded_model is None:
        return ProcessTemperature.COLD
    if loaded_model in pending_models:
        return ProcessTemperature.NEXT
    return ProcessTemperature.WARM


_HOT_PHRASES = {
    "INFERENCE_STARTING": "sampling",
    "POST_PROCESSING": "post-proc",
    "ALCHEMY_STARTING": "alchemy",
    "EVALUATING_SAFETY": "safety",
    "SAFETY_STARTING": "safety",
    "JOB_RECEIVED": "starting",
}
_PRIMING_PHRASES = {
    "PRELOADING_MODEL": "loading",
    "DOWNLOADING_MODEL": "downloading",
    "DOWNLOADING_AUX_MODEL": "aux dl",
    "PROCESS_STARTING": "starting",
}


def temperature_phrase(temperature: ProcessTemperature, state: str) -> str:
    """A short human phrase for the slot, paired with the temperature (e.g. ``Hot · sampling``).

    Hot and priming slots read out what they are doing (sampling, loading); ready slots read out their
    readiness (primed for an upcoming job, ready, idle) since the raw state is a uniform ``WAITING_FOR_JOB``.
    """
    if temperature == ProcessTemperature.HOT:
        return _HOT_PHRASES.get(state, "running")
    if temperature == ProcessTemperature.PRIMING:
        return _PRIMING_PHRASES.get(state, "loading")
    if temperature == ProcessTemperature.NEXT:
        return "primed"
    if temperature == ProcessTemperature.WARM:
        return "ready"
    if temperature == ProcessTemperature.COLD:
        return "idle"
    return state.replace("_", " ").lower()
