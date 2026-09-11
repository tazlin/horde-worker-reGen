"""Shared log signatures for the pop-governor spell boundaries.

The ``PopGovernorRegistry`` emits a grep-friendly ``ENTER``/``EXIT`` line at each governor spell boundary.
Two analysis surfaces parse those lines: the dominance detector (:mod:`.detectors`) and the duty-cycle
attribution (:mod:`.duty_log_report`), so the regexes and the human label map live here, in a module with
no analysis-package imports, to keep a single source of truth without coupling those two modules to each
other (which would form an import cycle through ``sessions``).
"""

from __future__ import annotations

import re

GOVERNOR_ENTER_RE = re.compile(r"Pop governor ENTER: (?P<name>\w+)")
"""Matches a governor spell opening; ``name`` is the governor's machine key."""

GOVERNOR_EXIT_RE = re.compile(r"Pop governor EXIT: (?P<name>\w+)")
"""Matches a governor spell closing; ``name`` is the governor's machine key."""

GOVERNOR_LABELS = {
    "whole_card_residency": "whole-card residency",
    "large_model_switch": "the large-model switch throttle",
    "large_model_reentry": "the large-model re-entry cooldown",
    "post_inference_backpressure": "post-inference backpressure",
    "unservable_model_holdback": "the unservable-model holdback",
    "consecutive_failure_pause": "the consecutive-failure pause",
    "pop_error_backoff": "pop error-backoff",
    "lora_pop_backoff": "the LoRA pop backoff",
    "self_throttle_pause": "the self-throttle pause",
    "megapixelstep_wait": "the megapixelstep wait",
    "model_stickiness": "model stickiness",
}
"""Machine governor key -> human phrase for report text; an unknown key falls back to the key itself."""

GOVERNOR_PLAIN_PHRASES = {
    "whole_card_residency": "The whole-card reservation",
    "large_model_switch": "The wait between large models",
    "large_model_reentry": "The wait before returning to a large model",
    "post_inference_backpressure": "The wait for the safety check to catch up",
    "unservable_model_holdback": "The hold on a model this card cannot run",
    "consecutive_failure_pause": "The pause after repeated failed jobs",
    "pop_error_backoff": "The wait after horde errors",
    "lora_pop_backoff": "The wait after LoRA errors",
    "self_throttle_pause": "The self-throttle pause",
    "megapixelstep_wait": "The wait for a large job to finish",
    "model_stickiness": "Staying on the current model",
}
"""Machine governor key -> the plain phrase a finding's headline uses, in the words of the style guide."""
