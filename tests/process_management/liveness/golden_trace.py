"""A per-tick record of what the scheduler decided and actuated while a dispatch-world scenario ran.

The trace is the parity oracle the admission-pipeline redesign is cut against: a refactor that only moves
code produces a byte-identical trace, and any difference is a behavior change that has to be named. What it
records per tick is the worker's own disclosure surface plus every command that left the parent:

- the decision-sink verdicts and the resource-state transitions the scheduler emitted,
- every reclaim rung the ladder actually performed through the world's recording actuator,
- every arbiter command batch run through ``_execute_preload_actuations``, with what each batch executed,
- every control message the lanes' pipes received, by flag, lane, model, job and, on a dispatch, the
  retention grant it carried,
- every dispatch, as the model and lane it was seated on and the jobs that first reached sampling,
- the safety context leaving and returning to the card.

Two things are normalised so a trace is comparable across runs rather than across one process: job ids are
random per run, so each is aliased by the order the trace first mentions it (``job-1``, ``job-2``, ...), and
every float is rounded to one decimal, which is finer than any figure the scheduler acts on and coarse enough
that a last-bit difference in an unrelated arithmetic path is not a diff.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordePreloadInferenceModelMessage,
    HordeStageModelMixin,
    HordeTextEncodeControlMessage,
    HordeVaeEncodeControlMessage,
)
from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommand

_JOB_CARRYING_CONTROL_MESSAGES = (
    HordePreloadInferenceModelMessage,
    HordeInferenceControlMessage,
    HordeTextEncodeControlMessage,
    HordeVaeEncodeControlMessage,
)
"""The control messages that name the job they are for, so a trace can attribute a send to a job."""

if TYPE_CHECKING:
    from tests.process_management.liveness._dispatch_world import _DispatchWorld

TRACE_SCHEMA_VERSION = 2
"""Bumped when the recorded field set changes, so a stale golden fails loudly instead of diffing as noise."""

_FLOAT_PLACES = 1
"""Decimal places every recorded float is rounded to."""

_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
)
"""Job ids as they appear inside subjects, reasons and decision inputs."""

TraceValue = str | int | float | bool | None | list["TraceValue"] | dict[str, "TraceValue"]
"""Anything a trace holds: JSON scalars and the containers of them, and nothing else."""


def _command_fields(command: ActuatorCommand) -> dict[str, Any]:
    """One arbiter command as its three identifying fields."""
    return {
        "kind": str(command.kind),
        "device_index": command.device_index,
        "target_process_id": command.target_process_id,
    }


class TraceRecorder:
    """Records one dispatch-world run as a per-tick trace, and renders it as stable JSON.

    Attaching wraps two seams on the world's scheduler (the arbiter command executor) and reads three the
    world already keeps (the decision and resource-state sinks, the ladder actuations, the safety events);
    the lanes' control messages are read off the recording pipes the world's processes already send through.
    Nothing here changes a decision: every wrap forwards its call unchanged and records the result.
    """

    def __init__(self, world: _DispatchWorld) -> None:
        """Attach to ``world``, taking the seams the trace is read from."""
        self._world = world
        self._aliases: dict[str, str] = {}
        self._ticks: list[dict[str, Any]] = []
        self._arbiter_batches: list[tuple[int, dict[str, Any]]] = []
        self._decisions_consumed = 0
        self._resource_states_consumed = 0
        self._ladder_consumed = 0
        self._dispatch_lanes_consumed = 0
        self._safety_pauses_consumed = 0
        self._safety_restores_consumed = 0
        self._first_dispatch_seen: set[str] = set()
        self._messages_consumed: dict[int, int] = {}
        self._wrap_arbiter_executor()

    # -- seams --------------------------------------------------------------------------------------------

    def _wrap_arbiter_executor(self) -> None:
        """Record every arbiter command batch the scheduler runs, forwarding the call unchanged."""
        scheduler = self._world.scheduler
        original = scheduler._execute_preload_actuations

        def recording_executor(
            commands: tuple[ActuatorCommand, ...],
            *,
            device_index: int | None,
            for_head_of_queue: bool,
        ) -> tuple[ActuatorCommand, ...]:
            executed = original(commands, device_index=device_index, for_head_of_queue=for_head_of_queue)
            self._arbiter_batches.append(
                (
                    self._world.tick,
                    {
                        "device_index": device_index,
                        "for_head_of_queue": for_head_of_queue,
                        "requested": [_command_fields(command) for command in commands],
                        "executed": [_command_fields(command) for command in executed],
                    },
                ),
            )
            return executed

        scheduler._execute_preload_actuations = recording_executor  # type: ignore[method-assign]

    # -- normalisation ------------------------------------------------------------------------------------

    def _alias(self, job_id: str) -> str:
        """The stable name for one job id, assigned by the order the trace first mentions it."""
        alias = self._aliases.get(job_id)
        if alias is None:
            alias = f"job-{len(self._aliases) + 1}"
            self._aliases[job_id] = alias
        return alias

    def _normalise_text(self, text: str) -> str:
        """Replace every job id in ``text`` with its alias, including the eight-character short form."""
        aliased = _UUID_PATTERN.sub(lambda match: self._alias(match.group(0)), text)
        for job_id, alias in self._aliases.items():
            aliased = aliased.replace(job_id[:8], alias)
        return aliased

    def _normalise(self, value: object) -> TraceValue:
        """Round floats, alias job ids in strings, and leave every other scalar as it stands."""
        if isinstance(value, bool) or value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, _FLOAT_PLACES)
        if isinstance(value, str):
            return self._normalise_text(value)
        if isinstance(value, dict):
            return {str(key): self._normalise(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, list | tuple):
            return [self._normalise(item) for item in value]
        return self._normalise_text(str(value))

    # -- per-tick capture ---------------------------------------------------------------------------------

    def _new_decisions(self) -> list[dict[str, Any]]:
        records = self._world.decision_records[self._decisions_consumed :]
        self._decisions_consumed = len(self._world.decision_records)
        return [
            {
                "kind": str(record.decision_kind),
                "subject": self._normalise(record.subject),
                "verdict": str(record.verdict),
                "reason": self._normalise(record.reason),
                "inputs": self._normalise(record.inputs),
            }
            for record in records
        ]

    def _new_resource_states(self) -> list[dict[str, Any]]:
        records = self._world.resource_state_records[self._resource_states_consumed :]
        self._resource_states_consumed = len(self._world.resource_state_records)
        return [
            {
                "kind": str(record.state_kind),
                "state": self._normalise(record.state),
                "device_index": record.device_index,
                "reason": self._normalise(record.reason),
                "inputs": self._normalise(record.inputs),
            }
            for record in records
        ]

    def _new_ladder_actuations(self) -> list[dict[str, Any]]:
        records = self._world.ladder_actuations[self._ladder_consumed :]
        self._ladder_consumed = len(self._world.ladder_actuations)
        return [
            {
                "kind": str(record.kind),
                "tenant": record.tenant_label,
                "target_process_id": record.target_process_id,
            }
            for record in records
        ]

    def _new_arbiter_batches(self, tick: int) -> list[dict[str, Any]]:
        batches = [payload for batch_tick, payload in self._arbiter_batches if batch_tick == tick]
        self._arbiter_batches = [entry for entry in self._arbiter_batches if entry[0] != tick]
        return [self._normalise(batch) for batch in batches]

    def _new_messages(self) -> list[dict[str, Any]]:
        """Every control message the lanes' pipes received since the last capture, in lane order."""
        messages: list[dict[str, Any]] = []
        for process_id, process in sorted(self._world._process_map.items()):
            pipe: object = process.pipe_connection
            if not isinstance(pipe, Mock):
                continue
            calls = pipe.send.call_args_list
            consumed = self._messages_consumed.get(process_id, 0)
            for call in calls[consumed:]:
                message = call.args[0] if call.args else None
                if not isinstance(message, HordeControlMessage):
                    continue
                model: str | None = None
                if isinstance(message, HordeControlModelMessage | HordeStageModelMixin):
                    model = message.horde_model_name
                job_id: str | None = None
                if isinstance(message, _JOB_CARRYING_CONTROL_MESSAGES) and message.sdk_api_job_info.id_ is not None:
                    job_id = self._alias(str(message.sdk_api_job_info.id_))
                keep_resident: bool | None = None
                if isinstance(message, HordeInferenceControlMessage):
                    keep_resident = message.keep_model_resident_after
                messages.append(
                    {
                        "process_id": process_id,
                        "control_flag": message.control_flag.name,
                        "model": model,
                        "job": job_id,
                        "keep_model_resident_after": keep_resident,
                    },
                )
            self._messages_consumed[process_id] = len(calls)
        return messages

    def _new_dispatches(self) -> list[dict[str, Any]]:
        seated = self._world.dispatch_lanes[self._dispatch_lanes_consumed :]
        self._dispatch_lanes_consumed = len(self._world.dispatch_lanes)
        return [{"model": model, "lane": lane} for model, lane in seated]

    def _new_sampling_starts(self) -> list[str]:
        started = [job_id for job_id in self._world.first_dispatch if job_id not in self._first_dispatch_seen]
        self._first_dispatch_seen.update(started)
        return [self._alias(job_id) for job_id in started]

    def _new_safety_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for _tick, _now, owner in self._world.safety_pause_events[self._safety_pauses_consumed :]:
            events.append({"event": "pause", "owner": str(owner)})
        self._safety_pauses_consumed = len(self._world.safety_pause_events)
        for _entry in self._world.safety_restore_events[self._safety_restores_consumed :]:
            events.append({"event": "restore", "owner": None})
        self._safety_restores_consumed = len(self._world.safety_restore_events)
        return events

    def capture_tick(self) -> None:
        """Fold everything the world and the scheduler produced during the tick just run into the trace."""
        tick = self._world.tick
        sections: dict[str, Any] = {
            "decisions": self._new_decisions(),
            "resource_states": self._new_resource_states(),
            "ladder": self._new_ladder_actuations(),
            "arbiter": self._new_arbiter_batches(tick),
            "messages": self._new_messages(),
            "dispatches": self._new_dispatches(),
            "sampling_started": self._new_sampling_starts(),
            "safety": self._new_safety_events(),
        }
        populated = {name: value for name, value in sections.items() if value}
        if not populated:
            return
        self._ticks.append({"tick": tick, **populated})

    # -- readback -----------------------------------------------------------------------------------------

    def trace(self, *, scenario: str, ticks: int) -> dict[str, Any]:
        """The recorded run as a plain dictionary, ready to serialise."""
        return {
            "schema": TRACE_SCHEMA_VERSION,
            "scenario": scenario,
            "ticks_driven": ticks,
            "jobs": len(self._aliases),
            "ticks": self._ticks,
        }


async def record(world: _DispatchWorld, ticks: int, *, scenario: str = "") -> dict[str, Any]:
    """Drive ``world`` for ``ticks`` scheduling ticks and return the trace of what the scheduler did.

    The world is stepped through its own ``step``, so what is recorded is a run of the ordinary tick, not a
    special recording mode. A caller that has to inject work mid-run drives the world itself and calls
    :meth:`TraceRecorder.capture_tick` after each step.
    """
    recorder = TraceRecorder(world)
    for _ in range(ticks):
        await world.step()
        recorder.capture_tick()
    return recorder.trace(scenario=scenario, ticks=ticks)


def serialise(trace: dict[str, Any]) -> str:
    """The trace as the JSON a golden file holds: sorted keys, two-space indent, one trailing newline."""
    return json.dumps(trace, indent=2, sort_keys=True) + "\n"


def _tick_index(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(entry["tick"]): entry for entry in trace.get("ticks", [])}


def diff(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    """Render the first divergence between two traces, with both sides of it; empty when they match.

    A trace is long enough that a whole-file diff is unreadable, and a scheduling difference propagates: the
    first tick that differs is the one that explains the rest, so that is the tick rendered.
    """
    for field in ("schema", "scenario", "ticks_driven", "jobs"):
        if expected.get(field) != actual.get(field):
            return f"trace header differs at {field}: expected {expected.get(field)!r}, got {actual.get(field)!r}"

    expected_ticks = _tick_index(expected)
    actual_ticks = _tick_index(actual)
    for tick in sorted(set(expected_ticks) | set(actual_ticks)):
        expected_entry = expected_ticks.get(tick)
        actual_entry = actual_ticks.get(tick)
        if expected_entry == actual_entry:
            continue
        rendered_expected = "(no recorded activity)" if expected_entry is None else serialise(expected_entry)
        rendered_actual = "(no recorded activity)" if actual_entry is None else serialise(actual_entry)
        return (
            f"traces diverge at tick {tick}\n"
            f"--- expected (golden) ---\n{rendered_expected}\n"
            f"--- actual (this run) ---\n{rendered_actual}"
        )
    return ""
