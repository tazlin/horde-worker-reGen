"""Out-of-process accelerator inventory, so torch-free callers can read device info without loading torch.

Reading the accelerator inventory (each device's index, name, and total VRAM) goes through hordelib's
backend-agnostic :func:`hordelib.utils.torch_memory.enumerate_accelerators`. That call has to interrogate
the active GPU backend, which loads torch (~500MB RSS) into whatever process makes it. The worker's
long-lived orchestrator process and the interactive config wizard are deliberately torch-free (see
``AGENTS.md``; only the inference/safety children, which need torch for their core function, should pay
that cost), so they must not enumerate accelerators in-process.

This module runs the enumeration in a short-lived subprocess and returns the result as plain validated
data. The subprocess pays the torch cost and frees it on exit, leaving the caller torch-free. It stays
backend-agnostic: the subprocess uses the same hordelib helper, so it reports whatever backend ComfyUI
supports (CUDA/ROCm, Intel XPU, Apple MPS, DirectML, CPU), not just NVIDIA.

Beyond the inventory, the subprocess also measures what a torch context costs on each card, for the streaming
forecast and the other consumers of ``scheduling/context_overhead_model``. Both figures are measured **per
card**, one card at a time: the probe materialises its own context on each device in turn and brings up a
sibling context pinned to that device, so a heterogeneous host gets each card's own figures rather than one
card's applied to all of them. Where a card cannot be measured (no NVML on a card that is not the active torch
device) its entry reports the figures as unmeasured and the consumer falls back to the worker-wide reduction.
The two figures are:

- the *first-context* cost: device-wide used after the probe's own context materialises, less the reading
  taken before any context existed. It sizes ``free_if_alone``. Because it is a before/after delta, whatever
  the device already held when the probe started (a desktop compositor, another tenant's process) cancels
  instead of being charged to the worker.
- the *marginal* cost of each additional sibling context: the device-wide used change across a second
  context-holding process pinned to the card. It sizes ``free_after_model_evict``.

Both figures are the cost of one context, and a clean launch reads them within a few tens of MB of each other.
All three readings are device-wide, so another program that allocates on the card between two of them is
counted in the figure that spans that window. :func:`probe_session_accelerators` compares the two figures per
card and repeats the probe while they disagree (see :func:`select_context_readings`), and logs the raw readings
behind each card's figures once at startup.

The readings come from NVML, which is device-wide on every platform. Where NVML cannot read the card the
probe falls back to torch's free-VRAM reading, which is per-process on Windows WDDM: there the marginal
reads 0 and the worker seeds a conservative per-additional-context constant
(``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``) rather than re-charging the first-context overhead
per context.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from enum import StrEnum, auto

from loguru import logger
from pydantic import BaseModel, ConfigDict

# The child only imports the torch-free ``torch_memory`` submodule, then calls the enumeration (which
# loads torch in *this* short-lived process). The result is emitted on stdout behind a sentinel prefix so
# parsing is robust against any stray stdout (logging/telemetry banners) the import might produce.
_RESULT_PREFIX = "ACCEL_PROBE_JSON:"

# A minimal second process that materialises *its own* backend context (a real kernel launch, so the
# runtime/context fully allocates; enumeration alone does not), announces it, then idles until the probe
# kills it. Run via ``python -c`` from the probe (below), so it stays a plain source string. It names no
# device: the probe launches it under hordelib's device mask for the card being measured, so the card it
# must land on is the only one it can see.
_HOLDER_SOURCE = """
import sys

try:
    import torch

    if torch.cuda.is_available():
        _dev = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        _dev = "xpu"
    else:
        _dev = None
    if _dev is not None:
        # A matmul loads cuBLAS, so the context materialises the way a real inference process's does (a bare
        # elementwise kernel under-counts it slightly); .item() forces the sync.
        _block = torch.ones((512, 512), device=_dev)
        float((_block @ _block).sum().item())
    sys.stdout.write("HOLDER_READY\\n")
    sys.stdout.flush()
    sys.stdin.readline()  # idle until the probe signals shutdown (or kills us)
except BaseException as exc:  # noqa: BLE001
    sys.stderr.write("HOLDER_ERR:" + repr(exc))
    sys.exit(4)
"""

_MAX_NVML_DEVICE_SCAN = 16
"""How far the child walks NVML device indices when reading the pre-context baselines.

hordelib's NVML wrapper exposes no device-count helper, so the child probes handles upward until one comes
back unreadable. The cap keeps a driver that answers for absurd indices from turning the baseline sweep into
an unbounded loop; no host the worker runs on has anything near this many accelerators."""

_PROBE_SOURCE = f"""
import json
import os
import subprocess
import sys
import threading

try:
    def _nvml_used_mb(_index):
        # NVML queries the driver without creating a CUDA context, and reports true device-wide usage on
        # every platform (including Windows WDDM, where the torch reading below is only this process's view).
        # Both properties are load-bearing for the before/after pair: a "before" reading taken through torch
        # would initialise CUDA and so already contain the context being measured, and a pair read on
        # different bases would not subtract. Indexed, so each card's pair is read on that card.
        try:
            from hordelib.utils.nvml import get_device_memory_mb

            _memory = get_device_memory_mb(_index)
            return None if _memory is None else int(_memory.used_mb)
        except BaseException:
            return None

    # Read every device's pre-existing baseline FIRST, before anything can touch any GPU: enumeration and
    # every torch memory helper initialise CUDA as a side effect. The walk stops at the first unreadable
    # handle, which is also how a host without NVML ends up with no baselines at all.
    _baselines = {{}}
    for _scan_index in range({_MAX_NVML_DEVICE_SCAN}):
        _scanned_mb = _nvml_used_mb(_scan_index)
        if _scanned_mb is None:
            break
        _baselines[_scan_index] = _scanned_mb

    from hordelib.utils.torch_memory import enumerate_accelerators
    # Device-wide free (mem_get_info), NOT comfy's per-process view (torch_memory.get_torch_free_vram_mb):
    # only the device-wide figure sees a *sibling* process's context, which is the whole point of the
    # second-context measurement below. (On Windows WDDM even this is per-process, so the marginal reads 0
    # there and the worker falls back; on the Linux servers the worker targets it is true device-wide.)
    from hordelib.api import get_torch_device_free_vram_mb, get_torch_total_vram_mb

    _accelerators = enumerate_accelerators()

    def _device_used_mb():
        # Reports the *active* torch device only: these helpers take no index. It is therefore the fallback
        # basis for the first enumerated card alone; every other card is measured through NVML or reported
        # unmeasured.
        return max(0, int(get_torch_total_vram_mb()) - int(get_torch_device_free_vram_mb()))

    def _materialize_context(_index):
        import torch
        if torch.cuda.is_available():
            _dev = "cuda:" + str(_index)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            _dev = "xpu:" + str(_index)
        else:
            return
        # Match the holder: a matmul loads cuBLAS so this process's context materialises like a real
        # inference process's, and the figures (first-context overhead and the marginal) stay comparable.
        _block = torch.ones((512, 512), device=_dev)
        float((_block @ _block).sum().item())

    def _holder_env(_accelerator):
        # Pin the holder to this one card through hordelib's single source of truth for device masking, so
        # the sibling context lands on the card whose delta is being read. The holder source stays
        # device-agnostic: under the mask the target card is the only one it can see. A backend that needs
        # no masking (cpu, mps) yields an empty patch, leaving the holder where it would have run anyway.
        _env = dict(os.environ)
        try:
            from hordelib.utils.device_pinning import device_pin_env

            _pin, _unused_args = device_pin_env(_accelerator.kind, int(_accelerator.index))
            _env.update(_pin)
        except BaseException:
            return _env
        return _env

    def _measure_marginal(_accelerator, _context_nvml_mb, _overhead_mb, _torch_basis):
        # Marginal cost of an *additional* sibling context on this card: bring up a second process pinned to
        # it that materialises its own context, then measure the device-wide used delta. The after-context
        # reading already includes this process's context and the card's prior tenants, so the delta is one
        # more context's cost (the per-context figure the forecast multiplies by (process count - 1) for
        # free_after_model_evict). Best-effort: any failure leaves it 0 and the worker falls back. The NVML
        # reading after the holder is returned as well, so the parent can log and compare the raw readings.
        _index = int(_accelerator.index)
        _marginal_mb = 0
        _marginal_note = "not attempted"
        _after_holder_mb = None
        _holder = None
        try:
            _holder = subprocess.Popen(
                [sys.executable, "-c", {_HOLDER_SOURCE!r}],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_holder_env(_accelerator),
            )
            _ready = {{"ok": False}}

            def _await_ready():
                # Scan the holder's stdout for the sentinel rather than reading one line: importing hordelib
                # prints telemetry banners to stdout first. An empty read is EOF (the holder died).
                while True:
                    line = _holder.stdout.readline()
                    if not line:
                        return
                    if line.strip() == "HOLDER_READY":
                        _ready["ok"] = True
                        return

            _waiter = threading.Thread(target=_await_ready, daemon=True)
            _waiter.start()
            _waiter.join(timeout=90)  # bound a hung holder so it never costs the basic device inventory
            if not _ready["ok"]:
                _marginal_note = "holder never signalled ready (timeout or early exit)"
            else:
                # Take the delta on the NVML basis, not the torch one. torch's device-free reading is
                # per-process on Windows WDDM, so it cannot see the holder's context at all and the
                # subtraction is structurally zero there; NVML queries the driver and is device-wide on
                # every platform. Falls back to the torch pair only where NVML is unavailable, and only on
                # the active torch device, since those helpers cannot be pointed at another card.
                _after_nvml_mb = _nvml_used_mb(_index)
                if _after_nvml_mb is not None and _context_nvml_mb is not None:
                    _after_holder_mb = _after_nvml_mb
                    _marginal_mb = max(0, _after_nvml_mb - _context_nvml_mb)
                    _marginal_note = f"nvml delta {{_after_nvml_mb}}-{{_context_nvml_mb}}"
                elif _torch_basis:
                    _marginal_mb = max(0, _device_used_mb() - _overhead_mb)
                    _marginal_note = "torch delta (nvml unavailable); per-process on WDDM, may read 0"
                else:
                    _marginal_note = (
                        "no device-wide reading for this card (nvml unavailable and it is not the "
                        "active torch device)"
                    )
                if _marginal_mb == 0 and _marginal_note.startswith(("nvml delta", "torch delta")):
                    _marginal_note += "; measured zero"
        except BaseException as _marginal_exc:
            _marginal_mb = 0
            _marginal_note = "raised: " + repr(_marginal_exc)
        finally:
            if _holder is not None:
                try:
                    _holder.kill()
                    # Drain both pipes after killing. stderr is a pipe so a holder failure is reportable
                    # rather than discarded, and an undrained pipe can block the writer once its buffer
                    # fills, which would turn a noisy holder into the hang the ready-timeout exists to bound.
                    _, _holder_err = _holder.communicate(timeout=10)
                    if _holder_err and _marginal_mb == 0:
                        _marginal_note += " | holder stderr: " + _holder_err.strip().replace("\\n", " ")[-300:]
                except BaseException:
                    pass
        return _marginal_mb, _marginal_note, _after_holder_mb

    def _measure_device(_accelerator):
        # One card's measurement, run to completion before the next card starts. The marginal is a
        # before/after pair around a sibling context, so two cards measured concurrently would let one
        # card's holder start inside the other's window and pollute a shared-driver reading.
        _index = int(_accelerator.index)
        _baseline_mb = _baselines.get(_index)
        # The torch memory helpers report the active device only, so they are a usable fallback basis for
        # the first enumerated card alone. Other cards degrade to "unmeasured" rather than to another
        # card's reading, which the parent turns into a worker-wide fallback.
        _torch_basis = _index == 0
        try:
            _materialize_context(_index)
            _overhead_mb = _device_used_mb() if _torch_basis else 0
            _context_nvml_mb = _nvml_used_mb(_index)
        except BaseException:
            _overhead_mb = 0
            _context_nvml_mb = None
        # First/sole context: the pair is reported raw and the parent takes the delta (see
        # _first_context_overhead_mb): the subtraction is plain arithmetic that must stay directly testable,
        # while this child only ever runs on real hardware. The NVML pair is used when both readings landed,
        # so the two figures share a basis; otherwise the baseline is dropped and the torch reading stands
        # alone (baseline-inclusive, over-counting in the safe direction).
        if _baseline_mb is None or _context_nvml_mb is None:
            _context_used_mb = _overhead_mb
            _baseline_mb = None
        else:
            _context_used_mb = _context_nvml_mb
        _marginal_mb, _marginal_note, _after_holder_mb = _measure_marginal(
            _accelerator,
            _context_nvml_mb,
            _overhead_mb,
            _torch_basis,
        )
        return {{
            "index": _index,
            "name": str(_accelerator.name),
            "total_vram_mb": int(_accelerator.total_vram_mb),
            "kind": str(_accelerator.kind),
            "context_device_used_mb": _context_used_mb,
            "device_baseline_mb": _baseline_mb,
            "after_holder_device_used_mb": _after_holder_mb,
            "marginal_overhead_mb": _marginal_mb,
            "marginal_note": _marginal_note,
        }}

    _payload = [_measure_device(a) for a in _accelerators]
except BaseException as exc:  # noqa: BLE001 - any failure means "no devices"; report and exit non-zero
    print("ACCEL_PROBE_ERR:" + repr(exc), file=sys.stderr)
    sys.exit(3)

print({_RESULT_PREFIX!r} + json.dumps(_payload))
"""

CONTEXT_FIGURE_TOLERANCE_FRACTION: float = 0.25
"""How far apart a card's first-context and marginal figures may read, as a fraction of the smaller one.

Both figures are the cost of one context. A clean launch reads them within about 15 MB of each other on every
card sampled (8 to 32 GB, Linux and Windows), while a foreign allocation inside the probe's window adds
hundreds of MB to one of them. A quarter of the smaller figure leaves room for driver and allocator noise on a
large context and still flags such an allocation."""

CONTEXT_FIGURE_TOLERANCE_FLOOR_MB: int = 64
"""The smallest tolerance (MB) between the two context figures.

A small card's context costs about 150 MB, where a quarter is under 40 MB. The floor keeps ordinary noise on
such a card from being read as a foreign allocation."""

MAX_CONTEXT_PROBE_RUNS: int = 3
"""How many probe runs startup takes at most: the first, plus up to two re-measures while a card is suspect.

Each re-measure costs a torch import and one holder process per card, and runs only after a suspect reading."""


class ProbedAccelerator(BaseModel):
    """One accelerator's identity and capacity, as returned by the out-of-process probe."""

    index: int
    name: str
    total_vram_mb: int
    kind: str = "cuda"
    """The accelerator backend reported by the probe (``cuda``/``rocm``/``xpu``/``directml``/...). Used to
    pin each inference process to its device with the right backend mask. Defaults to ``cuda`` for probes/
    serialisations that predate this field."""
    runtime_overhead_mb: int = 0
    """Approx. VRAM (MB) the *first/sole* fresh torch process's context costs on the card. Sizes
    free-if-alone. Derived by the parent as the before/after delta over :attr:`device_baseline_mb`, so VRAM
    other tenants already held is not charged to the worker. Defaults to 0 for probes/serialisations that
    predate this field."""
    context_device_used_mb: int = 0
    """Device-wide VRAM used (MB) measured after the probe materialised its context: the raw reading behind
    :attr:`runtime_overhead_mb`, before the baseline is netted out. An NVML reading whenever
    :attr:`device_baseline_mb` is set, else the torch reading of the active device."""
    device_baseline_mb: int | None = None
    """Device-wide VRAM used (MB) on this card *before* the probe created any context, read through NVML so
    the reading itself allocates nothing. None when unmeasurable (a non-NVIDIA backend, no driver), where the
    first-context overhead degrades to the baseline-inclusive raw reading."""
    after_holder_device_used_mb: int | None = None
    """Device-wide VRAM used (MB) read through NVML once the sibling holder's context had materialised: the
    raw reading behind :attr:`marginal_overhead_mb`. None when the marginal was not taken as an NVML delta
    (no NVML for the card, a holder that never started, a probe that predates this field)."""
    marginal_overhead_mb: int = 0
    """Approx. VRAM (MB) each *additional* sibling process's context costs, measured by bringing up a second
    context and taking the device-wide used delta. A clean reading is close to :attr:`runtime_overhead_mb`,
    since both are the cost of one context. Sizes free-after-model-evict. 0 when it could not be measured
    (single-context backends, probe failure), where the worker seeds a conservative per-additional-context
    constant (``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``) rather than re-charging the
    first-context ``runtime_overhead_mb`` against every context."""
    marginal_note: str = ""
    """Why :attr:`marginal_overhead_mb` reads what it does, for diagnosis when it reads 0.

    A zero marginal is indistinguishable at the value alone between a genuine measurement, a holder that
    never started, a raised exception, and a platform where the reading cannot see a sibling process at
    all. This carries which of those happened, including the holder's stderr where one failed, so the
    seeded fallback is a visible decision rather than a silent one. Empty for probes that predate it."""


class ContextReadingVerdict(StrEnum):
    """How a card's context figures were checked across the startup probe runs."""

    CONSISTENT = auto()
    """A run's first-context and marginal figures agreed within the tolerance, and that run supplies both."""
    INCONSISTENT = auto()
    """No run agreed. The run with the median first-context figure supplies both figures."""
    UNCHECKED = auto()
    """The first run did not measure both figures as NVML deltas, so they could not be compared and it stands."""


class SelectedContextReading(BaseModel):
    """Represents the probe reading chosen for one card, and the readings it was chosen from."""

    model_config = ConfigDict(frozen=True)

    accelerator: ProbedAccelerator
    """The reading whose figures the session uses."""
    readings: list[ProbedAccelerator]
    """This card's reading from every probe run that reported it, in run order."""
    verdict: ContextReadingVerdict
    """Whether the chosen reading's two figures agreed."""


def _first_context_overhead_mb(*, context_device_used_mb: int, device_baseline_mb: int | None) -> int:
    """Return the first/sole context's own VRAM cost (MB): the reading net of the device's prior baseline.

    The raw reading is device-wide, so it contains every tenant already on the card (a desktop compositor,
    another application, a second worker) as well as the context just created. Charged whole as a per-process
    overhead it is not a measurement of the worker at all, and the error scales with how busy the *host*
    is rather than with anything the worker does: a desktop machine's several-GB baseline, applied to a small
    card, removes more of that card's budget than the models it is being asked to serve. Subtracting the
    pre-context baseline leaves the context's own cost, which is what the term means.

    Clamped at 0: the baseline and the post-context reading are separate samples, so an unrelated tenant
    releasing memory in between can invert them, and a negative overhead would credit the worker VRAM it never
    freed. An unmeasurable baseline (None) leaves the raw reading, which over-counts in the safe direction.
    The opposite case, a tenant allocating in between, is caught by comparing against the marginal
    (:func:`select_context_readings`).

    Args:
        context_device_used_mb (int): Device-wide VRAM used (MB) after the probe materialised its context.
        device_baseline_mb (int | None): Device-wide VRAM used (MB) before it did, or None when unmeasurable.
    """
    if device_baseline_mb is None:
        return max(0, context_device_used_mb)
    return max(0, context_device_used_mb - device_baseline_mb)


def _context_figures_comparable(accelerator: ProbedAccelerator) -> bool:
    """Return whether both context figures of one reading are NVML deltas that can be compared.

    A missing baseline leaves the first-context figure baseline-inclusive, and a 0 marginal is the
    unmeasured fallback. Neither can be compared with the other figure.
    """
    return (
        accelerator.device_baseline_mb is not None
        and accelerator.after_holder_device_used_mb is not None
        and accelerator.marginal_overhead_mb > 0
    )


def _context_figures_agree(accelerator: ProbedAccelerator) -> bool:
    """Return whether one reading's first-context and marginal figures are within the tolerance."""
    smaller_mb = min(accelerator.runtime_overhead_mb, accelerator.marginal_overhead_mb)
    tolerance_mb = max(CONTEXT_FIGURE_TOLERANCE_FLOOR_MB, int(smaller_mb * CONTEXT_FIGURE_TOLERANCE_FRACTION))
    return abs(accelerator.runtime_overhead_mb - accelerator.marginal_overhead_mb) <= tolerance_mb


def _select_card_reading(readings: list[ProbedAccelerator]) -> SelectedContextReading:
    """Return the reading one card's session figures come from, given its readings in run order."""
    first_reading = readings[0]
    if not _context_figures_comparable(first_reading):
        return SelectedContextReading(
            accelerator=first_reading,
            readings=readings,
            verdict=ContextReadingVerdict.UNCHECKED,
        )
    comparable_readings = [reading for reading in readings if _context_figures_comparable(reading)]
    for reading in comparable_readings:
        if _context_figures_agree(reading):
            return SelectedContextReading(
                accelerator=reading,
                readings=readings,
                verdict=ContextReadingVerdict.CONSISTENT,
            )
    return SelectedContextReading(
        accelerator=_median_first_context_reading(comparable_readings),
        readings=readings,
        verdict=ContextReadingVerdict.INCONSISTENT,
    )


def _median_first_context_reading(comparable_readings: list[ProbedAccelerator]) -> ProbedAccelerator:
    """Return the reading with the median first-context figure, the larger of the middle two on an even count.

    Another program can move a reading either way. An allocation inside the probe's window inflates the
    first-context figure, and a release deflates it, down to the clamp at 0. The admission paths subtract the
    figure from the card's capacity, so work that does not fit is admitted when the figure is too low. The
    median drops one outlier on either side, and an even count resolves to the larger figure for the same
    reason. A figure clamped to 0 is not an estimate of a context's cost, so it is passed over while any other
    reading exists.
    """
    measured_readings = [reading for reading in comparable_readings if reading.runtime_overhead_mb > 0]
    ordered_readings = sorted(
        measured_readings or comparable_readings,
        key=lambda reading: reading.runtime_overhead_mb,
    )
    return ordered_readings[len(ordered_readings) // 2]


def select_context_readings(runs: Sequence[Sequence[ProbedAccelerator]]) -> list[SelectedContextReading]:
    """Return each card's chosen reading across one or more probe runs, in the first run's card order.

    The first-context figure and the marginal are both the cost of one context, so a clean reading has them
    within the tolerance (:data:`CONTEXT_FIGURE_TOLERANCE_FRACTION` of the smaller, at least
    :data:`CONTEXT_FIGURE_TOLERANCE_FLOOR_MB`). Per card, the first run whose figures agree supplies both. When
    no run agrees, the run with the median first-context figure supplies both, and a figure clamped to 0 is
    passed over (:func:`_median_first_context_reading`). A card whose first reading
    cannot be compared (no baseline, or an unmeasured marginal) keeps that reading. The first run sets the
    inventory. A card missing from a later run, as after a failed re-measure, keeps the readings it has.

    Args:
        runs (Sequence[Sequence[ProbedAccelerator]]): Each probe run's per-card readings, in run order.

    Returns:
        list[SelectedContextReading]: One entry per card of the first run. Empty when there are no runs.
    """
    if not runs:
        return []
    selections: list[SelectedContextReading] = []
    for first_run_reading in runs[0]:
        card_readings = [reading for run in runs for reading in run if reading.index == first_run_reading.index]
        selections.append(_select_card_reading(card_readings))
    return selections


def _format_reading_mb(reading_mb: int | None) -> str:
    """Return one raw reading for a log line, or ``unread`` when the probe did not take it."""
    return "unread" if reading_mb is None else f"{reading_mb}MB"


def _describe_raw_readings(accelerator: ProbedAccelerator) -> str:
    """Return a reading's three device-used readings and two figures as log text."""
    return (
        f"baseline {_format_reading_mb(accelerator.device_baseline_mb)}, "
        f"after context {_format_reading_mb(accelerator.context_device_used_mb)}, "
        f"after holder {_format_reading_mb(accelerator.after_holder_device_used_mb)}, "
        f"first {accelerator.runtime_overhead_mb}MB, marginal {accelerator.marginal_overhead_mb}MB"
    )


_VERDICT_LOG_TEXT: dict[ContextReadingVerdict, str] = {
    ContextReadingVerdict.CONSISTENT: "The two figures agree",
    ContextReadingVerdict.INCONSISTENT: (
        "The two figures disagree on every run, so the run with the median first-context figure is used"
    ),
    ContextReadingVerdict.UNCHECKED: "Not checked, because the marginal was not measured as an NVML delta",
}


def _log_context_selection(selection: SelectedContextReading, *, run_count: int) -> None:
    """Log one card's startup context figures, the raw readings behind them, and how they were checked."""
    accelerator = selection.accelerator
    run_word = "run" if run_count == 1 else "runs"
    logger.info(
        f"Accelerator {accelerator.index} context cost: {_describe_raw_readings(accelerator)}. "
        f"{_VERDICT_LOG_TEXT[selection.verdict]} ({run_count} probe {run_word}).",
    )
    if selection.verdict is not ContextReadingVerdict.INCONSISTENT:
        return
    every_reading = " | ".join(
        f"reading {position}: {_describe_raw_readings(reading)}"
        for position, reading in enumerate(selection.readings, start=1)
    )
    chosen_position = selection.readings.index(accelerator) + 1
    logger.warning(
        f"Accelerator {accelerator.index} first-context and marginal figures disagree on every probe run "
        f"({every_reading}). Using reading {chosen_position}, which has the median first-context figure. "
        "Another program allocating or releasing memory on this card while the worker starts changes the "
        "figures.",
    )


def _nvml_device_count() -> int:
    """Return how many NVML-visible devices this host has, or 0 when NVML cannot answer.

    Read through hordelib's torch-free NVML wrapper (which has no count helper, so the handles are walked
    until one comes back unreadable), because the caller must stay torch-free and the only thing this figure
    is used for is sizing the probe's subprocess timeout. A non-NVIDIA backend answers 0 and the caller
    falls back to a single-device budget, which is what the timeout was before per-device measurement.
    """
    try:
        from hordelib.utils.nvml import get_device_memory_mb
    except ImportError as import_error:
        logger.debug(f"NVML device count unavailable: {import_error}")
        return 0
    count = 0
    while count < _MAX_NVML_DEVICE_SCAN:
        if get_device_memory_mb(count) is None:
            break
        count += 1
    return count


def probe_accelerators(*, timeout_seconds: float = 120.0) -> list[ProbedAccelerator]:
    """Return the machine's accelerators by enumerating them in a short-lived subprocess.

    Keeps the calling (orchestrator/wizard) process torch-free: the subprocess loads torch, answers, and
    exits. Never raises: any failure (no backend, subprocess error or timeout, malformed output) is
    logged at debug and yields an empty list, so the caller degrades to "no devices detected" rather than
    crashing. The subprocess reuses this interpreter (``sys.executable``), so it sees the same hordelib.

    This is one run, with no check of the context figures. A caller that holds the figures for a session
    uses :func:`probe_session_accelerators`.

    Args:
        timeout_seconds (float): The budget for measuring *one* card. The child measures each card in turn
            (a first context, then a pinned sibling context, per device), so the subprocess is given this
            budget multiplied by the device count; a single-device host is bounded exactly as before.
    """
    device_budget = max(1, _nvml_device_count())
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=timeout_seconds * device_budget,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as run_error:
        logger.debug(f"Accelerator probe subprocess could not run: {type(run_error).__name__} {run_error}")
        return []

    if completed.returncode != 0:
        logger.debug(f"Accelerator probe exited {completed.returncode}: {completed.stderr.strip()}")
        return []

    for line in completed.stdout.splitlines():
        if not line.startswith(_RESULT_PREFIX):
            continue
        try:
            raw_entries = json.loads(line[len(_RESULT_PREFIX) :])
            accelerators = [ProbedAccelerator.model_validate(entry) for entry in raw_entries]
            # Reported once per probe, at info when the marginal is unmeasured. An unmeasured marginal is
            # not a detail: the forecast then prices every additional context from a seeded constant for
            # the whole session, which sizes free-after-model-evict and so the residency and process-count
            # decisions built on it. Left at debug when a real figure landed.
            for accelerator in accelerators:
                if accelerator.marginal_overhead_mb > 0:
                    logger.debug(
                        f"Accelerator {accelerator.index} marginal context cost: "
                        f"{accelerator.marginal_overhead_mb}MB ({accelerator.marginal_note})",
                    )
                else:
                    logger.info(
                        f"Accelerator {accelerator.index} marginal context cost unmeasured; the stream "
                        f"forecast will price additional contexts from its seeded constant. "
                        f"Reason: {accelerator.marginal_note or 'unreported'}",
                    )
            # The child reports the raw before/after pair; the first-context overhead is derived here. An entry
            # carrying no post-context reading (a serialisation that predates the pair) keeps whatever
            # ``runtime_overhead_mb`` it already had rather than being zeroed.
            return [
                (
                    accelerator
                    if accelerator.context_device_used_mb <= 0
                    else accelerator.model_copy(
                        update={
                            "runtime_overhead_mb": _first_context_overhead_mb(
                                context_device_used_mb=accelerator.context_device_used_mb,
                                device_baseline_mb=accelerator.device_baseline_mb,
                            ),
                        },
                    )
                )
                for accelerator in accelerators
            ]
        except (ValueError, TypeError) as parse_error:
            logger.debug(f"Could not parse accelerator probe output: {parse_error}")
            return []

    logger.debug("Accelerator probe produced no result line")
    return []


def probe_session_accelerators() -> list[ProbedAccelerator]:
    """Return the machine's accelerators with context figures checked for a foreign allocation.

    The figures are held for the whole session, so a reading inflated by another program allocating on the
    card during the probe would be charged until the worker restarts. The probe is repeated while any card's
    figures disagree, up to :data:`MAX_CONTEXT_PROBE_RUNS` runs in all, and each card takes its reading from
    :func:`select_context_readings`. The repeat runs a fresh subprocess, since a process creates its context
    once and cannot read the first-context figure again. Each run has the default per-card budget of
    :func:`probe_accelerators`. Logs one INFO line per card with the raw readings, and a WARNING for a card
    whose figures never agreed.

    Returns:
        list[ProbedAccelerator]: The first run's cards, each with its chosen reading. Empty when the first run
            found no accelerators.
    """
    runs: list[list[ProbedAccelerator]] = [probe_accelerators()]
    selections = select_context_readings(runs)
    while len(runs) < MAX_CONTEXT_PROBE_RUNS and any(
        selection.verdict is ContextReadingVerdict.INCONSISTENT for selection in selections
    ):
        runs.append(probe_accelerators())
        selections = select_context_readings(runs)
    for selection in selections:
        _log_context_selection(selection, run_count=len(runs))
    return [selection.accelerator for selection in selections]
