# The Fixed Model Pool

## The problem: swap cost versus queue exhaustion

A worker that serves more models than it has inference processes cannot keep them all resident. Every time a
popped job needs a model that is not loaded, a process must unload one model and load another, and on a slow
disk that swap can cost minutes during which the card does no useful work. The obvious defence, offering only a
handful of models, trades that swap cost for a different loss: the horde may have little or no queued work for
those few models, so the worker idles waiting for jobs it can run swap-free.

The fixed model pool is the deliberate middle path. The worker commits to a bounded set of **seats**, each
holding one model it keeps ready to serve, and shapes its pop requests so the horde preferentially returns work
for the seated models. The commitment is what makes the residency worth holding: because the worker keeps
advertising a seated model, its weights stay staged, and jobs for it arrive without a per-job swap. A separate
lane keeps reaching the wider demand so a seat set does not blind the worker to everything else.

The whole subsystem is off by default (`model_pool.enabled: false`), leaving model selection exactly as it was.
It supersedes the older `model_stickiness` knob, which only biased a single pop toward the already-loaded
models rather than committing to a standing set; a worker still carrying a positive `model_stickiness` with the
pool off is migrated onto a modest pool automatically (see
[Performance and Backpressure → Model stickiness](performance_and_backpressure.md#model-stickiness)).

## Seats, bench, and decay

The engine that owns seat state,
[`ModelPool`][horde_worker_regen.process_management.scheduling.model_pool.ModelPool], is pure: it reads no
clock, imports no torch or config, and is advanced one tick at a time against a freshly ranked candidate set by
the manager. Every seat change it makes is returned as a transition the caller logs and ledgers, so the pool's
behaviour is fully replayable from its inputs.

Seats fill in a fixed precedence:

1. **Manual pins first**, in affinity order. A pin is a model the operator names in `model_pool.pinned`, each
   with a persistent `affinity` in the half-open range `(0, 1]`. Affinity is a bias, never a lock: a higher
   value claims its seat earlier, extends the model's rotation window, and bolsters it in a re-contest, but a
   sufficiently stronger candidate can still take the seat, and a benched pin has to re-earn its place.
2. **Ranker candidates next**, when `ranker_enabled`, filling the seats the pins do not claim with the
   highest-scoring models the demand ranker returns.

Once seated, a model is protected by a **minimum dwell** (`min_dwell_minutes`) so a fresh seat is never churned
before it has had a fair chance to earn its place. After the dwell it becomes eligible for **rotation**: a
timed re-contest (`rotation_minutes`, extended by a pin's affinity) against the best challenger, which the
incumbent loses only to a meaningfully stronger candidate. A seat that stops earning its place is **demoted**:
sustained empty pops against near-zero measured demand, or a stretch with no fulfillment at all, unseat the
model. A demoted model **benches** with a cooldown, so it does not immediately re-seat and thrash between the
seat and the bench.

## The two pop lanes and the demand-weighted interleave

While the pool holds seats, a pop advertises one of two lanes, chosen by
[`decide_pool_lane`][horde_worker_regen.process_management.jobs.pool_lanes.decide_pool_lane]:

- The **fixed lane** narrows the offer to the seated models, so the horde returns work the card runs swap-free.
  A fixed-lane pop that comes back empty is the signal that charges a seat toward demotion.
- The **free lane** advertises the models the pool is *not* seating, so cold and rare-model demand still reaches
  the worker and a seated model does not monopolise the offer. An empty free-lane pop never charges a seat.

The worker interleaves the two as a smooth weighted round-robin. The fixed lane is weighted by how many seated
models it can offer this cycle; the free lane by how many inference slots are not committed to a seat, plus a
small boost when the recent run of fixed-lane pops all came back empty (so a fixed set the horde is not feeding
yields the offer back to the wider set rather than starving the worker). Two edge rules override the
round-robin: an empty fixed offer forces the free lane and an empty free offer forces the fixed lane, and when
both are empty the worker advertises the full eligible set. The TUI surfaces which lane the current pop used.

This lane choice **narrows the pop advertisement**, so it is one of the pop-shaping inputs described in
[Performance and Backpressure](performance_and_backpressure.md). The narrowing runs on top of the ordinary pop
gauntlet, not instead of it.

## Ranker scoring: demand per worker, weighted by local speed

The ranker, [`rank_candidates`][horde_worker_regen.process_management.scheduling.pool_ranker.rank_candidates],
orders the locally-servable candidates by a single deliberately-simple score. Live per-model demand comes from
[`ModelDemandPoller`][horde_worker_regen.process_management.scheduling.model_demand_poller.ModelDemandPoller],
which reads the horde's `/v2/status/models` endpoint on a jittered interval (`demand_poll_seconds`, floored at
60 seconds) and keeps the most recent good reading as an immutable snapshot.

- **Demand** is `log1p(queued_per_worker)`, where `queued_per_worker = queued / (worker_count + 1)`. Dividing
  by the serving-worker count is what keeps the pool from piling onto a model that already has ample workers;
  a model absent from the snapshot carries demand zero (the horde is not asking for it).
- **Speed** is this card's expected sampling rate for the model, normalised across the candidate set to
  `(0, 1]`. An unmeasured model takes the neutral midpoint rather than being excluded.
- **Score** is `demand * (0.5 + 0.5 * speed_normalised)`, so demand dominates while speed only breaks ties and
  tilts the order toward models this card runs efficiently. A zero-demand model always scores zero.

Ties in score are broken first by **shared-VAE cluster size** (favouring models that share components with
others the worker holds, which the component cache can keep warm) and then by name for determinism. The ranker
is pure and refuses to invent a job signature; the adapter that turns a bare model name into an expected rate
lives in [`pool_wiring.py`][horde_worker_regen.process_management.scheduling.pool_wiring], which scores an SDXL
model at 1024x1024 and every other baseline at 512x512 against the performance model.

## Rescue: a time-boxed seat for the most-starved model

Rescue (`rescue_enabled`, off by default) lets a single **starved** model briefly borrow the weakest ranker
seat so otherwise-impossible jobs still get done. A model counts as starved when its requester wait reaches
`rescue_eta_seconds`; the rescued model holds its borrowed seat for `rescue_window_minutes` and is then released
back to the ranker. Rescue never displaces a manual pin, and it engages at most one seat, so it donates spare
capacity to the queue's worst case without dismantling the committed set.

## Auto-download within a session budget

By default the pool only ever seats models already on disk (`download_budget_gb: 0.0`). A positive budget lets
the ranker seat a high-demand model the worker does not yet hold: the candidate becomes a **pending-download
seat**, the download is requested through the ordinary background download process, and the seat resolves once
the weights land (or is abandoned if the download fails). The budget bounds how much disk the pool may spend
this way over the session. This is the only path by which the pool itself initiates a download; see
[Model Downloads → Pool-initiated downloads](model_downloads.md#pool-initiated-downloads-within-a-budget).

## Residency protection and the pressure-yield precedence

A seat is a standing commitment, so the scheduler protects a seated model's staged weights from the ordinary
idle-unload that would otherwise reclaim a model the instant no job references it. That protection is not
absolute; under genuine memory pressure it yields in a strict precedence, weakest last:

1. **In-flight and imminent work is never evicted.** A model with a pending or in-progress job keeps its
   residency regardless of anything below.
2. **A whole-card residency holder is never evicted from VRAM**, even under budget pressure, because evicting
   it would break the residency convergence a heavy model depends on.
3. **Seats hold RAM residency** the way a recently-demanded model does, sparing a disk reload between a seat's
   jobs. Under true host-RAM pressure, a **seat with no live job yields** its staged components (the pool is
   told, and records the eviction without unseating the model, so it keeps advertising it). A busy seat does
   not yield.
4. **The recently-demanded grace** is the ordinary, lowest-precedence hold for a model that ran within the
   grace window but holds no seat.

The precedence is what lets the pool commit to residency without deadlocking the host: the worker keeps its
promise wherever it safely can, and sheds staged (not seated) weight first when the host genuinely runs short.
The measured-truth memory machinery this rides on is described in
[Performance and Backpressure → The VRAM and RAM budget](performance_and_backpressure.md#the-vram-and-ram-budget).

## Stale demand: a bench-hold, not a guess

The ranker's decisions are only as trustworthy as the demand reading behind them. When the latest snapshot is
too old to act on, the pool enters a **bench-hold**: rotations, re-contests, and both rescue engagement and
release are frozen, and only genuinely empty seats are still filled, and only from manual pins. The worker
therefore never churns its seats or sits emptier than the operator explicitly asked for on the strength of a
stale signal; it holds what it has until a fresh reading arrives.

## Who the pool is for

Two operator audiences reach the pool from opposite directions:

- **Pinners** name the exact models they want held and lean on `pinned` (optionally with the ranker off for a
  pins-only pool). This suits an operator who knows their audience and wants deterministic residency.
- **Max-throughput** operators want the worker to chase demand for them. The `max_throughput_mode` preset turns
  the pool on with its ranker and a 50GB auto-download budget in one switch, applied only where the matching
  `model_pool` fields were left at their defaults, so an explicit value always wins.

Both share the same engine; the difference is only how many seats the operator claims by hand versus leaves to
the ranker.

## Observing the pool

Every seat transition (seated, demoted with its reason, rescue engaged/released, download pending/ready) is
logged once at INFO and recorded as a `MODEL_POOL_*` action-ledger event, and the advertising lanes log their
narrowing edges the same way the residency-bias duty cycle does. The supervisor status snapshot (protocol v19)
carries an optional `model_pool` section: per-seat rows (model, source, state, dwell, last-fulfilled age,
empty pops, rescue countdown), the bench with cooldowns, the current lane, demand-snapshot age, and the
auto-download budget spent. The TUI renders this as a "Model pool" panel in the Insights view. With the pool
enabled the panel shows the live seats, lanes, and bench; with the pool disabled it shows a one-line
`Model pool: off` state and a note on what turning it on would trade, so the subsystem is discoverable rather
than invisible. Older supervisors simply never send the optional field, and the panel still reads as off.

## How to read the pool at a glance

The Insights view and the config editor share one vocabulary, so what you configure maps directly onto what you
watch:

- **A seat** is one committed model the worker keeps ready to serve. The seats table lists each seat's model,
  its **source** (`M`anual pin, `R`anker fill, re`S`cue), its **state** (serving, or `dl:` while it downloads a
  model to swap in), how long it has held the seat (**dwell**), how long since it last served work
  (**fulfilled**), its charged **empty** pops, and any **rescue** countdown. A seat with a fresh fulfilled age
  is earning its place; a seat with a growing fulfilled age and rising empty pops is heading for the bench.
- **The two lanes** are how the worker advertises while the pool holds seats. The **fixed** lane offers only the
  seated models (work the card runs swap-free); the **free** lane offers everything else, so cold and rare-model
  demand still reaches you. The status line shows which lane the most recent pool-routed pop used.
- **Demotion and the bench** are what unseating looks like on screen: a model that stops earning its seat
  (sustained empty pops against near-zero demand, or a stretch with no fulfillment) is demoted to the **bench**
  with a cooldown, shown as its own line under the seats. A benched model does not immediately re-seat, so it
  will not thrash between the seat and the bench.
- **The throughput-versus-variety choice** is the whole point of the pool. Off, the worker serves your entire
  model list evenly and earns the widest variety of jobs, but pays a model swap whenever a popped job needs a
  model that is not loaded. On, it commits to a small ready set for more kudos per wall-second, at the cost of
  serving fewer distinct models. Neither is "correct": it is an operator preference about what the card should
  optimise for.
- **When to leave the pool off:** if the worker is rarely idle and rarely swaps models (a small, evenly-demanded
  model list already fits its inference processes), the pool has little to add. It earns its keep when the card
  either idles waiting for rare-model demand or churns swapping between many models; the Insights advisors
  surface both cases (a "serving many models with the pool off" nudge, and, once on, seat-health lines) so you
  do not have to infer it from the raw numbers.

## See also

- [Performance and Backpressure](performance_and_backpressure.md): the pop gauntlet the lane narrowing runs on,
  model stickiness (which the pool supersedes), and the VRAM/RAM budget the residency precedence rides on
- [Model Downloads and Availability](model_downloads.md): the download process the pool's budgeted fetches use
- [`ModelPoolConfig`][horde_worker_regen.bridge_data.data_model.ModelPoolConfig]: the operator config fields
- [`ModelPool`][horde_worker_regen.process_management.scheduling.model_pool.ModelPool]: the pure seat engine
- [`build_pool_params`][horde_worker_regen.process_management.scheduling.pool_wiring.build_pool_params]: the
  config-to-engine adapter
