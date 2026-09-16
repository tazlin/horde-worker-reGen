# Text generation (the scribe role)

A worker with `scribe: true` serves the AI Horde's text jobs: a requester asks for text, this worker pops
the job, and the text comes back. The thing that actually generates the text is not part of the worker.
It is a separate program, the **text backend**, that you start yourself and that the worker talks to
over HTTP. The worker's part is the conversation with the horde on one side and the conversation with
that program on the other.

That makes text generation the odd one out among the worker's three roles. Image generation and alchemy
run in child processes the worker starts, loads with weights, and watches; they take VRAM out of the
worker's own budget. The text flow starts nothing, loads nothing, and takes no VRAM here. It is a single
loop in the worker's main process.

## Supported backends

- **koboldcpp** ([LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp)) is supported today.

Others are planned, sonar (aphrodite) next. Every backend so far serves the same KoboldAI HTTP API, so
they share one driver in the worker and differ only in what you launched: install and start the other
program, point `kai_url` at it, and the worker attaches exactly as it does to koboldcpp. The one thing
that is not interchangeable is how the horde spells the model name, covered below, which the worker
derives from which backend it is talking to.

## The three roles

Each role is an independent switch in `bridgeData.yaml`, and any combination is valid:

| Flag         | What it adds                                                          |
| ------------ | --------------------------------------------------------------------- |
| `dreamer`    | Image generation. On by default; this is the historical worker.        |
| `alchemist`  | Alchemy forms: upscaling, face-fixing, interrogation, captioning.     |
| `scribe`     | Text generation, through the backend you attach.                      |

With all three off, the worker has nothing to serve and says so at startup.

**Scribe only.** Set `dreamer: false`, `alchemist: false`, `scribe: true`. Nothing in the worker touches
the GPU, so this runs on a machine with no accelerator at all: whether the text generation itself uses a
GPU is between you and the backend. Your text worker registers under `scribe_name`, which must be unique
horde-wide and cannot reuse your `dreamer_name` or `alchemist_name`.

**Scribe plus dreamer.** Leave `dreamer: true` and add `scribe: true`. The two flows are independent:
the text flow never waits on image work and never delays it. What they do share is the machine. The
backend holds its model in VRAM and in system RAM beside the worker's own inference processes, and the
worker cannot see how much, so it reserves a fixed coarse allowance for a co-hosted scribe when it
decides how many inference processes to run. If you run a large text model next to image generation,
expect to size your image configuration down.

## Running the backend

By default the worker runs the backend for you (`text_backend_managed: true`). It obtains the program
(for koboldcpp, a pinned upstream release it downloads and verifies on first use, unless
`text_backend_executable` names one of your own: a binary, or a source checkout's `koboldcpp.py` beside its
compiled library, which the worker runs under its own interpreter), starts it on `text_backend_port` with the model file in
`text_model_path`, waits for it to report a loaded model, and from then on relaunches it if it exits and
stops it when the worker stops. The backend's own output goes to `logs/text_backend.log`. Which program is
launched, and with what command line, follows `text_backend_kind`; a kind the worker cannot launch yet is
refused by name at start-up with the supported list.

The backend runs on the card `text_gpu_device_index` names, or on the lowest driven card when unset. On a
multi-GPU host, naming a card dedicates it: image generation stops driving that card, so the two workloads
never compete for its VRAM. On a single-card host the card stays shared whatever is configured, and the
worker's VRAM accounting sees the backend as memory another process holds. The worker measures the
backend's footprint once, as the card's free-VRAM drop between launch and ready, and logs it.

Two process facts shape how the worker stops a backend, and they hold for any backend that behaves this
way. A packaged program may run its real server as a child of the process the worker started (koboldcpp
does), so the worker stops the whole process tree, children first, and records every process id so a
worker that died hard can reap them on its next start. And a server that survived a previous run still
holds its port, so the worker refuses to launch onto a held port and waits, rather than guessing another.

To run the backend yourself instead, set `text_backend_managed: false`, point `kai_url` at where it
listens (`http://localhost:5000` by default) and start it before, or at the same time as, the worker. The
worker polls that address until it reports a loaded model, then asks it what it can do, and only then
begins popping text jobs.

Waiting is the normal case, not an error. A packaged backend unpacks itself and then loads weights from
disk, which together take a minute or more on a first start. The worker polls quietly through that and
only says something when the wait has gone on long enough to suggest the backend is not coming: nothing
started, the wrong port, or a firewall.

One thing to know about credentials: a backend started with a password (koboldcpp's `--password`)
refuses a request carrying the wrong one in a way the worker cannot tell apart from a backend that is
down. If your worker sits in the readiness gate forever against a backend you can see running, check the
password before anything else. A backend you started for this worker needs no password, which is the
case the worker is written for.

## What the worker advertises, and why it must not be double-prefixed

What a pop offers the horde is a promise about what the backend will accept, so nothing is popped before
the backend has answered, and every figure offered is the lower of yours and the backend's:

- **The model name.** By default, whatever model the backend says it loaded. Set `text_model_name` to
  override it, spelled the way the text model reference spells it: `author/Model`, plus a quantisation
  suffix if your file has one.
- **`max_length` and `max_context_length`.** Your ceiling or the backend's, whichever is lower.
  Advertising more than the backend accepts just draws jobs it then refuses.
- **Soft prompts.** Exactly the list the backend exposes, which is usually empty. The worker passes a
  requested soft prompt through and has no way to supply one the backend lacks.
- **`text_threads`.** How many generations the backend can run at once. This is also the most text jobs
  the worker holds at a time. It has nothing to do with `max_threads`, which sizes the worker's own GPU
  inference processes.

The horde spells a text model's name with the backend serving it in front of it, and what that looks
like differs per backend: the koboldcpp spelling drops the author segment, so
`meta-llama/Llama-3.2-3B-Instruct` is offered as `koboldcpp/Llama-3.2-3B-Instruct`, while the aphrodite
spelling keeps the whole name. The worker derives that from which backend it is attached to, so
`text_model_name` takes the plain canonical name. A name that already carries a backend prefix would be
prefixed twice and match nothing the horde has work for, so the config refuses one and tells you to drop
it. An author segment is not a prefix and is kept.

## Watching a generation arrive

A text generation takes seconds to minutes, and a blocking request says nothing at all until it is over,
so the worker asks the backend to stream its answer instead. The text then arrives in pieces, and the
worker knows three things while a job is still running: how much has arrived, when the last of it did,
and how long the job has been going. That is what the dashboard shows, and it is also how the worker
tells a slow generation from a backend that has wedged.

**Token counts and a token rate are a separate question.** One stream record carries an arbitrary number
of tokens (fifty-five records carried a hundred and sixty tokens on a measured run), so nothing about
how many tokens have been produced can be worked out from the text. A backend built with the worker's
per-request statistics patch answers a route that says so, and a job on such a backend carries real
token counts and a real rate. A stock build has no such route, the worker asks once and then stops
asking, and the counts stay unknown rather than being guessed at from the characters. Either way the
finished job's counts come from what the backend itself reported: the statistics route where there is
one, the backend's own last-generation counters otherwise.

A backend with no stream route at all still works. The worker falls back to the blocking request for
that backend, says so once in the log, and generates exactly as it did before; it simply learns nothing
about a job until the job is done.

**The first generation after the backend starts is slow.** Its kernels are cold, and it produces a
couple of tokens in its first several seconds where the same backend runs at over a hundred a second
once warm. So a worker that launched the backend itself spends one short generation on warming it up
before it pops anything, and throws the answer away. A backend you run yourself is left alone, because
it may be serving other clients whose generation slot is not the worker's to spend.

## What happens when the backend misbehaves

The worker never drops a job it has popped. A popped job that is quietly abandoned holds its requester
until the server times it out minutes later, and the horde charges the worker for that, so every outcome
below ends in a submit, faulted where it has to be.

| What the backend does                      | What the worker does                                                                                              |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Says it is busy                            | Waits a moment and offers the same job again, a few times. The backend is healthy and the job is fine.            |
| Refuses the payload                        | Faults the job immediately. It will be refused again, so re-offering it only burns the same failure repeatedly.   |
| Cannot be reached, or fails the generation | Faults the job, then goes back to polling readiness. While the backend is down every job fails the same way.      |
| Takes longer than the deadline             | Asks the backend to abandon the generation, then faults the job.                                                  |
| Stops producing text part way through      | Asks the backend to abandon the generation and faults the job, without waiting the deadline out.                  |
| Breaks the stream part way through         | Faults the job. Half a generation reads as a whole one to a requester, so partial text is never submitted.        |

The per-generation deadline is `text_generation_timeout_seconds` when you set it, and otherwise is
derived from `max_length` at a deliberately slow tokens-per-second floor (`max_length / 2 + 10`, the
figure the older scribe bridge used). Set it explicitly if your backend is slower than that floor.

That deadline bounds a whole generation, and it cannot tell a backend that is running slowly from one
that has stopped: both are silent until it expires, and the difference is minutes of a requester's time.
Because the answer streams, silence is its own signal, so a generation that has produced nothing for
`text_stall_seconds` (thirty by default) is given up on there and then. The exception is the first token
after the backend starts, which is given the whole deadline, because a cold backend legitimately takes
seconds to produce anything.

An abandoned generation deserves one note, because a backend's answer to being told to stop can be
surprising: koboldcpp returns a success carrying the tokens it had produced so far. A truncated answer
that reads as complete is worse than no answer, so once the worker has given up on a generation it
discards whatever comes back for it and reports the job faulted.

On shutdown the worker stops popping, asks the backend to abandon everything still generating, waits a
bounded time for those jobs to report themselves faulted, and closes. The wait is bounded because the
backend is another program: it may have no abort route, or may ignore one, and the worker cannot be held
open on its behalf.

## Testing against the live horde without serving strangers

Put the scribe worker into maintenance on the horde (the worker's page, or `PUT /v2/workers/{id}` with
`maintenance: true`). The horde then refuses every pop from it, but still routes requests made under the
owner's own API key to it, so a request that names the worker in `workers` is served by it and nobody
else's traffic reaches it. The worker logs the hold once when it begins and once when pops resume.

## What a finished text job is counted in

A submitted text job earns kudos the same way an image job does, and the worker treats those earnings as
the account's rather than one flow's: the paid submit updates the session kudos total, the kudos/hr
clock, and the rolling kudos events, exactly as an alchemy form does.

Each finished job (paid or faulted) also lands as one per-job run-metrics record carrying the advertised
model name, the pop and submit times, the wait from pop to generation start, the generation's own
duration, the reward, and the generation-length cap the horde asked for. The record names its workload,
so a session's totals split three ways instead of collapsing text into the image numbers. Its prompt and
generated token counts are there when the backend reported them and unknown when it did not, which is
the difference between a patched build and a stock one described above.

The supervisor snapshot carries the flow's live state for the dashboard: jobs in flight, the session's
submitted and faulted totals, whether the readiness gate has passed, and the model, context length and
generation length the worker is advertising.

## The backend on the dashboard

The backend is a program the worker launched, not a child it speaks IPC with, so it is not in the process
map and never will be: the map's entries hold torch VRAM the worker accounts for and move through an
image-and-alchemy state vocabulary, and a text entry would be misread by every reaper, ledger and health
check that walks the map. It still appears wherever the map's processes do, as one more row, through
`TextBackendSupervisor.to_process_snapshot()` and the
[`SupervisedProcessSnapshotSource`][horde_worker_regen.process_management.ipc.supervisor_channel.SupervisedProcessSnapshotSource]
protocol the snapshot builder concatenates onto the map-derived rows.

The row says what a text backend can be held to. Its state is the supervisor's own (`launching`, `ready`,
`relaunching in 8 s`), its resident model is the one the backend reported loading, its identity in the id
column is the OS pid of the launched process, and its VRAM figure is the card's free-memory drop measured
across the launch, labelled as measured because it is not an allocator reading and cannot be compared with
one. The Live tab carries the rest as labelled lines: the port and backend kind, the context and
generation caps, how long it has been ready, how many times it has been launched, and the buffer sizes
llama.cpp printed while loading (see [Logs](../reference/logs.md)). Nothing on the row feeds admission or
pricing; the arbiter still sees the backend only as the card's foreign floor.

Readers that reason about inference lanes skip the row rather than counting it: a ready backend does not
make the worker read as serving images, does not end the image lanes' warm-up, and names no card whose
duty could be called low.

What the row cannot say for itself, the flow hands it at snapshot time as a `TextBackendActivity`: how
many generations the backend is producing output for, how many the worker is holding for it to take, and
the most recent token rate any of them reported. The supervisor watches the process and never sees a
generation, so a copy of these kept on the supervisor could only go stale; passing them per snapshot
means the row is busy exactly while the flow has work against it.

The flow is also where the backend's restarts become a fact about generation. A relaunch with nothing in
hand fails no generation and re-runs no readiness gate, so a cold-kernel flag cleared by the previous
process's first token would hold over a process that has produced nothing, and the first job after such a
relaunch would be held to the stall bound instead of the whole deadline. For a backend the worker owns,
the coordinator reads the supervisor's launch count through a `launch_count_provider` and treats any
change in it as the kernels being cold again. A backend the operator runs has no launch the worker can
count, and keeps the readiness gate as the only thing that can say it started over.

Two health findings follow from the flow's own patiences rather than from figures the dashboard picks. A
backend that has reported no loaded model for longer than the readiness gate's patience is a warning and
past twice that a fault, which separates a model still loading from a backend nobody started. A
generation that has produced nothing for longer than `text_stall_seconds` plus the snapshot interval is a
warning; the interval is there so the dashboard never warns about a job the worker is in the act of
faulting.

## What is not supported yet

- **A shared card is not priced.** When the backend shares a card with image generation, the worker's VRAM
  admission sees the backend's memory only as the card's foreign floor; it does not yet treat the backend
  as a tenant it can plan around, pause, or wake.
- **The backend cannot be swapped or reconfigured while the worker runs.** Changing which model the
  backend serves means restarting the backend; the worker will notice when the generation it is waiting
  on fails and re-run its readiness gate, but nothing coordinates the two. Which *kind* of backend is
  attached is likewise fixed for the run.
- **The config editor has no scribe fields.** Text generation reaches the dashboard's surfaces: a text
  panel in the Advanced Overview, in-flight generations in the work ledger and on the Simple home, text
  postures in the health checklist, a per-workload split on Stats, and the backend's own section on the
  native page. What is still missing is the editor page, so `scribe` and everything under it are edited
  in `bridgeData.yaml` by hand.
- **No activity feed carries text events.** Finished text jobs reach the Simple ticker through the
  recent-jobs list, so the feed is built from what finished rather than from events the worker emitted at
  the transitions themselves.
- **Turning `scribe` on takes a restart.** Unlike most configuration, the role is read when the worker
  builds its flows, so a hot reload of `bridgeData.yaml` will not start a text flow that was off at
  launch (it will, however, stop one that was on).

## Where it lives in the code

- [`text_backends`][horde_worker_regen.text_backends]: the whole of the conversation with the external
  program. Six verbs (`ready`, `describe`, `capabilities`, `generate`, `stop`, `close`), the result
  models, three exception types, and no HTTP visible above it. One driver serves every backend that
  speaks the KoboldAI API, streaming its generations and reporting them through a callback the flow
  hands in.
- [`TextGenerationCoordinator`][horde_worker_regen.process_management.jobs.text_generation_coordinator.TextGenerationCoordinator]:
  the flow itself, satisfying the same
  [`FlowCoordinator`][horde_worker_regen.process_management.scheduling.workload_flow.FlowCoordinator]
  surface as image generation and alchemy. Which backend it is talking to is one `TEXT_BACKENDS` value
  on it, the single thing a further backend changes.
- [`capabilities.enabled_workloads`][horde_worker_regen.capabilities.enabled_workloads]: the one place
  the role flags become the workloads the worker serves.

## See also

- [Architecture](architecture.md): the process model and the workload flows.
- [Bridge configuration](bridge_config.md): every field this page mentions, in context.
