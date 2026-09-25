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

## Choosing the model

One key says which model a worker-run backend loads: `text_model`. It takes either a catalogued model's
name, spelled the way the text model reference spells it, or the path to a model file of your own.

```yaml
# a catalogued model: the worker knows its file, its size and its digest
text_model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M"

# or a file you supply: used where it sits, never fetched
text_model: "T:/horde-text-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

Which of the two a value is, is decided by the filesystem: an existing file is that file, and anything
else is looked up as a name. A managed backend needs the key; the config is refused without it. A backend
you run yourself (`text_backend_managed: false`) has already been given its model, so the key is unused
there.

**The catalogue is part of the model reference, not a list beside it.** The horde's text model reference
is a name registry: its records carry the publisher's page and a parameter count, and deliberately no
per-file download entries, because a quantised conversion hosted anywhere is a valid thing for a worker to
run. koboldcpp needs one such file. So the worker contributes the models it has measured to the reference
as one more source (`horde_worker_regen/text_backends/model_catalogue.py`, registered under the source id
`regen_text`), and a read of the text category sees them beside the canonical records. Each contributed
record declares its file's name, size and SHA-256 in the fields the reference already has, and carries
what running the model showed (its quantisation, the context it was measured at, the VRAM it held, the
tokens per second, and the card it was measured on) in the record's `settings`. A canonical record of the
same name wins, so when the reference carries these files itself the worker's copies are shadowed and
nothing else changes.

Three models are in the catalogue today, each measured on one 16 GB card: Llama-3.2-3B-Instruct Q4_K_M
(2911 MB at 4096 context), Meta-Llama-3.1-8B-Instruct Q4_K_M (5954 MB at 8192), and
Mistral-Nemo-Instruct-2407 Q4_K_M (8633 MB at 8192). The list grows by measurement: a model nobody has run
has no footprint to price against, and an estimate priced as a measurement is how a card gets
over-committed.

**Where files land.** A file the worker fetches goes in `text_models_dir`, which defaults to
`<AIWORKER_CACHE_HOME>/text_models`. The model reference publishes an on-disk folder for each category
whose files it declares, and it declares none for text models, so this folder is the worker's own choice
rather than part of the shared layout; registering one in the reference is an open question. A
`text_model` naming a path is read where it sits and never copied here.

**The fetch happens while the backend is being provisioned**, in the same step that obtains the program,
so the dashboard's backend row reads `provisioning` for as long as it takes and a failure is reported
there rather than as a silent relaunch loop. The reference package's own downloader does the transfer: it
resumes an interrupted one with a `Range` request instead of starting over, hashes while it streams, and
fails the fetch if the result does not match the declared digest. A file already on disk at the declared
size is left alone, so only the first launch pays for it. The image model download process is not
involved at any point.

Each model in the catalogue names the public Hugging Face file its measurement was taken from, matched to
it by size and SHA-256, so a first launch fetches it. A catalogued model whose file has no recorded origin
cannot be fetched, and the worker says which file it wanted and where it looked for it. Put the file at the
named path (or set `text_models_dir` to where you keep it) and the catalogue still supplies its name, digest
and measured footprint.

## Running the backend

By default the worker runs the backend for you (`text_backend_managed: true`). It obtains the program
(for koboldcpp, a pinned upstream release it downloads and verifies on first use, unless
`text_backend_executable` names one of your own: a binary, or a source checkout's `koboldcpp.py` beside its
compiled library, which the worker runs under its own interpreter), obtains the model file `text_model`
names, starts it on `text_backend_port`, waits for it to report a loaded model, and from then on relaunches it if it exits and
stops it when the worker stops. The backend's own output goes to `logs/text_backend.log`. Which program is
launched, and with what command line, follows `text_backend_kind`; a kind the worker cannot launch yet is
refused by name at start-up with the supported list.

The backend runs on the card `text_gpu_device_index` names. When it is unset, the worker picks the driven
card with the most free VRAM at launch (the same per-card reading the safety process is placed by, or the
card's total where nothing has reported yet), because the model and its context have to fit beside
whatever already occupies the card. Cards whose free VRAM is within one model file of the roomiest count
as equal, and among those a card without the safety process wins. The choice is logged once with each
card's reading and the model file size. On a multi-GPU host, naming a card dedicates it: image generation
stops driving that card, so the two workloads never compete for its VRAM. An unnamed card stays shared. On
a single-card host the card stays shared whatever is configured. The worker measures the backend's footprint
once, as the card's free-VRAM drop between launch and ready, and logs it. While the backend serves, that
footprint is charged to its card as a standing floor (see the [VRAM arbiter](vram_arbiter.md#the-decision-pipeline)).
Image admission, the models offered for that card, the safety process's card choice and the start of a GPU
child all see the card as that much smaller, without waiting for the arbiter's learned foreign floor to observe
the backend. A relaunch re-measures the footprint, and a backend that stops serving is no longer charged.

With `text_threads` above 1 the worker launches koboldcpp with `--parallelrequests` set to that many, and
sizes its context for all of them: `--contextsize` is `text_threads` × `max_context_length`. koboldcpp's
parallel requests draw on one shared KV pool of `--contextsize` cells and clamp each request only to that
pool, while the worker advertises the full `max_context_length` on every job and holds `text_threads` jobs
at once. A pool of one context runs out as soon as two long requests overlap. The extra context costs VRAM
in proportion: the footprint line at ready states the cells opened and the multiplier, and when what the
backend allocated for the card (the larger of the measured footprint and the buffers it printed for a GPU
device) exceeds the card's total, the worker logs one warning naming both numbers. It launches anyway; what
does not fit spills to system memory and generation slows. One parallel request renders the context
exactly as configured.

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

Obtaining the program comes before any of that, and is part of the backend's life rather than a gap before
it: the worker's supervisor owns that step too, so the backend has a row on the dashboard reading
`provisioning` for as long as a first-run download takes, and the row states the port only once the command
line has been rendered. A program that cannot be obtained at all is not retried, because a download that
failed is not something a relaunch ladder recovers; the reason is logged once and carried on the row, where
the dashboard turns it into an error rather than waiting on a clock that would never run out.

The patience a managed backend is judged by is therefore measured from the launch attempt rather than from
the worker's start, and while the program is still being obtained the dashboard says so and raises nothing.
The wait for an attached backend is timed from the worker's first look at it, because the worker did not
start it and has no launch to measure from. Either way, a message about a backend that is not answering
names the address the worker is actually using: its own loopback port for a managed backend, `kai_url` for
one you run.

A backend you started with a password (koboldcpp's `--password`) is told that password through
`text_backend_password`, and the worker sends it as a bearer token on every request. Get it wrong and the
worker says so by name rather than waiting: a refused credential is not a backend that is still starting,
so nothing is popped, the reason is logged once, and the backend is re-checked a minute at a time until a
`text_backend_password` you corrected in `bridgeData.yaml` is accepted. The key is for a backend you run;
a backend the worker starts listens on a loopback port of the worker's own choosing with no password, so
none is sent there.

Where the refusal shows is worth knowing, because koboldcpp answers its model route to anyone: a worker
whose password is wrong sees a ready backend reporting a model called `koboldcpp/protected-model`, and
the refusal only arrives at the routes behind the guard. The worker therefore judges the credential at
those, not at the readiness probe.

## What the worker advertises, and why it must not be double-prefixed

What a pop offers the horde is a promise about what the backend will accept, so nothing is popped before
the backend has answered, and every figure offered is the lower of yours and the backend's:

- **The model name.** A catalogued `text_model` is offered under its record's name. A `text_model` that
  names a file of your own is offered under the file's stem, and an attached backend under whatever model
  it reports loading. Set `text_model_name` to override any of those, spelled the way the text model
  reference spells it: `author/Model`, plus a quantisation suffix if your file has one.
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
token counts and a real rate: the backend's own generation timing where it reports one live, else the
growth in its token count between two samples over the time between them. A stock build has no such route, the worker asks once and then stops
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
| Says it is busy                            | Waits a moment and offers the same job again until the job's ttl runs out. The job itself is fine.                |
| Says it is busy to everything, for minutes | Stops popping and, for a backend the worker launched, relaunches it. See the wedged backend section below.        |
| Refuses the payload                        | Faults the job immediately. It will be refused again, so re-offering it only burns the same failure repeatedly.   |
| Refuses the worker's password              | Faults the job, then holds off popping until a corrected `text_backend_password` is accepted.                     |
| Cannot be reached, or fails the generation | Faults the job, then goes back to polling readiness. While the backend is down every job fails the same way.      |
| Takes longer than the deadline             | Asks the backend to abandon the generation, then faults the job.                                                  |
| Stops producing text part way through      | Asks the backend to abandon the generation and faults the job, without waiting the deadline out.                  |
| Breaks the stream part way through         | Faults the job. Half a generation reads as a whole one to a requester, so partial text is never submitted.        |

A busy backend is also bounded by the horde rather than by the worker alone. A pop states a `ttl`, the
seconds the server will wait before it reassigns the generation, and the worker keeps that as a deadline
on the job: past it, the job is faulted instead of being offered again, and a job whose ttl has already
run out is never started. Generating an answer the server has stopped waiting for spends the backend on
nobody. A pop that states no ttl leaves the worker's own bounds as the only ones: such a job is offered
for one generation deadline and then faulted.

### A backend that is busy to everything

A backend can stop generating while still answering. koboldcpp run with parallel requests
(`--parallelrequests`, which the worker passes when `text_threads` is above 1) holds a generation lock,
queues a few requests behind it, and refuses the rest as busy. If a generation fails inside the backend
in a way that never releases that lock, every later request is refused as busy, indefinitely, while the
backend still reports its model loaded. To one job this looks exactly like a backend that is merely full.
The known way in is a context pool too small for the requests sharing it: the backend's own output then
shows `find_slot` or `failed to find a memory slot`. The pool must hold `text_threads` ×
`max_context_length`, which is why the managed launch sizes it that way (see
[Running the backend](#running-the-backend)).

The difference shows only across jobs, so the worker keeps one busy clock for the backend rather than
one per job. The clock starts at the first busy answer after the backend last produced text, and any
text from any job stops it. The worker judges the backend **wedged** when all of these hold:

- the clock has run for at least a minute, or one generation deadline when that is longer, because the
  worker itself tolerates one of its own generations staying silent that long;
- the backend has refused at least five offers in that time;
- no job the worker holds is receiving text. A backend streaming one job while refusing another is
  full, not wedged.

While the backend is wedged the worker pops nothing. The hold is the readiness gate's, so the dashboard
shows the backend as not ready and for how long. Jobs already in hand keep being offered until their
ttl runs out, in case the backend recovers.

What happens next depends on who runs the backend:

- **A backend the worker launched** is stopped and launched again, once per wedge, through the same
  relaunch path as a backend that exited, and the worker logs one warning naming the backend's address
  and how long it was busy. The hold ends when the new process answers, and the worker then warms it up
  and resumes popping. If three relaunches in a row each wedge again with no successful generation
  between them, the worker stops relaunching, keeps the hold, and logs one error saying so. That error
  points at `logs/text_backend.log` and the `find_slot` lines: with the pool already sized for every
  request, those lines mean it still ran out.
- **A backend you run yourself** is not the worker's to restart. The worker holds, logs one error per
  wedge naming the backend's address, and keeps checking readiness on the gate's usual cadence. Because
  a wedged backend still answers that check, the hold ends only on a successful generation or once the
  backend has been seen restarting: a check that finds it gone, then one that finds it back. The error
  asks you to restart it and to start it with `--contextsize` of `text_threads` × `max_context_length`.

Either error mentions `text_threads: 1` only as a way to confirm the cause: a backend that stops wedging
with one request at a time ran out of shared context. Parallel requests remain a supported setting, and
the fix is a context pool large enough for them. `horde-log diagnose` reports these episodes as
`text_backend_wedged`, and also recognises the run of busy-refused faults an older worker left behind.

The worker's own hold and the horde's maintenance flag are independent. The horde sets maintenance when
a worker drops too many jobs; that refuses pops at the server, and the worker keeps asking so it notices
when maintenance is lifted. The wedge hold stops the worker asking at all. A pop goes out only when the
wedge hold is off, and the horde answers it only when maintenance is off, so pops resume once both
holds have ended.

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

On shutdown the worker stops popping, asks the backend to abandon everything still generating, and waits
a bounded time for those jobs to report themselves faulted. The wait is bounded because the backend is
another program: it may have no abort route, or may ignore one, and the worker cannot be held open on its
behalf. What the wait does not finish is then ended and reported by the flow itself, under a bound of its
own, so every job the worker popped is accounted for exactly once and by one reporter.

A submit whose answer was lost in transit is retried, and the horde then says it already holds the
result. That is a delivery: the job is counted as the outcome it carried, with no kudos figure, because
the reward was paid against the attempt whose answer went missing.

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

The backend's posture also reaches the headline, not only the checklist. A scribe whose backend is being
obtained or started is warming up, in the same place on the phase ladder as a worker loading its first
image model, and an image role that is further along (serving a job, or loading its own models) keeps the
headline instead. Past the readiness patience the headline is a warning and past twice that an error
naming the remedy, which differs by who runs the backend: a managed one points at its row and
`logs/text_backend.log`, an attached one at `kai_url` and whether the program is running. A program that
could not be obtained at all is an error immediately, with the reason the download or the rendering gave.
And whatever the phase decides, the headline can never read OK while any check in the list beside it has
failed.

The whole-worker counters count every workload's work. The header's "done", the hero's submitted and
faulted figures, the Stats tab's job line, Simple's completed count, the native page's active-work figure
and the durable run record are the worker's totals across image, alchemy and text, so a scribe-only or
alchemist-only worker reports its own work rather than the image tracker's zero; "last pop" is the most
recent pop any flow made. Each flow contributes its delivered submits plus its fault reports, which is what
the image tracker's own movement counter means. The `alchemy_*` and `text_*` fields and the by-workload
totals remain the per-workload split, and the Stats tab shows that split whenever a workload other than
image generation has totals: a scribe-only worker's single row carries the mean generated tokens, which no
headline figure states.

## What is not supported yet

- **A shared card is charged, not planned around.** When the backend shares a card with image generation,
  the worker charges its measured footprint to that card as a standing floor; it does not yet pause, wake
  or shrink the backend to make room for image work.
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
- **Turning `scribe` on mid-run does not start a worker-run backend.** The flow itself follows the live
  configuration, so enabling the role by hot reload begins serving against a backend you run yourself
  (`text_backend_managed: false`) with no restart. The supervisor that launches a managed backend is
  still started on the role as it stood when the worker did, so switching the role on for one needs a
  restart.

## Where it lives in the code

- [`text_backends`][horde_worker_regen.text_backends]: the whole of the conversation with the external
  program. Six verbs (`ready`, `describe`, `capabilities`, `generate`, `stop`, `close`), the result
  models, four exception types, and no HTTP visible above it. One driver serves every backend that
  speaks the KoboldAI API, streaming its generations and reporting them through a callback the flow
  hands in.
- [`TextGenerationCoordinator`][horde_worker_regen.process_management.jobs.text_generation_coordinator.TextGenerationCoordinator]:
  the flow itself, satisfying the same
  [`FlowCoordinator`][horde_worker_regen.process_management.scheduling.workload_flow.FlowCoordinator]
  surface as image generation and alchemy, registered on every worker like those two and inert while
  `scribe` is off. Which backend it is talking to is one `TEXT_BACKENDS` value on it, the single thing a
  further backend changes.
- [`capabilities.enabled_workloads`][horde_worker_regen.capabilities.enabled_workloads]: the one place
  the role flags become the workloads the worker serves.

## See also

- [Architecture](architecture.md): the process model and the workload flows.
- [Bridge configuration](bridge_config.md): every field this page mentions, in context.
