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

## Attaching the backend

Point `kai_url` at where the backend listens (`http://localhost:5000` by default) and start it before,
or at the same time as, the worker. The worker polls that address until it reports a loaded model, then
asks it what it can do, and only then begins popping text jobs.

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

The per-generation deadline is `text_generation_timeout_seconds` when you set it, and otherwise is
derived from `max_length` at a deliberately slow tokens-per-second floor (`max_length / 2 + 10`, the
figure the older scribe bridge used). Set it explicitly if your backend is slower than that floor.

An abandoned generation deserves one note, because a backend's answer to being told to stop can be
surprising: koboldcpp returns a success carrying the tokens it had produced so far. A truncated answer
that reads as complete is worse than no answer, so once the worker has given up on a generation it
discards whatever comes back for it and reports the job faulted.

On shutdown the worker stops popping, asks the backend to abandon everything still generating, waits a
bounded time for those jobs to report themselves faulted, and closes. The wait is bounded because the
backend is another program: it may have no abort route, or may ignore one, and the worker cannot be held
open on its behalf.

## What is not supported yet

- **The worker does not own the backend process.** You start the backend and keep it running; the worker
  attaches to its URL. It does not launch it, restart it after a crash, or stop it on shutdown.
- **You cannot tell the worker which card to run text generation on.** That is a backend launch option
  today, so the worker's own VRAM accounting knows nothing about it beyond the coarse co-tenant
  allowance.
- **The backend cannot be swapped or reconfigured while the worker runs.** Changing which model the
  backend serves means restarting the backend; the worker will notice when the generation it is waiting
  on fails and re-run its readiness gate, but nothing coordinates the two. Which *kind* of backend is
  attached is likewise fixed for the run.
- **Text jobs do not appear in the dashboard.** The flow keeps its own counts of submitted and faulted
  jobs and logs each submit with its kudos, but it does not feed the per-job run metrics the image and
  alchemy panels read.
- **Turning `scribe` on takes a restart.** Unlike most configuration, the role is read when the worker
  builds its flows, so a hot reload of `bridgeData.yaml` will not start a text flow that was off at
  launch (it will, however, stop one that was on).

## Where it lives in the code

- [`text_backends`][horde_worker_regen.text_backends]: the whole of the conversation with the external
  program. Five verbs (`ready`, `describe`, `generate`, `stop`, `close`), two result models, three
  exception types, and no HTTP visible above it. One driver serves every backend that speaks the
  KoboldAI API.
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
