# Run a pricing corpus on your machine

This page is for a developer contributing measurements. You run a fixed workload on your card and send
two files back to the maintainer of the kudos cost model. You do not train anything, and you do not need
the training tooling.

You need a working reGen install (it can be the one you run a worker from), the card idle for a few
hours, and a CivitAI token in `bridgeData.yaml` (`civitai_api_token`).

## 1. Install the hordelib the maintainer names

The maintainer tells you which hordelib checkout to use. Install it into the worker's environment:

```bash
git clone https://github.com/Haidra-Org/hordelib.git ../hordelib
git -C ../hordelib checkout <branch or tag the maintainer named>
uv pip install -e ../hordelib
```

If that reinstall moved any ComfyUI package version, the preflight in the next step says which ones and
prints the install line to put them back.

## 2. Check the machine

Pick a machine id: lowercase, dash-separated, stable for this box, `<owner>-<gpu>` by convention.

```bash
horde-benchmark corpus-preflight --tier census --machine alice-l40s
```

One row per check with a `FIX` column. Run whatever the `FIX` column says, then run the preflight again
until every row is `OK`. Missing models are the usual finding; the fix line is a
`horde-benchmark download` command that fetches exactly what the tier needs.

## 3. Run the corpus

Use the tier the maintainer asked for. Nothing else may use the card while it runs, and do not start a
normal worker from this directory until it finishes.

```bash
horde-benchmark pricing-corpus --tier census --machine alice-l40s
```

| Tier | What it measures | Time |
|------|------------------|------|
| `smoke` | That the plumbing works; produces no usable rows | minutes |
| `standard` | The marginal cost of each payload axis on SD1.5 and SDXL | about 1.5 h |
| `census` | Every sampler, schedule, control type and post-processor the cost model prices | 2.5 h with `--job-budget 600`, 4 h by default |
| `heavy` | Flux, Qwen, Z-Image, Krea2 and Anima on a card that holds them | about 1.5 h |

If this machine is running the `heavy` tier for the first time, run one model as its own smoke first
and send that bundle before the full tier:

```bash
horde-benchmark pricing-corpus --tier heavy --machine alice-l40s --model "Flux.1-Schnell fp8 (Compact)"
```

The heavy families are the only ones that exercise the beta model opt-in and the samplers the small
tiers never touch, so a single-model run proves the whole path in a fraction of the time a failed full
tier would cost.

The run refuses to start if the preflight fails, and its last log line is
`Corpus finished: N/N jobs completed, 0 faulted`. A few faulted jobs are fine; the assembler drops them.

## 4. Send the bundle

When the run finishes it copies its own results into one directory and logs the path:

```
Bundled the run for hand-off: .../benchmark_results/corpus-alice-l40s-census-20260101T093000Z (send this whole directory).
```

The name is your machine id, the tier and the UTC time the run started, so runs from many people sort
and pair correctly on the maintainer's disk regardless of timezone. Inside are the run's definition
JSON, the stats parts written during the run, and a `bundle.json` listing them with their hashes.
Archive and send the whole directory:

```bash
tar czf corpus-alice-l40s-census-20260101T093000Z.tgz -C benchmark_results corpus-alice-l40s-census-20260101T093000Z
```

Nothing in it identifies you beyond the machine id and the GPU, driver and package versions the
preflight recorded. If the log instead says the run could not be bundled, send the definition JSON it
named earlier (`Wrote the corpus definition to ...`) together with every `stats-v*.jsonl` file in
`.horde_worker_regen/stats/` newer than it.
