# Run an alchemist-only worker

An *alchemist-only* worker serves only alchemy (interrogation/post-processing) jobs: upscaling,
face-fixing, background removal, captioning, interrogation, and NSFW classification. It does **not**
pop image-generation jobs. Use this when you want to contribute alchemy without running the dreamer
(image-generation) role, for example to leave a GPU free for other work, or because you have no usable
GPU at all.

There are two ways to end up alchemist-only:

- **Deliberately, on a GPU box:** set `dreamer: false` and `alchemist: true` in `bridgeData.yaml`.
- **Automatically, on a CPU install:** a CPU-only torch build cannot run image generation, so it is
  always alchemist-only regardless of the `dreamer` flag. See
  [Compute backends](../explanation/compute_backends.md#cpu--alchemist-only-mode-running-without-a-usable-gpu).

## Deliberate opt-in on a GPU

1. Open `bridgeData.yaml`.
2. Set the role flags:

   ```yaml
   dreamer: false
   alchemist: true
   ```

3. Give the worker a unique `alchemist_name` (it must be unique horde-wide and must not reuse your
   `dreamer_name`):

   ```yaml
   alchemist_name: "My Unique Alchemist"
   ```

4. (Optional) Choose which forms to offer with `forms:`. If unset, all of them are offered. Captioning
   additionally requires `alchemy_caption_enabled: true` because it loads BLIP.
5. Start the worker as usual.

### The role matrix

| `dreamer` | `alchemist` | Result                                            |
| --------- | ----------- | ------------------------------------------------- |
| `true`    | `false`     | Image generation only (the default)               |
| `true`    | `true`      | Both image generation and alchemy                 |
| `false`   | `true`      | **Alchemy only**                                  |
| `false`   | `false`     | Nothing to serve (a warning is logged)            |

## What changes in alchemist-only mode

- **One inference process.** Instead of the image-generation fleet, a single inference process is
  spawned per card; graph alchemy forms (upscale, face-fix, background removal) serialize through it.
  Text/CLIP forms (caption, interrogation, NSFW) run on the safety process.
- **No image models are loaded.** Any configured `models_to_load`/`dynamic_models` are coerced off, so
  the worker never advertises or pops an image job.
- **The dashboard reshapes around alchemy.** The overview shows an "ALCHEMIST-ONLY WORKER" identity, an
  alchemy-centric headline (forms submitted, in flight, pending), an alchemy job pipeline, and a longer
  recent-jobs view so sparse alchemy work stays visible over the session.

## Verifying it worked

- On startup the log should show no image models loaded and exactly one inference process per card.
- The dashboard overview should display the alchemist-only identity and the alchemy pipeline.
- Pop and complete a form (any offered form) and confirm it appears in the **Recent jobs** view (press
  the details view; alchemist-only retains more rows than a dreamer worker).

## Switching back

Set `dreamer: true` (keep or drop `alchemist` as you like) and restart the worker. The role flags
affect process sizing, which is decided at startup, so a change takes effect on the next start rather
than via hot-reload.

## See also

- [Compute backends: CPU / alchemist-only mode](../explanation/compute_backends.md#cpu--alchemist-only-mode-running-without-a-usable-gpu)
- [Architecture: Workloads (flows)](../explanation/architecture.md#workloads-flows)
- [Bridge configuration: alchemy](../explanation/bridge_config.md#alchemy)
