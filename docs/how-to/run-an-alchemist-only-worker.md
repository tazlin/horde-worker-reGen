# Run an alchemist-only worker

An *alchemist-only* worker serves only alchemy (interrogation/post-processing) jobs: upscaling,
face-fixing, background removal, captioning, interrogation, NSFW classification, vectorize, palette,
describe, and aesthetic forms. It does **not** pop image-generation jobs. Use this when you want to
contribute alchemy without running the dreamer (image-generation) role, for example to leave a GPU free
for other work, or because you have no usable GPU at all.

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

4. (Optional) Choose which forms to offer with `forms:`. If unset, all default forms are offered:
   `caption`, `nsfw`, `interrogation`, `post-process`, `vectorize`, `palette`, `describe`, and
   `aesthetic`. Captioning additionally requires `alchemy_caption_enabled: true` because it loads BLIP.
5. Start the worker as usual.

### The role matrix

| `dreamer` | `alchemist` | Result                                            |
| --------- | ----------- | ------------------------------------------------- |
| `true`    | `false`     | Image generation only (the default)               |
| `true`    | `true`      | Both image generation and alchemy                 |
| `false`   | `true`      | **Alchemy only**                                  |
| `false`   | `false`     | Nothing to serve (a warning is logged)            |

## What changes in alchemist-only mode

- **No inference processes.** No alchemy form runs on one, so none is started. Graph alchemy forms
  (upscale, face-fix) run on the dedicated post-processing lane; background removal and annotation run
  on the image-utilities lane (on by default, `enable_image_utilities`). Text/CLIP forms (caption,
  interrogation, NSFW, vectorize, palette, describe, aesthetic) run on the safety process, which is
  why it stays up; alchemy results are not safety-screened. The download process still runs to fetch
  the models those three need.
- **Pipeline disaggregation does nothing here.** Its text-encode and VAE lanes are stages of an image
  job and are not started.
- **No image models are loaded.** Any configured `models_to_load`/`dynamic_models` are coerced off, so
  the worker never advertises or pops an image job.
- **The forms decide which auxiliary models are downloaded.** A worker offering `post-process` fetches
  the upscalers and face-fixers, and (with `enable_image_utilities` on) the background-removal weight;
  one offering only text forms such as `interrogation` fetches neither. The caption model is fetched
  only when `caption` is offered and `alchemy_caption_enabled: true` is set. Editing `forms` while the
  worker runs applies without a restart.
- **The dashboard reshapes around alchemy.** The overview shows an "ALCHEMIST-ONLY WORKER" identity, an
  alchemy-centric headline (forms submitted, in flight, pending), an alchemy job pipeline, and a longer
  recent-jobs view so sparse alchemy work stays visible over the session.

## Verifying it worked

- On startup the log should show no image models loaded and no inference process started; the safety
  process and the enabled lanes come up once the on-disk scan completes.
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
