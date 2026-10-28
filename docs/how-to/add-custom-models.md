# Add custom models

Serving custom models requires the `customizer` role. Request it on
[Discord](https://discord.gg/3DxrhksKzn).

With the role, the easiest path is the dashboard **Config** tab:

1. Open **Config → Models**.
2. Use **Add custom model...** in the Custom models section.
3. Enter the served model name, choose the baseline, and enter the local model file path.
4. Leave **Also add this model name to the Offer list** checked unless you only want to define the model now.
5. Save and restart the worker.

If you are editing YAML by hand:

1. Download your model files locally.
2. Add them to `bridgeData.yaml`:

   ```yaml
   custom_models:
     - name: My Custom Model
       baseline: stable_diffusion_xl
       filepath: /path/to/model/file.safetensors
   ```

   The dashboard's baseline choices are generated from the installed
   `horde_model_reference` baseline vocabulary and hordelib's loading capabilities. The current YAML
   shape describes one fused checkpoint, so split-component baselines and Stable Cascade's two-stage
   weights are not offered. An unsupported value is rejected when the config is edited or loaded rather
   than being advertised and failing a job later.

3. Add the model `name` to your `models_to_load` list.

## Rules and limits

- Only Flux.schnell models are allowed. Flux.dev and its derivatives are **not** permitted.
- Custom model names cannot conflict with existing horde model names.
- The horde treats custom models as SD 1.5 for kudos and safety purposes.
- The checkpoint path must name a readable regular file on the worker host.

At worker startup, reGen validates every entry and atomically writes the legacy registry hordelib reads at
`.horde_worker_regen/custom_models.json`.
Only entries that pass validation and also appear in `models_to_load` are advertised. The Overview panel
shows `ready/configured` counts and lists any rejected model with its reason; the same details are written
to the worker log. An explicitly set `HORDELIB_CUSTOM_MODELS` remains operator-owned, so reGen checks that
its entries match `bridgeData.yaml` instead of overwriting it. The same applies to a historical root-level
`custom_models.json`, which the worker continues to discover as an external registry.

Changing `custom_models` requires a worker restart. A hot reload keeps the registry and offer from the
running process, logs that a restart is needed, and does not advertise the new definition prematurely.

See [Bridge configuration](../explanation/bridge_config.md#custom-models) for how custom models flow
into the pop request.

## Beta models

Some models are published to the model reference's "pending" (beta) queue before they are promoted
into the canonical reference. The worker opts every install into the image-generation beta by
default, so any beta checkpoint is available to load and serve without extra configuration. Reading the beta queue only needs a reader-level AI-Horde key, so the worker
reuses your configured `api_key` (the anonymous `0000000000` works too).

Being *available* is not the same as being *loaded*: as with any model, the worker only serves a
beta model once its `name` is in your `models_to_load` list (a literal entry, or via an "all"/"top"
selection that now includes it).

Every part of the worker that needs the list of models reads one merged view: the canonical reference
with the beta queue laid over it, where a beta record replaces a canonical record of the same name. The
config load (including `only_models_on_disk`, which keeps a beta model whose files are on disk), the
scheduler's model reference, the dashboard's model picker and the benchmark all use it. When the beta
queue cannot be reached, the view falls back to the canonical reference alone. The worker reads the view
once and reads it again after each periodic reference refresh, but the scheduler's model reference keeps
the set it started with until the worker restarts.

Only the main worker process reads the beta queue from the network. It writes the beta records it holds to
a `pending` folder beside the model reference files, at startup and after each successful refresh, and
the inference and other worker processes load their beta models from those files through
`HORDELIB_BETA_MODELS_DIR`. When a refresh cannot read the beta queue, the previous files stay in place.

When the config load drops a configured model, the log lists it. A name that is not in the reference and
a model whose files are missing under `only_models_on_disk` each get an INFO line listing the dropped
names.

To opt out, set the environment variable `HORDELIB_BETA_MODEL_CATEGORIES=""` before launching the
worker. An empty value disables the beta opt-in; any value you set yourself also takes precedence
over the worker's default.
