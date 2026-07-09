# Add custom models

Serving custom models requires the `customizer` role. Request it on
[Discord](https://discord.gg/3DxrhksKzn).

With the role, the easiest path is the dashboard **Config** tab:

1. Open **Config → Workload**.
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

   Supported baselines: `stable_diffusion_1`, `stable_diffusion_2_768`, `stable_diffusion_2_512`,
   `stable_diffusion_xl`, `stable_cascade`, `flux_1`.

3. Add the model `name` to your `models_to_load` list.

## Rules and limits

- Only Flux.schnell models are allowed. Flux.dev and its derivatives are **not** permitted.
- Custom model names cannot conflict with existing horde model names.
- The horde treats custom models as SD 1.5 for kudos and safety purposes.

See [Bridge configuration](../explanation/bridge_config.md#custom-models) for how custom models flow
into the pop request.

## Beta models

Some models are published to the model reference's "pending" (beta) queue before they are promoted
into the canonical reference. The worker opts every install into the image-generation beta by
default, so a beta checkpoint such as `Qwen-Image_fp8` is available to load and serve without any
extra configuration. Reading the beta queue only needs a reader-level AI-Horde key, so the worker
reuses your configured `api_key` (the anonymous `0000000000` works too).

Being *available* is not the same as being *loaded*: as with any model, the worker only serves a
beta model once its `name` is in your `models_to_load` list (a literal entry, or via an "all"/"top"
selection that now includes it).

To opt out, set the environment variable `HORDELIB_BETA_MODEL_CATEGORIES=""` before launching the
worker. An empty value disables the beta opt-in; any value you set yourself also takes precedence
over the worker's default.
