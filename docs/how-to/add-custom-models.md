# Add custom models

Serving custom models requires the `customizer` role. Request it on
[Discord](https://discord.gg/3DxrhksKzn).

With the role:

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
