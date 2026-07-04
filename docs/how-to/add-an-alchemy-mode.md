# How to add a new alchemy mode

Alchemy is the horde's post-processing / interrogation feature (the `/api/v2/interrogate/*`
endpoints), distinct from image generation. This guide is the procedure for adding a new alchemy
**form**, distilled from adding the image **vectorizer** (raster -> SVG) as a pilot. It spans
multiple repositories, so the goal here is to make the touch points and gotchas explicit.

## The one decision that determines everything: output shape

An alchemy form is image-in, but its *output* is either:

- **Text/JSON** (like `caption` / `interrogation` / `nsfw`, and the new `vectorize`): the result is
  delivered inline in the submit `result` dict. **No R2 image upload.** These forms run on the
  worker's **safety process** (it already owns the CLIP stack and is not comfy-loaded), and are
  *not* members of the server's `KNOWN_POST_PROCESSORS`.
- **Image** (like upscalers / face-fixers / `strip_background`): the result is a PNG/WebP uploaded
  to R2. These run through hordelib's ComfyUI graph on the **dedicated post-processing process**
  (see [Process lanes and job chaining](../explanation/process_lanes_and_chaining.md)), and *are*
  members of `KNOWN_POST_PROCESSORS`.

**Text-output, model-free modes are dramatically cheaper to add.** They do not touch hordelib or
horde-model-reference at all: no ComfyUI graph, no model-reference category, no downloadable model,
and they sidestep the image pipeline's "output is always a PIL image" assumption (e.g.
`HordeLib.post_process()` calls `Image.open()` unconditionally). The vectorizer is text-output and
uses `vtracer` (a pure-python/Rust wheel, no model download), so it touched only **three** repos:
the server, the SDK, and this worker.

If your mode produces an **image**, you additionally need the hordelib graph + model-reference work;
that path is not covered here.

## Touch points (text-output mode)

### 1. AI-Horde server: the one strict gate

The server validates incoming form names against a hardcoded enum. Add your form name to the three
places in `horde/apis/models/stable_v2.py` that read
`["caption", "interrogation", "nsfw"] + list(KNOWN_POST_PROCESSORS.keys())` (the async-request input,
the worker pop-request, and the pop-response models).

Deliberately **do not** add a text-output form to `KNOWN_POST_PROCESSORS` in `horde/consts.py`. That
set is a *routing switch*, not just a list: membership mints an `r2_upload` URL at pop, shortens the
result cache TTL, and scales kudos by image tiles. Leaving your form out routes it down the
text-result path (inline result, longer cache, flat default kudos via `Interrogation.set_forms`).

Everything else on the server is dynamic and needs no change: the DB models store the form `name` as
a free string, pop/worker matching is by string (`get_sorted_forms_filtered_to_worker` ->
`Worker.can_interrogate`), and submit accepts an arbitrary `result` dict.

**Gotcha — bridge capabilities do not gate alchemy.** `horde/bridge_reference.py`
(`BRIDGE_CAPABILITIES` / `check_bridge_capability`) is consulted only for image-generation params.
The alchemy pop path uses `Worker.can_interrogate`, which never calls it, so adding your form there
is a no-op for alchemy. Don't bother (the worker's advertised `forms` list is the real gate).

### 2. horde_sdk: discoverability, plus one hard coupling

Add the form to `horde_sdk/generation_parameters/alchemy/consts.py`: a member in
`KNOWN_ALCHEMY_FORMS` and `KNOWN_ALCHEMY_TYPES`, and an `is_<form>_form()` classifier mirroring
`is_strip_background_form`.

Two things to know about the SDK coupling:

- The **wire models are forward-compatible**: `AlchemyPopFormPayload.form` and
  `AlchemyAsyncRequestFormItem.name` are typed `KNOWN_ALCHEMY_TYPES | str` with a *warn-only*
  validator, so a brand-new form name round-trips as a plain string. The worker does **not** block
  on an SDK release to pop/submit the form.
- The **bridge-data `forms` config validator is NOT forward-compatible**:
  `CombinedHordeBridgeData.validate_alchemy_forms` *raises* on any form not in
  `KNOWN_ALCHEMY_FORMS.__members__`. Against a published SDK that predates your form, this makes a
  config that lists the form un-loadable. The worker works around this (see below) so it does not
  have to wait for an SDK release.

### 3. horde-worker-reGen: the implementation

- **Run the op.** Add a branch to the if/elif chain in `safety_process.py::start_alchemy` that sets
  `result_payload = {"<form>": <text>}`. The source image is already decoded to PIL there. The
  vectorizer's branch calls a small `_vectorize_image` helper.
- **Optional-deps probe.** If the op needs a package that isn't in the base install, add a probe like
  `vectorize_available()` in `capabilities.py` (a plain import probe for a worker-only dependency;
  hordelib-backed features instead read its `FEATURE_KIND` registry), and a matching extra in
  `pyproject.toml` (`vectorize = ["vtracer>=0.6.0"]`), then `uv lock`.
- **Offer the form.** In `alchemy_popper.py::expand_offered_forms`, append the form when configured
  and available; add it to `DEFAULT_ALCHEMY_FORMS` if it should be offered by default.
  `required_capability` needs no change for text-output forms: anything not an upscaler/face-fixer/
  strip_background already falls through to `ALCHEMY_CLIP` (the safety process).
- **Gate on server support.** The server validates a worker's offered pop `forms` against a fixed
  enum and rejects the *entire* pop if any form is unknown, so a new form must only be offered once
  the server lists it. `server_capabilities.py` reads the server's published Swagger form enum
  (`ModelInterrogationFormStable.name`) into a fail-closed, TTL-refreshed cache; a background loop in
  the process manager (`_periodic_server_capabilities_loop`) refreshes it off any hot path, and
  `expand_offered_forms` checks `server_supports_interrogation_form` alongside the local-deps probe.
  This lets the worker ship **ahead of the server's go-live**: it
  withholds the form until the server advertises it, then begins offering it within the TTL with no
  restart. Forms already in every server's enum (caption/nsfw/post-process) do not need this gate.
- **Accept it in config.** Override `validate_alchemy_forms` in `reGenBridgeData` to accept the
  worker-known form in addition to the SDK enum (this is the workaround for the hard SDK coupling in
  §2), still raising on real typos. The form-name constant + worker-known-extras set live in
  `horde_worker_regen/consts.py` as the single source of truth.

The submit path needs **zero changes**: `_submit_single_form` keys image-vs-text on
`result_message.result_payload is None`, so a set `result_payload` is submitted inline with no R2
upload.

## Copy-paste checklist (text-output mode)

1. `AI-Horde/horde/apis/models/stable_v2.py` — add the form name to the 3 validation enums.
2. `horde_sdk/generation_parameters/alchemy/consts.py` — `KNOWN_ALCHEMY_FORMS` + `KNOWN_ALCHEMY_TYPES`
   member + `is_<form>_form()` classifier.
3. `horde_worker_regen/consts.py` — form-name constant, `WORKER_KNOWN_EXTRA_ALCHEMY_FORMS`, `is_<form>_form`.
4. `horde_worker_regen/bridge_data/data_model.py` — override `validate_alchemy_forms`.
5. `horde_worker_regen/capabilities.py` — `<form>_available()` probe (only if it needs an optional dep).
6. `horde_worker_regen/pyproject.toml` — the optional extra (only if it needs a dep) + `uv lock`.
7. `horde_worker_regen/process_management/jobs/alchemy_popper.py` — `expand_offered_forms` branch (+ `DEFAULT_ALCHEMY_FORMS`).
8. `horde_worker_regen/process_management/workers/safety_process.py` — `start_alchemy` branch + the op helper.
9. `horde_worker_regen/server_capabilities.py` — gate the offer on the server's published form enum (lets the worker ship before server go-live).
10. Tests in `tests/test_alchemy_models.py` and `tests/test_server_capabilities.py`.

## Worked examples and variations (text-output)

Several text-output forms now exist; they vary only in their dependency and where the score comes
from, not in the touch-point shape above.

- **`palette`** (pure-Pillow, no dependency). Dominant-colour extraction via Pillow's median-cut
  quantizer. Because it needs no worker-only package, it has **no availability probe**: it gates only
  on server support, like the others, but `expand_offered_forms` does not call an `_available()` check
  for it. The cheapest possible form.
- **`describe`** (worker-only deps). Bundles a BlurHash string and perceptual hashes; its
  `describe_available()` probe imports `blurhash`/`imagehash`, so a lean install withholds it. Same
  shape as the `vectorize`/`vtracer` gate.
- **`aesthetic`** (model-backed, reuses the loaded CLIP). The LAION aesthetic head is a ~3.5 MB MLP
  over the **CLIP ViT-L/14 embedding the safety process already computes** for NSFW checks
  (`_interrogate_image` calls `image_to_features`). So the op (`_score_aesthetic`) embeds nothing new;
  it reuses that embedding and runs the head. The weight is **fetched once and cached** (SHA-256
  verified) by `process_management/workers/aesthetic_predictor.py` rather than vendored, so the worker
  never redistributes it. The head lives in its own module (it imports torch) so the **torch-free
  orchestrator** never loads it; only the safety process does. It needs no local-deps probe (the
  safety process always has torch + CLIP), so it gates on server support only, like `palette`.

### The same score, attached to every generation (gen_metadata)

A model-backed text-output op that is *free on the image-generation hot path* is worth surfacing
beyond on-request alchemy. The aesthetic score is also attached to **every** image generation as
`gen_metadata`, because the safety pass embeds each generated image with CLIP regardless:

- Computed in `safety_process.py::evaluate_safety` (requested via
  `HordeSafetyControlMessage.include_aesthetic_score`) and carried back on
  `HordeSafetyEvaluation.aesthetic_score`.
- Requested only when **both** the operator's `aesthetic_scoring_enabled` bridge flag is set **and**
  the server is known to accept the metadata type. `safety_orchestrator.py` sets
  `include_aesthetic_score` to `aesthetic_scoring_enabled and
  server_supports_generation_metadata_type(AESTHETIC_METADATA_TYPE)`, so no score is even computed
  while the server would reject it. See the server-capability gate below.
- Attached in `message_dispatcher.py` where each evaluation is applied, as a `GenMetadataEntry(type=
  aesthetic_score, value=see_ref, ref=str(score))` on the image's `generation_faults` (the worker's
  gen_metadata bucket). The float rides in `ref`; `value` is the categorical `see_ref` sentinel.
- The metadata `type` is emitted as the **string** `"aesthetic_score"` (a worker-side constant) so the
  worker ships ahead of the SDK release that adds `METADATA_TYPE.aesthetic_score`; the wire model's
  `type_` is `METADATA_TYPE | str` with a warn-only validator.

#### Gating the score on server support

The AI-Horde server validates every `gen_metadata` entry's `type` against the
`GenerationMetadataStable` `type` enum and **rejects the entire generation submit** if it sees a type
it does not list. So attaching `aesthetic_score` before the server's go-live would fail the submit of
an otherwise-good image. The score is therefore gated the same fail-closed way a new interrogation
form is (see the upscaler section below), but on a different enum:

- `server_capabilities.py` reads the server's published Swagger document once per pop-loop iteration
  and exposes `server_supports_generation_metadata_type(type)`, which reports whether the server lists
  a metadata type (fail-closed until the first successful probe). It parses the metadata-type enum
  (`GenerationMetadataStable.properties.type.enum`) alongside the interrogation-form enum from the
  same fetch.
- The refresh runs on its own supervised background loop in the process manager
  (`_periodic_server_capabilities_loop`), **not** inside a job pop loop: a slow or hung swagger fetch
  must never delay job popping. A single owner serves both the image and alchemy flows, and an
  image-generation-only worker keeps the gate current, beginning to attach the score within the probe
  TTL once the server enables the type, with no restart.
- The single server-side change is adding `aesthetic_score` to that enum; until it lands, workers
  carrying the feature stay dark automatically.

## Adding an upscaler model (not a new form)

Adding a new ESRGAN-family upscaler is lighter than adding a form: upscaling is already a form
(`post-process`), so there is no new pipeline, payload, or safety-process branch. A new upscaler is a
new *model* under an existing form, and it travels through two registries that must agree.

- **The weights** live in the esrgan model reference. New models are submitted to the
  horde-model-reference PRIMARY service (`models.aihorde.net`), not the GitHub `esrgan.json`: older
  workers read that file and cannot load newer architectures. Submitting puts the model in the
  service's pending queue, where it is served as a **beta** model. hordelib resolves and downloads it
  automatically once the worker opts the `esrgan` category into beta (the
  `HORDELIB_BETA_MODEL_CATEGORIES` default already includes it). The weight loads through spandrel's
  core registry, which auto-detects the architecture (esrgan, span, dat, realplksr, and so on) from
  the state dict, so the worker needs no per-architecture code.
- **The name** must be accepted by the SDK and the server. Add it to `KNOWN_UPSCALERS` in
  `horde_sdk/generation_parameters/alchemy/consts.py` (hordelib classifies a post-processor as an
  upscaler by membership there; an unknown name raises). Add it to `KNOWN_POST_PROCESSORS` in
  `AI-Horde/horde/consts.py` with a kudos multiplier (and `HEAVY_POST_PROCESSORS` if it is large);
  this is the server's strict gate and the go-live switch.

The server rejects an entire interrogation pop that offers a post-processor it does not list, so a new
upscaler name is gated the same way a new form is: list it in `WORKER_KNOWN_BETA_UPSCALERS`
(`horde_worker_regen/consts.py`) and `expand_offered_forms` withholds it until
`server_supports_interrogation_form` reports the server advertises it. This lets the worker ship ahead
of the server go-live and begin offering the model within the probe TTL once the server catches up,
with no restart. The long-standing upscalers are in every server's enum and are never gated.

### Checklist (upscaler model)

1. `horde_sdk/generation_parameters/alchemy/consts.py` — add the name to `KNOWN_UPSCALERS` (and mirror
   in `KNOWN_ALCHEMY_TYPES`).
2. `horde-model-reference` — submit a `LegacyEsrganRecord` to `/api/model_references/v1/esrgan`
   (`scripts/submit_esrgan_models.py`); it enters the pending queue and is served as beta.
3. `horde_worker_regen/consts.py` — add the name to `WORKER_KNOWN_BETA_UPSCALERS`.
4. `AI-Horde/horde/consts.py` — add the name to `KNOWN_POST_PROCESSORS` (+ `HEAVY_POST_PROCESSORS` if
   large). This is the go-live switch; deploy it last.
5. Tests in `tests/test_alchemy_models.py` (offered when the server advertises it, withheld otherwise).

## Adding a face-restoration model (not a new form)

Face-fixing is already a form (`post-process`), so adding a face restorer is the same shape as adding an
upscaler, with one engine difference: face restoration does **not** go through spandrel's auto-detecting
upscale path. hordelib's `facerestore_cf` node loads face models by architecture, dispatching on the
weight *filename*:

- A **GFPGAN-arch** weight (e.g. `GFPGANv1.3.pth`) loads through the node's existing GFPGAN path, which
  detects the StyleGAN2 state dict. No hordelib change is needed for a new GFPGAN-family weight.
- A **RestoreFormer** weight loads through spandrel's core registry (which detects the RestoreFormer
  arch); the node has a dedicated `restoreformer` filename branch for it. Its underlying module shares
  the node's call convention (a `[-1, 1]` tensor in, `(image, None)` out, ignoring the unused fidelity
  kwarg), so the restore loop drives it unchanged. Any *other* architecture (CodeFormer-style, GPEN,
  RestoreFormer++) needs a new node branch, which is why those are not drop-ins.

The weights live in the **`gfpgan`** model reference (which maps to the same `facerestore_models` folder
as `codeformer`), submitted to the PRIMARY service's pending queue and served as beta. The worker opts the
`gfpgan` category into beta via the `HORDELIB_BETA_MODEL_CATEGORIES` default. Names gate exactly like beta
upscalers: list them in `WORKER_KNOWN_BETA_FACEFIXERS` and `expand_offered_forms` withholds them until
`server_supports_interrogation_form` reports the server advertises them. `GFPGAN`/`CodeFormers` are in
every server's enum and are never gated. Prefer permissively-licensed weights: GFPGAN and RestoreFormer
are Apache-2.0, whereas CodeFormer and GPEN are non-commercial.

### Checklist (face-restoration model)

1. `horde_sdk/generation_parameters/alchemy/consts.py` — add the name to `KNOWN_FACEFIXERS` (and mirror
   in `KNOWN_ALCHEMY_TYPES`).
2. `hordelib` — only if the weight is neither GFPGAN-arch nor RestoreFormer: add a filename branch to
   `FaceRestoreModelLoader.load_model` (and a forward-call branch in `restore_face` if its call
   convention differs).
3. `horde-model-reference` — submit a `LegacyGfpganRecord` to `/api/model_references/v1/gfpgan`
   (`scripts/submit_facefixer_models.py`); it enters the pending queue and is served as beta.
4. `horde_worker_regen/consts.py` — add the name to `WORKER_KNOWN_BETA_FACEFIXERS`.
5. `AI-Horde/horde/consts.py` — add the name to `KNOWN_POST_PROCESSORS` (+ `HEAVY_POST_PROCESSORS` if
   large). This is the go-live switch; deploy it last.
6. Tests in `tests/test_alchemy_models.py` (offered when the server advertises it, withheld otherwise).

## Verify end-to-end

- `pytest tests/test_alchemy_models.py` (worker), `pytest -k alchemy` (SDK).
- Against a local AI-Horde dev server: `POST /api/v2/interrogate/async` with
  `forms: [{"name": "<form>"}]` and a small source image, run this worker as an alchemist offering
  the form, then poll `/api/v2/interrogate/status/{id}` and confirm the result carries your output.
  `AI-Horde/tests/integration/test_alchemy.py` is the template for the API calls.
