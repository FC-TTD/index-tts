# TTD Index-TTS Runtime Notes

This file records local TTD runtime decisions that are not upstream IndexTTS2 model behavior.

## Model and cache layout

IndexTTS 2.0 and 2.5 weights are kept in separate NAS directories:

- `/TTD-Data/index-tts/checkpoints-v2` is the retained 2.0 rollback model.
- `/TTD-Data/index-tts/checkpoints-v25` is the active 2.5 model.

The shared `/TTD-Data/index-tts/hf-cache` directory is mounted at both
`/opt/hf_cache` and `/app/checkpoints/hf_cache`. The first path remains the
process-wide Hugging Face cache; the second is required because the upstream
2.5 runtime resolves auxiliary models relative to `model_dir`. Both container
paths intentionally refer to the same NAS directory.

Do not replace or mix the two primary checkpoint directories. Rollback restores
the archived 2.0 image together with the `checkpoints-v2` mount.

## API compatibility

`POST /generate` remains the stable provider contract. IndexTTS 2.5 adds the
optional `language` form field with values `ZH`, `EN`, `JA`, `ES`, and `AR`;
omitting it preserves the historical Chinese behavior. The provider translates
the existing `speed` field to the 2.5 `duration_factor` runtime argument.

## Xique public ingress

`api-3` is the public Gradio/WebUI-facing instance for xique:

- internal service host: `http://xique`
- public host: `https://xique-tts.api.ttd.honeywave.net`
- mounted Gradio path: `/__gradio__`

The public route is intentionally a two-hop proxy:

```text
tx-derp public-caddy
  -> http://xique-tts-public-proxy
  -> index-tts_api-3:8000
```

`xique-tts-public-proxy` is a synthetic internal Caddy host generated from
`docker/docker-stack.yml`.  The outer public Caddy must proxy to this internal
host instead of proxying to `http://xique` directly.

Reason: Gradio generates absolute URLs from the request host and forwarded
scheme.  When the public route reaches the app as the internal `xique` host, the
HTML can leak `http://xique` URLs, causing browser mixed-content failures inside
`audio-nav`.  The synthetic proxy lets the internal Caddy pass:

- `Host: xique-tts.api.ttd.honeywave.net`
- `X-Forwarded-Host: xique-tts.api.ttd.honeywave.net`
- `X-Forwarded-Proto: https`
- `X-Forwarded-Port: 443`

`GRADIO_ALLOWED_HOSTS` must include both `xique` and the public host so the same
process keeps working for internal and public routes.

When changing or redeploying the Swarm stack, verify that the generated Caddy
config still contains a `http://xique-tts-public-proxy` site block.  Portainer
stack redeploys rebuild service labels from `docker/docker-stack.yml`; manual
label edits are not durable.

Minimum public smoke after ingress-related changes:

1. authorized `GET https://xique-tts.api.ttd.honeywave.net/` returns 200 HTML;
2. response HTML contains the public host and `/__gradio__`;
3. response HTML does not contain `http://xique`;
4. `/gradio_api/info` is reachable through the public route;
5. common unauthenticated API-style requests remain blocked by public ingress auth.

## Generated audio download filenames

Both the FastAPI `/generate` endpoint and the Gradio WebUI use the same helper in
`download_filename.py` to build output filenames:

```text
{source_basename}-{inference_text}-{epoch_ms}.wav
```

Rules:

- `epoch_ms = int(time.time() * 1000)`;
- preserve non-ASCII text, including Chinese;
- strip the source file extension and keep only its basename;
- replace path separators and other unsafe filename characters;
- truncate the source part to 48 characters and the text part to 40 characters;
- fall back to `source` / `text` for empty parts.

Reason: internal users download generated clips from multiple TTS services and
need filenames that identify the reference/source file and the inference text
without opening the WAV or matching it back to logs.  The epoch suffix keeps names
stable enough for humans while avoiding collisions during repeated generation.

Implementation notes:

- API downloads set `Content-Disposition` with both ASCII fallback and UTF-8
  `filename*` so Chinese names survive browser downloads.
- WebUI output paths are written under `outputs/` using the target basename;
  Gradio then exposes that basename as the downloadable file name.
- `download_filename.py` must be present in both default Docker and Swarm rsync
  include lists in `ansible/site.yml`, otherwise remote build contexts can miss
  the runtime helper.

## Gradio audio filename handling

Do not subclass `gr.Audio` just to retain `orig_name`.  In Gradio 5.45 this can be
serialized as a custom component type (for example `Namedaudio`), causing the
browser to request `/gradio_api/custom_component/.../client/...` assets that do
not exist in the deployed service.  The failure appears as stylesheet MIME errors
because the missing CSS route returns JSON 404.

The current implementation keeps the component type as standard `Audio` and wraps
only the component instance's `preprocess` function so the callback receives both
the temporary path and `payload.orig_name`.  After WebUI changes, confirm
`/gradio_api/info` reports `Audio` for `/gen_single` audio parameters, not a
custom component name.
