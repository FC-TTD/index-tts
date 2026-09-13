# Index Hub Runtime

This entry uses the shared `ttd-model-runtime` SDK for lifecycle, private control, activity tracking, passive health and audio tools. Upstream `api.py`, `entry.py`, `webui.py` and `indextts/` are unchanged on this branch.

- `__main__.py`: assembles API and the copied Gradio UI with one Runtime.
- `startup.py`: shared container startup arguments, cache paths and weight-file validation; no separate CLI client or model lifecycle manager.
- `http_compat.py`: preserves the API JSON home and the existing UI host/path aliases, without changing callers' URLs.
- `adapter.py`: native model constructor, model-local inference lock, completion/cache hooks and UI input mapping. The official device map and precision choices are retained.
- `service.py`: shared existing API inference/postprocessing pipeline. Both API and UI call it.
- `api.py`: existing multipart API contract registered on the foundation.
- `ui.py`: copy of the native UI from verified api baseline `b91f4a4`; layout and controls retained, manager wiring replaced.
- `parameters.py`: existing API defaults used by both entrypoints.

```sh
python -m hub_runtime --model_dir /app/checkpoints --host 0.0.0.0 --port 8000 --bf16 --device cuda:0
python -m hub_runtime describe
```

The unused legacy `--preload_model` / `--no-preload_model` options are removed. Residency and loading are controlled by Hub and Runtime; there is no second SmartModel timer or preload path. The copied UI uses the same startup configuration and weight validation as the API.

`describe` emits the model OpenAPI contract without loading weights, requiring Hub credentials or contacting a node. It is a build-time input for later Gateway declaration generation, not a second manually maintained configuration.

Configure `HUB_SERVICE_ID`, `HUB_NODE_URL`, `HUB_RUNTIME_TOKEN`, `HUB_RUNTIME_VERSION`, `HUB_RUNTIME_ACTOR_ID` and `HUB_RUNTIME_STATE_DIR` through deployment. The actor must match the node binding and use a fresh private credential; state is isolated per actor, while declared business mounts and the glossary path remain shared across replacement. The node must advertise `admission-intent-recovery` and `runtime-actor-identity`; older node versions fail compatibility checks before model execution. Use `HUB_GLOSSARY_PATH` for a declared writable glossary path. The Docker asset installs a fixed Runtime wheel from a fixed Hub image, using `--no-deps` to retain the native model environment.

UI defaults follow the existing API tuning: emotion weight 1.0, shared sampling defaults, and the API's silence removal/EQ/loudness processing. UI vectors are sent through the same API vector handling rather than receiving another UI-only normalization pass. A zero top_k follows the existing API omission behavior. These changes align effective execution, not just displayed numbers. Native UI controls remain present; CPU/GPU/offload placement is not rewritten.

Tests use real FastAPI/Gradio and audio libraries with a fake inference model. They cover matching API/UI audio and parameters, OpenAPI defaults, and the complete original Gradio upload/queue/generation callback. Both a current dependency environment and the observed online Index versions (FastAPI 0.116.2, Gradio 5.45.0, NumPy 2.2.6, SciPy 1.16.2, librosa 0.10.2.post1) pass. This branch has not yet been deployed or GPU-smoked; previous online evidence belongs to the earlier adapter version.

The old ttd-fastapi-utils subtree is retained for callers not yet migrated; this new entry does not import it. Existing fusion deployment assets remain available until an explicit rollout switches the service.

`index-hub` in the current Compose asset is an auxiliary address for validation before cutover. Formal adoption must take over the existing business domain and preserve its API/UI URLs so callers do not change addresses. The cutover changes Caddy's backend, not the public URL; rollback preserves that same URL. A redirect to a new `-hub` domain is not a substitute. Acceptance through the existing business URL and its callers is still pending.

`docker/compose.hub-business.yml` prepares the original `index-api`, `xique` and `xique-tts-public-proxy` routes. It replaces the auxiliary route labels and preserves the existing public Host/HTTPS headers. It requires a Hub image with `model-entry --public-origin` support and the SDK Gradio mount-prefix fix. Runtime port 8000 remains private to the model network; proxy header trust is enabled only in this overlay for the CPU entry hop. This overlay has been validated, not deployed; the current Hub fresh-release installer cannot perform the old Swarm actor/route handoff. Do not layer it onto the live service before that handoff is complete.

Expanded tests use the real `create_app` entry and native Gradio under both the internal UI Host and the existing public Host/HTTPS scheme. They check API JSON versus UI home behavior, original root-path upload/queue aliases, the `/__gradio__` config prefix, generated download origins and actual file retrieval. Model inference remains a fixture, not GPU acceptance.

`device_observation.py` records native tensor devices/dtypes and device maps for smoke evidence. It does not move tensors, change precision or reject official CPU/GPU mixed placement. The latest acceptance plan finishes Index on worker first, then expands to the second host; Compose uses `HUB_NODE_ID` for host-specific compilation caches. The newer actor integration has not yet been deployed.
