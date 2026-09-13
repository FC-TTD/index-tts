# Index Hub Runtime

This entry uses the shared `ttd-model-runtime` SDK for lifecycle, private control, activity tracking, passive health and audio tools. Upstream `api.py`, `entry.py`, `webui.py` and `indextts/` are unchanged on this branch.

- `__main__.py`: assembles API and the copied Gradio UI with one Runtime.
- `startup.py`: shared container startup arguments, cache paths and weight-file validation; no separate CLI client or model lifecycle manager.
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

Configure `HUB_SERVICE_ID`, `HUB_NODE_URL`, `HUB_RUNTIME_TOKEN`, `HUB_RUNTIME_VERSION` and `HUB_RUNTIME_STATE_DIR` through deployment. Use `HUB_GLOSSARY_PATH` for a declared writable glossary path. The node must advertise `admission-intent-recovery`; older node versions fail compatibility checks before model execution. The Docker asset installs a fixed Runtime wheel from a fixed Hub image, using `--no-deps` to retain the native model environment.

UI defaults follow the existing API tuning: emotion weight 1.0, shared sampling defaults, and the API's silence removal/EQ/loudness processing. UI vectors are sent through the same API vector handling rather than receiving another UI-only normalization pass. A zero top_k follows the existing API omission behavior. These changes align effective execution, not just displayed numbers. Native UI controls remain present; CPU/GPU/offload placement is not rewritten.

Tests use real FastAPI/Gradio and audio libraries with a fake inference model. They cover matching API/UI audio and parameters, OpenAPI defaults, and the complete original Gradio upload/queue/generation callback. Both a current dependency environment and the observed online Index versions (FastAPI 0.116.2, Gradio 5.45.0, NumPy 2.2.6, SciPy 1.16.2, librosa 0.10.2.post1) pass. This branch has not yet been deployed or GPU-smoked; previous online evidence belongs to the earlier adapter version.

The old ttd-fastapi-utils subtree is retained for callers not yet migrated; this new entry does not import it. Existing fusion deployment assets remain available until an explicit rollout switches the service.

`index-hub` in the current Compose asset is an auxiliary address for validation before cutover. Formal adoption must take over the existing business domain and preserve its API/UI URLs so callers do not change addresses. The cutover changes Caddy's backend, not the public URL; rollback preserves that same URL. A redirect to a new `-hub` domain is not a substitute. Acceptance through the existing business URL and its callers is still pending.
