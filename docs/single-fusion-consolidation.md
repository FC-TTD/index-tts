# Single Index fusion deployment

The approved target is one `index-tts_api-3` replica on worker GPU1. It already combines API and WebUI using the same model manager. Preserve the current model image, weights, caches, `/generate`, `index-api`, `xique` and the existing public-proxy boundary.

`docker/docker-stack.yml` is the final desired state. Its pinned image is the previously deployed and verified `h-54be7b525694` image, digest `7f3309bdacda8424cefb094cf09bee6f2b4499db1174bfa33da1b98378a3176a`. Build metadata in its environment describes that existing application image, not the new deployment-only commit.

Commit deployment inputs, then use the existing project entrypoint:

```sh
./deploy.sh consolidate --smoke-command 'python3 scripts/smoke_single_fusion.py'
./deploy.sh consolidate --apply --smoke-command 'python3 scripts/smoke_single_fusion.py'
```

The first command is read-only preparation. The second requires deployment and retirement authorization. It updates Portainer stack228 / endpoint4 in two stages: move all Index ingress onto api-3 while retaining old models; verify Caddy and real synthesis; drain old tasks; then prune api-1/2. Only service labels and equivalent immutable image notation change in the retained model specification. Any unexpected runtime drift or concurrent Portainer edit blocks the update.

Each run stores the previous management stack, a private rollback.json request including the original environment, transition and final stack, source commit, hashes, execution stage and failure status in a private evidence directory. If the transition smoke fails, both old instances remain available. Rollback uses Portainer `PUT /api/stacks/228?endpointId=4` with the recorded `rollback.yml`, original stack environment, `prune=true`, `pullImage=false`, then verifies service replicas, ingress and synthesis. Do not use direct `docker stack deploy` for this managed stack. A failure after pruning requires this explicit rollback; the script does not silently report success.

Tests cover retention before cutover, unexpected services, deployment drift and host-scoped route verification. The smoke sends repository reference audio to both normal and Premium `index-api` paths and the xique API, verifies actual WAV outputs, and checks the WebUI document. No shared data or unrelated model is deleted.

## Production acceptance — 2026-09-12

Completed with deployment source commit `558184cadfb91fe5b5125a504aaad728ebf6d04b`. Portainer stack228 and Swarm both contain only `api-3`, one healthy replica on worker GPU1. Final management and desired YAML SHA256 both equal `887eeabe6c11f2d10ee720c36d06d68099b20ce26757a1575ce5b841c26e08b2`; the application image digest stayed unchanged.

Real speech was generated before pruning and after convergence through ordinary `index-api`, `X-LB-Mode: Premium`, and `xique` API requests. Both batches produced valid WAV files and the xique Gradio document remained available. The initial smoke stopped before pruning because the isolated clone contained an LFS pointer instead of reference audio; the fixture was resolved from its verified repository SHA256, and readiness checks were added before retry. The complete first pre-change rollback record is `/tmp/index-consolidation-5a9_dj6e/rollback.json`; successful final evidence is `/tmp/index-consolidation-rgld1flz/release.json` and audio evidence `/tmp/index-fusion-smoke-jxoizc1l/result.json`.

The transition update recreated tasks despite retaining the same image because Docker's stack image metadata notation changed; all old service declarations were retained until the successful speech batch and drain checks. No shared models, cache, unrelated service or volume was removed. Final worker usage snapshot after removal: GPU0 5531MiB, GPU1 14060MiB, GPU2 998MiB; idle unloading and other callers can change this snapshot, so it is not an isolated memory benchmark. Existing public-proxy headers and domains were preserved, with no access-boundary change. Rollback payloads are retained and were not executed.
