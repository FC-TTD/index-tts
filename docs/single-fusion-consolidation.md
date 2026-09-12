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
