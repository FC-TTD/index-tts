# AGENTS.md

## Scope

These instructions apply to the whole `index-tts` repository.

When working on files under `packages/ttd_fastapi_utils/`, treat that package as a reusable library with its own packaging, test environment, and release path.
The package is only temporarily colocated in this repository for fast integration testing. It is not part of the `index-tts` service application itself.

## Repository Routing

- Service application work includes files such as `api.py`, `docker/`, `ansible/`, and runtime code under `indextts/`.
- Library work for `ttd_fastapi_utils` includes files under `packages/ttd_fastapi_utils/`.
- `ttd_fastapi_utils` should be reasoned about as an independent package subtree, even though it currently lives inside this repository.
- Do not assume that a library change should be deployed through the service deployment path.

## Release And Deployment Rules

- The repository has three distinct operator paths exposed by `deploy.sh`:
  - default / `--docker`: Docker Compose style deployment using `ansible/site.yml`
  - `--swarm` or `api`: Swarm API deployment using `ansible/site.yml`
  - `--pub` or `pypi`: package publication using `packages/pub.yml`
- For changes limited to `packages/ttd_fastapi_utils/`, the intended release path is package publication, not Docker or Swarm deployment.
- `packages/pub.yml` is the authoritative publication entry for `ttd_fastapi_utils` and publishes `ttd_fastapi_utils` through the `pypi_pub` role.
- Do not route `ttd_fastapi_utils` changes through `ansible/site.yml` unless the user explicitly asks for a service rollout that also consumes a new package version.
- Do not treat `index-tts` root runtime conventions, Docker deployment steps, or service rollout assumptions as automatically applicable to `ttd_fastapi_utils`.
- Before changing the package version in `packages/ttd_fastapi_utils/pyproject.toml`, confirm the intended version bump with the user if they have not explicitly specified it. Do not guess patch/minor/major policy for package publication.

## Working Rules

- When changing `packages/ttd_fastapi_utils/`, prefer validation at the package level first.
- `ttd_fastapi_utils` may use its own package-local virtualenv or publication environment; do not assume the repository root `.venv` is the source of truth for package validation.
- When running tests for `packages/ttd_fastapi_utils/` from the repository root, ensure imports resolve to the local package source under `packages/ttd_fastapi_utils/src` rather than an already installed site-package copy.
- Before publishing `ttd_fastapi_utils`, include at least one package-level integration test that mounts the development version of the plugin into a real FastAPI app and verifies routing through real HTTP requests.
- Before publishing `ttd_fastapi_utils`, include at least one real audio HTTP integration check for `postprocess` or `preset`; health-route verification alone is not sufficient for package publication.
- Before publishing `ttd_fastapi_utils`, verify remaining runtime-facing helpers that change service behavior, including `SmartModel`, `speed_control`, and notifier wiring.
- Treat that FastAPI integration check as part of the package publication path, not as part of the `index-tts` service deployment flow.
- If a task involves deployment, release, or CI/CD decisions, inspect `deploy.sh`, `ansible/site.yml`, and `packages/pub.yml` before acting.
