# Build and Publish to Private PyPI

This package uses Hatchling (PEP 517) as the build backend and is compatible with `python -m build`.

## 1) Prepare Environment

- Ensure your Python environment is activated (conda/venv)
- Install build and twine

```bash
python -m pip install --upgrade pip
python -m pip install build twine
```

## 2) Bump Version and Build Artifacts

```bash
# Run at repository root or inside the package directory
cd packages/fastapi_cuda_health

# Bump version in pyproject.toml (project.version), commit and tag
# e.g. version = "0.2.0"

python -m build
# artifacts will be placed in dist/
```

Expected outputs:

- `dist/fastapi-cuda-health-<version>-py3-none-any.whl`
- `dist/fastapi-cuda-health-<version>.tar.gz`

## 3) Upload to Private PyPI

Pick the example that matches your server implementation.

### Option A: Using ~/.pypirc (recommended)

```bash
twine upload -r <REPO_ALIAS> dist/*
```

Your ~/.pypirc should contain the repository URL and credentials for <REPO_ALIAS>.

### Option B: pypiserver

```bash
twine upload \
  --repository-url http://<HOST>:<PORT>/ \
  -u <USERNAME> -p <PASSWORD> \
  dist/*
```

### Option C: devpi

```bash
twine upload \
  --repository-url http://<HOST>:<PORT>/<USER>/<INDEX>/ \
  -u <USERNAME> -p <PASSWORD> \
  dist/*
```

### Option D: Nexus/Artifactory (PyPI hosted)

```bash
twine upload \
  --repository-url https://<HOST>/repository/<REPO_NAME>/ \
  -u <USERNAME> -p <PASSWORD> \
  dist/*
```

If you have credentials configured in `~/.pypirc`, you can instead use:

```bash
twine upload -r <REPO_ALIAS> dist/*
```

## 4) Install from Private PyPI

```bash
pip install \
  --index-url http://<HOST>:<PORT>/simple \
  --extra-index-url https://pypi.org/simple \
  fastapi-cuda-health
```

Replace placeholders with your actual server address and credentials.

Notes:

- This package includes a default notifier (`fastapi_cuda_health/notify.py`). If you plan to publish to a public index, make sure the webhook and behavior comply with your security policies, or override by passing `notifier=None` to `setup_cuda_health()`.
