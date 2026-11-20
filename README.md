# Digit classification API — MLflow, DagsHub, FastAPI, Docker, GitHub Actions

## Status at a glance

| Area | Status |
|---|---|
| Model training and logging experiment | Completed in `MLFlowSetup.ipynb` |
| Model artifact in MLflow 3 logged-model storage | Completed |
| `random_forest_digits` version `1` in DagsHub registry | Completed; status `READY` |
| Deployment code present in this project root | Completed |
| FastAPI `/predict` endpoint | Tested with the live `random_forest_digits` version `1` model in the built container |
| Docker image build | Completed; `digit-api:latest` built successfully from the registered model |
| Docker container test | Completed; `/predict` verified with the live registered model |
| GitHub Actions workflow file | Configured |
| GitHub Actions execution / GHCR publish / public GitHub repository | Pending |

Do not interpret the presence of the workflow file as evidence that CI, GHCR publishing, or public deployment has already happened.

---

## 1. Project root and structure

Run every command in this README from the project root:

```text
mlflow-with-daghub/
```

Current structure:

```text
mlflow-with-daghub/
├── app/
│   └── main.py                         # FastAPI app: only POST /predict
├── scripts/
│   └── download_model.py               # build-time DagsHub/MLflow model download
├── .github/
│   └── workflows/
│       └── docker-build.yml            # build + publish to GHCR
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── README.md
├── MLFlowSetup.ipynb                   # original training/logging notebook
└── data/
    └── raw_dataset.csv                 # local dataset file, not used by the API
```

The dataset CSV and local environment/IDE directories are excluded from the Docker context by `.dockerignore`; PDFs and ZIP files are excluded from Git by `.gitignore`. The `.gitignore` still contains legacy nested paths from the previous layout, so always run `git status --short` before committing and do not stage `data/` or `.idea/`. The API does not read `data/raw_dataset.csv`.

---

## 2. Model and registry facts

`MLFlowSetup.ipynb` trains a `RandomForestClassifier` on `sklearn.datasets.load_digits()`:

- input: 64 numeric pixel features from an 8×8 image;
- output: one digit class, 0–9;
- MLflow experiment: `RandomForest_V1.0.1`, experiment ID `0`;
- best run: `random_forest_model_run_V1.0.2`;
- run ID: `45e040c76f714bb995632bc76db25f5c`;
- accuracy: `0.9722222222222222`;
- MLflow 3 logged model: `m-12a229d2e94f46859ddbf818b342fd64`.

The notebook logs the model but does **not** contain a `mlflow.register_model(...)` cell. The registered model was created separately:

```text
Registered model: random_forest_digits
Version:          1
Status:           READY
Source:           models:/m-12a229d2e94f46859ddbf818b342fd64
Run ID:           45e040c76f714bb995632bc76db25f5c
```

The run’s normal Artifacts section appears empty because MLflow 3 stores logged models as separate model entities. The artifact files (`MLmodel`, `model.pkl`, environment files) live under the logged model above. **Do not retrain the model and do not use a guessed path such as `runs:/.../model`.**

To verify the registry without a token, open:

```text
https://dagshub.com/HanineKh/mlflow-with-dagshub.mlflow/api/2.0/mlflow/model-versions/get?name=random_forest_digits&version=1
```

Expected non-secret fields include `name=random_forest_digits`, `version=1`, and `status=READY`.

---

## 3. DagsHub and MLflow configuration

| Name | Kind | Value / meaning |
|---|---|---|
| `DAGSHUB_REPO_OWNER` | GitHub Actions variable / build arg | `HanineKh` |
| `DAGSHUB_REPO_NAME` | GitHub Actions variable / build arg | `mlflow-with-dagshub` |
| `MLFLOW_MODEL_NAME` | optional variable / build arg | `random_forest_digits`; recommended so the intended model is explicit |
| `DAGSHUB_USERNAME` | optional variable / build arg | DagsHub username; if omitted, the token is used as the username |
| `DAGSHUB_TOKEN` | GitHub Actions secret / BuildKit secret | Private DagsHub access token with access to this repository |

Create a token at **DagsHub → Settings → Access Tokens** if needed.

Never put `DAGSHUB_TOKEN` in:

- source code;
- the notebook;
- `README.md`;
- a committed `.env` file;
- a Dockerfile `ARG` or `ENV`;
- GitHub Actions variables (use **Secrets**, not Variables);
- chat output or build logs.

### Authentication troubleshooting

- `Provided DagsHub token is not valid`: the `DAGSHUB_TOKEN` in the Windows User environment is missing, expired, or a placeholder. Create a new DagsHub access token and update the **User** environment variable, then open a new terminal. The build command does not need to change.
- `AUTHORIZATION REQUIRED` with an OAuth URL: `dagshub.init()` did not receive its expected `DAGSHUB_USER_TOKEN`. The current Dockerfile maps the BuildKit secret value to that name for the model-download `RUN` step.

### Registration note

Registration is already complete. If the registry were ever empty, the correct source would be:

```python
mlflow.register_model(
    model_uri="models:/m-12a229d2e94f46859ddbf818b342fd64",
    name="random_forest_digits",
)
```

Do not run this again for the current assignment; doing so would create another model version.

---

## 4. How `scripts/download_model.py` works

The script runs only during the Docker build:

1. Requires `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME`.
2. Reads `DAGSHUB_TOKEN` from the BuildKit secret environment.
3. Configures the DagsHub tracking URI through `dagshub.init(..., mlflow=True)`.
4. Searches the MLflow Model Registry for `MLFLOW_MODEL_NAME`.
5. Keeps only versions that are `READY` and not `ARCHIVED`.
6. Selects the highest numeric version.
7. Downloads that version’s artifacts with `mlflow.artifacts.download_artifacts`.
8. Flattens the model directory so `MLmodel` is at the root of `MODEL_PATH`.
9. Writes `model_info.json` with the selected model name, version, source, run ID, and tracking URI.

With the current registry, passing `--model-name random_forest_digits` selects version `1`.

Important: the script selects the **latest READY version**, not a hard-coded version number. If another version is registered later, the build would select that newer version. For this assignment, the registry currently contains only version `1`.

---

## 5. Docker build and container run

The Dockerfile uses:

```dockerfile
# syntax=docker/dockerfile:1.10
```

This is required because `RUN --mount=type=secret,...,env=DAGSHUB_TOKEN` needs Dockerfile syntax 1.10 or newer.

Dependency installation is resumable. The Dockerfile mounts two BuildKit caches:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/wheelhouse \
    python -m pip download --retries 20 --timeout 120 -r requirements.txt -d /opt/wheelhouse \
 && python -m pip install --no-index --find-links=/opt/wheelhouse -r requirements.txt
```

`pip download` fills the wheelhouse, and `pip install` then installs offline from it. If Docker/TLS interrupts a large wheel, the next build reuses what was already downloaded instead of starting over. `PIP_NO_CACHE_DIR` is intentionally not set.

The token is passed only as a BuildKit secret:

```dockerfile
RUN --mount=type=secret,id=dagshub_token,env=DAGSHUB_TOKEN \
    DAGSHUB_USER_TOKEN="$DAGSHUB_TOKEN" \
    python scripts/download_model.py ...
```

The `dagshub` Python client reads `DAGSHUB_USER_TOKEN`, so the Dockerfile maps the secret value to that name **for this one `RUN` process only**. Without this mapping, `dagshub.init()` starts an interactive OAuth flow and the build fails.

The token is **not** declared as a Dockerfile `ARG`, `ENV`, image layer, or runtime environment variable.

### PowerShell commands

From `mlflow-with-daghub/`:

```powershell
# Set DAGSHUB_TOKEN from your own secret manager before this step.
# Do not paste it into this README, source code, or chat.

$env:DOCKER_BUILDKIT = '1'

docker build --progress=plain `
  --secret id=dagshub_token,env=DAGSHUB_TOKEN `
  --build-arg DAGSHUB_REPO_OWNER=HanineKh `
  --build-arg DAGSHUB_REPO_NAME=mlflow-with-dagshub `
  --build-arg MLFLOW_MODEL_NAME=random_forest_digits `
  -t digit-api .

docker run --rm -p 8000:8000 digit-api
```

Successful build output for this assignment:

```text
Selected model: random_forest_digits version 1 (stage: None)
Model saved to /app/model
naming to docker.io/library/digit-api:latest done
```

If the build reports a different version, inspect the DagsHub registry before testing the container. If the build fails, the failure is reported by Docker and must be fixed before container testing.

---

## 6. FastAPI `/predict` endpoint

The API exposes exactly one route:

```text
POST /predict
```

Interactive docs, ReDoc, and OpenAPI schema are disabled.

Each request must contain either:

- `features`: a list of exactly 64 finite numbers; or
- `instances`: a list of rows, each with exactly 64 finite numbers.

Do not send both fields and do not send neither. Invalid length, `NaN`, infinity, or malformed JSON returns HTTP `422`.

### Test with the running container

From a separate terminal:

Use PowerShell's `Invoke-RestMethod` rather than `curl.exe`; `curl.exe` argument quoting on Windows can corrupt the JSON body.

```powershell
$body = @'
{"features":[0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}
'@

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json
```

Verified response for the sample above:

```json
{
  "predictions": [0],
  "model_name": "random_forest_digits",
  "model_version": "1"
}
```

The digit comes from the model inside the container. `model_name` and `model_version` are omitted only if `model_info.json` is missing. A malformed request such as `{"features":[1,2,3]}` returns HTTP `422`.

MLflow logs non-blocking warnings because the model was saved with Python `3.13.7` and `pandas==3.1.1`, while the image runs Python `3.11` and `pandas 2.3.3`. Prediction requests still succeed.

Batch request:

```json
{
  "instances": [
    [64 numbers here],
    [64 numbers here]
  ]
}
```

### Run the app without Docker

This requires a downloaded MLflow model directory. From `mlflow-with-daghub/`:

```powershell
$env:MODEL_PATH = "model"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The directory in `MODEL_PATH` must contain `MLmodel`. At runtime the app only loads that local model; it does not contact DagsHub.

---

## 7. GitHub Actions workflow

Workflow file:

```text
.github/workflows/docker-build.yml
```

Triggers:

- push to `main`;
- manual `workflow_dispatch`.

The workflow:

1. checks out the repository;
2. validates required configuration;
3. sets up Docker Buildx;
4. logs in to GHCR using the automatic `GITHUB_TOKEN`;
5. builds the image with `MLFLOW_MODEL_NAME` and the `DAGSHUB_TOKEN` BuildKit secret;
6. pushes `ghcr.io/<owner>/<repo>:latest` and `:<commit-sha>`.

### Required GitHub configuration

In **Settings → Secrets and variables → Actions**:

Secrets:

- `DAGSHUB_TOKEN`

Variables:

- `DAGSHUB_REPO_OWNER` = `HanineKh`
- `DAGSHUB_REPO_NAME` = `mlflow-with-dagshub`
- `MLFLOW_MODEL_NAME` = `random_forest_digits` (recommended)
- `DAGSHUB_USERNAME` (optional)

`GITHUB_TOKEN` is supplied automatically by GitHub Actions.

### Publish the repository

After confirming that no secrets, PDFs, ZIP files, datasets, or local model directories are staged:

```bash
git init -b main
git add .
git status --short
git commit -m "Serve registered digit model with FastAPI, Docker and GitHub Actions"
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
git push -u origin main
```

If a repository already exists, skip `git init` and use the existing remote instead. Pushing to `main` triggers the workflow.

After a successful workflow run, make the repository and GHCR package public if the assignment requires public submission.

---

## 8. Requirement checklist

- [x] Existing training notebook inspected; no retraining required
- [x] MLflow 3 logged model identified
- [x] Model registered as `random_forest_digits` version `1`, status `READY`
- [x] FastAPI implementation with one `POST /predict` route
- [x] Build-time DagsHub registry download implemented
- [x] Token passed only as a BuildKit secret; no token in source/image config
- [x] Dockerfile uses syntax 1.10 for secret environment mount
- [x] GitHub Actions workflow configured for build and GHCR publish
- [x] Docker image built successfully in this environment (`digit-api:latest`)
- [x] Docker container started and `/predict` tested against the live model
- [ ] GitHub Actions workflow executed successfully
- [ ] Docker image published to GHCR
- [ ] Public GitHub repository ready for submission

Remaining pending items require repository push access, GitHub configuration, and a successful workflow run. They are not claimed as complete here.
