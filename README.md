# Digit classification API — MLflow, DagsHub, FastAPI, Docker, GitHub Actions, DVC/Azure

This repository contains the completed university assignment project:

- an MLflow experiment tracked through DagsHub;
- a registered Random Forest digit-classification model;
- a FastAPI service with one endpoint, `POST /predict`;
- a Dockerfile that downloads the registered model from the DagsHub MLflow registry at build time;
- a GitHub Actions workflow that publishes the image to GitHub Container Registry (GHCR);
- DVC data versioning backed by Azure Blob Storage, including the required two-version demonstration.

The notebook `MLFlowSetup.ipynb` was not retrained or replaced during deployment/DVC work.

---

## 1. Project overview and structure

Run all commands below from the project root:

```text
mlflow-with-daghub/
```

```text
mlflow-with-daghub/
├── .dvc/
│   ├── .gitignore              # ignores DVC cache/tmp
│   └── config                  # committed DVC remote configuration (no secrets)
├── .github/
│   └── workflows/
│       └── docker-build.yml    # builds and publishes Docker image to GHCR
├── app/
│   └── main.py                 # FastAPI app: only POST /predict
├── data/
│   ├── .gitignore              # DVC-generated; ignores local CSV contents
│   ├── raw_dataset.csv.dvc     # DVC pointer for the supplied dataset
│   ├── dvc_demo.csv.dvc        # DVC pointer for the two-version demo
│   ├── raw_dataset.csv         # local data only; never committed
│   └── dvc_demo.csv            # local data only; never committed
├── scripts/
│   └── download_model.py       # build-time DagsHub/MLflow model download
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── MLFlowSetup.ipynb
├── requirements.txt
└── README.md
```

Large CSV contents are excluded from Git. DVC stores the data in Azure Blob Storage and keeps lightweight `.dvc` pointer files in Git. `.dvc/config.local` is ignored and must never be committed.

---

## 2. MLflow and DagsHub configuration

### DagsHub project

```text
https://dagshub.com/HanineKh/mlflow-with-dagshub
```

Tracking URI:

```text
https://dagshub.com/HanineKh/mlflow-with-dagshub.mlflow
```

The notebook configures remote tracking with:

```python
import dagshub
dagshub.init(repo_owner="HanineKh", repo_name="mlflow-with-dagshub", mlflow=True)
```

### Experiment and runs

| Item | Value |
|---|---|
| Experiment | `RandomForest_V1.0.1` |
| Experiment ID | `0` |
| Run 1 | `random_forest_model_run_1` |
| Run 1 ID | `4ee7f4e7cc1d4ed5b421853a96a72ed5` |
| Run 1 accuracy | `0.9388888888888889` |
| Best run | `random_forest_model_run_V1.0.2` |
| Best run ID | `45e040c76f714bb995632bc76db25f5c` |
| Best run accuracy | `0.9722222222222222` |
| Logged model | `m-12a229d2e94f46859ddbf818b342fd64` |

The run’s normal Artifacts section may appear empty because MLflow 3 stores logged models as separate model entities. The real model files live under the logged model above.

### Registered model

```text
Name:    random_forest_digits
Version: 1
Status:  READY
Source:  models:/m-12a229d2e94f46859ddbf818b342fd64
Run ID:  45e040c76f714bb995632bc76db25f5c
Stage:   None
```

Verify the registry version with:

```text
https://dagshub.com/HanineKh/mlflow-with-dagshub.mlflow/api/2.0/mlflow/model-versions/get?name=random_forest_digits&version=1
```

Only one READY version currently exists. Do not register the model again; doing so would create another version.

---

## 3. FastAPI `/predict` service

The application exposes exactly one route:

```text
POST /predict
```

Interactive docs, ReDoc, and OpenAPI schema are disabled.

Each request must contain exactly one of:

- `features`: a list of exactly 64 finite numeric values; or
- `instances`: a list of rows, each with exactly 64 finite numeric values.

Malformed JSON, wrong feature count, both fields, neither field, `NaN`, or infinity returns HTTP `422`.

At runtime the service loads the model baked into `/app/model`; it does not contact DagsHub while serving predictions.

### Example single prediction

```powershell
$body = @'
{"features":[0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}
'@

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json
```

Verified response using the published image:

```json
{
  "predictions": [0],
  "model_name": "random_forest_digits",
  "model_version": "1"
}
```

Batch request format:

```json
{
  "instances": [
    [64 numbers here],
    [64 numbers here]
  ]
}
```

Verified batch response:

```json
{
  "predictions": [0, 0],
  "model_name": "random_forest_digits",
  "model_version": "1"
}
```

---

## 4. Docker build and run

The Dockerfile uses BuildKit secret support:

```dockerfile
# syntax=docker/dockerfile:1.10
```

The DagsHub token is passed only as a BuildKit secret and is never declared as a Dockerfile `ARG`, `ENV`, layer value, or runtime environment variable:

```dockerfile
RUN --mount=type=secret,id=dagshub_token,env=DAGSHUB_TOKEN \
    DAGSHUB_USER_TOKEN="$DAGSHUB_TOKEN" \
    python scripts/download_model.py ...
```

### Build locally

```powershell
$env:DOCKER_BUILDKIT = '1'

docker build --progress=plain `
  --secret id=dagshub_token,env=DAGSHUB_TOKEN `
  --build-arg DAGSHUB_REPO_OWNER=HanineKh `
  --build-arg DAGSHUB_REPO_NAME=mlflow-with-dagshub `
  --build-arg MLFLOW_MODEL_NAME=random_forest_digits `
  -t digit-api .
```

Successful build output observed:

```text
Selected model: random_forest_digits version 1 (stage: None)
Model saved to /app/model
naming to docker.io/library/digit-api:latest done
```

### Run locally

```powershell
docker run --rm -p 8000:8000 digit-api
```

The container exposes port `8000` and starts with:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The published image contains `/app/model/MLmodel` and `/app/model/model_info.json`, confirming the registered model was downloaded during the image build.

---

## 5. GitHub Actions and GHCR

Workflow file:

```text
.github/workflows/docker-build.yml
```

Triggers:

- push to `main`;
- manual `workflow_dispatch`.

Required GitHub configuration:

Secrets:

- `DAGSHUB_TOKEN`

Variables:

- `DAGSHUB_REPO_OWNER` = `HanineKh`
- `DAGSHUB_REPO_NAME` = `mlflow-with-dagshub`
- `MLFLOW_MODEL_NAME` = `random_forest_digits`
- `DAGSHUB_USERNAME` = optional

`GITHUB_TOKEN` is supplied automatically by GitHub Actions.

The workflow validates all four required values, builds the image, and publishes:

```text
ghcr.io/<lowercase-owner>/<repo>:latest
ghcr.io/<lowercase-owner>/<repo>:<commit-sha>
```

Verified workflow run:

```text
https://github.com/HanineKh/mlflow-with-dagshub/actions/runs/44806489155
```

Result:

- attempt 1 failed because repository variables/secret were not yet configured;
- attempt 2 completed successfully;
- all steps in attempt 2 succeeded, including `Build image and publish to GHCR`.

Published image:

```text
ghcr.io/haninekh/mlflow-with-dagshub:latest
ghcr.io/haninekh/mlflow-with-dagshub:48675fd7a2e222a458515b151a1c9f63a11d4726
```

Both tags were verified with `docker manifest inspect` and resolved to the same manifest.

---

## 6. DVC with Azure Blob Storage

### DVC installation

DVC with Azure support is required:

```powershell
python -m pip install -U "dvc[azure]"
dvc --version
```

Version used during this work: `3.67.1`.

### Azure remote configuration

Committed, non-secret DVC configuration:

```text
Remote name:    myazure
Remote URL:     azure://dvc
Account name:   wut2026
```

These values are stored in `.dvc/config`:

```ini
[core]
    remote = myazure
['remote "myazure"']
    url = azure://dvc
    account_name = wut2026
```

Verification commands:

```powershell
dvc remote list
dvc config remote.myazure.account_name
```

Expected:

```text
myazure azure://dvc     (default)
wut2026
```

### Secure SAS authentication

The Azure guide contains a live SAS token. It must never be printed, committed, uploaded, or placed in `.dvc/config`.

Authentication is stored locally only:

```powershell
# Use your own token from a secure source.
# Do not paste it into README, source code, chat, GitHub Actions, or .dvc/config.
$env:DVC_AZURE_SAS = '<your token from a secure source>'
dvc remote modify --local myazure sas_token "$env:DVC_AZURE_SAS"
Remove-Item Env:\DVC_AZURE_SAS
```

For this assignment, the token from the private PDF guide was read programmatically into a process-local variable and passed directly to `dvc remote modify --local`. It was not displayed and was not staged.

`.dvc/config.local` is ignored by both DVC and the root `.gitignore`. Verify with:

```powershell
git check-ignore -v --no-index -- .dvc/config.local
git status --short
```

### Dataset ignore rules

The root `.gitignore` no longer blanket-ignores `data/`. Instead:

- CSV contents are ignored by the DVC-generated `data/.gitignore`;
- `data/*.dvc` files are tracked;
- `data/.gitignore` is tracked;
- `.dvc/config.local` is ignored.

Verified with `git check-ignore`:

```text
IGNORED data/raw_dataset.csv
IGNORED data/dvc_demo.csv
NOT_IGNORED data/raw_dataset.csv.dvc
NOT_IGNORED data/dvc_demo.csv.dvc
NOT_IGNORED data/.gitignore
IGNORED .dvc/config.local
```

### Track and upload the supplied dataset

Commands used:

```powershell
dvc add data/raw_dataset.csv
git add data/.gitignore data/raw_dataset.csv.dvc
git commit -m "Add DVC tracking for raw dataset"
dvc push
```

Verification observed:

```text
Everything is up to date.
Data and pipelines are up to date.
Cache and remote 'myazure' are in sync.
```

The raw CSV itself was never committed. Its DVC pointer records the 23,324,132-byte file by hash.

---

## 7. DVC two-version demonstration

The experiment uses a small demo dataset so the real dataset is not affected.

### Version 1

`data/dvc_demo.csv` initially contained:

```csv
id,score
1,80
2,90
```

Commands used:

```powershell
dvc add data/dvc_demo.csv
git add data/.gitignore data/dvc_demo.csv.dvc
git commit -m "Add demo dataset version 1"
git tag dvc-demo-v1
dvc push
```

### Version 2

A third record was added:

```csv
3,95
```

`dvc status` detected the change:

```text
data\dvc_demo.csv.dvc:
    changed outs:
        modified:           data\dvc_demo.csv
```

Commands used:

```powershell
dvc add data/dvc_demo.csv
git add data/dvc_demo.csv.dvc
git commit -m "Add demo dataset version 2"
git tag dvc-demo-v2
dvc push
```

`dvc push` reported:

```text
1 file pushed
```

### Pointer history and comparison

```powershell
git log --oneline -- data/dvc_demo.csv.dvc
git diff dvc-demo-v1 dvc-demo-v2 -- data/dvc_demo.csv.dvc
```

The pointer hash changed from the two-row version to the three-row version, while Git stored only the pointer.

### Version-switch verification

The guide’s switching workflow was performed from a clean tree on `main`.

```powershell
$originalBranch = git branch --show-current

git switch --detach dvc-demo-v1
dvc pull
# verified 2 data rows

git switch --detach dvc-demo-v2
dvc pull
# verified 3 data rows

git switch $originalBranch
dvc pull
```

Verified results:

| State | Expected | Observed |
|---|---:|---:|
| `dvc-demo-v1` after `dvc pull` | 2 data rows | 2 |
| `dvc-demo-v2` after `dvc pull` | 3 data rows | 3 |
| `main` after return + `dvc pull` | version tracked by `main` | 3 data rows |
| Raw dataset restored on `main` | 23,324,132 bytes | 23,324,132 bytes |

`dvc gc` was intentionally not run.

---

## 8. Clone and retrieve data as a collaborator

```powershell
git clone https://github.com/HanineKh/mlflow-with-dagshub.git
cd mlflow-with-dagshub

python -m pip install -U "dvc[azure]"

# Configure a personal SAS token locally; never commit it.
dvc remote modify --local myazure sas_token "<your token from a secure source>"

dvc pull
```

After `dvc pull`, the local `data/` directory is populated from Azure according to the `.dvc` pointers in the current Git commit. Switching to `dvc-demo-v1` or `dvc-demo-v2` and running `dvc pull` restores the corresponding demo dataset version.

---

## 9. Testing summary

Verified during this assignment:

- MLflow experiment, two runs, parameters, metrics, and logged model are visible through the DagsHub public MLflow API.
- Registered model `random_forest_digits` version `1` is `READY` and uses the MLflow 3 logged-model source.
- `download_model.py` selected version `1` and downloaded it during Docker build.
- FastAPI returned single and batch predictions from the live registered model in the container.
- Malformed input returned HTTP `422`.
- GitHub Actions final attempt succeeded.
- GHCR published both `latest` and commit-SHA tags.
- GHCR image contains the embedded model metadata.
- DVC initialized successfully.
- Azure remote `myazure` and account name `wut2026` configured.
- SAS token stored only in ignored local config.
- Raw dataset tracked and uploaded with DVC.
- Both demo dataset versions pushed to Azure.
- Both versions restored successfully with `dvc pull`.
- Original branch `main` restored and verified.

---

## 10. Troubleshooting

| Problem | Cause | Action |
|---|---|---|
| `Provided DagsHub token is not valid` | Missing, expired, or placeholder DagsHub token | Create a valid token and set it only in the environment/BuildKit secret |
| `AUTHORIZATION REQUIRED` with OAuth URL | `DAGSHUB_USER_TOKEN` was not available to `dagshub.init()` | Use the Dockerfile secret mapping shown above |
| Missing four GitHub configuration values | Actions variables/secret not configured | Add the variables and secret listed in section 5 |
| `you are not inside of a DVC repository` | Command run outside the project or before `dvc init` | Run from project root after DVC initialization |
| Azure authentication failure | Missing/expired SAS token or wrong account name | Update local-only SAS token; verify `dvc config remote.myazure.account_name` |
| CSV not appearing after clone | DVC data is not in Git | Run `dvc pull` with valid local credentials |
| MLflow model warnings in container | Model was logged with a newer Python/pandas environment | Non-blocking; predictions were verified successfully |
| Local `curl` JSON quoting problems | Windows shell quoting | Use PowerShell `Invoke-RestMethod` examples above |

Do not run `dvc gc` unless you have reviewed exactly what it would delete and explicitly approved it.
