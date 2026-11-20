"""Download the latest READY registered MLflow model into the image.

With no --tracking-uri, DagsHub is configured from DAGSHUB_REPO_OWNER and
DAGSHUB_REPO_NAME, and authentication uses the DAGSHUB_TOKEN build secret.

Nothing printed by this script includes the token.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ARCHIVED = "ARCHIVED"
READY = "READY"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-owner", default=os.getenv("DAGSHUB_REPO_OWNER", ""))
    parser.add_argument("--repo-name", default=os.getenv("DAGSHUB_REPO_NAME", ""))
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", ""))
    parser.add_argument("--username", default=os.getenv("DAGSHUB_USERNAME", ""))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--dst", default=os.getenv("MODEL_PATH", "/app/model"))
    return parser.parse_args()


def version_number(version: object) -> int:
    try:
        return int(str(version.version).strip())  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return 0


def stage_rank(version: object) -> int:
    stage = str(getattr(version, "stage", "") or "").upper()
    return {"PRODUCTION": 3, "STAGING": 2, "NONE": 1, ARCHIVED: 0}.get(stage, 1)


def usable(version: object) -> bool:
    return (
        getattr(version, "status", None) == READY
        and str(getattr(version, "stage", "") or "").upper() != ARCHIVED
    )


def configure_tracking(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        # Optional non-DagsHub override, useful for local smoke tests.
        mlflow.set_tracking_uri(args.tracking_uri)
        print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
        return

    missing = [
        name
        for name, value in (
            ("DAGSHUB_REPO_OWNER", args.repo_owner),
            ("DAGSHUB_REPO_NAME", args.repo_name),
        )
        if not value
    ]
    if missing:
        fail(f"Missing DagsHub configuration: {', '.join(missing)}")

    token = os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not token:
        fail(
            "No DagsHub token found. Pass it as a BuildKit secret, for example: "
            "docker build --secret id=dagshub_token,env=DAGSHUB_TOKEN ..."
        )

    # DagsHub accepts either (username, token) or the token alone as username.
    os.environ["DAGSHUB_TOKEN"] = token
    if args.username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = args.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)

    try:
        import dagshub

        dagshub.init(
            repo_owner=args.repo_owner,
            repo_name=args.repo_name,
            mlflow=True,
        )
    except Exception as exc:
        fail(f"Could not connect to DagsHub repository: {type(exc).__name__}: {exc}")

    print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")


def select_version(client: MlflowClient, model_name: str) -> object:
    if model_name:
        versions = [
            version
            for version in client.search_model_versions(f"name='{model_name}'")
            if usable(version)
        ]
        if not versions:
            fail(
                f"No READY, non-archived versions found for registered model '{model_name}'. "
                "Check the model name in the DagsHub MLflow UI."
            )
        versions.sort(key=lambda version: version_number(version))
        return versions[-1]

    candidates: list[object] = []
    for model in client.search_registered_models():
        candidates.extend(version for version in getattr(model, "latest_versions", []) if usable(version))

    if not candidates:
        fail(
            "No READY registered model versions were found in the DagsHub MLflow registry. "
            "The supplied notebook logs runs but does not register a model; register it first "
            "or set MLFLOW_MODEL_NAME to an existing registered model."
        )

    # Prefer a promoted stage, then the most recently updated version.
    candidates.sort(
        key=lambda version: (
            stage_rank(version),
            getattr(version, "last_updated_timestamp", 0) or 0,
            version_number(version),
        )
    )
    return candidates[-1]


def flatten_model(root: Path, dst: Path) -> Path:
    mlmodel_files = list(root.rglob("MLmodel"))
    if not mlmodel_files:
        fail(f"Downloaded artifact has no MLmodel file under {root}")

    model_root = mlmodel_files[0].parent
    if model_root != dst:
        for item in list(model_root.iterdir()):
            target = dst / item.name
            if target == item:
                continue
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    if not (dst / "MLmodel").is_file():
        fail("Model artifact could not be placed at the root of the model directory")
    return dst


def main() -> None:
    args = parse_args()
    configure_tracking(args)

    try:
        client = MlflowClient()
        version = select_version(client, args.model_name)
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Could not read the MLflow model registry: {type(exc).__name__}: {exc}")

    print(f"Selected model: {version.name} version {version.version} (stage: {getattr(version, 'stage', None)})")

    dst = Path(args.dst).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(dst))
        )
    except Exception as exc:
        fail(f"Could not download model artifacts from {version.source}: {type(exc).__name__}: {exc}")

    model_root = flatten_model(downloaded, dst)

    info = {
        "model_name": version.name,
        "model_version": str(version.version),
        "stage": getattr(version, "stage", None),
        "run_id": getattr(version, "run_id", None),
        "source": version.source,
        "tracking_uri": mlflow.get_tracking_uri(),
    }
    (model_root / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Model saved to {model_root}")


if __name__ == "__main__":
    main()
