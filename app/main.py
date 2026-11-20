"""FastAPI service for the registered digit-classification model.

The model artifact is downloaded into MODEL_PATH at Docker image build time.
At runtime the service only loads that local MLflow model; it never calls
DagsHub while serving predictions.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

# sklearn.datasets.load_digits() produces one flat vector of 64 pixels (8x8).
N_FEATURES = 64
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model"))


def _model_info() -> dict[str, Any]:
    info_file = MODEL_PATH / "model_info.json"
    if info_file.is_file():
        try:
            return json.loads(info_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def _input_columns(model: Any) -> list[str] | None:
    """Use the logged input schema's column names when the model has one."""
    try:
        schema = model.metadata.get_input_schema()
        names = list(schema.column_names) if schema is not None else []
        if len(names) == N_FEATURES:
            return names
    except Exception:
        # Models logged without a signature (as in the supplied notebook)
        # are handled as plain numeric arrays below.
        pass
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (MODEL_PATH / "MLmodel").is_file():
        raise RuntimeError(
            f"No MLflow model found at {MODEL_PATH}. "
            "The model must be downloaded during the Docker build."
        )
    app.state.model = mlflow.pyfunc.load_model(str(MODEL_PATH))
    app.state.input_columns = _input_columns(app.state.model)
    app.state.model_info = _model_info()
    yield


app = FastAPI(
    title="Digit classification API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so 422 error bodies remain valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Any, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


class PredictRequest(BaseModel):
    """One sample via `features`, or several samples via `instances`."""

    features: list[float] | None = None
    instances: list[list[float]] | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        if (self.features is None) == (self.instances is None):
            raise ValueError("Provide exactly one of 'features' or 'instances'.")

        rows = [self.features] if self.features is not None else self.instances or []
        if not rows:
            raise ValueError("'instances' must contain at least one sample.")

        for index, row in enumerate(rows):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"Sample {index} has {len(row)} features; exactly {N_FEATURES} are required."
                )
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Sample {index} contains NaN or infinite values.")
        return self


class PredictResponse(BaseModel):
    predictions: list[int]
    model_name: str | None = None
    model_version: str | None = None


def _to_digit(value: Any) -> int:
    digit = int(value)
    if float(value) != digit:
        raise ValueError(f"Model returned a non-integer prediction: {value!r}")
    return digit


@app.post("/predict", response_model_exclude_none=True)
def predict(request: PredictRequest) -> PredictResponse:
    rows = [request.features] if request.features is not None else request.instances or []

    columns = app.state.input_columns
    if columns is not None:
        payload: Any = pd.DataFrame(rows, columns=columns, dtype=float)
    else:
        payload = np.asarray(rows, dtype=np.float64)

    try:
        raw_predictions = app.state.model.predict(payload)
        predictions = [_to_digit(value) for value in np.asarray(raw_predictions).reshape(-1)]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc

    info = app.state.model_info
    return PredictResponse(
        predictions=predictions,
        model_name=info.get("model_name"),
        model_version=info.get("model_version"),
    )
