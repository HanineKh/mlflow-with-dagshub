# syntax=docker/dockerfile:1.10
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model

WORKDIR /app

COPY requirements.txt .
# Both caches survive failed attempts: the HTTP cache and a wheelhouse.
# A failed download of one large wheel no longer forces re-downloading the rest.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/wheelhouse \
    python -m pip download --retries 20 --timeout 120 -r requirements.txt -d /opt/wheelhouse \
 && python -m pip install --no-index --find-links=/opt/wheelhouse -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Repo names are configuration, not secrets. The token never becomes an
# environment variable of the final image; it exists only inside this RUN.
ARG DAGSHUB_REPO_OWNER=""
ARG DAGSHUB_REPO_NAME=""
ARG MLFLOW_MODEL_NAME=""
ARG DAGSHUB_USERNAME=""

RUN --mount=type=secret,id=dagshub_token,env=DAGSHUB_TOKEN \
    DAGSHUB_USER_TOKEN="$DAGSHUB_TOKEN" \
    python scripts/download_model.py \
      --repo-owner "$DAGSHUB_REPO_OWNER" \
      --repo-name "$DAGSHUB_REPO_NAME" \
      --model-name "$MLFLOW_MODEL_NAME" \
      --username "$DAGSHUB_USERNAME" \
      --dst "$MODEL_PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
