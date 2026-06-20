# ── SprintGuard API Image ──────────────────────────────────────────────────
# Base: slim Python 3.11 — keeps the image small while matching the dev env.
FROM python:3.11-slim

# Prevents .pyc files and buffers stdout/stderr immediately (better logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────
# libgomp1  : required by scikit-fuzzy / numpy for OpenMP threading
# curl      : used by the Docker healthcheck probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY backend/ /app/backend/
COPY storage/models/ /app/storage/models/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini

EXPOSE 8000

# ── Runtime command ───────────────────────────────────────────────────────
# Alembic migration runs first (idempotent), then uvicorn starts.
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
