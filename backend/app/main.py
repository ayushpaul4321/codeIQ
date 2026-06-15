"""
SprintGuard - FastAPI Application Entry Point

Lifespan context manager loads all ML model services once at startup and
attaches them to app.state so that route handlers can retrieve them via
FastAPI dependency injection (see dependencies.py).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths to model assets — can be overridden via environment variables
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "storage/models")

_TRIAGE_MODEL_PATH = os.path.join(MODEL_DIR, "bert_triage", "mlp_classifier.pt")
_LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "bert_triage", "dev_label_map.json")
_EFFORT_MODEL_PATH = os.path.join(MODEL_DIR, "effort_estimator", "lstm_estimator.pt")


# ---------------------------------------------------------------------------
# Lifespan: load models once, attach to app.state
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: verify DB connectivity, then load every ML service and attach to app.state.
    Services that are missing their asset files will still be created but will report
    is_loaded=False so the health endpoint can surface them.
    """
    # --- Database connectivity check (Requirement 5.5) ---
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional; env vars may already be set

    try:
        from app.db.session import get_supabase_client
        sb = get_supabase_client()
        # Lightweight ping: fetch a single row from sprints (or empty result — just checks connectivity)
        sb.table("sprints").select("id").limit(1).execute()
        logger.info("Supabase DB connection verified.")
    except Exception as exc:
        logger.critical("DB is unreachable at startup: %s", exc)
        import sys
        sys.exit(1)

    # --- Triage Service ---
    try:
        from app.models.triage_service import TriageService
        triage = TriageService(
            model_path=_TRIAGE_MODEL_PATH,
            label_map_path=_LABEL_MAP_PATH,
        )
        triage.load()
        logger.info("TriageService loaded successfully.")
    except Exception as exc:
        logger.warning("TriageService failed to load: %s", exc)
        triage = _make_unloaded_stub("TriageService")

    # --- Effort Estimator Service ---
    try:
        from app.models.effort_estimator import EffortEstimatorService
        effort = EffortEstimatorService(model_path=_EFFORT_MODEL_PATH)
        effort.load()
        logger.info("EffortEstimatorService loaded successfully.")
    except Exception as exc:
        logger.warning("EffortEstimatorService failed to load: %s", exc)
        effort = _make_unloaded_stub("EffortEstimatorService")

    # --- Sprint Risk Engine ---
    try:
        from app.models.sprint_risk import SprintRiskEngine
        risk_engine = SprintRiskEngine()
        risk_engine.build()
        logger.info("SprintRiskEngine built successfully.")
    except Exception as exc:
        logger.warning("SprintRiskEngine failed to build: %s", exc)
        risk_engine = _make_unloaded_stub("SprintRiskEngine")

    app.state.triage_service = triage
    app.state.effort_service  = effort
    app.state.risk_engine     = risk_engine

    yield  # application runs here

    # Shutdown: nothing to clean up for in-memory models
    logger.info("SprintGuard shutting down.")


def _make_unloaded_stub(name: str) -> Any:
    """
    Return a minimal stub object that responds to is_loaded with False.
    Used when a model file is absent so the app can still start and report
    missing assets via /health without crashing.
    """

    class _Stub:
        is_loaded: bool = False
        _name: str = name

        def __repr__(self) -> str:
            return f"<UnloadedStub:{self._name}>"

    return _Stub()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SprintGuard",
    version="1.0.0",
    description="Agile Sprint Health & Auto Bug Triage — hybrid neural-fuzzy-evolutionary system",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health endpoint (must respond even when models are absent)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health():
    """
    Health check for Docker and load-balancer probes.

    Returns HTTP 200 in all cases.  When model assets are missing the response
    includes models_loaded=False and a list of missing asset filenames so that
    operators can diagnose problems without restarting the container.

    Response time target: < 200 ms (no disk/DB I/O on the hot path).
    """
    missing: list[str] = []

    triage = getattr(app.state, "triage_service", None)
    effort = getattr(app.state, "effort_service", None)
    risk   = getattr(app.state, "risk_engine", None)

    if not getattr(triage, "is_loaded", False):
        missing.append("mlp_classifier.pt")

    if not getattr(effort, "is_loaded", False):
        missing.append("lstm_estimator.pt")

    # SprintRiskEngine has no file to load — it is built in-memory.
    # Report it missing only if the attribute is absent entirely.
    if risk is None:
        missing.append("sprint_risk_engine")

    models_loaded = len(missing) == 0

    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "missing_assets": missing,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.routes import repo, bugs, sprint  # noqa: E402  (import after app creation)

app.include_router(repo.router,   prefix="/api/v1", tags=["Repo"])
app.include_router(bugs.router,   prefix="/api/v1", tags=["Bugs"])
app.include_router(sprint.router, prefix="/api/v1", tags=["Sprint"])
