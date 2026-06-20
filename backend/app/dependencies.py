"""
SprintGuard - FastAPI Dependency Injection Helpers

All heavy objects (ML models, DB sessions) are retrieved here so that
route handlers stay thin and testable.  Each function is a FastAPI
"dependency" — pass it to Depends() in route signatures.

Usage example:
    @router.post("/analyze")
    def analyze(
        triage: TriageService = Depends(get_triage_service),
        db:     Session       = Depends(get_db),
    ): ...
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    # Imported only for type-checkers — actual imports happen at runtime
    # inside each function to avoid circular imports and optional dependencies.
    from sqlalchemy.orm import Session
    from app.models.triage_service import TriageService
    from app.models.effort_estimator import EffortEstimatorService
    from app.models.sprint_risk import SprintRiskEngine
    from app.models.replanner import GAReplanner


# ---------------------------------------------------------------------------
# ML service dependencies
# ---------------------------------------------------------------------------

def get_triage_service(request: Request) -> "TriageService":
    """
    Return the TriageService attached to app.state at startup.

    Raises AttributeError if the service was never attached (should not
    happen after a complete startup, but guards against test misconfiguration).
    """
    return request.app.state.triage_service


def get_effort_service(request: Request) -> "EffortEstimatorService":
    """Return the EffortEstimatorService attached to app.state at startup."""
    return request.app.state.effort_service


def get_risk_engine(request: Request) -> "SprintRiskEngine":
    """Return the SprintRiskEngine attached to app.state at startup."""
    return request.app.state.risk_engine


def get_replanner(request: Request) -> "GAReplanner":
    """Return the GAReplanner attached to app.state at startup."""
    return request.app.state.replanner


# ---------------------------------------------------------------------------
# Database session dependency
# ---------------------------------------------------------------------------

def get_supabase():
    """
    Return the singleton supabase-py Client for HTTPS-based DB operations.

    This is the preferred DB accessor at runtime — it works through firewalls
    that block raw PostgreSQL ports.  Use this dependency in route handlers.
    """
    from app.db.session import get_supabase_client
    return get_supabase_client()


def get_db() -> "Generator[Session, None, None]":
    """
    Yield a SQLAlchemy Session (direct TCP connection).

    Kept for compatibility with code that uses SQLAlchemy ORM sessions.
    On networks that block port 5432, prefer get_supabase() instead.
    """
    try:
        from app.db.session import SessionLocal
        if SessionLocal is None:
            raise ImportError("SessionLocal not available")
    except ImportError as exc:
        raise RuntimeError(
            "SQLAlchemy SessionLocal is not available. "
            "Use get_supabase() for HTTPS-based DB access instead."
        ) from exc

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
