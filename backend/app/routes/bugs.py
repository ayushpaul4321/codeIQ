"""
SprintGuard - Bug Routes

POST /api/v1/bugs/analyze

Stub implementation — full pipeline (embedding → triage → effort → risk → persist)
will be wired in task 8.  The stub returns HTTP 501 with a clear message so that
the health endpoint and router mounting can be validated end-to-end before the ML
models are integrated.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/bugs/analyze", status_code=501)
def analyze_bug():
    """
    Analyse an incoming bug report and return triage + effort + risk data.

    Full implementation: task 8 (TriageService, EffortEstimatorService, SprintRiskEngine wiring).
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented yet — see task 8"},
    )
