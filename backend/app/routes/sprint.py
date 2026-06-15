"""
SprintGuard - Sprint Routes

GET  /api/v1/sprint/{sprint_id}/risk
POST /api/v1/sprint/{sprint_id}/replan

Stub implementation — full pipeline will be wired in tasks 12 and 13 respectively.
The stubs return HTTP 501 with a clear message so that the health endpoint and router
mounting can be validated end-to-end before the sprint service is implemented.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/sprint/{sprint_id}/risk", status_code=501)
def get_sprint_risk(sprint_id: str):
    """
    Return current sprint risk score and contributing factors.

    Full implementation: task 12 (SprintService + SprintRiskEngine wiring).
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented yet — see task 12"},
    )


@router.post("/sprint/{sprint_id}/replan", status_code=501)
def replan_sprint(sprint_id: str):
    """
    Run the GA Replanner and return re-planning suggestions.

    Full implementation: task 13 (GAReplanner wiring).
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented yet — see task 13"},
    )
