"""
SprintGuard - Sprint Routes

GET  /api/v1/sprint/{sprint_id}/risk   — implemented (task 12)
POST /api/v1/sprint/{sprint_id}/replan — implemented (task 13)

Requirements: 8.8, 8.10, 9.1, 9.9
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.dependencies import get_replanner, get_risk_engine, get_supabase
from app.schemas.sprint import ReplanRequest, ReplanResponse, ReplanSuggestion, SprintRiskResponse
from app.services.sprint_service import SprintService

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# GET /api/v1/sprint/{sprint_id}/risk
# ---------------------------------------------------------------------------

@router.get(
    "/sprint/{sprint_id}/risk",
    response_model=SprintRiskResponse,
    summary="Get current sprint risk",
)
def get_sprint_risk(
    sprint_id: str,
    risk_engine=Depends(get_risk_engine),
    supabase=Depends(get_supabase),
):
    """
    Compute and return the current sprint risk score and contributing factors.

    Steps:
        1. Fetch sprint row from DB via SprintService.
        2. Compute velocity_trend, bug_hours_today, days_remaining.
        3. Run SprintRiskEngine.compute() with those inputs.
        4. Return SprintRiskResponse.

    Returns HTTP 404 if the sprint_id is not found in the database.
    """
    svc = SprintService(supabase)

    # --- 1. Fetch sprint ---
    sprint = svc.get_sprint(sprint_id)
    if sprint is None:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found.")

    # --- 2. Compute inputs ---
    velocity_trend     = svc.get_velocity_trend(sprint)
    bug_hours_today    = svc.get_bug_hours_today(sprint_id)
    days_remaining_raw = svc.get_days_remaining(sprint)

    # Clamp days_remaining to a non-negative value for the FIS
    days_remaining = max(0.0, days_remaining_raw)

    # --- 3. Run fuzzy risk engine ---
    if risk_engine.is_loaded:
        try:
            result = risk_engine.compute(
                bug_hours_added=bug_hours_today,
                velocity_trend=velocity_trend,
                days_remaining=days_remaining,
            )
            risk_score  = result["risk_score"]
            risk_level  = result["risk_level"]
            factors     = result["factors"]
        except Exception as exc:
            logger.error("SprintRiskEngine.compute() failed for sprint %r: %s", sprint_id, exc)
            risk_score = 0.5
            risk_level = "MEDIUM"
            factors    = ["Risk computation unavailable"]
    else:
        logger.warning("SprintRiskEngine not loaded — returning safe defaults for sprint %r.", sprint_id)
        risk_score = 0.5
        risk_level = "MEDIUM"
        factors    = ["Risk engine not available"]

    return SprintRiskResponse(
        sprint_id=sprint_id,
        risk_level=risk_level,
        risk_score=risk_score,
        days_remaining=int(days_remaining_raw),
        velocity_trend=velocity_trend,
        bug_hours_added_today=bug_hours_today,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/sprint/{sprint_id}/replan
# ---------------------------------------------------------------------------

@router.post(
    "/sprint/{sprint_id}/replan",
    response_model=ReplanResponse,
    summary="Re-plan sprint using GA Replanner",
)
def replan_sprint(
    sprint_id: str,
    body: ReplanRequest,
    replanner=Depends(get_replanner),
    risk_engine=Depends(get_risk_engine),
    supabase=Depends(get_supabase),
):
    """
    Run the GA Replanner and return at least two re-planning suggestions.

    Steps:
        1. Fetch sprint row from DB via SprintService — HTTP 404 if absent.
        2. Gather sprint_state (velocity_trend, days_remaining, bug_hours_today).
        3. Compute current risk via SprintRiskEngine.
        4. Convert ReplanRequest stories to SprintStory dataclasses.
        5. Run GAReplanner.replan() to obtain candidate suggestions.
        6. Build and return ReplanResponse with recommended field set to
           the suggestion id that has the lowest projected_risk_score.

    Returns HTTP 404 if the sprint_id is not found in the database.
    """
    from app.models.replanner import SprintStory  # local import avoids circular dep at module load

    svc = SprintService(supabase)

    # --- 1. Validate sprint exists ---
    sprint = svc.get_sprint(sprint_id)
    if sprint is None:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found.")

    # --- 2. Gather sprint state ---
    velocity_trend  = svc.get_velocity_trend(sprint)
    bug_hours_today = svc.get_bug_hours_today(sprint_id)
    days_remaining  = max(0.0, svc.get_days_remaining(sprint))

    sprint_state = {
        "velocity_trend": velocity_trend,
        "days_remaining": days_remaining,
        "bug_hours_today": bug_hours_today,
    }

    # --- 3. Compute current sprint risk ---
    if risk_engine.is_loaded:
        try:
            risk_result  = risk_engine.compute(
                bug_hours_added=bug_hours_today,
                velocity_trend=velocity_trend,
                days_remaining=days_remaining,
            )
            current_risk = risk_result["risk_level"]
        except Exception as exc:
            logger.error("SprintRiskEngine.compute() failed in replan for sprint %r: %s", sprint_id, exc)
            current_risk = "UNKNOWN"
    else:
        logger.warning("SprintRiskEngine not loaded — defaulting current_risk to UNKNOWN for sprint %r.", sprint_id)
        current_risk = "UNKNOWN"

    # --- 4. Convert request stories to SprintStory dataclasses ---
    stories = [
        SprintStory(
            id=s.id,
            story_points=s.story_points,
            priority=s.priority,
            effort_hours=s.effort_hours,
            must_have=s.must_have,
        )
        for s in body.stories
    ]

    # --- 5. Run GA Replanner ---
    try:
        raw_suggestions = replanner.replan(
            stories=stories,
            available_capacity_hours=body.available_capacity_hours,
            sprint_state=sprint_state,
        )
    except Exception as exc:
        logger.error("GAReplanner.replan() failed for sprint %r: %s", sprint_id, exc)
        raise HTTPException(
            status_code=500,
            detail="GA Replanner encountered an internal error.",
        )

    # --- 6. Build response ---
    if not raw_suggestions:
        raise HTTPException(
            status_code=500,
            detail="GA Replanner returned no suggestions.",
        )

    suggestion_objects = [
        ReplanSuggestion(
            id=s["id"],
            action=s["action"],
            story_points_removed=s["story_points_removed"],
            projected_risk=s["projected_risk"],
            projected_risk_score=s["projected_risk_score"],
            note=s.get("note"),
        )
        for s in raw_suggestions
    ]

    # recommended = suggestion with lowest projected_risk_score (first after sort)
    recommended_id = suggestion_objects[0].id

    return ReplanResponse(
        sprint_id=sprint_id,
        current_risk=current_risk,
        suggestions=suggestion_objects,
        recommended=recommended_id,
    )
