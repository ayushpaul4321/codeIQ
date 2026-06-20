"""
SprintGuard - Bug Routes

POST /api/v1/bugs/analyze

Accepts a BugAnalyzeRequest, generates an embedding, runs the triage model,
computes effort estimate, computes sprint risk impact, persists Bug + Assignment
records via supabase-py, and returns BugAnalyzeResponse.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_triage_service, get_supabase, get_effort_service, get_risk_engine
from app.models.bug_embedding import BugEmbeddingService, embed_single
from app.schemas.bugs import (
    BugAnalyzeRequest,
    BugAnalyzeResponse,
    DevPrediction,
    EffortEstimate,
    SprintImpact,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_embedding_service = BugEmbeddingService()


@router.post("/bugs/analyze", response_model=BugAnalyzeResponse)
def analyze_bug(
    body: BugAnalyzeRequest,
    triage_service=Depends(get_triage_service),
    supabase=Depends(get_supabase),
    effort_service=Depends(get_effort_service),
    risk_engine=Depends(get_risk_engine),
):
    """
    Analyse an incoming bug report and return triage + effort + risk data.

    Pipeline:
        1. Build enriched text from title + description
        2. Embed with all-MiniLM-L6-v2 (384-dim)
        3. Predict developer assignment via TriageService (MLP)
        4. Persist Bug and Assignment records via supabase-py
        5. Return BugAnalyzeResponse

    Returns HTTP 503 when the triage model is not loaded.
    Returns HTTP 422 when title or description are blank (handled by Pydantic).
    """
    if not triage_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"detail": "Triage model is not loaded. Check server logs."},
        )

    start_ms = time.monotonic() * 1000

    # --- 1. Build enriched text and embed ---
    enriched = _embedding_service.build_enriched_text(
        title=body.title,
        description=body.description,
    )
    embedding = embed_single(enriched)

    # --- 2. Run triage model ---
    raw_preds = triage_service.predict(embedding, top_k=3)

    top = raw_preds[0]
    assigned_dev: str = top["dev"]
    assignment_confidence: float = top["probability"]
    top3_devs = [DevPrediction(dev=p["dev"], probability=p["probability"]) for p in raw_preds]

    # --- 3. Generate bug ID ---
    bug_id = uuid.uuid4()

    latency_ms = time.monotonic() * 1000 - start_ms
    logger.info(
        "BugAnalyze bug_id=%s assigned_dev=%s confidence=%.4f latency_ms=%.1f",
        bug_id,
        assigned_dev,
        assignment_confidence,
        latency_ms,
    )

    # --- 4. Persist via supabase-py (best-effort — errors logged, not raised) ---
    try:
        bug_record = {
            "id": str(bug_id),
            "title": body.title,
            "description": body.description,
            "reporter": body.reporter,
            "sprint_id": body.sprint_id or None,
            "file_paths": body.file_paths,
        }
        supabase.table("bugs").insert(bug_record).execute()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist Bug record for bug_id=%s: %s", bug_id, exc)

    try:
        assignment_record = {
            "bug_id": str(bug_id),
            "assigned_dev": assigned_dev,
            "confidence": assignment_confidence,
            "top3_devs": [p.model_dump() for p in top3_devs],
        }
        supabase.table("assignments").insert(assignment_record).execute()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist Assignment record for bug_id=%s: %s", bug_id, exc)

    # --- 5. Build response ---
    # Effort estimation via EffortEstimatorService (task 9)
    if effort_service.is_loaded:
        try:
            effort_result = effort_service.predict(embedding)
            effort = EffortEstimate(
                hours=effort_result["hours"],
                confidence_interval=effort_result["confidence_interval"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("EffortEstimatorService.predict() failed for bug_id=%s: %s", bug_id, exc)
            effort = EffortEstimate(hours=None, confidence_interval=None)
    else:
        effort = EffortEstimate(hours=None, confidence_interval=None)

    # --- Sprint risk impact via SprintRiskEngine (task 10) ---
    # Use default sprint-context values; full wiring with live sprint state happens in task 12.
    _DEFAULT_VELOCITY_TREND = 0.0
    _DEFAULT_DAYS_REMAINING = 7.0

    if risk_engine.is_loaded and effort_service.is_loaded:
        try:
            effort_hours = effort.hours if effort.hours is not None else 0.0

            # Risk BEFORE adding this bug (bug_hours_added = 0)
            result_before = risk_engine.compute(
                bug_hours_added=0.0,
                velocity_trend=_DEFAULT_VELOCITY_TREND,
                days_remaining=_DEFAULT_DAYS_REMAINING,
            )
            risk_score_before: float = result_before["risk_score"]
            risk_level_before: str   = result_before["risk_level"]

            # Risk AFTER adding this bug
            result_after = risk_engine.compute(
                bug_hours_added=effort_hours,
                velocity_trend=_DEFAULT_VELOCITY_TREND,
                days_remaining=_DEFAULT_DAYS_REMAINING,
            )
            risk_score_after: float = result_after["risk_score"]
            risk_level_after: str   = result_after["risk_level"]

            # Suggest re-planning if this bug pushed sprint over the HIGH threshold (≥ 0.70)
            replan_suggested = risk_score_after >= 0.70 and risk_score_before < 0.70

            sprint_impact = SprintImpact(
                risk_before=risk_level_before,
                risk_after=risk_level_after,
                risk_score=risk_score_after,
                factors=result_after["factors"],
                replan_suggested=replan_suggested,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("SprintRiskEngine.compute() failed for bug_id=%s: %s", bug_id, exc)
            sprint_impact = SprintImpact(
                risk_before=None,
                risk_after=None,
                risk_score=None,
                factors=["Risk computation unavailable"],
                replan_suggested=False,
            )
    else:
        # Risk engine or effort service not loaded — return safe defaults
        sprint_impact = SprintImpact(
            risk_before=None,
            risk_after=None,
            risk_score=None,
            factors=[],
            replan_suggested=False,
        )

    return BugAnalyzeResponse(
        bug_id=bug_id,
        assigned_dev=assigned_dev,
        assignment_confidence=assignment_confidence,
        top3_devs=top3_devs,
        effort_estimate=effort,
        sprint_impact=sprint_impact,
    )
