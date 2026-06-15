"""
SprintGuard - Pydantic Schemas for Sprint Endpoints

SprintRiskResponse : GET /api/v1/sprint/{id}/risk response
ReplanRequest      : POST /api/v1/sprint/{id}/replan request body
ReplanResponse     : replan response shape
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Risk endpoint
# ---------------------------------------------------------------------------

class SprintRiskResponse(BaseModel):
    sprint_id: str
    risk_level: str          # "LOW" | "MEDIUM" | "HIGH"
    risk_score: float
    days_remaining: int
    velocity_trend: float
    bug_hours_added_today: float
    factors: list[str]


# ---------------------------------------------------------------------------
# Replan endpoint
# ---------------------------------------------------------------------------

class SprintStoryRequest(BaseModel):
    id: str
    story_points: int = Field(..., ge=0)
    priority: int = Field(..., ge=1, le=5)
    effort_hours: float = Field(..., ge=0.0)
    must_have: bool = False


class ReplanRequest(BaseModel):
    stories: list[SprintStoryRequest] = Field(..., min_length=1)
    available_capacity_hours: float = Field(..., gt=0.0)


class ReplanSuggestion(BaseModel):
    id: str
    action: str
    story_points_removed: int
    projected_risk: str          # "LOW" | "MEDIUM" | "HIGH"
    projected_risk_score: float
    note: str | None = None


class ReplanResponse(BaseModel):
    sprint_id: str
    current_risk: str
    suggestions: list[ReplanSuggestion]
    recommended: str             # id of the best suggestion
