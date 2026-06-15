"""
SprintGuard - Pydantic Schemas for Bug Triage Endpoint

BugAnalyzeRequest  : POST /api/v1/bugs/analyze request body
BugAnalyzeResponse : successful response shape
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class BugAnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Short bug summary")
    description: str = Field(..., min_length=1, description="Full bug description")
    file_paths: list[str] = Field(default_factory=list)
    reporter: str = Field(default="unknown")
    sprint_id: str = Field(default="")

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_and_require(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("field cannot be blank or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------

class DevPrediction(BaseModel):
    dev: str
    probability: float


class EffortEstimate(BaseModel):
    hours: float | None
    confidence_interval: list[float] | None  # [lower, upper]


class SprintImpact(BaseModel):
    risk_before: str | None
    risk_after: str | None
    risk_score: float | None
    factors: list[str]
    replan_suggested: bool


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class BugAnalyzeResponse(BaseModel):
    bug_id: UUID
    assigned_dev: str
    assignment_confidence: float
    top3_devs: list[DevPrediction]
    effort_estimate: EffortEstimate
    sprint_impact: SprintImpact
