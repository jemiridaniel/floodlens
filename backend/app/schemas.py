"""Data contracts shared by the API, the scorer, the agent and the verifier."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Tier(StrEnum):
    low = "Low"
    moderate = "Moderate"
    high = "High"
    severe = "Severe"


class Action(StrEnum):
    approve = "approve"
    approve_note = "approve_note_risk"
    approve_with_insurance = "approve_with_insurance"
    adjust_terms = "adjust_terms"
    decline = "decline_or_change_collateral"


class Metric(BaseModel):
    id: str
    label: str
    value: float
    unit: str
    source: str
    decimals: int = 1

    def display(self) -> str:
        return f"{self.value:.{self.decimals}f}"


class Score(BaseModel):
    score: float = Field(ge=0, le=100)
    tier: Tier
    confidence: str  # low | medium | high
    contributions: dict[str, float]  # metric id -> points added
    missing: list[str]


class AssessmentRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    label: str | None = None
    note: str | None = None


class Assessment(BaseModel):
    id: str
    lat: float
    lon: float
    label: str | None = None
    metrics: list[Metric]
    score: Score
    allowed_actions: list[Action]
    evidence: dict[str, str] = {}
    notes: list[str] = []
