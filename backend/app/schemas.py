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


class Driver(BaseModel):
    """One metric the agent says is driving the score."""

    metric_id: str
    why: str


class Report(BaseModel):
    """The written assessment. Produced by the agent, or by the code-only fallback."""

    summary: str
    drivers: list[Driver] = []
    action: Action
    rationale: str
    caveats: list[str] = []
    explained: bool = True  # False = the model never produced a report the verifier accepted

    def text(self) -> str:
        """Everything a reader will see, joined for the verifier to check."""
        parts = [self.summary, self.rationale, *(d.why for d in self.drivers), *self.caveats]
        return "\n".join(p for p in parts if p)


class AgentStep(BaseModel):
    """One step of the agent run, for live progress in the UI and for debugging."""

    n: int
    kind: str  # model | tool | verify
    name: str
    detail: str = ""  # short, already truncated: safe to show
    ms: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0  # part of input_tokens, billed at the cached rate
    output_tokens: int = 0
    reasoning_tokens: int = 0  # part of output_tokens


class AgentRun(BaseModel):
    report: Report
    steps: list[AgentStep] = []
    model: str = ""  # the model actually called, which is what the cost is priced on
    reasoning_effort: str = ""
    calls: int = 0  # model calls used, capped by MAX_MODEL_CALLS
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = []  # verifier errors that forced a retry or the fallback
