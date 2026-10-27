"""FloodLens API."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .geo.sources import LayerSource, PlanetaryComputerSource
from .pipeline import assess
from .schemas import Assessment, AssessmentRequest

app = FastAPI(title="FloodLens", version="0.1.0")
_store: dict[str, Assessment] = {}  # swapped for Postgres later


def get_source() -> LayerSource:
    return PlanetaryComputerSource()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/assessments", response_model=Assessment)
async def create_assessment(req: AssessmentRequest, source: Annotated[LayerSource, Depends(get_source)]) -> Assessment:
    result = await run_in_threadpool(assess, req, source)
    _store[result.id] = result
    return result


@app.get("/assessments/{assessment_id}", response_model=Assessment)
def get_assessment(assessment_id: str) -> Assessment:
    if assessment_id not in _store:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _store[assessment_id]
