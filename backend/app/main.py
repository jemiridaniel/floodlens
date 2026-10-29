"""FloodLens API: start an assessment, watch it run, read the result."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from . import jobs as J
from .photos import MAX_PHOTOS
from .schemas import AssessmentRequest

app = FastAPI(title="FloodLens", version="0.2.0")

# The Next.js app in web/ runs here in development. Nothing else is allowed, and the API
# holds no credentials the browser could ever see.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_jobs: dict[str, J.Job] = {}  # swapped for Postgres later

# Test seams. Empty in production, where run_job reaches for the real source, the real
# search and the real agent. Tests fill these in so nothing touches the network.
HOOKS: dict[str, object] = {}
HEARTBEAT_S = 15  # keeps an idle SSE connection alive through the long steps


def _get(assessment_id: str) -> J.Job:
    job = _jobs.get(assessment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return job


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "demo_cache": "on" if J.demo_cache_on() else "off"}


@app.post("/assessments", status_code=202)
async def create_assessment(
    lat: Annotated[float, Form()],
    lon: Annotated[float, Form()],
    address: Annotated[str | None, Form()] = None,
    label: Annotated[str | None, Form()] = None,
    loan_amount: Annotated[float | None, Form()] = None,
    tenor_months: Annotated[int | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
    photos: Annotated[list[UploadFile], File()] = [],  # noqa: B006 (FastAPI reads the default)
) -> dict[str, str]:
    """Start an assessment and return its id. Progress is on /events."""
    try:
        AssessmentRequest(lat=lat, lon=lon, label=label, note=note)  # reuse the bounds check
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
    real = [p for p in photos if p.filename]
    if len(real) > MAX_PHOTOS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_PHOTOS} photos")

    job = J.Job(
        id=uuid.uuid4().hex[:12], lat=lat, lon=lon,
        address=(address or None), label=(label or None), note=(note or None),
        loan_amount=loan_amount, tenor_months=tenor_months,
        photo_bytes=[await p.read() for p in real],
    )
    _jobs[job.id] = job

    cached = J.load_demo(lat, lon) if J.demo_cache_on() and not real else None
    if cached is not None:
        J.replay(job, *cached)
    else:
        await J.start(job, on_done=J.save_demo if J.demo_cache_on() else None, **HOOKS)
    return {"id": job.id}


@app.get("/assessments/{assessment_id}/events")
async def assessment_events(assessment_id: str, request: Request) -> StreamingResponse:
    """Server-sent events: one per step, then a final `done`."""
    job = _get(assessment_id)

    async def stream():
        # A heartbeat during the quiet stretches. The explain step can run 45 s without
        # producing an event, which is long enough for an idle connection to be dropped
        # by a phone network or any proxy in between.
        beat = asyncio.create_task(asyncio.sleep(0))
        events = job.stream().__aiter__()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(events.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_S)
                if await request.is_disconnected():
                    return
                if not done:
                    yield ": keep-alive\n\n"
                    continue
                try:
                    event = pending.result()
                finally:
                    pending = None
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                if event["type"] == "done":
                    return
        except StopAsyncIteration:
            return
        finally:
            beat.cancel()
            if pending is not None:
                pending.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # tell any proxy not to sit on the stream
    })


@app.get("/assessments/{assessment_id}", response_model=J.AssessmentView)
def get_assessment(assessment_id: str, request: Request) -> J.AssessmentView:
    return J.view(_get(assessment_id), _base(request))


@app.get("/assessments/{assessment_id}/evidence/{name}.png")
def get_evidence(assessment_id: str, name: str) -> Response:
    png = _get(assessment_id).pngs.get(name)
    if png is None:
        raise HTTPException(status_code=404, detail=f"No evidence image named {name!r}")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})
