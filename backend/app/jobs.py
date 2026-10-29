"""Background assessment jobs, and the live progress the UI watches.

One job per assessment. The stages are the real pipeline, in the order the officer sees
them, and the first two run at the same time because neither needs the other:

    Area screen  ┐
                 ├─> Site photos -> Score -> Explain -> Verify
    Flood history┘

Everything here is about honesty under failure. A stage that fails is recorded as failed
and the job carries on: a dead land-cover tile must not cost the officer the flood
reports. The score is still computed by code, and a whole missing evidence group still
caps confidence at low, exactly as in the batch path.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .evidence import evidence_pngs
from .geo.aoi import Site
from .geo.sources import PlanetaryComputerSource
from .photos import MAX_PHOTOS, Photo, save_photo, segment_all, to_png
from .pipeline import assemble
from .schemas import AgentRun, Assessment, FloodReport, Metric, Report
from .scoring import GROUPS

log = logging.getLogger(__name__)

AREA, HISTORY, PHOTOS, SCORE, EXPLAIN, VERIFY = (
    "Area screen", "Flood history", "Site photos", "Score", "Explain", "Verify")
STEP_ORDER = (AREA, HISTORY, PHOTOS, SCORE, EXPLAIN, VERIFY)
DEMO_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "demo"


def demo_cache_on() -> bool:
    return os.environ.get("FLOODLENS_DEMO_CACHE", "") == "1"


# ----------------------------------------------------------------- view models
class StepView(BaseModel):
    seq: int
    step: str
    status: str  # pending | running | done | failed | skipped
    ms: int = 0
    detail: str = ""
    cost_usd: float = 0.0  # cumulative for the job so far


class FindingView(BaseModel):
    prompt: str
    coverage: float
    instances: int
    status: str
    overlay_url: str = ""


class PhotoView(BaseModel):
    name: str
    url: str
    findings: list[FindingView] = []
    error: str = ""


class GroupView(BaseModel):
    """Metrics grouped the way the report reads them, so the UI never regroups by id."""

    name: str
    present: bool  # False = this whole kind of evidence is missing, confidence is capped
    metrics: list[Metric] = []
    contributions: dict[str, float] = {}


class AssessmentView(BaseModel):
    id: str
    status: str  # running | done | failed
    lat: float
    lon: float
    address: str | None = None
    label: str | None = None
    loan_amount: float | None = None
    tenor_months: int | None = None
    note: str | None = None
    from_cache: bool = False

    assessment: Assessment | None = None
    groups: list[GroupView] = []
    reports: list[FloodReport] = []
    photos: list[PhotoView] = []
    report: Report | None = None
    run: AgentRun | None = None
    evidence_urls: dict[str, str] = {}
    steps: list[StepView] = []
    cost_usd: float = 0.0
    error: str | None = None


# ----------------------------------------------------------------------- job
@dataclass
class Job:
    id: str
    lat: float
    lon: float
    address: str | None = None
    label: str | None = None
    loan_amount: float | None = None
    tenor_months: int | None = None
    note: str | None = None
    photo_bytes: list[bytes] = field(default_factory=list)

    status: str = "running"
    error: str | None = None
    from_cache: bool = False
    steps: dict[str, StepView] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0

    assessment: Assessment | None = None
    cached: Any = None  # a saved AssessmentView, when this job is a demo replay
    history: Any = None
    photos: list[Photo] = field(default_factory=list)
    run: AgentRun | None = None
    pngs: dict[str, bytes] = field(default_factory=dict)

    _seq: int = 0
    _waiters: list[asyncio.Queue] = field(default_factory=list)
    _loop: Any = None
    done: Any = None

    def __post_init__(self) -> None:
        self.done = asyncio.Event()
        for name in STEP_ORDER:
            self.steps[name] = StepView(seq=0, step=name, status="pending")

    # --- events -------------------------------------------------------------
    def _publish(self, event: dict) -> None:
        self.events.append(event)
        for q in list(self._waiters):
            q.put_nowait(event)

    def emit(self, event: dict) -> None:
        """Safe from the worker threads the blocking stages run in."""
        self._seq += 1
        event["seq"] = self._seq
        if self._loop is not None and asyncio.get_event_loop_policy() and not _in_loop(self._loop):
            self._loop.call_soon_threadsafe(self._publish, event)
        else:
            self._publish(event)

    def step(self, name: str, status: str, ms: int = 0, detail: str = "") -> None:
        view = StepView(seq=self._seq + 1, step=name, status=status, ms=ms, detail=detail,
                        cost_usd=round(self.cost_usd, 6))
        self.steps[name] = view
        self.emit({"type": "step", **view.model_dump()})

    def finish(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.emit({"type": "done", "status": status, "error": error,
                   "cost_usd": round(self.cost_usd, 6)})
        if self._loop is not None and not _in_loop(self._loop):
            self._loop.call_soon_threadsafe(self.done.set)
        else:
            self.done.set()

    async def stream(self):
        """Replay what has happened, then follow. Late subscribers miss nothing."""
        q: asyncio.Queue = asyncio.Queue()
        self._waiters.append(q)
        try:
            seen = 0
            for event in list(self.events):
                seen = event["seq"]
                yield event
                if event["type"] == "done":
                    return
            while True:
                event = await q.get()
                if event["seq"] <= seen:
                    continue
                seen = event["seq"]
                yield event
                if event["type"] == "done":
                    return
        finally:
            if q in self._waiters:
                self._waiters.remove(q)


def _in_loop(loop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


async def start(job: Job, on_done: Any = None, **kw) -> Any:
    """Launch the job in the background and hand back its task."""
    async def runner():
        await run_job(job, **kw)
        if on_done is not None and job.status == "done":
            try:
                on_done(job, view(job))
            except Exception:  # saving a demo copy must never fail the assessment
                log.exception("could not save the demo copy of %s", job.id)

    return asyncio.create_task(runner())


# ------------------------------------------------------------------ the stages
async def run_job(job: Job, source: Any = None, history_lookup: Any = None,
                  explain: Any = None, segment: Any = None,
                  uploads: Path | None = None) -> None:
    """Drive one assessment. Never raises: failure is a status, not an exception."""
    job._loop = asyncio.get_running_loop()
    site = Site(job.lat, job.lon)
    try:
        layers, history = await asyncio.gather(
            _area_screen(job, site, source),
            _flood_history(job, site, history_lookup),
        )
        if layers is None:
            job.finish("failed", "the area screen could not be measured")
            return
        job.history = history

        await _site_photos(job, segment, uploads)

        t0 = time.monotonic()
        job.step(SCORE, "running")
        from .schemas import AssessmentRequest
        req = AssessmentRequest(lat=job.lat, lon=job.lon, label=job.label, note=job.note)
        job.assessment = await run_in_threadpool(assemble, req, layers, history)
        job.assessment.id = job.id
        s = job.assessment.score
        job.step(SCORE, "done", _ms(t0),
                 f"{s.score}/100, {s.tier.value}, confidence {s.confidence}"
                 + (f"; missing: {', '.join(s.missing_groups)}" if s.missing_groups else ""))

        await _explain(job, explain)
        job.finish("done")
    except Exception as exc:  # the job must always end in a state the UI can render
        log.exception("assessment %s failed", job.id)
        job.finish("failed", f"{exc.__class__.__name__}: {exc}")


async def _area_screen(job: Job, site: Site, source: Any):
    t0 = time.monotonic()
    job.step(AREA, "running")
    try:
        src = source or PlanetaryComputerSource()
        layers = await run_in_threadpool(src.fetch, site, None)
        job.pngs.update(await run_in_threadpool(evidence_pngs, layers))
        have = [n for n in ("optical", "radar", "jrc_occurrence", "dem", "landcover")
                if getattr(layers, n, None) is not None or n in layers.item_ids]
        job.step(AREA, "done", _ms(t0), f"{len(have)} layers: {', '.join(have)}")
        return layers
    except Exception as exc:
        job.step(AREA, "failed", _ms(t0), f"{exc.__class__.__name__}: {exc}")
        return None


async def _flood_history(job: Job, site: Site, history_lookup: Any):
    t0 = time.monotonic()
    job.step(HISTORY, "running")
    try:
        if history_lookup is None:
            from .agent.history import lookup as history_lookup
        history = await run_in_threadpool(history_lookup, site, address=job.address)
        job.cost_usd += getattr(history, "cost_usd", 0.0)
        kept = len(history.reports)
        street = sum(1 for r in history.reports if r.match_level == "street")
        detail = f"{kept} report(s) kept, {street} street-level, {len(history.dropped)} rejected"
        if getattr(history, "from_cache", False):
            detail += " (already searched for this street)"
        if getattr(history, "stopped_early", False):
            detail += " - search stopped at the time limit, so this is partial"
        job.step(HISTORY, "done", _ms(t0), detail)
        return history
    except Exception as exc:
        # No history is not "no floods": the metric is dropped and confidence is capped.
        job.step(HISTORY, "failed", _ms(t0), f"{exc.__class__.__name__}: {exc}")
        return None


async def _site_photos(job: Job, segment: Any, uploads: Path | None) -> None:
    t0 = time.monotonic()
    if not job.photo_bytes:
        job.step(PHOTOS, "skipped", 0, "no photos uploaded")
        return
    job.step(PHOTOS, "running")
    try:
        stored = [await run_in_threadpool(save_photo, b, uploads)
                  for b in job.photo_bytes[:MAX_PHOTOS]]
        for name, image in stored:
            job.pngs[Path(name).stem] = to_png(image)
        job.photos, made = await segment_all(stored, segment=segment)
        job.pngs.update(made)
        hits = sum(1 for p in job.photos for f in p.findings if f.instances)
        job.step(PHOTOS, "done", _ms(t0),
                 f"{len(job.photos)} photo(s), {hits} finding(s). Evidence only: no effect "
                 "on the score.")
    except Exception as exc:
        job.step(PHOTOS, "failed", _ms(t0), f"{exc.__class__.__name__}: {exc}")


async def _explain(job: Job, explain: Any) -> None:
    t0 = time.monotonic()
    job.step(EXPLAIN, "running")
    try:
        if explain is None:
            from .agent.loop import explain
        run = await run_in_threadpool(explain, job.assessment, None)
        job.run = run
        job.cost_usd += run.cost_usd
        model_ms = sum(s.ms for s in run.steps if s.kind == "model")
        job.step(EXPLAIN, "done", model_ms or _ms(t0),
                 f"{run.calls} model call(s), {run.input_tokens + run.output_tokens} tokens")
    except Exception as exc:
        job.step(EXPLAIN, "failed", _ms(t0), f"{exc.__class__.__name__}: {exc}")
        job.step(VERIFY, "skipped", 0, "nothing to verify")
        return

    checks = [s for s in job.run.steps if s.kind in ("verify", "schema")]
    ok = job.run.report.explained
    job.step(VERIFY, "done" if ok else "failed", sum(s.ms for s in checks),
             "accepted" if ok else "; ".join(job.run.errors)[:200]
             or "the model's report was rejected; showing the metrics-only report")


def _ms(t0: float) -> int:
    return int(1000 * (time.monotonic() - t0))


# ------------------------------------------------------------------ the view
def view(job: Job, base: str = "") -> AssessmentView:
    if job.cached is not None:  # a replay is served exactly as it was recorded
        return job.cached.model_copy(update={
            "id": job.id, "from_cache": True,
            "evidence_urls": {n: f"{base}/assessments/{job.id}/evidence/{n}.png"
                              for n in job.pngs},
            "photos": [p.model_copy(update={
                "url": f"{base}/assessments/{job.id}/evidence/{Path(p.name).stem}.png",
                "findings": [f.model_copy(update={
                    "overlay_url": (f"{base}/assessments/{job.id}/evidence/"
                                    f"{f.overlay_url.rsplit('/', 1)[-1]}" if f.overlay_url else "")})
                    for f in p.findings]}) for p in job.cached.photos],
        })
    a = job.assessment
    groups: list[GroupView] = []
    if a is not None:
        by_id = {m.id: m for m in a.metrics}
        for name, ids in GROUPS.items():
            present = [by_id[i] for i in ids if i in by_id]
            groups.append(GroupView(
                name=name, present=bool(present), metrics=present,
                contributions={i: a.score.contributions[i] for i in ids
                               if i in a.score.contributions}))
    photos = [
        PhotoView(name=p.name, url=f"{base}/assessments/{job.id}/evidence/{Path(p.name).stem}.png",
                  error=p.error,
                  findings=[FindingView(
                      prompt=f.prompt, coverage=f.coverage, instances=f.instances, status=f.status,
                      overlay_url=(f"{base}/assessments/{job.id}/evidence/{f.overlay}.png"
                                   if f.overlay else ""))
                      for f in p.findings])
        for p in job.photos
    ]
    return AssessmentView(
        id=job.id, status=job.status, lat=job.lat, lon=job.lon, address=job.address,
        label=job.label, loan_amount=job.loan_amount, tenor_months=job.tenor_months,
        note=job.note, from_cache=job.from_cache,
        assessment=a, groups=groups,
        reports=list(getattr(job.history, "reports", []) or []),
        photos=photos,
        report=job.run.report if job.run else None,
        run=job.run,
        evidence_urls={n: f"{base}/assessments/{job.id}/evidence/{n}.png" for n in job.pngs},
        steps=[job.steps[n] for n in STEP_ORDER],
        cost_usd=round(job.cost_usd, 6),
        error=job.error,
    )


# --------------------------------------------------------------- demo cache
# FLOODLENS_DEMO_CACHE=1 makes a finished assessment replay instantly. It exists so a
# three-minute demo is not at the mercy of a cold Planetary Computer read, and it only
# ever replays a run that really happened: results are written by real runs, never faked.
def demo_key(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def scoring_fingerprint() -> str:
    """Identifies the scoring rules a cached result was produced under.

    A demo must never replay a verdict the current code would not produce. The Okota entry
    was recorded before the street-level tier floor existed and kept replaying Moderate
    after the rule would have made it High, which is exactly the kind of quiet staleness
    that makes a cached demo a lie. Change a weight, a ramp, a tier cut or the floor and
    this changes, so the entry is ignored and re-run.
    """
    from .scoring import FEATURES, GROUPS, TIER_CUTS, TIER_FLOOR, TIER_FLOOR_STREET_EVENTS
    raw = json.dumps({
        "features": [[f.metric_id, f.weight, f.zero_at, f.full_at] for f in FEATURES],
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "cuts": [[c, t.value] for c, t in TIER_CUTS],
        "floor": [TIER_FLOOR_STREET_EVENTS, TIER_FLOOR.value],
    }, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def save_demo(job: Job, rendered: AssessmentView, root: Path | None = None) -> None:
    root = DEMO_CACHE if root is None else root
    folder = root / demo_key(job.lat, job.lon)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "view.json").write_text(rendered.model_dump_json(indent=2))
    (folder / "scoring.txt").write_text(scoring_fingerprint())
    for name, png in job.pngs.items():
        (folder / f"{name}.png").write_bytes(png)


def load_demo(lat: float, lon: float, root: Path | None = None
              ) -> tuple[AssessmentView, dict[str, bytes]] | None:
    root = DEMO_CACHE if root is None else root
    folder = root / demo_key(lat, lon)
    saved = folder / "view.json"
    stamp = folder / "scoring.txt"
    if not saved.exists():
        return None
    if not stamp.exists() or stamp.read_text().strip() != scoring_fingerprint():
        log.info("demo copy for %s predates the current scoring rules: re-running", folder.name)
        return None
    rendered = AssessmentView.model_validate_json(saved.read_text())
    pngs = {p.stem: p.read_bytes() for p in folder.glob("*.png")}
    return rendered, pngs


def replay(job: Job, rendered: AssessmentView, pngs: dict[str, bytes]) -> None:
    """Fill a job from a saved run and emit its steps, so the UI behaves identically."""
    job.pngs = dict(pngs)
    job.cached = rendered
    job.from_cache = True
    job.cost_usd = rendered.cost_usd
    for step in rendered.steps:
        job.steps[step.step] = step
        job.emit({"type": "step", **step.model_dump()})
    job.assessment = rendered.assessment
    job.run = rendered.run
    job.finish(rendered.status, rendered.error)
