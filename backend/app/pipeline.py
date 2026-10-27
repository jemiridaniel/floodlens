"""One assessment: fetch layers -> measure -> score. (The Muse Spark agent loop wraps this next.)"""
from __future__ import annotations

import uuid
from collections.abc import Callable

from .geo.aoi import Site
from .geo.sources import LayerSource
from .metrics import compute_metrics
from .schemas import Assessment, AssessmentRequest
from .scoring import ALLOWED_ACTIONS, score


def assess(req: AssessmentRequest, source: LayerSource,
           flood_history: Callable[[Site], object] | None = None) -> Assessment:
    """Measure, then score. flood_history is optional because it costs money and needs the
    network: without it the assessment is the area screen alone."""
    site = Site(req.lat, req.lon)
    layers = source.fetch(site, req.wet_date)
    metrics = compute_metrics(layers)
    history = None
    if flood_history is not None:
        from .agent.history import history_metric  # local: keeps the offline path import-free
        history = flood_history(site)
        found = history_metric(history) if history is not None else None
        if found is not None:
            metrics.append(found)
    s = score(metrics)
    evidence = dict(layers.item_ids)
    if layers.optical:
        evidence["optical"] = layers.optical.item_id
    if layers.radar:
        evidence["radar"] = layers.radar.item_id
    notes = list(layers.notes)
    if history is not None:
        evidence.update({f"report_{i}": r.url for i, r in enumerate(history.reports, 1)})
        notes += [f"flood history: {d}" for d in history.dropped]
    return Assessment(
        id=uuid.uuid4().hex[:12],
        lat=req.lat, lon=req.lon, label=req.label,
        metrics=metrics, score=s,
        allowed_actions=ALLOWED_ACTIONS[s.tier],
        evidence=evidence,
        notes=notes if flood_history is not None else layers.notes,
    )
