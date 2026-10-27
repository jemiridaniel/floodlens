"""One assessment: fetch layers -> measure -> score. (The Muse Spark agent loop wraps this next.)"""
from __future__ import annotations

import uuid

from .geo.aoi import Site
from .geo.sources import LayerSource
from .metrics import compute_metrics
from .schemas import Assessment, AssessmentRequest
from .scoring import ALLOWED_ACTIONS, score


def assess(req: AssessmentRequest, source: LayerSource) -> Assessment:
    site = Site(req.lat, req.lon)
    layers = source.fetch(site, req.wet_date)
    metrics = compute_metrics(layers)
    s = score(metrics)
    evidence = dict(layers.item_ids)
    if layers.optical:
        evidence["optical"] = layers.optical.item_id
    if layers.radar:
        evidence["radar"] = layers.radar.item_id
    return Assessment(
        id=uuid.uuid4().hex[:12],
        lat=req.lat, lon=req.lon, label=req.label,
        metrics=metrics, score=s,
        allowed_actions=ALLOWED_ACTIONS[s.tier],
        evidence=evidence,
        notes=layers.notes,
    )
