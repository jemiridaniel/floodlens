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
    history = flood_history(site) if flood_history is not None else None
    return assemble(req, layers, history)


def assemble(req: AssessmentRequest, layers, history=None) -> Assessment:
    """Build the Assessment from layers and history that are already in hand.

    Split out of assess() so the API can run the area screen and the flood history at the
    same time and still end up with exactly one definition of what an Assessment is.
    """
    metrics = compute_metrics(layers)
    street_events = 0
    if history is not None:
        from .agent.history import distinct_events, history_metric  # keeps the offline path clean
        found = history_metric(history)
        if found is not None:
            metrics.append(found)
        street_events = len(distinct_events(
            [r for r in history.reports if r.match_level == "street"]))
    s = score(metrics, street_events=street_events)
    evidence = dict(layers.item_ids)
    if layers.optical:
        evidence["optical"] = layers.optical.item_id
    if layers.radar:
        evidence["radar"] = layers.radar.item_id
    notes = list(layers.notes)
    street = False
    if history is not None:
        evidence.update({f"report_{i}": r.url for i, r in enumerate(history.reports, 1)})
        notes += [f"flood history: {d}" for d in history.dropped]
        street = any(r.match_level == "street" for r in history.reports)
    for group in s.missing_groups:
        notes.append(f"NO {group.upper()} EVIDENCE: that whole group is missing for this site, so "
                     "the score rests on the others and confidence is held at low.")
    return Assessment(
        id=uuid.uuid4().hex[:12],
        lat=req.lat, lon=req.lon, label=req.label,
        metrics=metrics, score=s,
        allowed_actions=ALLOWED_ACTIONS[s.tier],
        evidence=evidence,
        notes=notes,
        has_street_reports=street,
        street_flood_events=street_events,
    )
