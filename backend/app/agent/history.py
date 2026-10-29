"""Local flood history from news reports, verified against the search engine's citations.

Why this exists: the satellite features cannot tell a flooding Lagos street from a dry one
(AUC 0.578 against 30 random built-up points). They screen an *area* for coastal and
riverine exposure. What a loan officer actually needs to know -- did this street flood? --
is in local news, so we go and read it.

The model is not trusted with any of it. Muse Spark runs the Meta web_search tool and then
submits structured reports, and a report survives only if:
  1. its URL appears in the API's own url_citation annotations, so the model cannot invent
     a plausible-looking link to a real publisher, and
  2. it carries a date we can parse, inside the last 5 years.
Everything dropped is recorded with the reason, because "we found nothing" and "the model
made something up and we caught it" must never look the same in a report.

Reporting bias is the known weakness: Lagos news covers some neighbourhoods far better
than others, so no reports found is weak evidence of no flooding. m_reported_floods is a
positive signal only (see scoring.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html as html_lib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from pydantic import ValidationError

from ..config import META_BASE_URL, get_settings
from ..geo.aoi import Site
from ..schemas import FloodHistory, FloodReport, MatchLevel, Metric
from .loop import PRICES

log = logging.getLogger(__name__)

YEARS_BACK = 5
MAX_MODEL_CALLS = 3
REQUEST_TIMEOUT_S = 180  # a search that hangs must not stall a whole validation run
SEARCH_CONTEXT = "medium"
MAX_QUOTE_WORDS = 30
USD_PER_SEARCH = 2.50 / 1000
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "FloodLens/0.1 flood-history lookup (OSM Nominatim policy: 1 req/s)"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "history"

# A street-level report is direct evidence about this address; an estate-level one is
# weaker. Area matches score NOTHING: 27 of 30 random Lagos addresses turned up an
# area-level report, because a report naming a district matches everywhere inside it.
# They are still kept and shown, as context for the reader, but they move no number.
MATCH_WEIGHT = {MatchLevel.street: 1.0, MatchLevel.estate: 0.6, MatchLevel.area: 0.0}
SAME_EVENT_DAYS = 3  # reports within three days are one flood, not several

# Reading the page itself. Grounding says the search tool fetched the URL; it does not say
# the article mentions this street. Publishers block unknown clients, so a normal browser
# user agent, and a failure is "unknown" rather than evidence against the report.
FETCH_TIMEOUT_S = 10
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
SCRIPTS = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.S | re.I)
MARKUP = re.compile(r"<[^>]+>")
PLACE_WORDS = 3  # a place name leads with its distinctive words
QUOTE_WORDS = 8


@dataclass(frozen=True)
class Place:
    """What the site is called, from most to least specific."""

    street: str | None
    estate: str | None
    area: str | None
    lga: str | None
    state: str = "Lagos"
    neighbourhood: str | None = None  # from the coarser lookup; often the name locals use
    # What the loan officer typed. It widens the SEARCH only: OSM often lacks the street,
    # and the officer knows the address. It is deliberately kept out of street_names, so
    # match_level still rests on names the geocoder produced and scoring stays comparable
    # with validation, which never has a typed address.
    typed: str | None = None

    @property
    def names(self) -> list[str]:
        seen: list[str] = []
        for n in (self.typed, self.street, self.estate, self.neighbourhood, self.area, self.lga):
            if n and flatten(n) not in {flatten(x) for x in seen}:
                seen.append(n)
        return seen

    @property
    def street_names(self) -> list[str]:
        return [n for n in (self.street,) if n]

    @property
    def estate_names(self) -> list[str]:
        return [n for n in (self.estate, self.neighbourhood) if n]

    @property
    def area_names(self) -> list[str]:
        return [n for n in (self.area, self.lga) if n]

    def describe(self) -> str:
        return ", ".join(self.names + [self.state]) or self.state

    def key(self) -> str:
        raw = "|".join(n or "" for n in (self.typed, self.street, self.estate,
                                         self.neighbourhood, self.area, self.lga, self.state))
        slug = re.sub(r"[^a-z0-9]+", "-", raw.casefold()).strip("-")[:60]
        return f"{slug or 'unknown'}-{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def _reverse(site: Site, zoom: int, c: httpx.Client) -> dict:
    r = c.get(NOMINATIM, params={"lat": site.lat, "lon": site.lon, "format": "jsonv2",
                                 "zoom": zoom, "addressdetails": 1})
    r.raise_for_status()
    time.sleep(1.1)  # Nominatim allows one request a second
    return r.json()


_PLACES: dict[tuple[float, float], Place] = {}  # per process: sampling and scoring ask twice


def reverse_geocode(site: Site, client: httpx.Client | None = None) -> Place:
    """Two Nominatim lookups: zoom 18 for the street, and a coarser zoom 14 for the name
    locals and reporters actually use. Both run for every site, validation and baseline
    alike, so the comparison stays fair.

    Memoised per process: the baseline sampler asks for the state and the scorer then asks
    for the names, and Nominatim allows only one request a second.
    """
    memo = (round(site.lat, 6), round(site.lon, 6))
    if memo in _PLACES:
        return _PLACES[memo]
    c = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
    try:
        fine = _reverse(site, 18, c)
        coarse = _reverse(site, 14, c)
    finally:
        if client is None:
            c.close()
    a = fine.get("address", {})
    ca = coarse.get("address", {})
    first = lambda d, *keys: next((d[k] for k in keys if d.get(k)), None)  # noqa: E731
    _PLACES[memo] = Place(
        street=first(a, "road", "pedestrian", "residential_road"),
        estate=first(a, "neighbourhood", "residential", "quarter", "hamlet"),
        area=first(a, "suburb", "city_district", "town", "village"),
        lga=first(a, "county", "municipality", "city", "state_district"),
        state=first(a, "state") or "Lagos",
        neighbourhood=coarse.get("name") or first(ca, "suburb", "neighbourhood", "town"),
    )
    return _PLACES[memo]


REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reports": {
            "type": "array",
            "description": "One entry per news report of flooding. Empty if you found none.",
            "items": {
                "type": "object",
                "properties": {
                    "place_named": {"type": "string",
                                    "description": "The place the article names, copied from the article."},
                    "match_level": {
                        "type": "string", "enum": ["street", "estate", "area"],
                        "description": "street: the article names this street. estate: it names this "
                                       "estate or neighbourhood. area: it only names the wider district.",
                    },
                    "event_date": {"type": "string",
                                   "description": "When the flooding happened, YYYY-MM-DD, or YYYY-MM if "
                                                  "the article gives only a month. Not the publication date "
                                                  "unless they are the same."},
                    "publisher": {"type": "string"},
                    "url": {"type": "string", "description": "The exact source URL you were given."},
                    "quote": {"type": "string",
                              "description": "At most 30 words, copied verbatim from the article."},
                },
                "required": ["place_named", "match_level", "event_date", "publisher", "url", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reports"],
    "additionalProperties": False,
}


def _instructions(place: Place, since: dt.date) -> str:
    return (
        f"Find news reports of flooding at this place in Lagos, Nigeria:\n{place.describe()}\n\n"
        f"Only events on or after {since.isoformat()} count.\n\n"
        "Use web_search. Search the specific names above, the most specific first, together with "
        "words like flood, flooding, flooded, submerged. Nigerian outlets cover this best.\n\n"
        "Rules:\n"
        "1. Only report what a source you actually opened says. Never guess a URL: a URL you did "
        "not receive from search will be discarded and the report with it.\n"
        "2. event_date is when the water was there, not when the piece was published.\n"
        "3. match_level says how closely the article's place matches the place above. Be strict: "
        "use 'area' if the article only names the district.\n"
        "4. quote is 30 words or fewer, copied exactly.\n"
        "5. If you find nothing, call submit_flood_history with an empty list. That is a fine answer. "
        "Do not pad it with reports about other parts of Lagos.\n\n"
        "Finish by calling submit_flood_history exactly once."
    )


# "James Kunle St" and "James Kunle Street" are the same street. Both sides of every
# comparison are expanded, so it does not matter which form the article uses.
ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "cres": "crescent", "cl": "close", "dr": "drive", "ln": "lane", "blvd": "boulevard",
    "hwy": "highway", "expy": "expressway", "expwy": "expressway", "sq": "square",
    "pl": "place", "jct": "junction", "jcn": "junction", "mkt": "market", "est": "estate",
}


def flatten(text: str) -> str:
    """Lowercase words only, abbreviations expanded, so punctuation, curly quotes and
    markup spacing cannot break a match."""
    words = re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()
    return " ".join(ABBREVIATIONS.get(w, w) for w in words)


def says(flat_page: str, phrase: str) -> bool:
    """Does the page contain this phrase as whole words?

    Whole words matter: "Ojo" must not match "Ojota", which a plain substring test would.
    """
    flat = flatten(phrase)
    return bool(flat) and f" {flat} " in f" {flat_page} "


def page_text(url: str, client: Any) -> str | None:
    """The readable text of a page, or None if we could not get it."""
    try:
        r = client.get(url, timeout=FETCH_TIMEOUT_S, follow_redirects=True,
                       headers={"User-Agent": BROWSER_UA,
                                "Accept": "text/html,application/xhtml+xml,*/*"})
    except Exception:
        return None
    if getattr(r, "status_code", 0) >= 400:
        return None
    body = getattr(r, "text", "") or ""
    if not body.strip():
        return None
    return html_lib.unescape(MARKUP.sub(" ", SCRIPTS.sub(" ", body)))


def level_on_page(flat_page: str, place: Place) -> MatchLevel | None:
    """How closely the page matches OUR site, judged on OUR names.

    This is the check that decides match_level, not the model's opinion of it. A report
    is street-level only when the article contains the street name the reverse geocode
    gave us and the search was given. None means the article never names this place at
    all, which is grounds to drop it.
    """
    for level, names in ((MatchLevel.street, place.street_names),
                         (MatchLevel.estate, place.estate_names),
                         (MatchLevel.area, place.area_names)):
        if any(says(flat_page, n) for n in names):
            return level
    return None


def check_text(reports: list[FloodReport], place: Place,
               client: Any | None = None) -> tuple[list[FloodReport], list[str], list[str]]:
    """Open each page and ask whether it is about our site. Drops explicit contradictions.

    Grounding proved the search tool fetched the URL. It did not prove the article is
    about this street, and the model's own match_level is just its opinion. So the page
    is searched for our names, and match_level is set from what is actually there. The
    model's phrase and its quote are kept as secondary checks and recorded, never
    decisive.
    """
    own = client is None
    if own:
        client = httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT_S)
    kept: list[FloodReport] = []
    dropped: list[str] = []
    corrections: list[str] = []  # kept: a demotion is not a rejection and must not read as one
    try:
        for r in reports:
            body = page_text(r.url, client)
            if body is None:
                r.text_verified, r.quote_found, r.phrase_found = "unknown", None, None
                kept.append(r)  # a paywall is not evidence against a report
                continue
            flat = flatten(body)
            level = level_on_page(flat, place)
            r.phrase_found = says(flat, r.place_named)
            r.quote_found = says(flat, r.quote.rstrip("…"))
            if level is None:
                r.text_verified = "false"
                dropped.append(f"{r.place_named} / {r.event_date}: the page at {r.url} names "
                               f"none of {place.describe()}")
                continue
            if level != r.match_level:
                direction = ("demoted" if MATCH_WEIGHT[level] < MATCH_WEIGHT[r.match_level]
                             else "promoted")
                corrections.append(f"{r.place_named} / {r.event_date}: {direction} from "
                                   f"{r.match_level.value} to {level.value} by the page text")
            r.match_level = level
            r.text_verified = "true"
            kept.append(r)
    finally:
        if own:
            client.close()
    return kept, dropped, corrections


def _normalise(url: str) -> str:
    u = url.strip().casefold()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def parse_event_date(raw: str) -> str | None:
    """YYYY-MM-DD or YYYY-MM only. A bare year cannot place an event, so it is not a date."""
    raw = (raw or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        try:
            dt.date.fromisoformat(raw)
        except ValueError:
            return None
        return raw
    if re.fullmatch(r"\d{4}-\d{2}", raw):
        try:
            dt.date.fromisoformat(f"{raw}-01")
        except ValueError:
            return None
        return raw
    return None


def _as_date(event_date: str) -> dt.date:
    return dt.date.fromisoformat(event_date if len(event_date) == 10 else f"{event_date}-01")


def trim_quote(quote: str, max_words: int = MAX_QUOTE_WORDS) -> str:
    words = quote.split()
    return quote.strip() if len(words) <= max_words else " ".join(words[:max_words]) + "…"


def keep_reports(raw: list[dict[str, Any]], cited: set[str], since: dt.date) -> tuple[list[FloodReport], list[str]]:
    """Verification. Returns (kept, dropped-with-reasons). Nothing here trusts the model."""
    kept: list[FloodReport] = []
    dropped: list[str] = []
    for item in raw:
        url = str(item.get("url", "")).strip()
        where = f"{item.get('place_named', '?')} / {item.get('event_date', '?')}"
        if _normalise(url) not in cited:
            dropped.append(f"{where}: url was never retrieved by the search tool ({url or 'empty'})")
            continue
        date = parse_event_date(str(item.get("event_date", "")))
        if date is None:
            dropped.append(f"{where}: no parseable event date ({item.get('event_date')!r})")
            continue
        if _as_date(date) < since:
            dropped.append(f"{where}: older than {YEARS_BACK} years")
            continue
        try:
            kept.append(FloodReport(place_named=str(item["place_named"]), match_level=item["match_level"],
                                    event_date=date, publisher=str(item.get("publisher", "")).strip(),
                                    url=url, quote=trim_quote(str(item.get("quote", "")))))
        except (ValidationError, KeyError) as exc:
            dropped.append(f"{where}: does not fit the schema ({exc.__class__.__name__})")
    return kept, dropped


def distinct_events(reports: list[FloodReport]) -> list[tuple[str, float]]:
    """Collapse reports into flood events, keeping the closest match for each.

    The same flood is covered by several outlets and dated a day or two apart, so reports
    within SAME_EVENT_DAYS are one event and the strongest match_level wins.
    """
    events: list[tuple[str, float]] = []
    for r in sorted(reports, key=lambda r: r.event_date):
        weight = MATCH_WEIGHT[MatchLevel(r.match_level)]
        if events and abs((_as_date(r.event_date) - _as_date(events[-1][0])).days) <= SAME_EVENT_DAYS:
            events[-1] = (events[-1][0], max(events[-1][1], weight))
        else:
            events.append((r.event_date, weight))
    return events


def history_metric(history: FloodHistory) -> Metric | None:
    """m_reported_floods: weighted distinct flood events in the last YEARS_BACK years.

    A search that ran and found nothing scores 0, which is a real (if weakly evidenced)
    measurement. A search that never ran returns None, so the weight is rescaled away
    rather than silently counted as "no floods here".
    """
    if not history.searched:
        return None
    events = distinct_events(history.reports)
    total = sum(w for _, w in events)
    levels = sorted({r.match_level.value for r in history.reports})
    unread = sum(1 for r in history.reports if r.text_verified != "true")
    area_only = sum(1 for r in history.reports if r.match_level == MatchLevel.area)
    detail = (f"{len(events)} scoring event(s), {'/'.join(levels)} match"
              if history.reports else "none found")
    if area_only:
        detail += f"; {area_only} area-level report(s) listed as context but scoring 0"
    if unread:
        detail += f"; {unread} report(s) whose page could not be read to confirm the place"
    return Metric(
        id="m_reported_floods",
        label=f"Flooding reported here in news, last {YEARS_BACK} years (weighted by how closely "
              "the place matches)",
        value=total, unit="events",
        source=f"Meta web search over news, verified against citations: {detail}. "
               "Absence is weak evidence: news covers some areas far better than others.")


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def grounded_urls(resp: Any) -> set[str]:
    """Every URL the API itself retrieved or cited, which is what a report is checked against.

    Two sources, both of them the API's own record rather than the model's prose:
    - completed web_search_call items whose action is open_page. This is the page the
      search tool actually fetched.
    - url_citation annotations on output_text, as documented.

    In practice the annotations come back empty when the final answer is a function call
    instead of prose, so open_page is what does the work. Either way, a URL the model
    merely typed into submit_flood_history appears in neither and is dropped.
    """
    out: set[str] = set()
    for item in getattr(resp, "output", None) or []:
        if _field(item, "type") == "web_search_call" and _field(item, "status") == "completed":
            url = _field(_field(item, "action"), "url")
            if url:
                out.add(_normalise(str(url)))
        for block in getattr(item, "content", None) or []:
            for ann in (_field(block, "annotations") or []):
                if _field(ann, "type") == "url_citation" and _field(ann, "url"):
                    out.add(_normalise(str(_field(ann, "url"))))
    return out


def _search_calls(resp: Any) -> int:
    """Billable search tool calls. Both a query and a page open are a web_search_call, and
    the docs do not say which bill, so count them all: this over-states rather than hides."""
    return sum(1 for i in (getattr(resp, "output", None) or [])
               if _field(i, "type") == "web_search_call")


def lookup(site: Site, client: Any | None = None, cache_dir: Path | None = None,
           place: Place | None = None, today: dt.date | None = None,
           http: Any | None = None, address: str | None = None) -> FloodHistory:
    """Flood history for one site. Cached per place, so neighbouring sites pay once.

    `address` is what the loan officer typed. It joins the search terms but not the
    match-level test: see Place.typed.
    """
    place = place or reverse_geocode(site)
    if address and address.strip():
        place = replace(place, typed=address.strip())
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    cached = cache_dir / f"{place.key()}.json"
    if cached.exists():
        got = FloodHistory.model_validate_json(cached.read_text())
        got.from_cache = True
        if not got.text_checked:
            # Written before the page check existed. The search is the expensive part, so
            # upgrade the entry in place rather than paying to search again.
            got.reports, failed, corrected = check_text(got.reports, place, http)
            got.dropped += failed
            got.corrections += corrected
            got.text_checked = True
            cached.write_text(got.model_dump_json(indent=2))
        return got

    history = _ask(place, client, today or dt.date.today(), http)
    if history.searched:  # never cache a failed call
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(history.model_dump_json(indent=2))
    log.info("flood history for %s: %d kept, %d dropped, %d searches, est $%.4f",
             place.describe(), len(history.reports), len(history.dropped), history.searches,
             history.cost_usd)
    return history


def _ask(place: Place, client: Any | None, today: dt.date, http: Any | None = None) -> FloodHistory:
    settings = get_settings()
    own = client is None
    if own:
        if not settings.model_api_key:
            raise RuntimeError("MODEL_API_KEY is not set")
        client = OpenAI(base_url=META_BASE_URL, api_key=settings.model_api_key,
                        timeout=REQUEST_TIMEOUT_S, max_retries=1)

    since = today.replace(year=today.year - YEARS_BACK)
    out = FloodHistory(place=place.describe())
    tools = [
        {"type": "web_search", "search_context_size": SEARCH_CONTEXT,
         "user_location": {"country": "NG", "city": "Lagos"}},
        {"type": "function", "name": "submit_flood_history", "strict": True,
         "description": "Submit the reports you verified. Call exactly once, last.",
         "parameters": REPORT_SCHEMA},
    ]
    items: list[Any] = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": _instructions(place, since)}]}]
    cited: set[str] = set()
    try:
        for _ in range(MAX_MODEL_CALLS):
            try:
                resp = client.responses.create(
                    model=settings.spark_model, input=items, tools=tools,
                    reasoning={"effort": settings.reasoning_effort})
            except Exception as exc:
                out.dropped.append(f"search call failed: {exc.__class__.__name__}: {exc}")
                return out
            usage = getattr(resp, "usage", None)
            out.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            out.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            out.searches += _search_calls(resp)
            cited |= grounded_urls(resp)

            calls = [i for i in (getattr(resp, "output", None) or [])
                     if getattr(i, "type", "") == "function_call" and i.name == "submit_flood_history"]
            if not calls:
                items.append({"type": "message", "role": "user", "content": [{"type": "input_text",
                              "text": "Now call submit_flood_history with what you verified."}]})
                continue
            try:
                raw = json.loads(calls[0].arguments or "{}").get("reports") or []
            except json.JSONDecodeError as exc:
                out.dropped.append(f"submit_flood_history was not valid JSON: {exc}")
                raw = []
            kept, dropped = keep_reports(raw if isinstance(raw, list) else [], cited, since)
            kept, failed_text, corrected = check_text(kept, place, http)
            out.reports = kept
            out.dropped += dropped + failed_text
            out.corrections += corrected
            out.searched = True
            out.text_checked = True
            break
        else:
            out.dropped.append(f"no submit_flood_history call in {MAX_MODEL_CALLS} model calls")
    finally:
        if own:
            client.close()

    price = PRICES.get(settings.spark_model)
    tokens = 0.0 if price is None else (out.input_tokens * price.input + out.output_tokens * price.output) / 1e6
    out.cost_usd = round(out.searches * USD_PER_SEARCH + tokens, 6)
    return out
