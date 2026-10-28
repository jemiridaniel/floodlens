"""Flood history, offline: a canned Responses API with canned search citations.

The point of these tests is the verification step. A report the model typed but the
search engine never cited must not reach a loan officer.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest

from app.agent import history as H
from app.geo.aoi import Site
from app.schemas import MatchLevel

TODAY = dt.date(2026, 10, 27)
PLACE = H.Place(street="Ago Palace Way", estate=None, area="Cele", lga="Oshodi/Isolo")
CITED = "https://allafrica.com/stories/202607010113.html"
UNCITED = "https://www.vanguardngr.com/2026/07/invented-story.html"


# ------------------------------------------------------------------ canned API
@dataclass
class Ann:
    url: str
    type: str = "url_citation"
    title: str = "Flooding in Lagos"


@dataclass
class Block:
    annotations: list = field(default_factory=list)
    type: str = "output_text"


@dataclass
class Msg:
    content: list
    type: str = "message"


@dataclass
class SearchCall:
    """A query. Carries no url: querying is not grounding."""

    action: dict = field(default_factory=lambda: {"type": "search", "query": "lagos flood"})
    status: str = "completed"
    type: str = "web_search_call"


@dataclass
class OpenPage:
    """The API fetched this page. This is what a report's url is checked against."""

    url: str
    status: str = "completed"
    type: str = "web_search_call"

    @property
    def action(self) -> dict:
        return {"type": "open_page", "url": self.url}


@dataclass
class Call:
    arguments: str
    name: str = "submit_flood_history"
    call_id: str = "c1"
    type: str = "function_call"


@dataclass
class Usage:
    input_tokens: int = 900
    output_tokens: int = 400


@dataclass
class Resp:
    output: list
    usage: Usage = field(default_factory=Usage)


class FakeClient:
    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        assert self.turns, "more model calls than the test scripted"
        return self.turns.pop(0)


def turn(reports, cited_urls=(), searches=1, via="open_page"):
    """One response: searches, the pages the tool opened, and the structured submission.

    via="open_page" mirrors what the live API does (annotations empty, grounding in the
    web_search_call records); via="annotation" exercises the documented citation path.
    """
    out: list = [SearchCall() for _ in range(searches)]
    anns = []
    if via == "open_page":
        out += [OpenPage(url=u) for u in cited_urls]
    else:
        anns = [Ann(url=u) for u in cited_urls]
    out.append(Msg(content=[Block(annotations=anns)]))
    out.append(Call(arguments=json.dumps({"reports": reports})))
    return Resp(output=out)


def report(**over):
    base = {"place_named": "Ago Palace Way, Okota", "match_level": "street",
            "event_date": "2026-06-30", "publisher": "AllAfrica", "url": CITED,
            "quote": "Residents of Ago Palace Way waded through knee-deep water."}
    return {**base, **over}


def run(client, **kw):
    """lookup() with a throwaway cache. http defaults to a client that reads nothing, so
    tests that do not care about the page check get text_verified="unknown"."""
    return H.lookup(Site(6.5026016, 3.3051669), client=client, place=PLACE, today=TODAY,
                    cache_dir=kw.pop("cache_dir", None) or _tmp(),
                    http=kw.pop("http", None) or FakeHttp({}), **kw)


_TMP: list = []


def _tmp():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    _TMP.append(d)
    return d


# ------------------------------------------------------------------ the four cases
def test_a_cited_report_is_kept():
    h = run(FakeClient(turn([report()], cited_urls=[CITED])))
    assert h.searched is True
    assert len(h.reports) == 1 and h.dropped == []
    r = h.reports[0]
    assert r.url == CITED and r.match_level == MatchLevel.street and r.event_date == "2026-06-30"
    assert h.searches == 2  # the query and the page open are both billable calls
    assert h.cost_usd > 2 * H.USD_PER_SEARCH


def test_an_uncited_url_is_dropped():
    """The model typed a real-looking Vanguard link the search never returned."""
    h = run(FakeClient(turn([report(url=UNCITED, publisher="Vanguard")], cited_urls=[CITED])))
    assert h.reports == []
    assert len(h.dropped) == 1 and "never retrieved by the search tool" in h.dropped[0]
    assert UNCITED in h.dropped[0]


def test_a_cited_report_is_kept_via_the_documented_annotation_path():
    """The API also documents url_citation annotations; both routes must verify."""
    h = run(FakeClient(turn([report()], cited_urls=[CITED], via="annotation")))
    assert len(h.reports) == 1 and h.dropped == []


def test_a_queried_but_unopened_url_is_not_grounding():
    """Running a search is not evidence; only a page the tool actually opened is."""
    h = run(FakeClient(turn([report()], cited_urls=[], searches=4)))
    assert h.reports == [] and "never retrieved" in h.dropped[0]
    assert h.searches == 4


def test_an_undated_report_is_dropped():
    cases = ["sometime last year", "2026", "", "June 2026"]
    h = run(FakeClient(turn([report(event_date=d) for d in cases], cited_urls=[CITED])))
    assert h.reports == []
    assert len(h.dropped) == len(cases)
    assert all("no parseable event date" in d for d in h.dropped)


def test_match_level_weighting():
    """1.0 street, 0.6 estate, 0.0 area. Area reports are context, not score."""
    assert H.MATCH_WEIGHT == {MatchLevel.street: 1.0, MatchLevel.estate: 0.6, MatchLevel.area: 0.0}
    rows = [report(event_date="2026-06-30", match_level="street"),
            report(event_date="2025-05-02", match_level="estate"),
            report(event_date="2024-09-14", match_level="area")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])))  # unreadable pages: levels stand
    assert len(h.reports) == 3
    m = H.history_metric(h)
    assert m.value == pytest.approx(1.0 + 0.6)
    assert m.id == "m_reported_floods" and m.unit == "events"


def test_area_reports_are_still_listed_as_context():
    rows = [report(event_date="2024-09-14", match_level="area")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])))
    assert len(h.reports) == 1  # kept and shown
    m = H.history_metric(h)
    assert m.value == 0.0 and "context but scoring 0" in m.source


# ------------------------------------------------------------------ the rest
def test_one_flood_reported_by_three_outlets_is_one_event():
    rows = [report(event_date="2026-06-29", publisher="AllAfrica", match_level="area"),
            report(event_date="2026-06-30", publisher="Punch", match_level="street"),
            report(event_date="2026-06-30", publisher="Vanguard", match_level="estate")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])))
    events = H.distinct_events(h.reports)
    assert len(events) == 1  # within SAME_EVENT_DAYS
    assert events[0][1] == 1.0  # the closest match wins
    assert H.history_metric(h).value == pytest.approx(1.0)


def test_events_more_than_a_week_apart_count_separately():
    rows = [report(event_date="2026-06-01"), report(event_date="2026-07-15")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])))
    assert len(H.distinct_events(h.reports)) == 2


def test_reports_older_than_five_years_are_dropped():
    h = run(FakeClient(turn([report(event_date="2019-07-02")], cited_urls=[CITED])))
    assert h.reports == [] and "older than 5 years" in h.dropped[0]


def test_nothing_found_scores_zero_but_a_failed_search_scores_nothing():
    found_none = run(FakeClient(turn([], cited_urls=[])))
    assert found_none.searched is True
    m = H.history_metric(found_none)
    assert m is not None and m.value == 0.0  # a search that ran and found nothing is a measurement

    class Broken:
        def __init__(self):
            self.responses = SimpleNamespace(create=self._boom)

        def _boom(self, **kw):
            raise RuntimeError("503")

    failed = run(Broken())
    assert failed.searched is False
    assert H.history_metric(failed) is None  # never counted as "no floods here"
    assert any("503" in d for d in failed.dropped)


def test_quotes_are_trimmed_to_thirty_words():
    long_quote = " ".join(f"w{i}" for i in range(60))
    h = run(FakeClient(turn([report(quote=long_quote)], cited_urls=[CITED])))
    kept = h.reports[0].quote
    assert len(kept.split()) == 30 and kept.endswith("…")  # the ellipsis rides the 30th word
    assert kept.startswith("w0 w1 ") and "w30" not in kept


def test_citation_matching_ignores_scheme_www_and_trailing_slash():
    h = run(FakeClient(turn([report(url="HTTP://WWW.AllAfrica.com/stories/202607010113.html/")],
                            cited_urls=[CITED])))
    assert len(h.reports) == 1


def test_the_search_tool_and_strict_schema_are_declared():
    client = FakeClient(turn([report()], cited_urls=[CITED]))
    run(client)
    tools = client.requests[0]["tools"]
    assert {"type": "web_search", "search_context_size": H.SEARCH_CONTEXT,
            "user_location": {"country": "NG", "city": "Lagos"}} in tools
    fn = next(t for t in tools if t.get("name") == "submit_flood_history")
    assert fn["strict"] is True
    item = fn["parameters"]["properties"]["reports"]["items"]
    assert item["properties"]["match_level"]["enum"] == ["street", "estate", "area"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])  # strict mode needs all of them
    assert "Ago Palace Way" in json.dumps(client.requests[0]["input"])


def test_results_are_cached_per_place():
    cache = _tmp()
    first = H.lookup(Site(6.5, 3.3), client=FakeClient(turn([report()], cited_urls=[CITED])),
                     place=PLACE, today=TODAY, cache_dir=cache)
    assert first.from_cache is False
    # a second site in the same street must not pay for another search
    again = H.lookup(Site(6.51, 3.31), client=None, place=PLACE, today=TODAY, cache_dir=cache)
    assert again.from_cache is True
    assert [r.url for r in again.reports] == [r.url for r in first.reports]
    assert len(list(cache.glob("*.json"))) == 1


def test_a_failed_search_is_not_cached():
    cache = _tmp()

    class Broken:
        def __init__(self):
            self.responses = SimpleNamespace(create=self._boom)

        def _boom(self, **kw):
            raise RuntimeError("503")

    H.lookup(Site(6.5, 3.3), client=Broken(), place=PLACE, today=TODAY, cache_dir=cache)
    assert list(cache.glob("*.json")) == []


def test_place_names_come_from_the_address_most_specific_first():
    p = H.Place(street="Ago Palace Way", estate="Light Estate", area="Cele", lga="Oshodi/Isolo",
                neighbourhood="Okota")
    assert p.names == ["Ago Palace Way", "Light Estate", "Okota", "Cele", "Oshodi/Isolo"]
    assert p.street_names == ["Ago Palace Way"]
    assert p.estate_names == ["Light Estate", "Okota"]  # the coarse lookup joins the estate level
    assert p.area_names == ["Cele", "Oshodi/Isolo"]
    duplicated = H.Place(street=None, estate="Cele", area="cele", lga="Ojo")
    assert duplicated.names == ["Cele", "Ojo"]  # the same name twice is listed once
    assert p.describe().endswith("Lagos")
    assert H.Place(None, None, None, "Ojo").describe() == "Ojo, Lagos"


def test_a_model_that_never_submits_is_given_up_on():
    quiet = Resp(output=[Msg(content=[Block()])])
    h = run(FakeClient(*[quiet] * H.MAX_MODEL_CALLS))
    assert h.searched is False
    assert any("no submit_flood_history call" in d for d in h.dropped)


# ------------------------------------------------- reading the page itself
@dataclass
class Page:
    text: str = ""
    status_code: int = 200


class FakeHttp:
    """Mocked httpx: url -> Page, or an exception to raise."""

    def __init__(self, pages: dict):
        self.pages, self.asked, self.kwargs = pages, [], []

    def get(self, url, **kw):
        self.asked.append(url)
        self.kwargs.append(kw)
        page = self.pages.get(url, Page(status_code=404))
        if isinstance(page, Exception):
            raise page
        return page

    def close(self):
        pass


ARTICLE = ("<html><head><style>.x{color:red}</style></head><body><p>Among the worst-hit areas "
           "were <b>Ago&nbsp;Palace Way</b> in Okota and sections of Gbagada, where rising water "
           "entered homes.</p><script>track()</script></body></html>")
# PLACE.street is "Ago Palace Way", so ARTICLE is a street-level match for our site.


def test_a_page_that_names_the_place_is_text_verified():
    http = FakeHttp({CITED: Page(ARTICLE)})
    quoted = "Among the worst-hit areas were Ago Palace Way in Okota and sections of Gbagada"
    h = run(FakeClient(turn([report(quote=quoted)], cited_urls=[CITED])), http=http)
    assert len(h.reports) == 1
    r = h.reports[0]
    assert r.text_verified == "true"
    assert r.quote_found is True
    assert http.asked == [CITED]
    assert http.kwargs[0]["timeout"] == 10
    assert "Mozilla/5.0" in http.kwargs[0]["headers"]["User-Agent"]  # publishers block bots


def test_a_page_that_never_names_our_place_is_dropped():
    """Grounded url, real article, but it is about somewhere else."""
    other = "<html><body><p>Flooding hit Festac Town and Amuwo Odofin on Monday.</p></body></html>"
    http = FakeHttp({CITED: Page(other)})
    h = run(FakeClient(turn([report()], cited_urls=[CITED])), http=http)
    assert h.reports == []
    assert len(h.dropped) == 1 and "names none of" in h.dropped[0]


def test_an_unreadable_page_is_unknown_and_kept():
    """Paywalls, blocks and timeouts are not evidence against a report."""
    for page in (Page(status_code=403), Page(status_code=500), Page(""),
                 httpx.ConnectTimeout("too slow")):
        http = FakeHttp({CITED: page})
        h = run(FakeClient(turn([report()], cited_urls=[CITED])), http=http)
        assert len(h.reports) == 1, page
        assert h.reports[0].text_verified == "unknown"
        assert h.reports[0].quote_found is None
        assert h.dropped == []


def test_the_match_uses_our_street_name_not_the_models_wording():
    """The model's phrasing varies; our reverse-geocoded street name does not."""
    http = FakeHttp({CITED: Page(ARTICLE)})
    h = run(FakeClient(turn([report(place_named="somewhere around Okota, Lagos")],
                            cited_urls=[CITED])), http=http)
    r = h.reports[0]
    assert r.text_verified == "true" and r.match_level == MatchLevel.street  # our street is there
    assert r.phrase_found is False  # the model's own wording was not, and that is only noted


def test_a_generic_tail_cannot_carry_the_match():
    """'area of Lagos' matches any Lagos article and must never verify on its own."""
    page = Page("<html><body>Heavy rain across the area of Lagos on Monday.</body></html>")
    h = run(FakeClient(turn([report(place_named="Ago Palace Way in Okota area of Lagos")],
                            cited_urls=[CITED])), http=FakeHttp({CITED: page}))
    assert h.reports == [] and "names none of" in h.dropped[0]


def test_street_level_requires_our_street_on_the_page():
    """The model claims street level; the page only names the district, so it is demoted."""
    district_only = Page("<html><body><p>Flooding across Oshodi/Isolo on Monday.</p></body></html>")
    h = run(FakeClient(turn([report(match_level="street")], cited_urls=[CITED])),
            http=FakeHttp({CITED: district_only}))
    assert h.reports[0].match_level == MatchLevel.area
    assert h.dropped == []  # a demotion keeps the report: it is not a rejection
    assert any("demoted from street to area" in c for c in h.corrections)
    assert H.history_metric(h).value == 0.0  # area matches score nothing


def test_an_area_claim_is_promoted_when_our_street_is_on_the_page():
    h = run(FakeClient(turn([report(match_level="area")], cited_urls=[CITED])),
            http=FakeHttp({CITED: Page(ARTICLE)}))
    assert h.reports[0].match_level == MatchLevel.street
    assert H.history_metric(h).value == pytest.approx(1.0)


def test_street_names_match_across_abbreviations():
    page = Page("<html><body><p>James Kunle St was under water for two days.</p></body></html>")
    place = H.Place(street="James Kunle Street", estate=None, area=None, lga="Ojo")
    kept, dropped, corrected = H.check_text(
        [H.FloodReport(place_named="James Kunle Street", match_level="street",
                       event_date="2026-06-28", publisher="P", url=CITED, quote="under water")],
        place, FakeHttp({CITED: page}))
    assert kept and kept[0].match_level == MatchLevel.street
    assert dropped == [] and corrected == []


def test_whole_word_matching_so_ojo_is_not_ojota():
    page = Page("<html><body><p>Ojota was flooded on Monday.</p></body></html>")
    place = H.Place(street=None, estate=None, area=None, lga="Ojo")
    kept, dropped, _ = H.check_text(
        [H.FloodReport(place_named="Ojo", match_level="area", event_date="2026-06-28",
                       publisher="P", url=CITED, quote="flooded")],
        place, FakeHttp({CITED: page}))
    assert kept == [] and "names none of" in dropped[0]


def test_a_quote_that_is_not_on_the_page_is_flagged_but_kept():
    """The place is corroborated, so the report stands; the quote is marked not found."""
    http = FakeHttp({CITED: Page(ARTICLE)})
    h = run(FakeClient(turn([report(quote="Residents said the water reached the rooftops overnight "
                                          "and destroyed every shop on the street.")],
                            cited_urls=[CITED])), http=http)
    assert len(h.reports) == 1
    assert h.reports[0].text_verified == "true"
    assert h.reports[0].quote_found is False


def test_the_metric_flags_reports_whose_page_could_not_be_read():
    http = FakeHttp({CITED: Page(status_code=403)})
    h = run(FakeClient(turn([report()], cited_urls=[CITED])), http=http)
    m = H.history_metric(h)
    assert m.value == pytest.approx(1.0)
    assert "could not be read" in m.source


def test_an_old_cache_entry_is_upgraded_not_re_searched():
    """The search is the expensive part. Entries written before the page check get the
    page check added in place, without paying for another search."""
    cache = _tmp()
    H.lookup(Site(6.5, 3.3), client=FakeClient(turn([report()], cited_urls=[CITED])),
             place=PLACE, today=TODAY, cache_dir=cache, http=FakeHttp({CITED: Page(ARTICLE)}))
    stored = json.loads(next(cache.glob("*.json")).read_text())
    stored["text_checked"] = False  # pretend it predates the page check
    for r in stored["reports"]:
        r["text_verified"], r["quote_found"] = "unknown", None
    next(cache.glob("*.json")).write_text(json.dumps(stored))

    http = FakeHttp({CITED: Page(ARTICLE)})
    again = H.lookup(Site(6.5, 3.3), client=None, place=PLACE, today=TODAY, cache_dir=cache, http=http)
    assert again.from_cache is True and again.text_checked is True
    assert again.reports[0].text_verified == "true"
    assert http.asked == [CITED]  # the page was fetched; no model call was made
    assert json.loads(next(cache.glob("*.json")).read_text())["text_checked"] is True


# ------------------------------------------------- three-day event merging
def test_reports_within_three_days_are_one_event():
    http = FakeHttp({CITED: Page(ARTICLE)})
    rows = [report(event_date="2026-06-28"), report(event_date="2026-06-30")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])), http=http)
    assert len(H.distinct_events(h.reports)) == 1


def test_reports_four_days_apart_are_two_events():
    http = FakeHttp({CITED: Page(ARTICLE)})
    rows = [report(event_date="2026-06-26"), report(event_date="2026-06-30")]
    h = run(FakeClient(turn(rows, cited_urls=[CITED])), http=http)
    assert H.SAME_EVENT_DAYS == 3
    assert len(H.distinct_events(h.reports)) == 2


def test_a_demotion_is_a_correction_not_a_rejection():
    """These were being counted together, which overstated rejections nearly threefold."""
    district_only = Page("<html><body><p>Flooding across Oshodi/Isolo on Monday.</p></body></html>")
    h = run(FakeClient(turn([report(match_level="street")], cited_urls=[CITED])),
            http=FakeHttp({CITED: district_only}))
    assert len(h.reports) == 1 and h.dropped == []
    assert len(h.corrections) == 1 and "demoted" in h.corrections[0]

    nowhere = Page("<html><body><p>Flooding in Kano.</p></body></html>")
    h2 = run(FakeClient(turn([report()], cited_urls=[CITED])), http=FakeHttp({CITED: nowhere}))
    assert h2.reports == [] and len(h2.dropped) == 1 and h2.corrections == []
