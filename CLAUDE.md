# FloodLens: notes for Claude

Flood-risk checks on loan collateral for Nigerian microfinance lenders. Entry for the Meta Global AI Developer Hackathon (Indie track). Owner: Daniel (MLOps/AI engineer, Lagos).

## Hackathon rules that constrain the code

- Submission due **Wed Nov 4, 9:00 PM ET** (Thu 3:00 AM Lagos). Treat Wed evening Lagos as the real deadline.
- Repo is public. All code must be written during the event (from Oct 27, 2026). Do not paste in code from other projects.
- Meta models must be in the core path: SAM 3.1 and Muse Spark via Meta Model API.
- Open-licensed data only. **Never use real bank or customer data**; demo loans are synthetic. Free Meta credits mean prompts may be used for product improvement.
- Not a "simple image analyzer": the multi-step agent and the verifier are the point. No medical or political features.
- A 3-minute demo video must show only features built here.

## What FloodLens claims (changed Oct 27, after the baseline run)

FloodLens does **not** claim to tell a flooding Lagos street from a dry one using satellite
data. It was measured: AUC 0.578 against 30 random built-up points, no better than chance.
Drainage decides which street floods, and drainage is not visible from orbit. The product
is now three parts, and each is only allowed to claim what it can support:

1. **Area screen (satellite, open data).** Coastal and riverine exposure for the
   *neighbourhood*, plus terrain and surface proxies. Weight 0.50. Never phrase its output
   as a verdict on an address.
2. **Local flood history (`app/agent/history.py`).** News reports of flooding at this
   street, estate or district in the last 5 years, found with Meta's `web_search` tool.
   A report is kept only if it clears three checks: the URL is one the API itself
   retrieved (an `open_page` record or a `url_citation` annotation — never a URL the model
   merely typed), the event date parses and sits inside the window, and the fetched page
   actually names one of our reverse-geocoded place names. Weight 0.40, and it is a
   **positive signal only**: reports raise the score, none found does not clear a site,
   because Lagos news covers some neighbourhoods far better than others.
3. **Rainfall.** Weight 0.10. A ~10 km grid, so it separates years, not addresses.

Three rules that fall out of this, all enforced in code:
- **Area-level news reports score zero.** 27 of 30 random Lagos addresses turn up one, so
  they carry no information about an address. They are still listed as context.
- **A report is street-level only if the article names the street our reverse geocode gave
  us.** The model's own `match_level` is corrected against the fetched page, up or down.
- **Where no street-level report was found, the report must say so**, in these words: "No
  street-level flood reports found. This does not mean the site is safe." The verifier
  rejects any report without it.

## SCORING IS FROZEN (Oct 27)

The weights, ramps and match-level weights in `app/scoring.py` and `MATCH_WEIGHT` in
`app/agent/history.py` are **not to be changed** without **new, independent validation
sites** — sites not drawn from news reports, and including dry ones. They are reasoned
priors, and the current set (4 flood sites, 0 dry, a 30-point contaminated baseline) cannot
support fitting them. Tuning on it would turn the only honest test into a training set.

`docs/validation.md` is the log of every run and what changed after it. Read it before
proposing a scoring change. `test_the_three_groups_carry_their_documented_weight` pins the
split so it cannot drift by accident. The right fix for a bad score is a better
measurement — see the end of `docs/validation.md` for the four that would actually help.

## Core design decisions (do not undo without asking)

1. **Code measures, the LLM explains.** Every number in a report comes from `app/metrics.py`. The score is a fixed formula in `app/scoring.py`. Muse Spark never invents or changes numbers or tiers.
2. **Verifier gate.** `app/agent/verifier.py` rejects reports with numbers not in the metrics, a wrong tier, or an action the tier does not allow. Keep it strict. Dates are the one special case: a `YYYY-MM-DD` token is allowed only if some metric's label or source states it, and dates are cut from the text before the number rule runs, so `2025-06-14` can never make `6` or `14` quotable. Nothing else in a label or source is quotable — a source reading `threshold -14.0 dB` does not make `14%` a measured value.
3. **SAM 3.1 is not used to measure site water.** A pre-build test (`docs/day0-sam-test.md`) showed it finds nothing at 2 km / 10 m and only the lagoon at 5 km. SAM is for site photos and a 5 km context map.
4. **The web search is never taken on trust.** `history.py` checks a report against the
   API's own retrieval record (`grounded_urls`), then against the live page, which must
   name one of the places the reverse geocode gave us. The model's wording and its quote
   are recorded as secondary checks and are never decisive; its `match_level` is corrected
   against the page. A page we cannot read is "unknown" and the report is kept, because a
   paywall is not evidence against it. Every drop is recorded with its reason: "we found
   nothing" and "the model made something up and we caught it" must never look the same.
   Results are cached per place in `data/cache/history/`.
5. **The agent explains, it never measures.** `app/agent/loop.py` hands Muse Spark the metrics and the evidence PNGs and takes back one `submit_report` call. Numbers, tier and allowed actions are fixed before the model is called; `submit_report`'s action enum is the tier's allowed actions, so a disallowed action cannot even be expressed. The tool is `strict: true`, so the API enforces the field types and only content errors should reach the verifier. Every report goes through the verifier: one rejection is retried with the errors, a second returns a metrics-only report with `explained: false`. `_report_schema` must stay inside the strict subset or the call 400s; a test guards that.
6. **Cloud is normal over Lagos.** Optical water uses Sentinel-2 SCL and only clear pixels; radar and JRC history are core inputs. Every layer is optional and the score rescales weights, lowering confidence.

## Layout

```
backend/app/
  config.py        env settings, model IDs, endpoints
  schemas.py       Metric, Score, Assessment, Tier, Action (pydantic)
  geo/aoi.py       Site, Grid, bbox, ring masks (250 m / 500 m)
  geo/raster.py    read a COG window around a site
  geo/sources.py   Planetary Computer + Open-Meteo fetchers -> SiteLayers
  geo/water.py     pure numpy: NDWI, SCL, radar Otsu, distance, ground height
  metrics.py       SiteLayers -> list[Metric]
  scoring.py       FEATURES, tiers, ALLOWED_ACTIONS
  evidence.py      SiteLayers -> the named PNGs (optical, water_now, water_history, water_wet)
  agent/verifier.py
  agent/history.py Nominatim reverse geocode -> web_search -> cited, dated flood reports
  agent/loop.py    Muse Spark: get_metrics / view_evidence / submit_report, verify, retry, fallback
  sam/client.py    SAM 3.1 segment() + mask union
  pipeline.py      fetch -> measure -> score
  main.py          FastAPI
backend/scripts/   assess_site.py (real run, --debug, --explain), validate.py, sam_photos.py
data/              validation_sites.csv (flood/dry ground truth, one wet_date per flood site)
backend/tests/     offline, synthetic layers in conftest.py
```

## Commands (from backend/, Python 3.11 venv)

```bash
pip install -e ".[dev]"
pytest -q && ruff check .
python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island" --debug --explain
python scripts/validate.py                       # open data only, free
python scripts/validate.py --baseline 30         # + 30 random built-up Lagos points
python scripts/validate.py --baseline 30 --history   # + news flood history (costs money)
uvicorn app.main:app --reload
```

Tests must stay offline (no network, no API keys). Add a test with every change to metrics, scoring or the verifier.

## Models and keys

Keys live in `.env` at the repo root (never commit it; see `.env.example`).

| Use | Model ID | Endpoint |
| --- | --- | --- |
| Agent, vision, report | `muse-spark-1.3` (dev: `muse-spark-1.3-contributor`, 150 RPM) | `https://api.meta.ai/v1`, key `MODEL_API_KEY` |
| Segmentation | `sam-3.1` (Responses API, streamed; parse with `meta_sam_parser`) | same |
| Portfolio screener | `meta-models/muse-spark-open-1026` (exact ID) | `https://inference.baseten.co/v1`, key `BASETEN_API_KEY` |

Prices (USD per 1M tokens, `PRICES` in `app/agent/loop.py`, looked up by the model actually called):

| Model | Input | Cached input | Output |
| --- | --- | --- | --- |
| `muse-spark-1.3` | 1.25 | 0.15 | 4.25 |
| `muse-spark-1.3-contributor` | 0.10 | 0.002 | 0.20 |

Reasoning effort is `FLOODLENS_REASONING_EFFORT` (default `medium`), sent as `reasoning={"effort": ...}`; reasoning tokens come back in `output_tokens_details.reasoning_tokens` and are counted in the run. `max` is not available on the Contributor variant. Low was not cheaper in practice on Lagos Island: it needed a verifier retry and so used more input tokens than medium.

Meta's API is OpenAI-compatible: use the `openai` SDK with `base_url`. Docs: https://dev.meta.ai/docs (tool calling, structured output, image understanding). Cookbook: https://github.com/meta-models/meta-model-cookbook (SAM example in `07_sam/01_api_basics`). Budget: $150 Meta credit, about $0.06 per assessment.

## Known issues (Lagos Island and the validation run, Oct 27)

- **Sentinel-2 water near clouds is unreliable.** In the first Lagos run, cloud shadow was labelled water 72 m from a site that JRC and radar show is 500+ m from water. Optical water now needs 100 m clearance from cloud/shadow, and distance/height use JRC history first. JRC occurrence of 0% within 500 m of Lagos Island was correct.
- **Tall-building shadows read as water.** `m_water_now_250` reported 14.6% water within 250 m of Lagos Island, where JRC has seen no water within 500 m and the nearest regular water is 749 m away. The pixels are not near cloud, so they are building shadows that Sentinel-2's SCL labels water. The metric now counts an optical water pixel only where JRC occurrence is above 0, and says "(unconfirmed)" in its label when there is no JRC history to check against.
- **Ground height was 7.5 m**, probably rooftops: Copernicus GLO-30 is a surface model. Now uses the 10th percentile within 100 m of the site. Re-check.
- **CONFIRMED: radar cannot see urban flooding.** Not just the Lagos Island dry-date run: on the validation set, every flood site read 0.0-0.9% water from Sentinel-1 *on the flood date itself* (`wet_date` pointed it at the right scene, and the Otsu threshold still hit the -14 dB clip). Building double-bounce defeats it. `m_wet_water_250` holds 0.10 weight and contributes almost nothing in the city; a wet-minus-dry change measure is the next thing to try, and until it works radar is dead weight on dense sites.
- **The formula measured the wrong kind of flood (fixed, needs re-checking).** The first validation run scored four flood-verified sites 0.0 / 1.2 / 2.1 / 30.9. Every feature measured proximity to a body of water, so only riverine and coastal risk could score; three of the four sites have no water within 2 km and flood because rain cannot drain away. Added `m_local_low_spot`, `m_built_up_500` and `m_wetland_500`, made "no water in view" saturate `m_dist_water` at the 1 km grid edge instead of dropping it, and split the weights 0.45 pluvial / 0.45 riverine / 0.10 rain as reasoned priors (see the `scoring.py` docstring; they are **not** fitted to these four sites). The same four sites now score 28.0 / 35.2 / 36.7 / 39.4, all Moderate. Still unproven: with no dry sites in the set there is nothing to show it does not simply score everything Moderate.
- **`m_wetland_500` contributes nothing so far** (0.0-0.6% on all four sites). WorldCover finds almost no wetland inside dense Lagos, so the 0.10 weight may be buying nothing; re-check once there are dry and peri-urban sites.
- **Ojota's -9.2 m low spot is real terrain, not a bridge deck.** Checked by saving the DEM window (`docs/ojota-low-spot.png`, site marked): the high ground is three broad blobs 450-1500 m across at 60-73% fill, i.e. hillsides, while a 30 m flyover deck would be ~3 px wide and linear. The site's own 100 m neighbourhood tops out at 5.8 m, so there is no deck near it. It sits on the floor of a ~500 m wide valley with plateaus ~18 m higher on both sides, which is a genuine pluvial signal. `low_spot_m` was therefore left on the median; Ojota is also the only site where p50 and p25 of the 1 km neighbourhood diverge much (-9.24 vs -1.88 m; the others differ by under 1.3 m), so this is terrain, not a systematic DSM bias.
- **AUC 0.856 at Run 4 (0.578 satellite-only, 0.622 at Run 3). Still n=3, and read `docs/validation.md` before believing it:** the baseline changed sample (5 Ogun points removed), most of the gain is the baseline falling rather than flood sites rising, and one site carries much of it. The number that genuinely improved is the random street-level report rate, 5/30 -> 2/30, after the page check started matching our own street names instead of the model's wording.
- **(Run 3) With local history, AUC 0.622 (was 0.578 on satellite alone).** Flood sites now land at the 100th, 57th and 30th percentile of 30 random built-up points. The gain is real but it rests on one site: Ago Palace Way has 3 street-level events and tops the city; Ojota barely moves; Lekki Phase 1 got **0 reports and fell to Low**.
- **Area-level news matching is close to worthless. 27 of 30 random Lagos addresses have *some* flood report.** Reports naming a whole district match almost everywhere, so the 0.3 area weight mostly adds noise. The selective signal is street level: 5/30 random points versus 2/4 flood sites. Consider dropping area matches from the count, or weighting them near zero.
- **Lekki's miss is our reverse geocoding, not the search.** Nominatim at zoom 18 returned `Olarenwaju Ninalowo Crescent, Maroko, Eti Osa`, so the search never looked for "Lekki Phase 1" and missed the well-documented 2024 flood. Fix: a second, coarser lookup for the neighbourhood name, applied to baseline points too so the comparison stays fair. Do **not** feed the CSV `label` into the search: those labels came from the news headlines, which would be maximal circularity.
- **Verification earns its place.** Across 70 submitted reports the URL check caught one the model typed that the search tool never retrieved (a real-looking Punch link). One API timeout was recorded, not cached, and not counted as "no floods here".
- **Transient Planetary Computer failures were silently degrading scores.** A first run had two validation sites quietly lose the whole coastal/riverine group, rescaling 0.25 of the weight onto the rest and inflating one score from 56.5 to 69.2. `validate.py` now prints a `!` warning per degraded site and writes `notes` to the CSV. Always check for that line before comparing runs.
- **The satellite-only rebalance raised the scores but did not make them discriminate. AUC 0.578.** Against 30 random built-up Lagos points (`scripts/validate.py --baseline 30`, seed 20261027), the three well-pinned flood sites land at the 43rd, 60th and 70th percentile; a random point outscores a flood site in 42.2% of pairs, and the highest-scoring point in the city (60.2) is a random one, not a verified flood. Per-feature, nothing separates: `m_water_history_500`, `m_wet_water_250` and `m_wetland_500` are ~0 on every site in the city (0.35 of the weight doing no work), `m_dist_water` is saturated at 1000 m for 21 of 30 baseline points, and the two features that do vary vary just as much in the baseline (`m_local_low_spot` baseline p25 -3.5 m vs flood -1.0/-1.7/-9.2; `m_built_up_500` baseline median 89% vs flood 61-98%). The next move is a measurement that separates, not another weighting. Caveat: a random built-up point is **not** a dry site, so this AUC understates true performance by an unknown amount -- it needs real dry sites to be read properly.
- **`m_local_low_spot`'s ramp saturates too early for Lagos.** 0 to -3 m means a quarter of random built-up points already sit at full risk. Consider expressing it as a percentile of the local terrain distribution rather than in absolute metres.
- **`m_max_daily_rain` cannot separate sites in one city.** Open-Meteo's archive is a ~10 km grid, so one storm gives every Lagos address nearly the same number; `scripts/validate.py --baseline` prints the spread across the baseline so this stays visible. Its threshold is **not** to be tuned on the validation set.
- **`m_max_daily_rain` may have the wrong threshold.** Its prior treats 60 mm/day as the start of risk, but these floods happened on 56-75 mm days, so it reads near-zero risk for days that actually flooded Lagos. Lowering it would be tuning on the validation set, so it is left alone and flagged.
- The score weights in `scoring.py` are hand-set; tune them with `scripts/validate.py` on `data/validation_sites.csv`. The flood sites are in and sourced; **the two dry sites are still missing**, so there is nothing to separate against yet.
- **Two validation pins are weak.** `isashi_light_estate` is the rounded OSM village node (the street and Light Estate are not in OSM) and `lekki_phase_1` is a neighbourhood centroid wider than the 2 km grid. Both need a hand-placed pin before any weight tuning.

## Next steps

1. Fix the issues above using `--debug` output.
2. ~~Muse Spark agent loop~~ built: `app/agent/loop.py`, Responses API, 3 tools, verifier with one retry then a metrics-only fallback. Still to do: stream the steps over SSE and add `segment_photo`.
3. SAM 3.1 on site photos (`scripts/sam_photos.py` first, then a `segment_photo` tool).
4. Replace the weak validation pins, add the dry sites, then tune the weights on the table from `scripts/validate.py`.
5. Next.js web app from the design (new assessment, agent at work with live steps over SSE, report, portfolio).
6. Portfolio mode with the Baseten screener; demo video.
