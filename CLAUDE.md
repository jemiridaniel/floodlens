# FloodLens: notes for Claude

Flood-risk checks on loan collateral for Nigerian microfinance lenders. Entry for the Meta Global AI Developer Hackathon (Indie track). Owner: Daniel (MLOps/AI engineer, Lagos).

## Hackathon rules that constrain the code

- Submission due **Wed Nov 4, 9:00 PM ET** (Thu 3:00 AM Lagos). Treat Wed evening Lagos as the real deadline.
- Repo is public. All code must be written during the event (from Oct 27, 2026). Do not paste in code from other projects.
- Meta models must be in the core path: SAM 3.1 and Muse Spark via Meta Model API.
- Open-licensed data only. **Never use real bank or customer data**; demo loans are synthetic. Free Meta credits mean prompts may be used for product improvement.
- Not a "simple image analyzer": the multi-step agent and the verifier are the point. No medical or political features.
- A 3-minute demo video must show only features built here.

## Core design decisions (do not undo without asking)

1. **Code measures, the LLM explains.** Every number in a report comes from `app/metrics.py`. The score is a fixed formula in `app/scoring.py`. Muse Spark never invents or changes numbers or tiers.
2. **Verifier gate.** `app/agent/verifier.py` rejects reports with numbers not in the metrics, a wrong tier, or an action the tier does not allow. Keep it strict. Dates are the one special case: a `YYYY-MM-DD` token is allowed only if some metric's label or source states it, and dates are cut from the text before the number rule runs, so `2025-06-14` can never make `6` or `14` quotable. Nothing else in a label or source is quotable — a source reading `threshold -14.0 dB` does not make `14%` a measured value.
3. **SAM 3.1 is not used to measure site water.** A pre-build test (`docs/day0-sam-test.md`) showed it finds nothing at 2 km / 10 m and only the lagoon at 5 km. SAM is for site photos and a 5 km context map.
4. **The agent explains, it never measures.** `app/agent/loop.py` hands Muse Spark the metrics and the evidence PNGs and takes back one `submit_report` call. Numbers, tier and allowed actions are fixed before the model is called; `submit_report`'s action enum is the tier's allowed actions, so a disallowed action cannot even be expressed. The tool is `strict: true`, so the API enforces the field types and only content errors should reach the verifier. Every report goes through the verifier: one rejection is retried with the errors, a second returns a metrics-only report with `explained: false`. `_report_schema` must stay inside the strict subset or the call 400s; a test guards that.
5. **Cloud is normal over Lagos.** Optical water uses Sentinel-2 SCL and only clear pixels; radar and JRC history are core inputs. Every layer is optional and the score rescales weights, lowering confidence.

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
python scripts/validate.py          # every validation site, open data only, free
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

## Known issues (first real run, Lagos Island, Oct 27)

- **Sentinel-2 water near clouds is unreliable.** In the first Lagos run, cloud shadow was labelled water 72 m from a site that JRC and radar show is 500+ m from water. Optical water now needs 100 m clearance from cloud/shadow, and distance/height use JRC history first. JRC occurrence of 0% within 500 m of Lagos Island was correct.
- **Tall-building shadows read as water.** `m_water_now_250` reported 14.6% water within 250 m of Lagos Island, where JRC has seen no water within 500 m and the nearest regular water is 749 m away. The pixels are not near cloud, so they are building shadows that Sentinel-2's SCL labels water. The metric now counts an optical water pixel only where JRC occurrence is above 0, and says "(unconfirmed)" in its label when there is no JRC history to check against.
- **Ground height was 7.5 m**, probably rooftops: Copernicus GLO-30 is a surface model. Now uses the 10th percentile within 100 m of the site. Re-check.
- **Radar found 0% water**; the Otsu threshold hit the -14 dB clip. Urban flooding is hard for radar (building double-bounce). Consider a wet-minus-dry change measure.
- The score weights in `scoring.py` are hand-set; tune them with `scripts/validate.py` on `data/validation_sites.csv`. The flood sites are in and sourced; **the two dry sites are still missing**, so there is nothing to separate against yet.
- **Two validation pins are weak.** `isashi_light_estate` is the rounded OSM village node (the street and Light Estate are not in OSM) and `lekki_phase_1` is a neighbourhood centroid wider than the 2 km grid. Both need a hand-placed pin before any weight tuning.

## Next steps

1. Fix the issues above using `--debug` output.
2. ~~Muse Spark agent loop~~ built: `app/agent/loop.py`, Responses API, 3 tools, verifier with one retry then a metrics-only fallback. Still to do: stream the steps over SSE and add `segment_photo`.
3. SAM 3.1 on site photos (`scripts/sam_photos.py` first, then a `segment_photo` tool).
4. Replace the weak validation pins, add the dry sites, then tune the weights on the table from `scripts/validate.py`.
5. Next.js web app from the design (new assessment, agent at work with live steps over SSE, report, portfolio).
6. Portfolio mode with the Baseten screener; demo video.
