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
2. **Verifier gate.** `app/agent/verifier.py` rejects reports with numbers not in the metrics, a wrong tier, or an action the tier does not allow. Keep it strict.
3. **SAM 3.1 is not used to measure site water.** A pre-build test (`docs/day0-sam-test.md`) showed it finds nothing at 2 km / 10 m and only the lagoon at 5 km. SAM is for site photos and a 5 km context map.
4. **Cloud is normal over Lagos.** Optical water uses Sentinel-2 SCL and only clear pixels; radar and JRC history are core inputs. Every layer is optional and the score rescales weights, lowering confidence.

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
  agent/verifier.py
  sam/client.py    SAM 3.1 segment() + mask union
  pipeline.py      fetch -> measure -> score
  main.py          FastAPI
backend/scripts/   assess_site.py (real run, --debug), sam_photos.py
backend/tests/     offline, synthetic layers in conftest.py
```

## Commands (from backend/, Python 3.11 venv)

```bash
pip install -e ".[dev]"
pytest -q && ruff check .
python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island" --debug
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

Meta's API is OpenAI-compatible: use the `openai` SDK with `base_url`. Docs: https://dev.meta.ai/docs (tool calling, structured output, image understanding). Cookbook: https://github.com/meta-models/meta-model-cookbook (SAM example in `07_sam/01_api_basics`). Budget: $150 Meta credit, about $0.06 per assessment.

## Known issues (first real run, Lagos Island, Oct 27)

- **Sentinel-2 water near clouds is unreliable.** In the first Lagos run, cloud shadow was labelled water 72 m from a site that JRC and radar show is 500+ m from water. Optical water now needs 100 m clearance from cloud/shadow, and distance/height use JRC history first. JRC occurrence of 0% within 500 m of Lagos Island was correct.
- **Ground height was 7.5 m**, probably rooftops: Copernicus GLO-30 is a surface model. Now uses the 10th percentile within 100 m of the site. Re-check.
- **Radar found 0% water**; the Otsu threshold hit the -14 dB clip. Urban flooding is hard for radar (building double-bounce). Consider a wet-minus-dry change measure.
- The score weights in `scoring.py` are hand-set; tune them on `data/validation_sites.csv` once it has real, sourced sites.

## Next steps

1. Fix the issues above using `--debug` output.
2. Muse Spark agent loop (`app/agent/loop.py`): Responses API with tool calling over the existing functions, structured JSON report, verifier, one retry with the verifier's errors, then fall back to metrics only.
3. SAM 3.1 on site photos (`scripts/sam_photos.py` first, then a `segment_photo` tool).
4. Next.js web app from the design (new assessment, agent at work with live steps over SSE, report, portfolio).
5. Portfolio mode with the Baseten screener; validation sites; demo video.
