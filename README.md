# FloodLens

**Flood-risk checks on loan collateral for African microfinance lenders.**

A loan officer pins a shop, farm or house, adds a few site photos, and FloodLens returns a risk tier, the evidence behind it and a lending action: approve, approve with flood insurance, adjust terms, or decline. Every number in the report is measured from open satellite and weather data and checked before it is shown.

Built for the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (Indie track, *Build for Your World*).

> **Built during the hackathon.** All code in this repository was written from Oct 27, 2026 onward, during the event. No code is reused from any earlier project.

## How it works

| Step | What happens | Done by |
| --- | --- | --- |
| Gather | Clearest Sentinel-2 scene over the site, Sentinel-1 radar after the wettest recent days, JRC water history since 1984, Copernicus elevation, Open-Meteo rainfall | Code (`app/geo`) |
| Measure | Water now and on wet dates inside 250 m / 500 m rings, distance and height above nearest water, peak rainfall | Code (`app/metrics.py`) |
| See | Site photos: standing water, water stains, drains, door step vs road. Wide-area water map on a 5 km tile | SAM 3.1 (`app/sam`) |
| Score | Fixed, documented formula, 0-100, tier and allowed actions | Code (`app/scoring.py`) |
| Explain | Reads the evidence and writes the report for the loan officer | Muse Spark 1.3 (next) |
| Verify | Rejects any number not in the measured metrics, a wrong tier, or an action the tier does not allow | Code (`app/agent/verifier.py`) |

### Why SAM 3.1 reads photos, not satellite tiles

A pre-build test on central Lagos (Sentinel-2, 24 Jan 2026) found SAM 3.1 returned the lagoon as "water" in a 5 km true-colour tile (IoU 0.66 against Sentinel-2's own water class), but nothing at 2 km, nothing for buildings or roads, and nothing on water-index or radar images. At 10 m a building is one or two pixels. So site water is measured in code, and SAM works where it is strong: street-level photos and the wide context map. Details: [docs/day0-sam-test.md](docs/day0-sam-test.md).

## Run it

Python 3.11+.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp ../.env.example ../.env    # add MODEL_API_KEY for SAM / Muse Spark steps
pytest -q                     # offline tests, synthetic data

# real assessment from open data (no API key needed for this step)
python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island"

# API
uvicorn app.main:app --reload  # POST /assessments {"lat": 6.4541, "lon": 3.3947}

# SAM 3.1 on site photos (needs MODEL_API_KEY)
python scripts/sam_photos.py photos/*.jpg
```

## Data

Open data only: Copernicus Sentinel-1 and Sentinel-2 and Copernicus DEM (via Microsoft Planetary Computer), JRC Global Surface Water, Open-Meteo. No bank or customer data is used; demo loans are synthetic.

FloodLens is decision support for credit officers, not a property survey.

## Status

- [x] Open-data water metrics, scoring, verifier, API skeleton, offline tests (Oct 27)
- [ ] Muse Spark agent loop with tool calls and structured report
- [ ] SAM 3.1 on site photos
- [ ] Web app (Next.js + MapLibre)
- [ ] Portfolio mode (Muse Spark Open screener on Baseten)
- [ ] Validation on known flooded and dry Lagos sites

## License

MIT
