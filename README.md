# FloodLens

**Flood-risk checks on loan collateral for African microfinance lenders.**

A loan officer pins a shop, farm or house, adds a few site photos, and FloodLens returns a risk tier, the evidence behind it and a lending action: approve, approve with flood insurance, adjust terms, or decline. Every number in the report is measured from open satellite and weather data and checked before it is shown.

Built for the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (Indie track, *Build for Your World*).

> **Built during the hackathon.** All code in this repository was written from Oct 27, 2026 onward, during the event. No code is reused from any earlier project.

## Try it

**[floodlens-psi.vercel.app](https://floodlens-psi.vercel.app)**

Three saved sites load instantly and cost nothing: **Ago Palace Way, Okota** (High, held
there by the street-level evidence rule), **Lekki Phase 1** (Low, and it shows what a site
with no street-level reports looks like) and **Ogba**. Pin anywhere else and it runs for
real, which takes a minute or two.

New assessments are capped at 10 per hour per address, because each one spends real money
on a web search and two or three model calls. The saved sites are never counted against
that, so the demo keeps working after you hit the cap.

API: `https://jemiridaniel--floodlens-web.us-east.modal.run` ([/health](https://jemiridaniel--floodlens-web.us-east.modal.run/health), [/limits](https://jemiridaniel--floodlens-web.us-east.modal.run/limits)).

## Deployment

| Part | Where | Notes |
| --- | --- | --- |
| Backend | Modal (`backend/modal_app.py`) | One warm container. Jobs, event streams and the rate limiter are in memory, so `max_containers=1` is load-bearing, not tuning. Cache and uploads on a Volume; keys in a Modal Secret. |
| Frontend | Vercel (`web/`, root directory `web`) | Connected to this repo: every push to `main` redeploys. `NEXT_PUBLIC_API_URL` is the only setting, and no key ever reaches the browser. |

```bash
modal deploy backend/modal_app.py
modal volume put floodlens-data data/cache/demo /cache/demo --force   # seed the demo sites
modal volume put floodlens-data data/cache/history /cache/history --force
```

The Volume is re-read at most once a minute, so seeding takes effect without a redeploy.

## Running it

Two parts: a FastAPI backend and a Next.js frontend. The backend holds every credential;
the frontend has one setting and no keys.

### 1. Keys

Copy `.env.example` to `.env` **in the repository root** (not in `backend/`) and fill in
`MODEL_API_KEY`. `BASETEN_API_KEY` is optional.

### 2. Backend

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check .            # 168 tests, all offline
uvicorn app.main:app --reload        # http://localhost:8000
```

Useful flags:

```bash
# replay finished assessments instantly, for demos (only ever replays a real run)
FLOODLENS_DEMO_CACHE=1 uvicorn app.main:app --reload

# one site on the command line, with the agent's explanation
python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island" --explain

# the validation set, and the random-baseline comparison (see docs/validation.md)
python scripts/validate.py --baseline 30 --history
```

### 3. Frontend

```bash
cd web
cp .env.local.example .env.local     # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                          # http://localhost:3000
```

The backend allows requests from `localhost:3000` only. Start the backend first: the
frontend has nothing to show without it.

### What you will see

1. **New assessment** — tap the map to drop the pin (rings show the 250 m and 500 m
   measurement radii), add the address, loan amount, tenor, a note and up to five photos.
   The typed address widens the news search; photo location data is stripped before
   anything is stored.
2. **Working on it** — the six steps as they run, with the evidence images appearing as
   they are produced, elapsed time and running cost. The area screen and the flood history
   run at the same time, so the first two steps start together.
3. **Risk report** — tier, score, recommended action and whether the agent's numbers were
   verified; drivers grouped by evidence type; flood-report cards with the match level,
   date, quote and a link to the source; site photos with SAM overlays; and what the
   report cannot tell you.

Portfolio is a placeholder; the batch screener comes after the single-site flow is solid.

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
