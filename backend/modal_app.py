"""FloodLens on Modal: the FastAPI app as one warm, single-container web endpoint.

Why one container. Jobs, their event streams and the rate limiter all live in memory, so
a second replica would answer /assessments/{id}/events for an id it has never heard of.
max_containers=1 makes that impossible; min_containers=1 keeps it warm so a demo does not
open on a cold start. The real fix is Postgres and a broker, and this is honest about
being a hackathon deployment rather than pretending otherwise.

    modal deploy backend/modal_app.py
    modal volume put floodlens-data data/cache/demo /cache/demo   # seed the demo sites
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "floodlens"
DATA = Path("/data")
VOLUME_RELOAD_S = 60  # pick up anything seeded from the CLI without needing a redeploy

image = (
    modal.Image.debian_slim(python_version="3.11")
    # rasterio and pyproj need the system GDAL/PROJ libraries
    .apt_install("libgdal-dev", "gdal-bin", "libproj-dev", "proj-bin", "libgeos-dev")
    .pip_install(
        "fastapi>=0.115", "uvicorn[standard]>=0.30", "pydantic>=2.7", "httpx>=0.27",
        "openai>=1.40", "meta_sam_parser>=0.1", "numpy>=1.26", "scipy>=1.12",
        "rasterio>=1.3.10", "pyproj>=3.6", "pystac-client>=0.8", "planetary-computer>=1.0",
        "Pillow>=10.3", "python-dotenv>=1.0", "python-multipart>=0.0.9", "anyio>=4.0",
    )
    .add_local_python_source("app")
)

# Cache and uploads outlive a container. The demo cache lives here too, so a replay is
# instant and free whichever container answers.
volume = modal.Volume.from_name("floodlens-data", create_if_missing=True)

# MODEL_API_KEY and BASETEN_API_KEY. Never in code, never in the image.
secret = modal.Secret.from_name("floodlens-keys")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[secret],
    volumes={str(DATA): volume},
    timeout=900,  # a cold assessment is ~2 min; this is headroom, not an expectation
    min_containers=1,  # stay warm: a demo should not wait on a cold start
    max_containers=1,  # in-memory jobs and streams: see the module docstring
    scaledown_window=1200,
)
@modal.concurrent(max_inputs=50)  # SSE streams sit open; they must not each pin a container
@modal.asgi_app()
def web():
    import os

    # A Volume written by another process (modal volume put) is not visible to a running
    # container until it reloads. Without this the seeded demo sites would silently run
    # for real, which costs money and defeats the point of seeding them.
    volume.reload()

    # Point the app's on-disk state at the Volume before anything imports it.
    os.environ.setdefault("FLOODLENS_DEMO_CACHE", "1")
    os.environ.setdefault("FLOODLENS_WEB_ORIGIN", "https://floodlens-psi.vercel.app")
    for name, path in (("FLOODLENS_CACHE_DIR", DATA / "cache"),
                       ("FLOODLENS_UPLOAD_DIR", DATA / "uploads")):
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))

    # Seeding the Volume from the CLI is invisible to this container until it reloads, and
    # a redeploy is skipped when the code has not changed, so "modal volume put" alone
    # silently did nothing. A cheap periodic reload means seeding works on its own: once
    # it did not, a fresh search overwrote a seeded entry with a thinner result and the
    # demo quietly lost a flood event.
    import time as _time

    from app.main import app as fastapi_app

    last = {"reload": _time.monotonic()}

    @fastapi_app.middleware("http")
    async def refresh_volume(request, call_next):
        if _time.monotonic() - last["reload"] > VOLUME_RELOAD_S:
            last["reload"] = _time.monotonic()
            try:
                volume.reload()
            except Exception:  # a stale view is far better than a failed request
                pass
        return await call_next(request)

    return fastapi_app
