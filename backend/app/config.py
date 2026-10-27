"""Runtime settings, read from the environment (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

META_BASE_URL = "https://api.meta.ai/v1"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True)
class Settings:
    model_api_key: str | None
    baseten_api_key: str | None
    spark_model: str
    sam_model: str = "sam-3.1"
    screener_model: str = "meta-models/muse-spark-open-1026"


def get_settings() -> Settings:
    return Settings(
        model_api_key=os.environ.get("MODEL_API_KEY") or None,
        baseten_api_key=os.environ.get("BASETEN_API_KEY") or None,
        spark_model=os.environ.get("FLOODLENS_SPARK_MODEL", "muse-spark-1.3"),
    )
