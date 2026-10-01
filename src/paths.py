"""Central place for project paths, so every script agrees on where things live."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
CHARTS_DIR = PROJECT_ROOT / "charts"

for d in (RAW_DIR, DERIVED_DIR, MODELS_DIR, CACHE_DIR, CHARTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
