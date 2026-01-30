import json
import math
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 30

DATA_DIR = Path(__file__).parent.parent / "data"

FALLBACK_ENABLED = True


def load_hospitals():
    """Load all hospitals from data/hospitals.json and return as list of dicts."""
    path = DATA_DIR / "hospitals.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["hospitals"]


def load_resources():
    """Load resource inventory and type definitions from data/resources.json."""
    path = DATA_DIR / "resources.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hospitals_with_resources():
    """Load hospitals and merge resource inventory from resources.json."""
    hospitals = load_hospitals()
    resources_data = load_resources()
    inventory = resources_data.get("inventory", {})
    for h in hospitals:
        h["resources"] = inventory.get(h["hospital_id"], {})
    return hospitals


def _euclidean_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    deg = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
    return deg * 111.0


def detect_terrain(lat: float, lon: float) -> str:
    """Detect terrain type from coordinates using regions.json.

    Returns the terrain_type of the closest matching region,
    or "urban" as default if no region matches.
    """
    path = DATA_DIR / "regions.json"
    with open(path, "r", encoding="utf-8") as f:
        regions = json.load(f)["regions"]

    best = None
    best_dist = float("inf")
    for r in regions:
        dist = _euclidean_km(r["lat_center"], r["lon_center"], lat, lon)
        if dist <= r["radius_km"] and dist < best_dist:
            best = r
            best_dist = dist

    return best["terrain_type"] if best else "urban"
