import json
import math
from langchain_core.tools import tool
from app.config import DATA_DIR


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _euclidean_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    deg = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
    return deg * 111.0


@tool
def get_hospital_data(hospital_id: str) -> str:
    """Fetch detailed data for a specific hospital by its ID.

    Args:
        hospital_id: The hospital identifier, e.g. "H1".

    Returns:
        JSON string of hospital data or an error message.
    """
    data = _load_json("hospitals.json")
    for h in data["hospitals"]:
        if h["hospital_id"] == hospital_id:
            return json.dumps(h, ensure_ascii=False)
    return f"Hospital {hospital_id} not found."


@tool
def get_nearby_hospitals(lat: float, lon: float, radius_km: float = 20.0) -> str:
    """Fetch all hospitals within a given radius of the coordinates.

    Args:
        lat: Latitude of the center point.
        lon: Longitude of the center point.
        radius_km: Search radius in kilometers. Default 20.

    Returns:
        JSON array of hospital objects within the radius, sorted by distance.
    """
    data = _load_json("hospitals.json")
    results = []
    for h in data["hospitals"]:
        dist = _euclidean_km(h["lat"], h["lon"], lat, lon)
        if dist <= radius_km:
            h_copy = dict(h)
            h_copy["distance_km"] = round(dist, 2)
            results.append(h_copy)
    results.sort(key=lambda x: x["distance_km"])
    return json.dumps(results, ensure_ascii=False)


@tool
def search_hospitals_by_type(hospital_type: str) -> str:
    """Fetch all hospitals of a given type.

    Args:
        hospital_type: One of "regular", "coast_guard", or "marine_ambulance".

    Returns:
        JSON array of matching hospital objects.
    """
    data = _load_json("hospitals.json")
    results = [h for h in data["hospitals"] if h["hospital_type"] == hospital_type]
    return json.dumps(results, ensure_ascii=False)


@tool
def get_region_data(lat: float, lon: float) -> str:
    """Determine which region the given coordinates fall in and return its data.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        JSON string of the matching region with terrain_type and characteristics.
    """
    data = _load_json("regions.json")
    best = None
    best_dist = float("inf")
    for r in data["regions"]:
        dist = _euclidean_km(r["lat_center"], r["lon_center"], lat, lon)
        if dist <= r["radius_km"] and dist < best_dist:
            best = r
            best_dist = dist
    if best:
        return json.dumps(best, ensure_ascii=False)
    return json.dumps({"region_id": "unknown", "terrain_type": "unknown", "characteristics": []})


@tool
def query_resources(hospital_id: str) -> str:
    """Fetch the resource inventory for a specific hospital.

    Args:
        hospital_id: The hospital identifier, e.g. "H1".

    Returns:
        JSON string of resource counts for the hospital.
    """
    data = _load_json("resources.json")
    inv = data.get("inventory", {}).get(hospital_id, {})
    return json.dumps(inv, ensure_ascii=False)
