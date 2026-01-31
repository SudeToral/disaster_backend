import json
from langchain_core.tools import tool
from app.store import data_store


@tool
def get_hospital_data(hospital_id: str) -> str:
    """Fetch detailed data for a specific hospital by its ID.

    Args:
        hospital_id: The hospital identifier, e.g. "H1".

    Returns:
        JSON string of hospital data or an error message.
    """
    h = data_store.get_hospital(hospital_id)
    if h:
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
    results = data_store.get_nearby_hospitals(lat, lon, radius_km)
    return json.dumps(results, ensure_ascii=False)


@tool
def search_hospitals_by_type(hospital_type: str) -> str:
    """Fetch all hospitals of a given type.

    Args:
        hospital_type: One of "regular", "coast_guard", or "marine_ambulance".

    Returns:
        JSON array of matching hospital objects.
    """
    results = data_store.get_hospitals_by_type(hospital_type)
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
    region = data_store.get_region_by_coords(lat, lon)
    if region:
        return json.dumps(region, ensure_ascii=False)
    return json.dumps({"region_id": "unknown", "terrain_type": "unknown", "characteristics": []})


@tool
def query_resources(hospital_id: str) -> str:
    """Fetch the resource inventory for a specific hospital.

    Args:
        hospital_id: The hospital identifier, e.g. "H1".

    Returns:
        JSON string of resource counts for the hospital.
    """
    inv = data_store.get_resources(hospital_id)
    return json.dumps(inv, ensure_ascii=False)
