import json
import math
from langchain_core.messages import HumanMessage
from app.agents.llm import get_llm
from app.config import DATA_DIR


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _euclidean_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    deg = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
    return deg * 111.0


ANALYSIS_PROMPT = """You are a disaster response database analyst. You are given
region data, nearby hospitals, and resource inventories for a disaster scenario.
Provide a brief summary (3-5 sentences) of the available resources and any
concerns (e.g. coastal region but no marine ambulances nearby, limited capacity).
Write in English."""


def database_agent_node(state: dict) -> dict:
    """LangGraph node: fetches relevant data programmatically, then asks LLM to summarize."""

    coords = state.get("coordinates")
    zone_terrains = []
    for z in state.get("zones", []):
        terrain = z.terrain_type if hasattr(z, "terrain_type") else z.get("terrain_type", "unknown")
        zone_terrains.append(terrain)

    # --- 1. Determine region ---
    region_info = {"region_id": "unknown", "terrain_type": "unknown", "characteristics": []}
    if coords:
        regions_data = _load_json("regions.json")
        best = None
        best_dist = float("inf")
        for r in regions_data["regions"]:
            dist = _euclidean_km(r["lat_center"], r["lon_center"], coords[0], coords[1])
            if dist <= r["radius_km"] and dist < best_dist:
                best = r
                best_dist = dist
        if best:
            region_info = best

    terrain_type = region_info.get("terrain_type", "unknown")
    characteristics = region_info.get("characteristics", [])

    # --- 2. Fetch nearby hospitals (20km default, expand to 50km if few) ---
    hospitals_data = _load_json("hospitals.json")
    radius_km = 20.0
    nearby = []

    if coords:
        for h in hospitals_data["hospitals"]:
            dist = _euclidean_km(h["lat"], h["lon"], coords[0], coords[1])
            if dist <= radius_km:
                h_copy = dict(h)
                h_copy["distance_km"] = round(dist, 2)
                nearby.append(h_copy)

        # Expand radius if too few results
        if len(nearby) < 3:
            radius_km = 50.0
            nearby = []
            for h in hospitals_data["hospitals"]:
                dist = _euclidean_km(h["lat"], h["lon"], coords[0], coords[1])
                if dist <= radius_km:
                    h_copy = dict(h)
                    h_copy["distance_km"] = round(dist, 2)
                    nearby.append(h_copy)

        nearby.sort(key=lambda x: x["distance_km"])
    else:
        nearby = hospitals_data["hospitals"]

    # --- 3. If coastal, also fetch specialized facilities ---
    specialized = []
    if terrain_type == "coastal" or "coastal" in characteristics:
        for h in hospitals_data["hospitals"]:
            if h.get("hospital_type") in ("coast_guard", "marine_ambulance"):
                if h["hospital_id"] not in {n["hospital_id"] for n in nearby}:
                    specialized.append(h)

    # --- 4. Fetch resource inventory for all relevant hospitals ---
    resources_data = _load_json("resources.json")
    inventory = resources_data.get("inventory", {})

    all_hospitals = nearby + specialized
    for h in all_hospitals:
        h["resources"] = inventory.get(h["hospital_id"], {})

    # Deduplicate
    seen_ids = set()
    enriched_hospitals = []
    for h in all_hospitals:
        if h["hospital_id"] not in seen_ids:
            seen_ids.add(h["hospital_id"])
            enriched_hospitals.append(h)

    # --- 5. Ask LLM to summarize findings ---
    summary_data = {
        "region": region_info,
        "terrain_type": terrain_type,
        "nearby_hospitals_count": len(nearby),
        "specialized_facilities_count": len(specialized),
        "search_radius_km": radius_km,
        "hospitals": [
            {
                "id": h["hospital_id"],
                "name": h.get("name", ""),
                "type": h.get("hospital_type", "regular"),
                "ambulances": h.get("ambulances", 0),
                "distance_km": h.get("distance_km", "N/A"),
                "resources": h.get("resources", {}),
            }
            for h in enriched_hospitals
        ],
    }

    prompt = (
        f"Disaster type: {state['disaster_type']}\n"
        f"Region: {json.dumps(region_info, ensure_ascii=False)}\n"
        f"Zone terrains: {zone_terrains}\n\n"
        f"Data fetched:\n{json.dumps(summary_data, ensure_ascii=False, indent=2)}\n\n"
        f"Please summarize the available resources briefly."
    )

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=f"{ANALYSIS_PROMPT}\n\n{prompt}")])
        db_reasoning = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        db_reasoning = f"Data fetched: {len(enriched_hospitals)} hospitals in {terrain_type} region. LLM summary unavailable: {e}"

    # --- 6. Enrich zone terrains from region data ---
    enriched_zones = []
    for z in state.get("zones", []):
        if hasattr(z, "model_copy"):
            zc = z.model_copy()
        else:
            zc = z
        # Auto-detect terrain if zone uses default "urban"
        if (hasattr(zc, "terrain_type") and zc.terrain_type == "urban") or \
           (isinstance(zc, dict) and zc.get("terrain_type", "urban") == "urban"):
            zone_lat = zc.lat if hasattr(zc, "lat") else zc.get("lat")
            zone_lon = zc.lon if hasattr(zc, "lon") else zc.get("lon")
            if zone_lat and zone_lon:
                # Find matching region for zone coordinates
                detected = region_info.get("terrain_type", "urban")
                # Check per-zone: zone might be in a different region than epicenter
                for r in _load_json("regions.json")["regions"]:
                    d = _euclidean_km(r["lat_center"], r["lon_center"], zone_lat, zone_lon)
                    if d <= r["radius_km"]:
                        detected = r["terrain_type"]
                        break
                if hasattr(zc, "terrain_type"):
                    zc.terrain_type = detected
                elif isinstance(zc, dict):
                    zc["terrain_type"] = detected
        enriched_zones.append(zc)

    return {
        "enriched_hospitals": enriched_hospitals,
        "enriched_zones": enriched_zones,
        "db_reasoning": db_reasoning,
    }
