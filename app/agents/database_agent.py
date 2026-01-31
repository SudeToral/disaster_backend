import json
from langchain_core.messages import HumanMessage
from app.agents.llm import get_llm
from app.store import data_store


ANALYSIS_PROMPT = """You are a disaster response database analyst. You are given
region data, nearby hospitals, and resource inventories for a disaster scenario.
Provide a brief summary (3-5 sentences) of the available resources and any
concerns (e.g. flood zone but no rescue boats nearby, fire area but no fire trucks, limited capacity).
Write in English."""


def database_agent_node(state: dict) -> dict:
    """LangGraph node: fetches relevant data from the store, then asks LLM to summarize."""

    coords = state.get("coordinates")
    zone_terrains = []
    for z in state.get("zones", []):
        terrain = z.terrain_type if hasattr(z, "terrain_type") else z.get("terrain_type", "unknown")
        zone_terrains.append(terrain)

    # --- 1. Determine region ---
    region_info = {"region_id": "unknown", "terrain_type": "unknown", "characteristics": []}
    if coords:
        region = data_store.get_region_by_coords(coords[0], coords[1])
        if region:
            region_info = region

    terrain_type = region_info.get("terrain_type", "unknown")
    characteristics = region_info.get("characteristics", [])

    # --- 2. Fetch nearby hospitals (20km default, expand to 50km if few) ---
    radius_km = 20.0
    nearby = []

    if coords:
        nearby = data_store.get_nearby_hospitals(coords[0], coords[1], radius_km)

        # Expand radius if too few results
        if len(nearby) < 3:
            radius_km = 50.0
            nearby = data_store.get_nearby_hospitals(coords[0], coords[1], radius_km)
    else:
        nearby = data_store.get_hospitals_with_resources()

    # --- 3. Fetch specialized facilities based on disaster type ---
    disaster_type = state.get("disaster_type", "")
    specialized = []
    nearby_ids = {n["hospital_id"] for n in nearby}

    # Disaster-specific specialized facility types
    specialized_types = []
    if terrain_type == "coastal" or "coastal" in characteristics:
        specialized_types.append("coast_guard")
    if disaster_type == "fire":
        specialized_types.append("fire_station")
    if disaster_type in ("earthquake", "fire"):
        specialized_types.append("helibase")

    for htype in specialized_types:
        for h in data_store.get_hospitals_by_type(htype):
            if h["hospital_id"] not in nearby_ids:
                h_copy = dict(h)
                h_copy["resources"] = data_store.get_resources(h["hospital_id"])
                specialized.append(h_copy)

    all_hospitals = nearby + specialized

    # Deduplicate
    seen_ids = set()
    enriched_hospitals = []
    for h in all_hospitals:
        if h["hospital_id"] not in seen_ids:
            seen_ids.add(h["hospital_id"])
            # Ensure resources are attached
            if "resources" not in h:
                h["resources"] = data_store.get_resources(h["hospital_id"])
            enriched_hospitals.append(h)

    # --- 4. Ask LLM to summarize findings ---
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

    # --- 5. Enrich zone terrains from region data ---
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
                detected = region_info.get("terrain_type", "urban")
                region_match = data_store.get_region_by_coords(zone_lat, zone_lon)
                if region_match:
                    detected = region_match["terrain_type"]
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
