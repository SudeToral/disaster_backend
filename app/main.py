from fastapi import FastAPI
from app.optimizer.schemas import AllocationRequest, ZoneResult, Hospital
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context
from app.optimizer.impact import estimate_impact
from app.agents.orchestrator import orchestrator_graph
from app.agents.crisis_manager_agent import CrisisManagerAgent
from app.config import load_hospitals_with_resources, detect_terrain
from app.store import data_store
from app.routes.data import router as data_router
from app.routes.crisis import router as crisis_router

# Initialize data store from JSON files
data_store.load_from_json()

app = FastAPI(title="MED-ARES Optimization Engine")
app.include_router(data_router)
app.include_router(crisis_router)


def _get_hospitals() -> list:
    """Load hospitals with resource inventory and convert to Hospital models."""
    return [Hospital(**h) for h in load_hospitals_with_resources()]


@app.post("/optimize")
async def optimize(req: AllocationRequest):
    """Single endpoint that does EVERYTHING:

    1. Impact estimation (coordinates + severity → zones)
    2. Resource allocation (agent or direct)
    3. Crisis session start
    4. Full lifecycle resolution (return → reoptimize loop)

    Frontend calls this ONCE and gets the complete result.
    """
    hospitals = _get_hospitals()

    # --- 1. Impact estimation ---
    if not req.zones and req.coordinates and req.severity is not None:
        req.zones = estimate_impact(
            epicenter=req.coordinates,
            disaster_type=req.disaster_type,
            severity=req.severity,
            impact_radius_km=req.impact_radius_km,
        )

    # Auto-detect terrain for zones that use the default "urban"
    for z in req.zones:
        if z.terrain_type == "urban":
            z.terrain_type = detect_terrain(z.lat, z.lon)

    # --- 2. Allocation (agent or direct) ---
    agent_reasoning = ""
    db_reasoning = ""
    events_applied = []
    fallback_used = False

    if req.use_agent:
        initial_state = {
            "disaster_type": req.disaster_type,
            "zones": req.zones,
            "hospitals": hospitals,
            "events": req.events,
            "coordinates": req.coordinates,
            "natural_language_query": req.natural_language_query,
            "enriched_hospitals": [],
            "enriched_zones": [],
            "db_reasoning": "",
            "context": {},
            "allocation_results": [],
            "events_applied": [],
            "strategy_reasoning": "",
            "fallback_used": False,
            "error": None,
        }
        try:
            final_state = orchestrator_graph.invoke(initial_state)
            allocation_results = final_state.get("allocation_results", [])
            agent_reasoning = final_state.get("strategy_reasoning", "")
            db_reasoning = final_state.get("db_reasoning", "")
            events_applied = final_state.get("events_applied", [])
            fallback_used = final_state.get("fallback_used", False)
        except Exception as e:
            # Agent failed → fallback to direct allocation
            context = events_to_context(req.events) if req.events else {}
            G = build_graph(req.zones, hospitals, context, disaster_type=req.disaster_type)
            allocation_results = allocate_resources(G, req.zones, hospitals, disaster_type=req.disaster_type)
            agent_reasoning = f"Agent pipeline error: {str(e)}. Used direct allocation."
            events_applied = [f"{ev.event_type}: {ev.params}" for ev in req.events]
            fallback_used = True
    else:
        context = events_to_context(req.events) if req.events else {}
        G = build_graph(req.zones, hospitals, context, disaster_type=req.disaster_type)
        allocation_results = allocate_resources(G, req.zones, hospitals, disaster_type=req.disaster_type)

    # --- 3. Start crisis session ---
    crisis = data_store.start_crisis(
        disaster_type=req.disaster_type,
        zones=[z.model_dump() for z in req.zones],
        allocation_results=allocation_results,
        events=[e.model_dump() for e in req.events],
    )

    # --- 4. Auto-resolve crisis lifecycle ---
    crisis_manager = CrisisManagerAgent(max_rounds=20)
    lifecycle = crisis_manager.run_lifecycle()

    # --- 5. Build response ---
    results = [
        ZoneResult(**r) if isinstance(r, dict) else r
        for r in allocation_results
    ]

    return {
        "crisis_id": crisis["crisis_id"],
        "disaster_type": req.disaster_type,
        "initial_allocation": [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
        "lifecycle": lifecycle,
        "agent_reasoning": agent_reasoning,
        "db_reasoning": db_reasoning,
        "crisis_manager_reasoning": lifecycle.get("crisis_manager_reasoning", ""),
        "events_applied": events_applied,
        "fallback_used": fallback_used,
    }
