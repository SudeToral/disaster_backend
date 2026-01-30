from fastapi import FastAPI
from app.optimizer.schemas import AllocationRequest, AgentOptimizeResponse, ZoneResult, Hospital
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context
from app.agents.orchestrator import orchestrator_graph
from app.config import load_hospitals_with_resources, detect_terrain

app = FastAPI(title="MED-ARES Optimization Engine")


def _get_hospitals() -> list:
    """Load hospitals with resource inventory and convert to Hospital models."""
    return [Hospital(**h) for h in load_hospitals_with_resources()]


@app.post("/optimize")
async def optimize(req: AllocationRequest):
    hospitals = _get_hospitals()

    # Auto-detect terrain for zones that use the default "urban"
    for z in req.zones:
        if z.terrain_type == "urban":
            z.terrain_type = detect_terrain(z.lat, z.lon)

    if not req.use_agent:
        context = events_to_context(req.events) if req.events else {}
        G = build_graph(req.zones, hospitals, context)
        return allocate_resources(G, req.zones, hospitals)

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
        return AgentOptimizeResponse(
            results=[ZoneResult(**r) if isinstance(r, dict) else r
                     for r in final_state.get("allocation_results", [])],
            agent_reasoning=final_state.get("strategy_reasoning", ""),
            events_applied=final_state.get("events_applied", []),
            data_sources=[final_state.get("db_reasoning", "")],
            fallback_used=final_state.get("fallback_used", False),
        )
    except Exception as e:
        context = events_to_context(req.events) if req.events else {}
        G = build_graph(req.zones, hospitals, context)
        results = allocate_resources(G, req.zones, hospitals)
        return AgentOptimizeResponse(
            results=[ZoneResult(**r) if isinstance(r, dict) else r for r in results],
            agent_reasoning=f"Agent pipeline error: {str(e)}. Used direct allocation.",
            events_applied=[f"{ev.event_type}: {ev.params}" for ev in req.events],
            data_sources=[],
            fallback_used=True,
        )
