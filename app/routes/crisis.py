from fastapi import APIRouter, HTTPException
from app.store import data_store
from app.optimizer.schemas import (
    BatchReturnRequest, Zone, Hospital, NextRoundRequest,
)
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context, zone_weight
from app.config import load_hospitals_with_resources

router = APIRouter(prefix="/crisis", tags=["crisis"])


@router.get("/status")
def crisis_status():
    """Get current crisis state."""
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")

    total_initial = sum(zs["initial_demand"] for zs in crisis["zone_states"].values())
    total_remaining = sum(zs["remaining_demand"] for zs in crisis["zone_states"].values())
    total_served = sum(zs["served"] for zs in crisis["zone_states"].values())
    active_dispatches = [d for d in crisis["dispatches"] if d["status"] == "dispatched"]

    return {
        **crisis,
        "summary": {
            "total_initial_demand": total_initial,
            "total_remaining_demand": total_remaining,
            "total_served": total_served,
            "active_dispatches": len(active_dispatches),
            "optimization_round": crisis["optimization_round"],
            "is_resolved": data_store.is_crisis_resolved(),
        },
    }


@router.post("/return")
def batch_return(req: BatchReturnRequest):
    """Return multiple dispatches at once. Only updates demand — does NOT re-optimize.

    Call POST /crisis/reoptimize after this to trigger a new allocation round.
    """
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")

    returned = []
    errors = []
    for item in req.returns:
        try:
            data_store.return_dispatch(item.dispatch_id, item.served_demand)
            returned.append(item.dispatch_id)
        except ValueError as e:
            errors.append({"dispatch_id": item.dispatch_id, "error": str(e)})

    total_remaining = sum(
        zs["remaining_demand"] for zs in crisis["zone_states"].values()
    )

    return {
        "returned": returned,
        "errors": errors,
        "total_remaining_demand": total_remaining,
        "is_resolved": data_store.is_crisis_resolved(),
    }


@router.post("/reoptimize")
def reoptimize():
    """Re-optimize resource allocation for remaining unserved demand.

    Call this after POST /crisis/return to trigger a new dispatch round.
    """
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")

    if data_store.is_crisis_resolved():
        final = data_store.end_crisis()
        return {
            "status": "crisis_resolved",
            "message": "All zone demands have been met",
            "final_state": final,
        }

    unserved_zones = data_store.get_unserved_zones()
    if not unserved_zones:
        return {"status": "no_unserved_demand"}

    zones = [Zone(**z) for z in unserved_zones]
    hospitals = [Hospital(**h) for h in load_hospitals_with_resources()]

    disaster_type = crisis.get("disaster_type")
    context = events_to_context(crisis.get("events", [])) if crisis.get("events") else {}
    G = build_graph(zones, hospitals, context, disaster_type=disaster_type)
    new_results = allocate_resources(G, zones, hospitals, disaster_type=disaster_type)

    data_store.record_reoptimization(new_results)

    return {
        "status": "reoptimized",
        "optimization_round": crisis["optimization_round"],
        "new_allocations": new_results,
        "remaining_zones": [
            {
                "zone_id": z.zone_id,
                "remaining_demand": z.demand,
                "priority": "HIGH" if zone_weight(z) > 3.0
                           else "MEDIUM" if zone_weight(z) > 2.0
                           else "LOW",
            }
            for z in zones
        ],
    }


@router.post("/next-round")
def next_round(req: NextRoundRequest = None):
    """Execute the next round of crisis management interactively.

    Accepts optional new events that get appended to the crisis before
    reoptimization. Returns round results, LLM reasoning, and prompts
    for the next action.
    """
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")

    if req and req.events:
        new_event_dicts = [e.model_dump() for e in req.events]
        data_store.append_events(new_event_dicts)

    from app.agents.crisis_manager_agent import CrisisManagerAgent
    agent = CrisisManagerAgent()
    return agent.run_single_round()


@router.get("/dispatches")
def list_dispatches():
    """List all dispatches in the active crisis."""
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")
    return {
        "dispatches": crisis["dispatches"],
        "active": [d for d in crisis["dispatches"] if d["status"] == "dispatched"],
        "returned": [d for d in crisis["dispatches"] if d["status"] == "returned"],
    }


@router.post("/auto-resolve")
def auto_resolve():
    """Run the full crisis lifecycle automatically.

    Uses CrisisManagerAgent to repeatedly return dispatches and
    reoptimize until all zone demands are met or max rounds reached.
    """
    crisis = data_store.get_crisis()
    if not crisis:
        raise HTTPException(404, "No active crisis")

    from app.agents.crisis_manager_agent import CrisisManagerAgent
    agent = CrisisManagerAgent(max_rounds=20)
    result = agent.run_lifecycle()
    return result


@router.post("/end")
def end_crisis():
    """Manually end the active crisis."""
    final = data_store.end_crisis()
    if not final:
        raise HTTPException(404, "No active crisis")
    return {"status": "ended", "final_state": final}
