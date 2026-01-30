import json
from langchain_core.messages import HumanMessage
from app.agents.llm import get_llm
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context


ANALYSIS_PROMPT = """You are a disaster response strategy analyst. You are given
allocation results produced by an optimization algorithm. Your job is to
explain the reasoning behind the allocation and highlight important points.

Rules:
- If a road collapse or bridge outage blocked a hospital from a zone, explain
  the impact on resource distribution.
- For zones with unserved demand, explain why and suggest alternatives.
- Keep your analysis concise (3-5 sentences max).
- Write in English."""


def strategy_agent_node(state: dict) -> dict:
    """LangGraph node: runs allocation programmatically, then asks LLM to analyze."""

    # --- Use enriched data if available, else original ---
    hospitals = state.get("enriched_hospitals") or state.get("hospitals", [])
    zones = state.get("enriched_zones") or state.get("zones", [])
    events = state.get("events", [])

    # --- Normalize items to model objects ---
    from app.optimizer.schemas import Zone, Hospital

    def to_zone(z):
        if isinstance(z, Zone):
            return z
        if hasattr(z, "model_dump"):
            return z
        return Zone(**z)

    def to_hospital(h):
        if isinstance(h, Hospital):
            return h
        if hasattr(h, "model_dump"):
            return h
        return Hospital(**h)

    zones = [to_zone(z) for z in zones]
    hospitals = [to_hospital(h) for h in hospitals]

    # --- 1. Apply events to context ---
    context = events_to_context(events) if events else {}
    events_applied = []
    if events:
        for e in events:
            if hasattr(e, "event_type"):
                events_applied.append(f"{e.event_type}: {e.params}")
            else:
                events_applied.append(f"{e.get('event_type')}: {e.get('params')}")

    # --- 2. Build graph & allocate ---
    G = build_graph(zones, hospitals, context)
    allocation_results = allocate_resources(G, zones, hospitals)

    # --- 3. Ask LLM to analyze the results ---
    def serialize(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return obj

    results_json = json.dumps(allocation_results, ensure_ascii=False, indent=2)
    events_desc = json.dumps([serialize(e) for e in events], ensure_ascii=False) if events else "None"
    context_desc = json.dumps(context, ensure_ascii=False) if context else "None"

    prompt = (
        f"Disaster type: {state['disaster_type']}\n"
        f"Events: {events_desc}\n"
        f"Context applied: {context_desc}\n\n"
        f"Allocation results:\n{results_json}\n\n"
        f"Please analyze these results briefly."
    )

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=f"{ANALYSIS_PROMPT}\n\n{prompt}")])
        strategy_reasoning = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        strategy_reasoning = f"Allocation completed. LLM analysis unavailable: {e}"

    return {
        "allocation_results": allocation_results,
        "events_applied": events_applied,
        "strategy_reasoning": strategy_reasoning,
    }
