import asyncio
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.database_agent import database_agent_node
from app.agents.strategy_agent import strategy_agent_node
from app.agents.llm import is_ollama_available
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context


def fallback_node(state: dict) -> dict:
    """Direct allocation without LLM - used when Ollama is unavailable."""
    zones = state.get("zones", [])
    hospitals = state.get("enriched_hospitals") or state.get("hospitals", [])
    events = state.get("events", [])
    context = events_to_context(events) if events else {}
    G = build_graph(zones, hospitals, context)
    results = allocate_resources(G, zones, hospitals)

    applied = [
        f"{e.event_type}: {e.params}" if hasattr(e, "event_type")
        else f"{e.get('event_type')}: {e.get('params')}"
        for e in events
    ]
    return {
        "allocation_results": results,
        "fallback_used": True,
        "strategy_reasoning": "Fallback mode: Ollama unavailable, used direct allocation without agent reasoning.",
        "db_reasoning": "Fallback mode: used request data directly.",
        "events_applied": applied,
    }


def check_ollama_route(state: dict) -> str:
    """Conditional edge: route to agents or fallback based on Ollama availability."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            available = loop.run_in_executor(pool, lambda: asyncio.run(is_ollama_available()))
    except RuntimeError:
        available = asyncio.run(is_ollama_available())

    if available:
        return "database_agent"
    return "fallback"


def build_orchestrator():
    workflow = StateGraph(AgentState)

    workflow.add_node("database_agent", database_agent_node)
    workflow.add_node("strategy_agent", strategy_agent_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_conditional_entry_point(
        check_ollama_route,
        {
            "database_agent": "database_agent",
            "fallback": "fallback",
        }
    )

    workflow.add_edge("database_agent", "strategy_agent")
    workflow.add_edge("strategy_agent", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


orchestrator_graph = build_orchestrator()
