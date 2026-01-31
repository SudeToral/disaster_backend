import json
import networkx as nx
from langchain_core.tools import tool
from app.optimizer.allocation import build_graph, allocate_resources
from app.optimizer.schemas import Zone, Hospital

# Module-level graph context (set per-request by the strategy agent node)
_current_graph: nx.Graph | None = None
_current_zones: list[Zone] = []
_current_hospitals: list[Hospital] = []


def set_graph_context(graph: nx.Graph, zones: list[Zone], hospitals: list[Hospital]):
    """Called by strategy_agent_node to set context before agent runs."""
    global _current_graph, _current_zones, _current_hospitals
    _current_graph = graph
    _current_zones = zones
    _current_hospitals = hospitals


def get_graph_context():
    """Return current graph context."""
    return _current_graph, _current_zones, _current_hospitals


@tool
def build_graph_with_context(zones_json: str, hospitals_json: str, context_json: str = "{}") -> str:
    """Build a bipartite graph connecting hospitals to disaster zones with context.

    The context can contain blocked_roads, weather_penalty, road_block_penalty
    to modify edge costs.

    Args:
        zones_json: JSON array of zone objects with zone_id, damage_severity,
                    population_density, demand, lat, lon.
        hospitals_json: JSON array of hospital objects with hospital_id,
                       ambulances, lat, lon.
        context_json: JSON object with optional keys: blocked_roads (array),
                     weather_penalty (float), road_block_penalty (float).

    Returns:
        Summary string with node and edge counts.
    """
    global _current_graph, _current_zones, _current_hospitals
    zones_data = json.loads(zones_json)
    hospitals_data = json.loads(hospitals_json)
    context = json.loads(context_json)

    _current_zones = [Zone(**z) for z in zones_data]
    _current_hospitals = [Hospital(**h) for h in hospitals_data]
    _current_graph = build_graph(_current_zones, _current_hospitals, context)

    return (
        f"Graph built: {_current_graph.number_of_nodes()} nodes, "
        f"{_current_graph.number_of_edges()} edges. "
        f"Context applied: {list(context.keys()) if context else 'none'}."
    )


@tool
def apply_events_to_context(events_json: str) -> str:
    """Convert a list of events into a context dict for the graph builder.

    Supported event types:
    - road_collapse: adds to blocked_roads list. Params: hospital_id, zone_id.
    - weather: sets weather_penalty. Params: penalty (float).
    - flood: sets weather_penalty to a high value. Params: severity_factor (float).
    - bridge_out: adds to blocked_roads. Params: hospital_id, zone_id.

    Args:
        events_json: JSON array of event objects with event_type and params.

    Returns:
        JSON string of the resulting context dict.
    """
    events = json.loads(events_json)
    context: dict = {
        "blocked_roads": [],
        "weather_penalty": 1.0,
        "road_block_penalty": 10.0,
    }

    applied = []
    for event in events:
        etype = event.get("event_type", "")
        params = event.get("params", {})

        if etype in ("road_collapse", "bridge_out"):
            context["blocked_roads"].append({
                "hospital_id": params.get("hospital_id", ""),
                "zone_id": params.get("zone_id", ""),
            })
            applied.append(f"{etype}: {params.get('hospital_id')}->{params.get('zone_id')}")

        elif etype == "weather":
            context["weather_penalty"] = params.get("penalty", 2.0)
            applied.append(f"weather: penalty={context['weather_penalty']}")

        elif etype == "flood":
            factor = params.get("severity_factor", 5.0)
            context["weather_penalty"] = max(context["weather_penalty"], factor)
            applied.append(f"flood: severity_factor={factor}")

    return json.dumps({
        "context": context,
        "events_applied": applied,
    }, ensure_ascii=False)


@tool
def run_allocation() -> str:
    """Run the severity-weighted greedy allocation algorithm on the current graph.

    The graph must have been built first using build_graph_with_context.

    Returns:
        JSON array of allocation results per zone.
    """
    global _current_graph, _current_zones, _current_hospitals
    if _current_graph is None:
        return "Error: Graph not built yet. Call build_graph_with_context first."
    results = allocate_resources(_current_graph, _current_zones, _current_hospitals)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def find_shortest_path(source: str, target: str) -> str:
    """Find the shortest path between two nodes in the graph.

    Args:
        source: Source node ID, e.g. "H:H1" or "Z:Z1".
        target: Target node ID, e.g. "H:H2" or "Z:Z2".

    Returns:
        JSON with path and total cost, or error message.
    """
    global _current_graph
    if _current_graph is None:
        return "Error: Graph not built yet."
    try:
        path = nx.shortest_path(_current_graph, source, target, weight="cost")
        cost = nx.shortest_path_length(_current_graph, source, target, weight="cost")
        return json.dumps({"path": path, "cost": round(cost, 4)})
    except nx.NetworkXNoPath:
        return f"No path exists between {source} and {target}."
    except nx.NodeNotFound as e:
        return f"Node not found: {e}"


@tool
def get_graph_info() -> str:
    """Get summary information about the current graph state.

    Returns:
        JSON with node count, edge count, connected components, and isolated nodes.
    """
    global _current_graph
    if _current_graph is None:
        return "Error: Graph not built yet."
    isolated = list(nx.isolates(_current_graph))
    components = nx.number_connected_components(_current_graph)
    return json.dumps({
        "nodes": _current_graph.number_of_nodes(),
        "edges": _current_graph.number_of_edges(),
        "connected_components": components,
        "isolated_nodes": isolated,
    })
