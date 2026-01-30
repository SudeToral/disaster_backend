import networkx as nx
import math


# ---------------------------------------------------------------------------
# Resource type configuration
# ---------------------------------------------------------------------------

RESOURCE_TYPES = {
    "ambulance": {
        "speed_kmh": 60,
        "terrain_capability": {"urban", "rural", "mountain", "coastal"},
        "ignores_road_block": False,
    },
    "marine_ambulance": {
        "speed_kmh": 30,
        "terrain_capability": {"coastal"},
        "ignores_road_block": True,
    },
    "helicopter": {
        "speed_kmh": 200,
        "terrain_capability": {"urban", "rural", "mountain", "coastal"},
        "ignores_road_block": True,
    },
    "search_and_rescue": {
        "speed_kmh": 20,
        "terrain_capability": {"urban", "mountain"},
        "ignores_road_block": False,
    },
}

DEFAULT_SPEED_KMH = 60

# Preferred resource order per terrain type (specialists first)
TERRAIN_PREFERRED_RESOURCES = {
    "coastal": ["marine_ambulance", "ambulance", "helicopter"],
    "mountain": ["search_and_rescue", "ambulance", "helicopter"],
    "urban": ["ambulance", "search_and_rescue", "helicopter"],
    "rural": ["ambulance", "helicopter"],
}


# ---------------------------------------------------------------------------
# Events → context
# ---------------------------------------------------------------------------

def events_to_context(events) -> dict:
    """Convert a list of Event objects (or dicts) into a context dict for build_graph.

    Supported event types:
    - road_collapse / bridge_out  → blocked_roads
    - weather                     → weather_penalty
    - flood                       → weather_penalty (high value)
    """
    context: dict = {
        "blocked_roads": [],
        "weather_penalty": 1.0,
        "road_block_penalty": 10.0,
    }
    for event in events:
        if hasattr(event, "event_type"):
            etype = event.event_type
            params = event.params
        else:
            etype = event.get("event_type", "")
            params = event.get("params", {})

        if etype in ("road_collapse", "bridge_out"):
            context["blocked_roads"].append({
                "hospital_id": params.get("hospital_id", ""),
                "zone_id": params.get("zone_id", ""),
            })
        elif etype == "weather":
            context["weather_penalty"] = params.get("penalty", 2.0)
        elif etype == "flood":
            factor = params.get("severity_factor", 5.0)
            context["weather_penalty"] = max(context["weather_penalty"], factor)

    return context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distance(a, b):
    return math.sqrt((a.lat - b.lat)**2 + (a.lon - b.lon)**2)


def zone_weight(z):
    return 1 + 2 * z.damage_severity + z.population_density


def is_road_blocked(h, z, context):
    blocked = context.get("blocked_roads", [])
    return any(
        b["hospital_id"] == h.hospital_id and
        b["zone_id"] == z.zone_id
        for b in blocked
    )


def can_serve_terrain(resource_type: str, terrain_type: str) -> bool:
    """Check if a resource type can operate on the given terrain."""
    rt_info = RESOURCE_TYPES.get(resource_type, {})
    return terrain_type in rt_info.get("terrain_capability", set())


def _get_hospital_resources(h) -> dict:
    """Extract resource inventory from a Hospital object.

    If h.resources is populated (agent path with enrichment), use it.
    Otherwise fall back to {"ambulance": h.ambulances} (direct path).
    """
    if hasattr(h, "resources") and h.resources:
        return dict(h.resources)
    return {"ambulance": h.ambulances}


# ---------------------------------------------------------------------------
# Edge cost (resource-type aware)
# ---------------------------------------------------------------------------

def compute_edge_cost(h, z, context, resource_type="ambulance"):
    """Compute effective cost of sending resource_type from hospital h to zone z.

    Cost model:
      effective_distance = euclidean_distance / speed_factor
      base_cost = effective_distance / zone_weight(z)
      Then apply penalties (road_block, weather) unless resource is exempt.
    """
    rt_info = RESOURCE_TYPES.get(resource_type, {})
    speed = rt_info.get("speed_kmh", DEFAULT_SPEED_KMH)
    speed_factor = speed / DEFAULT_SPEED_KMH

    effective_distance = distance(h, z) / speed_factor
    base_cost = effective_distance / zone_weight(z)

    multiplier = 1.0

    if is_road_blocked(h, z, context):
        if not rt_info.get("ignores_road_block", False):
            multiplier *= context.get("road_block_penalty", 10.0)

    multiplier *= context.get("weather_penalty", 1.0)

    return base_cost * multiplier


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(zones, hospitals, context=None):
    context = context or {}
    G = nx.Graph()

    for h in hospitals:
        resources = _get_hospital_resources(h)
        G.add_node(
            f"H:{h.hospital_id}",
            resources=resources,
            capacity=h.ambulances,
            lat=h.lat,
            lon=h.lon,
        )

    for z in zones:
        G.add_node(
            f"Z:{z.zone_id}",
            demand=z.medical_demand,
            weight=zone_weight(z),
            terrain_type=z.terrain_type,
            lat=z.lat,
            lon=z.lon,
        )

    for h in hospitals:
        for z in zones:
            edge_data = {"costs": {}}
            road_blocked = is_road_blocked(h, z, context)
            for rtype in _get_hospital_resources(h):
                if not can_serve_terrain(rtype, z.terrain_type):
                    continue
                # Road blocked → only air/sea resources can pass
                rt_info = RESOURCE_TYPES.get(rtype, {})
                if road_blocked and not rt_info.get("ignores_road_block", False):
                    continue
                cost = compute_edge_cost(h, z, context, resource_type=rtype)
                edge_data["costs"][rtype] = cost

            # Legacy cost field (for find_shortest_path etc.)
            if "ambulance" in edge_data["costs"]:
                edge_data["cost"] = edge_data["costs"]["ambulance"]
            elif edge_data["costs"]:
                edge_data["cost"] = min(edge_data["costs"].values())
            else:
                edge_data["cost"] = float("inf")

            G.add_edge(f"H:{h.hospital_id}", f"Z:{z.zone_id}", **edge_data)

    return G


# ---------------------------------------------------------------------------
# Multi-resource allocation
# ---------------------------------------------------------------------------

def allocate_resources(G, zones, hospitals):
    results = []

    # Mutable inventory: {"H:H1": {"ambulance": 15, "helicopter": 1}, ...}
    hospital_inv = {}
    for h in hospitals:
        h_node = f"H:{h.hospital_id}"
        hospital_inv[h_node] = dict(G.nodes[h_node].get("resources", {"ambulance": h.ambulances}))

    # Total supply = sum of ALL resource units
    total_supply = sum(
        sum(inv.values()) for inv in hospital_inv.values()
    )
    total_weight = sum(zone_weight(z) for z in zones)

    # Severity-driven target per zone
    zone_targets = {
        z.zone_id: min(
            z.medical_demand,
            int(total_supply * zone_weight(z) / total_weight)
        )
        for z in zones
    }

    # Process zones by severity (highest first)
    for z in sorted(zones, key=zone_weight, reverse=True):
        zone_node = f"Z:{z.zone_id}"
        remaining = zone_targets[z.zone_id]
        assignments = {}  # {"H1": {"ambulance": 5, "helicopter": 1}, ...}

        terrain = z.terrain_type
        resource_order = TERRAIN_PREFERRED_RESOURCES.get(terrain, ["ambulance", "helicopter"])

        for rtype in resource_order:
            if remaining <= 0:
                break

            # Hospitals that have this resource type AND can serve this zone's terrain
            candidates = []
            for h_node in G.neighbors(zone_node):
                edge_costs = G[h_node][zone_node].get("costs", {})
                if rtype in edge_costs and hospital_inv[h_node].get(rtype, 0) > 0:
                    candidates.append((h_node, edge_costs[rtype]))

            candidates.sort(key=lambda x: x[1])

            for h_node, _cost in candidates:
                if remaining <= 0:
                    break
                available = hospital_inv[h_node].get(rtype, 0)
                if available <= 0:
                    continue
                used = min(available, remaining)
                hospital_inv[h_node][rtype] -= used
                remaining -= used

                h_id = h_node.split(":")[1]
                if h_id not in assignments:
                    assignments[h_id] = {}
                assignments[h_id][rtype] = assignments[h_id].get(rtype, 0) + used

        # Build AssignedResource list
        assigned_list = []
        for h_id, breakdown in assignments.items():
            assigned_list.append({
                "hospital": h_id,
                "ambulances": breakdown.get("ambulance", 0),
                "resource_breakdown": breakdown,
            })

        # Priority
        w = zone_weight(z)
        if w > 3.0:
            priority = "HIGH"
        elif w > 2.0:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        # Confidence
        total_assigned = sum(sum(bd.values()) for bd in assignments.values())
        confidence = round(
            total_assigned / z.medical_demand if z.medical_demand else 1,
            2
        )

        # Resource summary for the zone
        resource_summary = {}
        for bd in assignments.values():
            for rt, count in bd.items():
                resource_summary[rt] = resource_summary.get(rt, 0) + count

        results.append({
            "zone_id": z.zone_id,
            "priority": priority,
            "confidence": confidence,
            "assigned_resources": assigned_list,
            "unserved": z.medical_demand - total_assigned,
            "resource_summary": resource_summary,
        })

    return results
