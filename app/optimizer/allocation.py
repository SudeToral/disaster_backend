import networkx as nx
import math


# ---------------------------------------------------------------------------
# Resource type configuration
# ---------------------------------------------------------------------------

RESOURCE_TYPES = {
    "ambulance": {
        "speed_kmh": 60,
        "capacity_per_unit": 1,
        "terrain_capability": {"urban", "rural", "mountain", "coastal"},
        "ignores_road_block": False,
        "applicable_disasters": {"earthquake", "flood", "fire"},
    },
    "helicopter": {
        "speed_kmh": 200,
        "capacity_per_unit": 5,
        "terrain_capability": {"urban", "rural", "mountain", "coastal", "forest"},
        "ignores_road_block": True,
        "applicable_disasters": {"earthquake", "flood", "fire"},
    },
    "search_and_rescue": {
        "speed_kmh": 20,
        "capacity_per_unit": 4,
        "terrain_capability": {"urban", "mountain", "rural"},
        "ignores_road_block": False,
        "applicable_disasters": {"earthquake", "flood"},
    },
    "fire_truck": {
        "speed_kmh": 50,
        "capacity_per_unit": 10,
        "terrain_capability": {"urban", "rural", "coastal"},
        "ignores_road_block": False,
        "applicable_disasters": {"fire"},
    },
    "fire_helicopter": {
        "speed_kmh": 180,
        "capacity_per_unit": 15,
        "terrain_capability": {"urban", "rural", "mountain", "coastal", "forest"},
        "ignores_road_block": True,
        "applicable_disasters": {"fire"},
    },
    "rescue_boat": {
        "speed_kmh": 25,
        "capacity_per_unit": 6,
        "terrain_capability": {"coastal", "rural"},
        "ignores_road_block": True,
        "applicable_disasters": {"flood"},
    },
    "heavy_rescue": {
        "speed_kmh": 15,
        "capacity_per_unit": 3,
        "terrain_capability": {"urban", "mountain", "rural"},
        "ignores_road_block": False,
        "applicable_disasters": {"earthquake"},
    },
    "food_supply": {
        "speed_kmh": 40,
        "capacity_per_unit": 0,
        "terrain_capability": {"urban", "rural", "mountain", "coastal"},
        "ignores_road_block": False,
        "applicable_disasters": {"earthquake"},
    },
}

DEFAULT_SPEED_KMH = 60

# Preferred resource order per disaster+terrain combination
DISASTER_TERRAIN_PREFERRED_RESOURCES = {
    "earthquake": {
        "urban":    ["heavy_rescue", "search_and_rescue", "ambulance", "helicopter", "food_supply"],
        "mountain": ["helicopter", "search_and_rescue", "heavy_rescue", "ambulance", "food_supply"],
        "coastal":  ["ambulance", "heavy_rescue", "helicopter", "food_supply"],
        "rural":    ["heavy_rescue", "ambulance", "food_supply", "search_and_rescue", "helicopter"],
    },
    "flood": {
        "urban":    ["rescue_boat", "search_and_rescue", "ambulance", "helicopter"],
        "mountain": ["helicopter", "search_and_rescue", "ambulance"],
        "coastal":  ["rescue_boat", "helicopter", "ambulance", "search_and_rescue"],
        "rural":    ["rescue_boat", "ambulance", "search_and_rescue", "helicopter"],
    },
    "fire": {
        "urban":    ["fire_truck", "ambulance", "fire_helicopter", "helicopter"],
        "mountain": ["fire_helicopter", "helicopter", "ambulance"],
        "coastal":  ["fire_truck", "fire_helicopter", "ambulance", "helicopter"],
        "rural":    ["fire_truck", "fire_helicopter", "ambulance", "helicopter"],
        "forest":   ["fire_helicopter", "helicopter"],
    },
}

# Fallback: terrain-only preference (used when disaster_type is unknown)
TERRAIN_PREFERRED_RESOURCES = {
    "coastal":  ["ambulance", "rescue_boat", "helicopter"],
    "mountain": ["search_and_rescue", "ambulance", "helicopter"],
    "urban":    ["ambulance", "search_and_rescue", "helicopter"],
    "rural":    ["ambulance", "helicopter"],
    "forest":   ["helicopter", "fire_helicopter"],
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


def is_applicable_for_disaster(resource_type: str, disaster_type: str) -> bool:
    """Check if a resource type is relevant for the given disaster."""
    if not disaster_type:
        return True
    rt_info = RESOURCE_TYPES.get(resource_type, {})
    applicable = rt_info.get("applicable_disasters")
    if not applicable:
        return True
    return disaster_type in applicable


def get_preferred_resources(disaster_type: str, terrain_type: str) -> list[str]:
    """Get the preferred resource order for a disaster+terrain combination."""
    if disaster_type and disaster_type in DISASTER_TERRAIN_PREFERRED_RESOURCES:
        terrain_prefs = DISASTER_TERRAIN_PREFERRED_RESOURCES[disaster_type]
        if terrain_type in terrain_prefs:
            return terrain_prefs[terrain_type]
    return TERRAIN_PREFERRED_RESOURCES.get(terrain_type, ["ambulance", "helicopter"])


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

def build_graph(zones, hospitals, context=None, disaster_type=None):
    context = context or {}
    G = nx.Graph()

    for h in hospitals:
        resources = _get_hospital_resources(h)
        # Filter resources applicable for this disaster type
        if disaster_type:
            resources = {
                rtype: count for rtype, count in resources.items()
                if is_applicable_for_disaster(rtype, disaster_type)
            }
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
            demand=z.demand,
            weight=zone_weight(z),
            terrain_type=z.terrain_type,
            lat=z.lat,
            lon=z.lon,
        )

    for h in hospitals:
        for z in zones:
            edge_data = {"costs": {}}
            road_blocked = is_road_blocked(h, z, context)
            h_resources = G.nodes[f"H:{h.hospital_id}"].get("resources", {})
            for rtype in h_resources:
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

def allocate_resources(G, zones, hospitals, disaster_type=None):
    results = []

    # Mutable inventory: {"H:H1": {"ambulance": 15, "helicopter": 1}, ...}
    hospital_inv = {}
    for h in hospitals:
        h_node = f"H:{h.hospital_id}"
        hospital_inv[h_node] = dict(G.nodes[h_node].get("resources", {"ambulance": h.ambulances}))

    # Total supply = sum of demand-serving capacity across all resources
    total_supply = sum(
        count * RESOURCE_TYPES.get(rtype, {}).get("capacity_per_unit", 1)
        for inv in hospital_inv.values()
        for rtype, count in inv.items()
    )
    total_weight = sum(zone_weight(z) for z in zones)

    # Severity-driven target per zone
    zone_targets = {
        z.zone_id: min(
            z.demand,
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
        resource_order = get_preferred_resources(disaster_type, terrain)

        for rtype in resource_order:
            if remaining <= 0:
                break

            capacity = RESOURCE_TYPES.get(rtype, {}).get("capacity_per_unit", 1)
            if capacity <= 0:
                continue

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
                units_needed = math.ceil(remaining / capacity)
                used = min(available, units_needed)
                hospital_inv[h_node][rtype] -= used
                remaining = max(0, remaining - used * capacity)

                h_id = h_node.split(":")[1]
                if h_id not in assignments:
                    assignments[h_id] = {}
                assignments[h_id][rtype] = assignments[h_id].get(rtype, 0) + used

        # Build AssignedResource list
        assigned_list = []
        for h_id, breakdown in assignments.items():
            assigned_list.append({
                "hospital": h_id,
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

        # Confidence (based on demand covered, not raw unit count)
        total_demand_covered = sum(
            count * RESOURCE_TYPES.get(rt, {}).get("capacity_per_unit", 1)
            for bd in assignments.values()
            for rt, count in bd.items()
        )
        confidence = round(
            min(1.0, total_demand_covered / z.demand) if z.demand else 1,
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
            "planned_coverage": confidence,
            "assigned_resources": assigned_list,
            "allocation_gap": max(0, z.demand - total_demand_covered),
            "resource_summary": resource_summary,
        })

    return results
