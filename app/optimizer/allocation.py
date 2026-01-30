import networkx as nx
import math

def distance(a, b):
    return math.sqrt((a.lat - b.lat)**2 + (a.lon - b.lon)**2)

def zone_weight(z):
    # Core driver of allocation
    return 1 + 2 * z.damage_severity + z.population_density

def build_graph(zones, hospitals):
    G = nx.Graph()

    for h in hospitals:
        G.add_node(
            f"H:{h.hospital_id}",
            capacity=h.ambulances,
            lat=h.lat,
            lon=h.lon
        )

    for z in zones:
        G.add_node(
            f"Z:{z.zone_id}",
            demand=z.medical_demand,
            weight=zone_weight(z),
            lat=z.lat,
            lon=z.lon
        )

    for h in hospitals:
        for z in zones:
            G.add_edge(
                f"H:{h.hospital_id}",
                f"Z:{z.zone_id}",
                cost=distance(h, z)
            )

    return G

def allocate_resources(G, zones, hospitals):
    results = []

    # Remaining hospital capacities
    hospital_caps = {
        f"H:{h.hospital_id}": h.ambulances
        for h in hospitals
    }

    total_supply = sum(h.ambulances for h in hospitals)
    total_weight = sum(zone_weight(z) for z in zones)

    # Compute target allocation per zone (severity-driven)
    zone_targets = {
        z.zone_id: min(
            z.medical_demand,
            round(total_supply * zone_weight(z) / total_weight)
        )
        for z in zones
    }

    # Process zones strictly by severity
    for z in sorted(zones, key=zone_weight, reverse=True):
        zone_node = f"Z:{z.zone_id}"
        remaining = zone_targets[z.zone_id]
        assignments = []

        neighbors = sorted(
            G.neighbors(zone_node),
            key=lambda h: G[h][zone_node]["cost"]
        )

        for h_node in neighbors:
            if remaining <= 0:
                break

            available = hospital_caps[h_node]
            if available <= 0:
                continue

            used = min(available, remaining)
            hospital_caps[h_node] -= used
            remaining -= used

            assignments.append({
                "hospital": h_node.split(":")[1],
                "ambulances": used
            })

        # Priority is MOCKED for now: intrinsic severity only
        w = zone_weight(z)
        if w > 3.0:
            priority = "HIGH"
        elif w > 2.0:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        confidence = round(
            (zone_targets[z.zone_id] - remaining) / z.medical_demand
            if z.medical_demand else 1,
            2
        )

        results.append({
            "zone_id": z.zone_id,
            "priority": priority,
            "confidence": confidence,
            "assigned_resources": assignments
        })

    return results
