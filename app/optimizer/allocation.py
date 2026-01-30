from ortools.graph.python import min_cost_flow

import math


def distance(a, b):
    # Simple Euclidean distance (MVP-safe)
    return math.sqrt((a.lat - b.lat)**2 + (a.lon - b.lon)**2)


def compute_priority_weight(zone):
    return 1 + 2 * zone.damage_severity + zone.population_density


def optimize_allocation(zones, hospitals):
    mcf = min_cost_flow.SimpleMinCostFlow()


    # Node indexing
    source = 0
    hospital_offset = 1
    zone_offset = hospital_offset + len(hospitals)
    sink = zone_offset + len(zones)

    # Source → Hospitals
    for j, h in enumerate(hospitals):
        mcf.add_arc_with_capacity_and_unit_cost(
            source,
            hospital_offset + j,
            h.ambulances,
            0
        )

    # Hospitals → Zones
    for j, h in enumerate(hospitals):
        for i, z in enumerate(zones):
            cost = int(
                distance(h, z) * 100 / compute_priority_weight(z)
            )
            mcf.add_arc_with_capacity_and_unit_cost(
                hospital_offset + j,
                zone_offset + i,
                999,
                cost
            )

    # Zones → Sink
    for i, z in enumerate(zones):
        mcf.add_arc_with_capacity_and_unit_cost(
            zone_offset + i,
            sink,
            z.medical_demand,
            0
        )

    # Supplies
    mcf.set_node_supply(source, sum(h.ambulances for h in hospitals))
    mcf.set_node_supply(sink, -sum(z.medical_demand for z in zones))

    for i in range(hospital_offset, sink):
        mcf.set_node_supply(i, 0)

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError("Optimization failed")

    return mcf, hospital_offset, zone_offset

def build_zone_results(mcf, zones, hospitals, hospital_offset, zone_offset):
    results = []

    for i, zone in enumerate(zones):
        allocated = 0
        assignments = []

        for arc in range(mcf.NumArcs()):
            if mcf.Head(arc) == zone_offset + i:
                flow = mcf.Flow(arc)
                if flow > 0:
                    hospital_idx = mcf.Tail(arc) - hospital_offset
                    assignments.append({
                        "hospital": hospitals[hospital_idx].hospital_id,
                        "ambulances": flow
                    })
                    allocated += flow

        served_ratio = allocated / zone.medical_demand if zone.medical_demand else 1

        priority = (
            "HIGH" if served_ratio < 0.6 else
            "MEDIUM" if served_ratio < 0.9 else
            "LOW"
        )

        confidence = round(1 - abs(served_ratio - 1), 2)

        results.append({
            "zone_id": zone.zone_id,
            "priority": priority,
            "confidence": confidence,
            "assigned_resources": assignments
        })

    return results
