import math
from app.store import data_store
from app.optimizer.schemas import Zone


# Base impact radius per disaster type (km)
BASE_IMPACT_RADIUS = {
    "earthquake": 10.0,
    "flood": 6.0,
    "fire": 4.0,
}

# Injury rate: fraction of affected population needing medical aid
INJURY_RATES = {
    "earthquake": 0.008,   # 0.8%
    "flood": 0.005,        # 0.5%
    "fire": 0.007,         # 0.7%
}

# Base damage severity ceiling per type
BASE_SEVERITY = {
    "earthquake": 0.9,
    "flood": 0.7,
    "fire": 0.8,
}

# Terrain modifies injury/damage rates per disaster type
TERRAIN_IMPACT_MODIFIER = {
    "earthquake": {"urban": 1.3, "mountain": 1.1, "coastal": 1.0, "rural": 0.8},
    "flood":      {"urban": 1.0, "mountain": 0.7, "coastal": 1.3, "rural": 1.4},
    "fire":       {"urban": 0.9, "mountain": 1.4, "coastal": 0.8, "rural": 1.2, "forest": 1.5},
}


def _circle_overlap_fraction(
    epicenter_lat: float, epicenter_lon: float, impact_radius_km: float,
    region_lat: float, region_lon: float, region_radius_km: float,
) -> float:
    """Estimate overlap fraction between impact circle and region circle."""
    deg_dist = math.sqrt(
        (epicenter_lat - region_lat) ** 2 + (epicenter_lon - region_lon) ** 2
    )
    center_dist_km = deg_dist * 111.0

    combined_radius = impact_radius_km + region_radius_km
    if center_dist_km >= combined_radius:
        return 0.0

    if center_dist_km + impact_radius_km <= region_radius_km:
        return 1.0

    overlap_fraction = 1.0 - (center_dist_km / combined_radius)
    return max(0.0, min(1.0, overlap_fraction))


def estimate_impact(
    epicenter: list[float],
    disaster_type: str,
    severity: float = 0.5,
    impact_radius_km: float | None = None,
) -> list[Zone]:
    """Estimate affected zones from epicenter + severity.

    Args:
        epicenter: [lat, lon]
        disaster_type: earthquake | flood | fire
        severity: 0-1 impact severity from external model
        impact_radius_km: optional override

    Returns:
        List of Zone objects with demand derived from real population data.
    """
    severity = max(0.0, min(1.0, severity))

    if impact_radius_km is None:
        base_radius = BASE_IMPACT_RADIUS.get(disaster_type, 5.0)
        # severity scales radius: 0.5 severity = base, 1.0 = 2x base
        impact_radius_km = base_radius * (0.5 + severity)

    injury_rate = INJURY_RATES.get(disaster_type, 0.005)
    base_dmg_severity = BASE_SEVERITY.get(disaster_type, 0.8)

    regions = data_store.get_all_regions()
    zones = []
    zone_counter = 1

    for region in regions:
        overlap = _circle_overlap_fraction(
            epicenter[0], epicenter[1], impact_radius_km,
            region["lat_center"], region["lon_center"], region["radius_km"],
        )
        if overlap <= 0:
            continue

        # Use real region population data
        total_population = region.get("total_population", 10000)
        density = region.get("population_density_per_sq_km", 1000)

        # Terrain impact modifier
        terrain = region.get("terrain_type", "urban")
        terrain_mod = TERRAIN_IMPACT_MODIFIER.get(disaster_type, {}).get(terrain, 1.0)

        # Affected population = total population × overlap × severity × terrain modifier
        affected_population = total_population * overlap * severity * terrain_mod

        # Demand calculation varies by disaster type + terrain
        if disaster_type == "fire" and terrain == "forest":
            # Forest fire: demand based on affected area, not population
            area_sq_km = overlap * math.pi * region["radius_km"] ** 2
            demand = max(1, int(area_sq_km * severity * terrain_mod * 10))
        else:
            # Population-based demand
            demand = max(1, int(affected_population * injury_rate))

        # Damage severity: base × severity × distance falloff × terrain modifier
        deg_dist = math.sqrt(
            (epicenter[0] - region["lat_center"]) ** 2 +
            (epicenter[1] - region["lon_center"]) ** 2
        )
        dist_km = deg_dist * 111.0
        distance_factor = max(0.3, 1.0 - dist_km / (impact_radius_km * 2))
        damage_severity = min(1.0, round(base_dmg_severity * severity * distance_factor * terrain_mod, 2))

        # Population density normalized (0-1)
        pop_density_norm = min(1.0, round(density / 10000, 2))

        zone = Zone(
            zone_id=f"Z{zone_counter}",
            damage_severity=damage_severity,
            population_density=pop_density_norm,
            demand=demand,
            lat=region["lat_center"],
            lon=region["lon_center"],
            terrain_type=region["terrain_type"],
        )
        zones.append(zone)
        zone_counter += 1

    return zones
