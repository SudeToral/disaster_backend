import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"


def _euclidean_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    deg = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
    return deg * 111.0


class DataStore:
    """In-memory data store for hospitals, regions, and resources.

    Initialized from JSON files at startup. All reads/writes go through
    this store so data can be updated at runtime via API endpoints.
    """

    def __init__(self):
        self.hospitals: dict[str, dict] = {}
        self.regions: dict[str, dict] = {}
        self.resource_inventory: dict[str, dict] = {}
        self.resource_types: list[dict] = []
        self.active_crisis: Optional[dict] = None

    def load_from_json(self):
        """Load initial data from JSON files in data/."""
        # Hospitals
        with open(DATA_DIR / "hospitals.json", "r", encoding="utf-8") as f:
            for h in json.load(f)["hospitals"]:
                self.hospitals[h["hospital_id"]] = h

        # Regions
        with open(DATA_DIR / "regions.json", "r", encoding="utf-8") as f:
            for r in json.load(f)["regions"]:
                self.regions[r["region_id"]] = r

        # Resources
        with open(DATA_DIR / "resources.json", "r", encoding="utf-8") as f:
            res_data = json.load(f)
            self.resource_types = res_data.get("resource_types", [])
            self.resource_inventory = res_data.get("inventory", {})

    # ---- Hospitals ----

    def get_all_hospitals(self) -> list[dict]:
        return list(self.hospitals.values())

    def get_hospital(self, hospital_id: str) -> Optional[dict]:
        return self.hospitals.get(hospital_id)

    def update_hospital(self, hospital_id: str, data: dict) -> Optional[dict]:
        if hospital_id not in self.hospitals:
            return None
        self.hospitals[hospital_id].update(data)
        return self.hospitals[hospital_id]

    def get_nearby_hospitals(self, lat: float, lon: float, radius_km: float = 20.0) -> list[dict]:
        results = []
        for h in self.hospitals.values():
            dist = _euclidean_km(h["lat"], h["lon"], lat, lon)
            if dist <= radius_km:
                h_copy = dict(h)
                h_copy["distance_km"] = round(dist, 2)
                h_copy["resources"] = dict(self.resource_inventory.get(h["hospital_id"], {}))
                results.append(h_copy)
        results.sort(key=lambda x: x["distance_km"])
        return results

    def get_hospitals_by_type(self, hospital_type: str) -> list[dict]:
        return [h for h in self.hospitals.values() if h.get("hospital_type") == hospital_type]

    def get_hospitals_with_resources(self) -> list[dict]:
        result = []
        for h in self.hospitals.values():
            h_copy = dict(h)
            h_copy["resources"] = dict(self.resource_inventory.get(h["hospital_id"], {}))
            result.append(h_copy)
        return result

    # ---- Resources ----

    def get_resources(self, hospital_id: str) -> dict:
        return dict(self.resource_inventory.get(hospital_id, {}))

    def dispatch_resource(self, hospital_id: str, resource_type: str, count: int = 1) -> dict:
        """Decrement resource count when dispatching (e.g. ambulance sent out)."""
        inv = self.resource_inventory.get(hospital_id, {})
        available = inv.get(resource_type, 0)
        if count > available:
            raise ValueError(
                f"{hospital_id} has {available} {resource_type}, cannot dispatch {count}"
            )
        inv[resource_type] = available - count
        self.resource_inventory[hospital_id] = inv
        return dict(inv)

    def return_resource(self, hospital_id: str, resource_type: str, count: int = 1) -> dict:
        """Increment resource count when resource returns."""
        inv = self.resource_inventory.setdefault(hospital_id, {})
        inv[resource_type] = inv.get(resource_type, 0) + count
        return dict(inv)

    # ---- Regions ----

    def get_all_regions(self) -> list[dict]:
        return list(self.regions.values())

    def get_region(self, region_id: str) -> Optional[dict]:
        return self.regions.get(region_id)

    def update_region(self, region_id: str, data: dict) -> Optional[dict]:
        if region_id not in self.regions:
            return None
        self.regions[region_id].update(data)
        return self.regions[region_id]

    def get_region_by_coords(self, lat: float, lon: float) -> Optional[dict]:
        best = None
        best_dist = float("inf")
        for r in self.regions.values():
            dist = _euclidean_km(r["lat_center"], r["lon_center"], lat, lon)
            if dist <= r["radius_km"] and dist < best_dist:
                best = r
                best_dist = dist
        return best

    def _capacity_for(self, resource_type: str) -> int:
        """Get capacity_per_unit for a resource type."""
        from app.optimizer.allocation import RESOURCE_TYPES
        return RESOURCE_TYPES.get(resource_type, {}).get("capacity_per_unit", 1)

    # ---- Crisis Session ----

    def start_crisis(self, disaster_type: str, zones: list[dict],
                     allocation_results: list[dict], events: list = None) -> dict:
        """Start a new crisis session from optimization results."""
        crisis_id = str(uuid.uuid4())[:8]
        zone_states = {}
        for z in zones:
            zid = z["zone_id"] if isinstance(z, dict) else z.zone_id
            demand = z["demand"] if isinstance(z, dict) else z.demand
            zone_states[zid] = {
                "initial_demand": demand,
                "remaining_demand": demand,
                "served": 0,
                "zone_data": z if isinstance(z, dict) else z.model_dump(),
            }

        dispatches = []
        for result in allocation_results:
            r = result if isinstance(result, dict) else result.model_dump()
            for res in r.get("assigned_resources", []):
                for rtype, count in res.get("resource_breakdown", {}).items():
                    if count <= 0:
                        continue
                    dispatches.append({
                        "dispatch_id": str(uuid.uuid4())[:8],
                        "hospital_id": res["hospital"],
                        "zone_id": r["zone_id"],
                        "resource_type": rtype,
                        "count": count,
                        "status": "dispatched",
                        "optimization_round": 1,
                        "capacity_served": count * self._capacity_for(rtype),
                    })
                    # Actually dispatch from inventory
                    try:
                        self.dispatch_resource(res["hospital"], rtype, count)
                    except ValueError:
                        dispatches[-1]["status"] = "failed"

        self.active_crisis = {
            "crisis_id": crisis_id,
            "disaster_type": disaster_type,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "zone_states": zone_states,
            "dispatches": dispatches,
            "events": [e if isinstance(e, dict) else e.model_dump()
                       for e in (events or [])],
            "optimization_round": 1,
        }
        return self.active_crisis

    def get_crisis(self) -> Optional[dict]:
        return self.active_crisis

    def return_dispatch(self, dispatch_id: str, served_demand: int) -> dict:
        """Mark a dispatch as returned. Updates zone demand and hospital inventory.

        Args:
            dispatch_id: ID of the dispatch to return
            served_demand: how many demand units this dispatch actually served

        Returns:
            Updated crisis state
        """
        if not self.active_crisis:
            raise ValueError("No active crisis")

        dispatch = None
        for d in self.active_crisis["dispatches"]:
            if d["dispatch_id"] == dispatch_id and d["status"] == "dispatched":
                dispatch = d
                break

        if not dispatch:
            raise ValueError(f"Dispatch {dispatch_id} not found or already returned")

        # Return resources to hospital
        self.return_resource(dispatch["hospital_id"], dispatch["resource_type"], dispatch["count"])
        dispatch["status"] = "returned"

        # Reduce zone remaining demand
        zstate = self.active_crisis["zone_states"].get(dispatch["zone_id"])
        if zstate:
            actual_served = min(served_demand, zstate["remaining_demand"])
            zstate["remaining_demand"] -= actual_served
            zstate["served"] += actual_served

        return self.active_crisis

    def get_unserved_zones(self) -> list[dict]:
        """Get zones that still have remaining demand."""
        if not self.active_crisis:
            return []
        result = []
        for _zid, zstate in self.active_crisis["zone_states"].items():
            if zstate["remaining_demand"] > 0:
                zone_data = dict(zstate["zone_data"])
                zone_data["demand"] = zstate["remaining_demand"]
                result.append(zone_data)
        return result

    def record_reoptimization(self, new_results: list[dict]):
        """Record a new optimization round and its dispatches."""
        if not self.active_crisis:
            raise ValueError("No active crisis")

        self.active_crisis["optimization_round"] += 1

        for result in new_results:
            r = result if isinstance(result, dict) else result.model_dump()
            for res in r.get("assigned_resources", []):
                for rtype, count in res.get("resource_breakdown", {}).items():
                    if count <= 0:
                        continue
                    self.active_crisis["dispatches"].append({
                        "dispatch_id": str(uuid.uuid4())[:8],
                        "hospital_id": res["hospital"],
                        "zone_id": r["zone_id"],
                        "resource_type": rtype,
                        "count": count,
                        "status": "dispatched",
                        "optimization_round": self.active_crisis["optimization_round"],
                        "capacity_served": count * self._capacity_for(rtype),
                    })
                    try:
                        self.dispatch_resource(res["hospital"], rtype, count)
                    except ValueError:
                        self.active_crisis["dispatches"][-1]["status"] = "failed"

    def end_crisis(self) -> Optional[dict]:
        """End the active crisis. Returns final state."""
        if not self.active_crisis:
            return None
        self.active_crisis["status"] = "resolved"
        self.active_crisis["ended_at"] = datetime.now(timezone.utc).isoformat()
        result = self.active_crisis
        self.active_crisis = None
        return result

    def is_crisis_resolved(self) -> bool:
        """Check if all zones have remaining_demand <= 0."""
        if not self.active_crisis:
            return True
        return all(
            zs["remaining_demand"] <= 0
            for zs in self.active_crisis["zone_states"].values()
        )


# Singleton instance
data_store = DataStore()
