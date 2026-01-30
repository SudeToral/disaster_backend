from pydantic import BaseModel
from typing import List, Optional


class Zone(BaseModel):
    zone_id: str
    damage_severity: float        # 0–1
    population_density: float     # 0–1
    medical_demand: int           # abstract units
    lat: float
    lon: float
    terrain_type: str = "urban"   # coastal | mountain | urban | rural


class Hospital(BaseModel):
    hospital_id: str
    ambulances: int
    lat: float
    lon: float
    hospital_type: str = "regular"       # regular | coast_guard | marine_ambulance
    name: str = ""
    region_id: str = ""
    capacity_beds: int = 0
    specialties: List[str] = []
    resources: dict = {}                 # e.g. {"ambulance": 15, "helicopter": 1}


class Event(BaseModel):
    event_type: str              # road_collapse | flood | weather | bridge_out
    params: dict = {}


class AllocationRequest(BaseModel):
    disaster_type: str
    zones: List[Zone]
    events: List[Event] = []
    use_agent: bool = True
    coordinates: Optional[List[float]] = None     # [lat, lon] epicenter
    natural_language_query: Optional[str] = None


class AssignedResource(BaseModel):
    hospital: str
    ambulances: int = 0
    resource_breakdown: dict = {}        # e.g. {"ambulance": 5, "helicopter": 1}


class ZoneResult(BaseModel):
    zone_id: str
    priority: str
    confidence: float
    assigned_resources: List[AssignedResource]
    unserved: int = 0
    resource_summary: dict = {}          # e.g. {"ambulance": 10, "helicopter": 2}


class AgentOptimizeResponse(BaseModel):
    results: List[ZoneResult]
    agent_reasoning: str = ""
    events_applied: List[str] = []
    data_sources: List[str] = []
    fallback_used: bool = False
