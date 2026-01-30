from pydantic import BaseModel
from typing import List


class Zone(BaseModel):
    zone_id: str
    damage_severity: float        # 0–1
    population_density: float     # 0–1
    medical_demand: int           # abstract units
    lat: float
    lon: float


class Hospital(BaseModel):
    hospital_id: str
    ambulances: int
    lat: float
    lon: float


class AllocationRequest(BaseModel):
    disaster_type: str
    zones: List[Zone]
    hospitals: List[Hospital]


class AssignedResource(BaseModel):
    hospital: str
    ambulances: int


class ZoneResult(BaseModel):
    zone_id: str
    priority: str
    confidence: float
    assigned_resources: List[AssignedResource]
