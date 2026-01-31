from fastapi import APIRouter, HTTPException
from app.store import data_store
from app.optimizer.impact import estimate_impact
from app.optimizer.schemas import ImpactEstimateRequest, DispatchRequest, ReturnRequest

router = APIRouter(prefix="/data", tags=["data"])


# ---- Hospitals ----

@router.get("/hospitals")
def list_hospitals():
    return data_store.get_all_hospitals()


@router.get("/hospitals/{hospital_id}")
def get_hospital(hospital_id: str):
    h = data_store.get_hospital(hospital_id)
    if not h:
        raise HTTPException(404, f"Hospital {hospital_id} not found")
    return h


@router.put("/hospitals/{hospital_id}")
def update_hospital(hospital_id: str, updates: dict):
    h = data_store.update_hospital(hospital_id, updates)
    if not h:
        raise HTTPException(404, f"Hospital {hospital_id} not found")
    return h


# ---- Resources ----

@router.get("/hospitals/{hospital_id}/resources")
def get_resources(hospital_id: str):
    if not data_store.get_hospital(hospital_id):
        raise HTTPException(404, f"Hospital {hospital_id} not found")
    return data_store.get_resources(hospital_id)


@router.post("/hospitals/{hospital_id}/dispatch")
def dispatch_resource(hospital_id: str, req: DispatchRequest):
    if not data_store.get_hospital(hospital_id):
        raise HTTPException(404, f"Hospital {hospital_id} not found")
    try:
        return data_store.dispatch_resource(hospital_id, req.resource_type, req.count)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/hospitals/{hospital_id}/return")
def return_resource(hospital_id: str, req: ReturnRequest):
    if not data_store.get_hospital(hospital_id):
        raise HTTPException(404, f"Hospital {hospital_id} not found")
    return data_store.return_resource(hospital_id, req.resource_type, req.count)


# ---- Regions ----

@router.get("/regions")
def list_regions():
    return data_store.get_all_regions()


@router.get("/regions/{region_id}")
def get_region(region_id: str):
    r = data_store.get_region(region_id)
    if not r:
        raise HTTPException(404, f"Region {region_id} not found")
    return r


@router.put("/regions/{region_id}")
def update_region(region_id: str, updates: dict):
    r = data_store.update_region(region_id, updates)
    if not r:
        raise HTTPException(404, f"Region {region_id} not found")
    return r


# ---- Impact Estimation ----

@router.post("/estimate-impact")
def estimate_impact_endpoint(req: ImpactEstimateRequest):
    zones = estimate_impact(
        epicenter=req.epicenter,
        disaster_type=req.disaster_type,
        severity=req.severity,
        impact_radius_km=req.impact_radius_km,
    )
    return {
        "disaster_type": req.disaster_type,
        "severity": req.severity,
        "epicenter": req.epicenter,
        "generated_zones": [z.model_dump() for z in zones],
        "total_demand": sum(z.demand for z in zones),
    }
