from fastapi import FastAPI
from app.optimizer.schemas import AllocationRequest, ZoneResult, Hospital
from app.optimizer.allocation import build_graph, allocate_resources, events_to_context
from app.optimizer.impact import estimate_impact
from app.agents.orchestrator import orchestrator_graph
from app.agents.crisis_manager_agent import CrisisManagerAgent
from app.config import load_hospitals_with_resources, detect_terrain
from app.store import data_store
from app.routes.data import router as data_router
from app.routes.crisis import router as crisis_router

# Initialize data store from JSON files
data_store.load_from_json()
import os
import uuid
import shutil
from typing import Any, Dict
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.model.model import predict_with_type

app = FastAPI(title="MED-ARES Optimization Engine")
app.include_router(data_router)
app.include_router(crisis_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
OVERLAY_DIR = os.path.join(UPLOAD_DIR, "overlay")


os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# overlay/original dosyalarını URL ile servis etmek için:
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")



def _get_hospitals() -> list:
    """Load hospitals with resource inventory and convert to Hospital models."""
    return [Hospital(**h) for h in load_hospitals_with_resources()]


@app.post("/optimize")
async def optimize(req: AllocationRequest):
    """Single endpoint that does EVERYTHING:

    1. Impact estimation (coordinates + severity → zones)
    2. Resource allocation (agent or direct)
    3. Crisis session start
    4. Full lifecycle resolution (return → reoptimize loop)

    Frontend calls this ONCE and gets the complete result.
    """
    hospitals = _get_hospitals()

    # --- 1. Impact estimation ---
    if not req.zones and req.coordinates and req.severity is not None:
        req.zones = estimate_impact(
            epicenter=req.coordinates,
            disaster_type=req.disaster_type,
            severity=req.severity,
            impact_radius_km=req.impact_radius_km,
        )

    # Auto-detect terrain for zones that use the default "urban"
    for z in req.zones:
        if z.terrain_type == "urban":
            z.terrain_type = detect_terrain(z.lat, z.lon)

    # --- 2. Allocation (agent or direct) ---
    agent_reasoning = ""
    db_reasoning = ""
    events_applied = []
    fallback_used = False

    if req.use_agent:
        initial_state = {
            "disaster_type": req.disaster_type,
            "zones": req.zones,
            "hospitals": hospitals,
            "events": req.events,
            "coordinates": req.coordinates,
            "natural_language_query": req.natural_language_query,
            "enriched_hospitals": [],
            "enriched_zones": [],
            "db_reasoning": "",
            "context": {},
            "allocation_results": [],
            "events_applied": [],
            "strategy_reasoning": "",
            "fallback_used": False,
            "error": None,
        }
        try:
            final_state = orchestrator_graph.invoke(initial_state)
            allocation_results = final_state.get("allocation_results", [])
            agent_reasoning = final_state.get("strategy_reasoning", "")
            db_reasoning = final_state.get("db_reasoning", "")
            events_applied = final_state.get("events_applied", [])
            fallback_used = final_state.get("fallback_used", False)
        except Exception as e:
            # Agent failed → fallback to direct allocation
            context = events_to_context(req.events) if req.events else {}
            G = build_graph(req.zones, hospitals, context, disaster_type=req.disaster_type)
            allocation_results = allocate_resources(G, req.zones, hospitals, disaster_type=req.disaster_type)
            agent_reasoning = f"Agent pipeline error: {str(e)}. Used direct allocation."
            events_applied = [f"{ev.event_type}: {ev.params}" for ev in req.events]
            fallback_used = True
    else:
        context = events_to_context(req.events) if req.events else {}
        G = build_graph(req.zones, hospitals, context, disaster_type=req.disaster_type)
        allocation_results = allocate_resources(G, req.zones, hospitals, disaster_type=req.disaster_type)

    # --- 3. Start crisis session ---
    crisis = data_store.start_crisis(
        disaster_type=req.disaster_type,
        zones=[z.model_dump() for z in req.zones],
        allocation_results=allocation_results,
        events=[e.model_dump() for e in req.events],
    )

    # --- 4. Auto-resolve crisis lifecycle ---
    crisis_manager = CrisisManagerAgent(max_rounds=20)
    lifecycle = crisis_manager.run_lifecycle()

    # --- 5. Build response ---
    results = [
        ZoneResult(**r) if isinstance(r, dict) else r
        for r in allocation_results
    ]

    return {
        "crisis_id": crisis["crisis_id"],
        "disaster_type": req.disaster_type,
        "initial_allocation": [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
        "lifecycle": lifecycle,
        "agent_reasoning": agent_reasoning,
        "db_reasoning": db_reasoning,
        "crisis_manager_reasoning": lifecycle.get("crisis_manager_reasoning", ""),
        "events_applied": events_applied,
        "fallback_used": fallback_used,
    }

    try:
        final_state = orchestrator_graph.invoke(initial_state)
        return AgentOptimizeResponse(
            results=[ZoneResult(**r) if isinstance(r, dict) else r
                     for r in final_state.get("allocation_results", [])],
            agent_reasoning=final_state.get("strategy_reasoning", ""),
            events_applied=final_state.get("events_applied", []),
            data_sources=[final_state.get("db_reasoning", "")],
            fallback_used=final_state.get("fallback_used", False),
        )
    except Exception as e:
        context = events_to_context(req.events) if req.events else {}
        G = build_graph(req.zones, hospitals, context)
        results = allocate_resources(G, req.zones, hospitals)
        return AgentOptimizeResponse(
            results=[ZoneResult(**r) if isinstance(r, dict) else r for r in results],
            agent_reasoning=f"Agent pipeline error: {str(e)}. Used direct allocation.",
            events_applied=[f"{ev.event_type}: {ev.params}" for ev in req.events],
            data_sources=[],
            fallback_used=True,
        )
@app.post("/predict")
async def predict(
    disaster_type: str = Form(...),
    save: bool = Form(False),
    file: UploadFile = File(...),
):
    # 1) input kontrol
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # 2) orijinal görüntüyü kaydet
    request_id = str(uuid.uuid4())
    original_path = os.path.join(ORIGINAL_DIR, f"{request_id}{ext}")

    try:
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 3) overlay hedef path (save=true ise model buraya yazabilir)
    overlay_path = os.path.join(OVERLAY_DIR, f"{request_id}_overlay.png")

    # 4) inference
    try:
        result: Dict[str, Any] = predict_with_type(
            image_path=original_path,
            disaster_type=disaster_type,
            save=save,
            overlay_path=overlay_path
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # 5) response
    response = {
        "request_id": request_id,
        "disaster_type": disaster_type,
        "original_url": f"/uploads/original/{request_id}{ext}",
        "is_disaster": result.get("is_disaster", False),
        "coverage_ratio": result.get("coverage_ratio", 0.0),
        "damaged_regions": result.get("damaged_regions", 0),
        "details": result.get("details", {}),
        "overlay_url": None
    }

    if save and os.path.exists(overlay_path):
        response["overlay_url"] = f"/uploads/overlay/{request_id}_overlay.png"

    return JSONResponse(content=response)
