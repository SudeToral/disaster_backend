from fastapi import FastAPI
from app.optimizer.schemas import AllocationRequest
from app.optimizer.allocation import build_graph, allocate_resources

app = FastAPI(title="MED-ARES Optimization Engine")


@app.post("/optimize")
def optimize(req: AllocationRequest):
    G = build_graph(req.zones, req.hospitals)
    return allocate_resources(G, req.zones, req.hospitals)


