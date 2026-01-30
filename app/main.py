from fastapi import FastAPI
from app.optimizer.schemas import AllocationRequest
from app.optimizer.allocation import optimize_allocation, build_zone_results

app = FastAPI(title="MED-ARES Optimization Engine")


@app.post("/optimize")
def optimize(request: AllocationRequest):
    mcf, h_off, z_off = optimize_allocation(
        request.zones,
        request.hospitals
    )

    return build_zone_results(
        mcf,
        request.zones,
        request.hospitals,
        h_off,
        z_off
    )

