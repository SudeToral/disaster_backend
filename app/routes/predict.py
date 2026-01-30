from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse
from app.model.model import predict

router = APIRouter(prefix="/predict")

@router.post("", response_model=PredictResponse)
def run(req: PredictRequest):
    result = predict(req.input)
    return {"result": result}
