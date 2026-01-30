from pydantic import BaseModel

class PredictRequest(BaseModel):
    input: float

class PredictResponse(BaseModel):
    result: float
