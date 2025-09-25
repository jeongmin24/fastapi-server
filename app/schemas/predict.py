from pydantic import BaseModel

class PredictRequest(BaseModel):
    line: str
    time: str

class PredictResponse(BaseModel):
    line: str
    time: str
    congestion: str