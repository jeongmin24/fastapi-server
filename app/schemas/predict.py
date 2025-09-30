from pydantic import BaseModel

class PredictStatsRequest(BaseModel):
    line: str
    station: str

class PredictStatsResponse(BaseModel):
    line: str
    station: str
    predicted_count: float