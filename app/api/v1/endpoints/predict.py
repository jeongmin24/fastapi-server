# 예측 API

from fastapi import APIRouter
from app.schemas.predict import PredictStatsRequest, PredictStatsResponse
from app.services.model import predict_stats

router = APIRouter()

@router.post("/", response_model=PredictStatsResponse)
def predict_stats_api(req: PredictStatsRequest):
    pred = predict_stats(req.line, req.station)
    return PredictStatsResponse(line=req.line, station=req.station, predicted_count=pred)