# 예측 API

from fastapi import APIRouter, status
from app.schemas.predict import PredictStatsRequest, PredictStatsResponse, CongestionRequest, CongestionResponse
from app.services.model import predict_stats
from app.services.prediction_service import get_congestion_prediction

router = APIRouter()

@router.post("/", response_model=PredictStatsResponse)
def predict_stats_api(req: PredictStatsRequest):
    pred = predict_stats(req.line, req.station)
    return PredictStatsResponse(line=req.line, station=req.station, predicted_count=pred)

@router.post("/predict/train_congestion", response_model=CongestionResponse,status_code=status.HTTP_200_OK, summary="지하철 칸별 혼잡도 예측")
def train_congestion_api(req: CongestionRequest):
    return get_congestion_prediction(req)