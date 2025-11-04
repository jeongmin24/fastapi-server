# 예측 API

from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse, CongestionRequest, CongestionResponse
from app.services.predict import parse_datetime_kst, predict_single
from app.services.prediction_service import get_congestion_prediction
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

router = APIRouter()


# 모델 로드 (test 코드 기준)
bundle = joblib.load("models/lines_CardSubwayTime_model_20251104.pkl")
model = bundle["model"]
line_encoder = bundle["line_encoder"]
station_encoder = bundle["station_encoder"]

@router.post("/predict", response_model=PredictSingleResponse)
def predict_endpoint(req: PredictSingleRequest):
    # datetime 변환
    try:
        dt_kst = parse_datetime_kst(req.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="datetime은 ISO8601 형식이어야 합니다.")


    # line/station 인코딩 및 DataFrame 생성
    try:
        gton, gtoff, feats = predict_single(req.line, req.station, dt_kst, model=model, line_encoder=line_encoder, station_encoder=station_encoder)
    except FileNotFoundError as e:
        # 특정 호선/역 모델이 없는 경우
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    # features_used 정수형 필드만 필터링
    features_used = {k: int(v) for k, v in feats.items() if isinstance(v, (int, bool))}

    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(),  # KST 기준으로 변환된 시각
        pred_gton=gton,
        pred_gtoff=gtoff,
        predicted_count=gton + gtoff,
        features_used=features_used
    )

@router.post("/predict/train_congestion", response_model=CongestionResponse,status_code=status.HTTP_200_OK, summary="지하철 칸별 혼잡도 예측")
def train_congestion_api(req: CongestionRequest):
    return get_congestion_prediction(req)