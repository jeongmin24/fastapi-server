# 예측 API (predict_router.py 또는 유사한 파일)

import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse, CongestionRequest, CongestionResponse
from app.services.predict import parse_datetime_kst, predict_single  # app.services.predict에 필요한 함수가 있다고 가정합니다.
from app.services.prediction_service import \
    get_congestion_prediction  # app.services.prediction_service에 필요한 함수가 있다고 가정합니다.
from datetime import datetime
from huggingface_hub import hf_hub_download  # Hugging Face Hub 라이브러리 추가

router = APIRouter()

@router.post("/predict", response_model=PredictSingleResponse)
def predict_endpoint(req: PredictSingleRequest):
    """
    단일 요청에 대한 지하철 승하차 인원 예측
    """
    # datetime 변환
    try:
        dt_kst = parse_datetime_kst(req.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="datetime은 ISO8601 형식이어야 합니다.")

    # line/station 인코딩 및 DataFrame 생성
    try:
        # predict_single 함수는 app.services.predict에 정의되어 있다고 가정합니다.
        gton, gtoff, feats = predict_single(
            req.line,
            req.station,
            dt_kst,
            model=model,
            line_encoder=line_encoder,
            station_encoder=station_encoder
        )
    except FileNotFoundError as e:
        # 특정 호선/역 모델이 없는 경우 (모델 로드 시 처리되었으나, 혹시 모를 내부 로직을 위해 유지)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # 기타 예측 실패
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    # features_used 정수형 필드만 필터링
    # numpy 데이터 타입이 직렬화 문제를 일으킬 수 있으므로 기본 Python 타입으로 변환
    features_used = {k: (int(v) if isinstance(v, (int, bool, np.integer)) else v) for k, v in feats.items()}

    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(),  # KST 기준으로 변환된 시각
        pred_gton=float(gton) if isinstance(gton, np.number) else gton,  # numpy 타입을 float으로 변환
        pred_gtoff=float(gtoff) if isinstance(gtoff, np.number) else gtoff,  # numpy 타입을 float으로 변환
        predicted_count=(float(gton) + float(gtoff)) if isinstance(gton, np.number) else (gton + gtoff),
        features_used=features_used
    )


@router.post("/predict/train_congestion", response_model=CongestionResponse, status_code=status.HTTP_200_OK,
             summary="지하철 칸별 혼잡도 예측")
def train_congestion_api(req: CongestionRequest):
    """
    지하철 칸별 혼잡도 예측 API (get_congestion_prediction 함수에 의존)
    """
    # get_congestion_prediction 함수는 app.services.prediction_service에 정의되어 있다고 가정합니다.
    return get_congestion_prediction(req)
