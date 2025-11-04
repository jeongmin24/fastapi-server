import numpy as np
from fastapi import APIRouter, HTTPException, status
# app.services.predict에서 필요한 함수만 import
from app.services.predict import parse_datetime_kst, predict_single
# app.schemas와 prediction_service는 기존대로 import
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse, CongestionRequest, CongestionResponse
from app.services.prediction_service import get_congestion_prediction
from datetime import datetime

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

    # 예측 실행
    try:
        gton, gtoff, feats = predict_single(
            req.line,
            req.station,
            dt_kst
        )

    except FileNotFoundError as e:
        # 특정 호선/역 모델 파일이 없을 때
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")
    except Exception as e:
        # 기타 예측 실패
        raise HTTPException(status_code=500, detail=f"기타 예측 실패: {e}")

    # features_used 정수형 필드만 필터링
    features_used = {k: (int(v) if isinstance(v, (int, bool, np.integer)) else v) for k, v in feats.items()}

    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(),
        # numpy 타입을 float으로 변환
        pred_gton=float(gton) if isinstance(gton, np.number) else gton,
        pred_gtoff=float(gtoff) if isinstance(gtoff, np.number) else gtoff,
        predicted_count=(float(gton) + float(gtoff)) if isinstance(gton, np.number) else (gton + gtoff),
        features_used=features_used
    )


# 혼잡도 예측 API는 그대로 유지
@router.post("/predict/train_congestion", response_model=CongestionResponse, status_code=status.HTTP_200_OK,
             summary="지하철 칸별 혼잡도 예측")
def train_congestion_api(req: CongestionRequest):
    """
    지하철 칸별 혼잡도 예측 API (get_congestion_prediction 함수에 의존)
    """
    return get_congestion_prediction(req)