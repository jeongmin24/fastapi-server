# 예측 API

from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse
from app.services.predict import parse_datetime_kst, predict_single
router = APIRouter()

@router.post("", response_model=PredictSingleResponse)
def predict_endpoint(req: PredictSingleRequest):
    try:
        dt_kst = parse_datetime_kst(req.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="datetime은 ISO8601 형식이어야 합니다.")

    try:
        gton, gtoff, feats = predict_single(req.line, req.station, dt_kst)
    except FileNotFoundError as e:
        # 특정 호선/역 모델이 없는 경우
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(), # KST 기준으로 변환된 시각
        pred_gton=gton,
        pred_gtoff=gtoff,
        features_used={k:int(v) for k,v in feats.items() if isinstance(v, (int, bool))}
    )