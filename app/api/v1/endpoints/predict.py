# 예측 API

from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    return {
        "line": req.line,
        "time": req.time,
        "congestion": "중간"  # TODO: 모델 예측으로 변경
    }
