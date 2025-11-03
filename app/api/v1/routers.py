from fastapi import APIRouter
from app.api.v1.endpoints import predict, subway, train

api_router = APIRouter()
# 단일 역 예측 API (line, station, datetime)
api_router.include_router(predict.router, prefix="", tags=["예측"])

# 지하철 관련 API
api_router.include_router(subway.router, prefix="/subway", tags=["지하철"])
api_router.include_router(predict.router, prefix="/predict-stats", tags=["통계 예측"])

# 모델 학습/관리 관련 API (옵션)
api_router.include_router(train.router, prefix="/train", tags=["학습"])