from fastapi import APIRouter
from app.api.v1.endpoints import predict, subway

api_router = APIRouter()
api_router.include_router(predict.router, prefix="",tags=["예측"])
api_router.include_router(subway.router, prefix="/subway", tags=["지하철"])
api_router.include_router(predict.router, prefix="/predict-stats", tags=["통계 예측"])
