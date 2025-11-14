from fastapi import FastAPI
from app.api.v1.routers import api_router
from app.config.settings import settings
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    app = FastAPI(
        title="NoRush ML API",
        version="0.0.1",
        description="대중교통 혼잡도 예측 기반 API 서버"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Swagger 요청 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 기본 헬스체크 엔드포인트
    @app.get("/")
    def home():
        return {"message": "FastAPI 서버 정상 실행 중!"}

    # 설정에서 불러온 prefix로 라우터 등록
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_app()
#
# print("라우터 목록:")
# for route in app.routes:
#     print(route.path)

# for route in app.routes:
#     if "predict" in route.path:
#         print(route.path, route.methods)


