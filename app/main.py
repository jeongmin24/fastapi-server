from fastapi import FastAPI
from app.api.v1.routers import api_router

app = FastAPI()


@app.get("/")
def home():
    return {"message": "FastAPI 서버 정상 실행 중!"}

app.include_router(api_router, prefix="/api/v1")


