from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    line: str
    time: str

@app.get("/")
def home():
    return {"message": "FastAPI 서버 정상 실행 중!"}

@app.post("/predict")
def predict(req: PredictRequest):
    return {"line": req.line, "time": req.time, "congestion": "중간"}
