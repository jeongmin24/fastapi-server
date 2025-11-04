from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
from app.config.settings import KST
from app.services.preprocessing import preprocess_stats_time_response
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse

router = APIRouter()

# 모델 파일 정보
HF_REPO_ID = "gcanoca/SubwayCongestionPkl"
MODEL_FILENAME = "lines_CardSubwayTime_model_20251105.pkl"

# 전역 캐시 (처음 한 번만 로드됨)
model = None
line_encoder = None
station_encoder = None

FEATURE_COLUMNS_V1 = [
    "year",
    "month",
    "hour",
    "line_encoded",
    "station_encoded"
]


# ---------------------
# 공용 함수들
# ---------------------
def parse_datetime_kst(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def build_feature_row(dt_kst, line, station, line_encoder, station_encoder):
    return {
        "year": dt_kst.year,
        "month": dt_kst.month,
        "hour": dt_kst.hour,
        "line_encoded": int(line_encoder.transform([line])[0]),
        "station_encoded": int(station_encoder.transform([station])[0])
    }


def predict_single(line: str, station: str, dt_kst: datetime, model, line_encoder, station_encoder):
    feats = build_feature_row(dt_kst, line, station, line_encoder, station_encoder)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)

    yhat = model.predict(X)[0]
    pred_gton = max(0, int(round(yhat[0])))
    pred_gtoff = max(0, int(round(yhat[1])))

    return pred_gton, pred_gtoff, feats


# ---------------------
# 실제 엔드포인트
# ---------------------
@router.post("/predict", response_model=PredictSingleResponse)
def predict_endpoint(req: PredictSingleRequest):
    global model, line_encoder, station_encoder

    # ❗ 모델이 아직 로드 안 됐으면, 요청 시점에 한 번만 로드
    if model is None:
        try:
            print(f"🔄 Lazy-loading model from Hugging Face: {HF_REPO_ID}/{MODEL_FILENAME}")
            downloaded_file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                repo_type="dataset",
                cache_dir="/tmp"  # Render의 임시 디스크 사용 (RAM 절약)
            )

            bundle = joblib.load(downloaded_file_path)
            model = bundle["model"]
            line_encoder = bundle["line_encoder"]
            station_encoder = bundle["station_encoder"]
            print("✅ Model loaded successfully (lazy load).")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}")

    # 요청 처리
    try:
        dt_kst = parse_datetime_kst(req.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="datetime은 ISO8601 형식이어야 합니다.")

    try:
        gton, gtoff, feats = predict_single(
            req.line, req.station, dt_kst,
            model=model,
            line_encoder=line_encoder,
            station_encoder=station_encoder
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    # 변환 및 응답 생성
    features_used = {k: (int(v) if isinstance(v, (int, bool, np.integer)) else v)
                     for k, v in feats.items()}

    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(),
        pred_gton=float(gton),
        pred_gtoff=float(gtoff),
        predicted_count=float(gton + gtoff),
        features_used=features_used
    )
