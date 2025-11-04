# services/predict.py

from fastapi import HTTPException
from datetime import datetime
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
# app.config.settings는 KST 정의를 제공한다고 가정합니다.
from app.config.settings import KST

# app.services.preprocessing는 이 파일에서 사용하지 않으므로 삭제했습니다.
# app.schemas.predict는 이 파일에서 사용하지 않으므로 삭제했습니다.


# 모델 파일 정보 (유지)
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
# 공용 함수들 (유지)
# ---------------------
def parse_datetime_kst(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


# 🌟 수정: build_feature_row 함수가 전역 변수를 사용하도록 변경
def build_feature_row(dt_kst: datetime, line: str, station: str):
    global line_encoder, station_encoder

    # 모델 로드가 완료되었는지 (즉, 인코더가 있는지) 확인하는 로직이 필요하다면 추가
    if line_encoder is None or station_encoder is None:
        raise RuntimeError("인코더가 로드되지 않았습니다. predict_single 함수를 먼저 호출해야 합니다.")

    # 인코더가 전역 변수에 로드되어 있다고 가정하고 사용합니다.
    return {
        "year": dt_kst.year,
        "month": dt_kst.month,
        "hour": dt_kst.hour,
        "line_encoded": int(line_encoder.transform([line])[0]),
        "station_encoded": int(station_encoder.transform([station])[0])
    }


def predict_single(line: str, station: str, dt_kst: datetime):
    global model, line_encoder, station_encoder

    # 1. Lazy Loading (첫 요청 시 모델 로드)
    if model is None:
        try:
            print(f"Lazy-loading model from Hugging Face: {HF_REPO_ID}/{MODEL_FILENAME}")
            downloaded_file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                repo_type="dataset",
                cache_dir="/tmp"
            )

            bundle = joblib.load(downloaded_file_path)
            model = bundle["model"]
            line_encoder = bundle["line_encoder"]
            station_encoder = bundle["station_encoder"]
            print(" Model loaded successfully (lazy load).")

        except Exception as e:
            # 모델 로드 실패 시, endpoints에서 잡을 수 있도록 RuntimeError 발생
            raise RuntimeError(f"모델 로드 실패: {e}")

    # 2. 특징 추출 (build_feature_row는 이제 모델/인코더 인자를 받지 않습니다)
    feats = build_feature_row(dt_kst, line, station)

    # 3. 예측 실행
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)

    yhat = model.predict(X)[0]
    pred_gton = max(0, int(round(yhat[0])))
    pred_gtoff = max(0, int(round(yhat[1])))

    return pred_gton, pred_gtoff, feats