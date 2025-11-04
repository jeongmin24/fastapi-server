from fastapi import APIRouter, HTTPException, status
from app.schemas.predict import PredictSingleRequest, PredictSingleResponse, CongestionRequest, CongestionResponse
from app.services.predict import parse_datetime_kst, predict_single
from app.services.prediction_service import get_congestion_prediction
from ..config.settings import HF_REPO_ID, PREDICTION_MODEL_FILENAME
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download  # Hugging Face Hub 라이브러리

router = APIRouter()

# 1. 모델 파일 정보 (상수)
HF_REPO_ID = "gcanoca/SubwayCongestionPkl"
MODEL_FILENAME = "lines_CardSubwayTime_model_20251105.pkl"

# 2. 모델 객체를 저장할 전역 변수 초기화 (앱 시작 시점에는 None)
MODEL_BUNDLE = {
    "model": None,
    "line_encoder": None,
    "station_encoder": None
}

# 3. 모델 로드 로직을 지연 실행 함수로 정의 
def load_model_bundle():
    """모델이 로드되지 않았을 경우에만 Hugging Face에서 다운로드 및 로드합니다."""
    # 이미 로드되었는지 확인 -> 두 번째 요청부터는 재로드 방지
    if MODEL_BUNDLE["model"] is not None:
        return

    print(f"Starting Lazy Load: Downloading and loading model from {HF_REPO_ID}/{MODEL_FILENAME}...")
    try:
        # Hugging Face Hub에서 모델 파일 다운로드 (임시 디렉토리에 저장)
        downloaded_file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="dataset"
        )

        # joblib을 사용하여 다운로드된 파일에서 모델 로드
        bundle = joblib.load(downloaded_file_path)
        
        # 전역 변수에 저장하여 재사용
        MODEL_BUNDLE["model"] = bundle["model"]
        MODEL_BUNDLE["line_encoder"] = bundle["line_encoder"]
        MODEL_BUNDLE["station_encoder"] = bundle["station_encoder"]

        print(" Model loaded successfully (first time).")

    except Exception as e:
        error_message = f"FATAL: Model loading failed. Check configuration or file. Error: {e}"
        print(error_message)
        # 로드 실패는 배포 실패를 유발해야 합니다.
        raise RuntimeError(error_message)


# ------------------ 라우트 핸들러 ------------------

@router.post("/predict", response_model=PredictSingleResponse)
def predict_endpoint(req: PredictSingleRequest):
    """
    단일 요청에 대한 지하철 승하차 인원 예측
    """
    
    # 요청이 들어왔을 때 로드 함수 호출
    # 앱 시작 시점의 메모리 초과를 막고, 첫 요청 시 로드를 실행
    load_model_bundle()
    
    # 로드된 모델 객체 사용
    model = MODEL_BUNDLE["model"]
    line_encoder = MODEL_BUNDLE["line_encoder"]
    station_encoder = MODEL_BUNDLE["station_encoder"]

    # datetime 변환
    try:
        dt_kst = parse_datetime_kst(req.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="datetime은 ISO8601 형식이어야 합니다.")

    # line/station 인코딩 및 DataFrame 생성 및 예측 실행
    try:
        gton, gtoff, feats = predict_single(
            req.line,
            req.station,
            dt_kst,
            model=model,
            line_encoder=line_encoder,
            station_encoder=station_encoder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    # features_used 정수형 필드 필터링 및 타입 변환
    features_used = {k: (int(v) if isinstance(v, (int, bool, np.integer)) else v) for k, v in feats.items()}

    # 응답 반환 시 numpy 타입 -> float/int 변환
    return PredictSingleResponse(
        line=req.line,
        station=req.station,
        datetime=dt_kst.isoformat(),
        pred_gton=float(gton) if isinstance(gton, np.number) else gton,
        pred_gtoff=float(gtoff) if isinstance(gtoff, np.number) else gtoff,
        predicted_count=(float(gton) + float(gtoff)) if isinstance(gton, np.number) else (gton + gtoff),
        features_used=features_used
    )


@router.post("/predict/train_congestion", response_model=CongestionResponse, status_code=status.HTTP_200_OK,
             summary="지하철 칸별 혼잡도 예측")
def train_congestion_api(req: CongestionRequest):
    """
    지하철 칸별 혼잡도 예측 API (get_congestion_prediction 함수에 의존)
    """
    return get_congestion_prediction(req)
