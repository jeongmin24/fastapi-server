import pickle
import numpy as np
from fastapi import HTTPException
# from ..config.settings import MODEL_PATH
from ..schemas.predict import CongestionDetail, CongestionResponse, CongestionRequest
from ..utils.feature_utils import create_features_for_prediction
from huggingface_hub import hf_hub_download

# 1. Hugging Face 저장소 정보
HF_REPO_ID = "gcanoca/SubwayCongestionPkl"
# 2. 저장소에 있는 모델 파일의 정확한 이름
MODEL_FILENAME = "train_congestion_model.pkl"

# 모델 로드 (서버 시작 시 한 번만)
congestion_model = None

try:
    # ------------------------------------------------------------------
    # 3. hf_hub_download를 사용하여 모델 파일을 로컬 임시 경로로 다운로드
    # ------------------------------------------------------------------
    MODEL_PATH_LOCAL = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        # 다운로드할 로컬 캐시 폴더 (선택 사항)
        # cache_dir="/tmp/hf_cache"
        repo_type="dataset"
    )

    # 4. 다운로드된 로컬 경로를 사용하여 파일 열기 및 로드
    with open(MODEL_PATH_LOCAL, 'rb') as f:
        congestion_model = pickle.load(f)

    print(f"[Service] 모델 로드 성공: {MODEL_PATH_LOCAL} (Hub: {HF_REPO_ID}/{MODEL_FILENAME})")

    # (선택적) 모델 로드 후 임시 파일 삭제
    # os.remove(MODEL_PATH_LOCAL)

except ImportError:
    print("[Service CRITICAL ERROR] 'huggingface_hub' 라이브러리가 설치되지 않았습니다. pip install huggingface_hub 필요.")
    congestion_model = None
except Exception as e:
    print(f"[Service CRITICAL ERROR] Hugging Face 모델 로드 실패 또는 파일 문제: {e}")
    congestion_model = None


def get_congestion_prediction(request: CongestionRequest) -> CongestionResponse:
    """
    1호차부터 10호차까지의 혼잡도를 예측하고 결과를 포맷하는 서비스 로직
    """
    if congestion_model is None:
        raise HTTPException(status_code=503, detail="예측 모델이 로드되지 않았습니다. 서버 설정을 확인하세요.")

    all_car_results = []
    total_predicted_passengers = 0.0

    # 1호차부터 10호차까지 반복하며 예측
    for car_num in range(1, 11):
        try:
            # 특징 생성 (utils 사용)
            X_data = create_features_for_prediction(request, car_num, congestion_model)

            # 예측 실행
            predicted_passengers = congestion_model.predict(X_data.iloc[0]).iloc[0]
            passenger_count = max(0.0, predicted_passengers)
            congestion_percent = (passenger_count / 1.6)  # 기준 혼잡도 1.6

            detail = CongestionDetail(
                car_number=car_num,
                predicted_passengers=round(passenger_count, 2),
                predicted_congestion_percent=round(congestion_percent, 2)
            )
            all_car_results.append(detail)
            total_predicted_passengers += passenger_count

        except Exception as e:
            print(f"[CRITICAL] 예측 중 알 수 없는 오류: {e}")
            raise HTTPException(status_code=500, detail=f"예측 처리 중 오류: {e}")

    return CongestionResponse(
        status="success",
        request_info=request,
        total_predicted_passengers=round(total_predicted_passengers, 2),
        car_congestion_details=all_car_results
    )