import os
import joblib
from functools import lru_cache
from huggingface_hub import hf_hub_download  # Hugging Face Hub import

# --- ⚠️ **필수 수정 사항:** 이 정보를 사용자님의 것으로 변경하세요. ---
# 모델 파일이 포함된 Hugging Face Repository ID
HF_REPO_ID = "gcanoca/SubwayCongestionPkl"
# 저장소에 있는 모델 파일의 정확한 이름
FIXED_MODEL_NAME = "lines_CardSubwayTime_model_20251104.pkl"
# Private Repository 접근 시 사용할 토큰 (Public이면 None 유지)
HF_TOKEN = None
# ------------------------------------------------------------------

FEATURE_COLUMNS_V1 = ["year", "month", "hour"]


# @lru_cache를 사용하여 모델 객체가 메모리에 한 번만 로드되도록 합니다.
@lru_cache(maxsize=1)
# 💡 함수명을 'load_latest_model'로 복구하여 ImportError를 해결합니다.
def load_latest_model(line: str = None, station: str = None):
    """
    Hugging Face Hub에서 고정된 모델을 로드합니다.
    (기존 코드와의 호환성을 위해 'load_latest_model' 함수명을 사용합니다.)
    """

    repo_id = HF_REPO_ID
    filename = FIXED_MODEL_NAME

    print(f"모델 로딩 시작 (HF Hub): {repo_id}/{filename}")

    try:
        # 1. Hugging Face Hub에서 모델 파일 다운로드
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,  # Private Repository 접근 시 사용
            repo_type = "dataset"
        )

        print(f"모델 파일 다운로드 완료. 경로: {downloaded_file_path}")

        # 2. joblib을 사용하여 다운로드된 파일에서 모델 로드
        model_data = joblib.load(downloaded_file_path)

        # dict로 저장된 경우 실제 모델 및 인코더 꺼내기
        if isinstance(model_data, dict):
            model_key = "model" if "model" in model_data else "estimator"
            if model_key in model_data:
                # 모델, line_encoder, station_encoder를 튜플로 반환
                return (
                    model_data.get(model_key),
                    model_data.get("line_encoder"),
                    model_data.get("station_encoder")
                )
            else:
                raise ValueError("모델 파일 딕셔너리에 'model' 또는 'estimator' 키가 없습니다. 내부 구조를 확인하세요.")

        # 딕셔너리가 아닌 경우 (단일 모델 객체만 저장된 경우)
        return model_data

    except Exception as e:
        error_message = f"FATAL: Hugging Face Hub에서 모델 로드 실패: {e}"
        print(error_message)
        # 애플리케이션 시작을 막기 위해 예외 발생
        raise RuntimeError(error_message)
