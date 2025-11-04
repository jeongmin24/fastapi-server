import os, joblib
from functools import lru_cache
from app.config.settings import settings

FEATURE_COLUMNS_V1 = ["year", "month", "hour"]  # 현재 학습 컬럼 (train.py와 동일해야 함)

# 고정된 모델 파일 이름
FIXED_MODEL_NAME = "lines_CardSubwayTime_model_20251104.pkl"

def _fixed_model_path() -> str:
    """지정된 고정 모델 파일 경로 반환"""
    model_path = os.path.join(settings.MODEL_DIR, FIXED_MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    return model_path

@lru_cache(maxsize=1)
def load_latest_model(line: str = None, station: str = None):
    """고정 모델만 로드 (line, station 무시)"""
    path = _fixed_model_path()
    print(f"모델 로딩: {path}")
    model_data = joblib.load(path)

    # dict로 저장된 경우 실제 모델 꺼내기
    if isinstance(model_data, dict):
        if "model" in model_data:
            return model_data["model"]
        elif "estimator" in model_data:
            return model_data["estimator"]
        else:
            raise ValueError("모델 파일에 'model' 키가 없습니다. 내부 구조를 확인하세요.")
    return model_data
