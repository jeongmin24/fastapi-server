import os, glob, re, joblib
from functools import lru_cache
from app.config.settings import settings

# # models/ 디렉토리에서 가장 최신 모델 파일 경로를 반환
# def get_latest_model_path(models_dir: str = "models") -> str:
#     model_files = glob.glob(os.path.join(models_dir, "stats_model_*.pkl"))
#     if not model_files:
#         raise FileNotFoundError("모델 파일이 존재하지 않습니다.")
#
#     # 파일명 기준으로 정렬 (가장 최신 날짜 모델이 맨뒤로)
#     model_files.sort()
#     latest_model = model_files[-1]
#     print(f"최신 모델 로딩: {latest_model}")
#     return latest_model


FEATURE_COLUMNS_V1 = ["year", "month", "hour"]  # 현재 학습 컬럼 (train.py와 동일해야 함)

def _safe_line(line: str) -> str:
    return line.replace("호선", "")

def _safe_station(station: str) -> str:
    return station.replace(" ", "_")

def _latest_model_path(line: str, station: str) -> str | None:
    pattern = os.path.join(settings.MODEL_DIR, f"{_safe_line(line)}_{_safe_station(station)}_*.pkl")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    def extract_date(p: str) -> str:
        m = re.search(r"_(\d{8})\.pkl$", p)
        return m.group(1) if m else "00000000"
    return sorted(candidates, key=extract_date, reverse=True)[0]

@lru_cache(maxsize=1024)
def load_latest_model(line: str, station: str):
    path = _latest_model_path(line, station)
    if not path:
        raise FileNotFoundError(f"모델 없음: {line} {station}")
    return joblib.load(path)