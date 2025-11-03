# from app.services.model import load_model
from app.services.preprocessing import preprocess_stats_time_response

from datetime import datetime
import pandas as pd
from app.config.settings import KST
from app.utils.model_loader import load_latest_model, FEATURE_COLUMNS_V1


# ---- Feature 확장 훅 ----
class FeatureJoiner:
    """
    서버 내부에서 feature를 점진적으로 확장.
    모델에 필요한 컬럼만 슬라이스해서 넣기 때문에
    여기서 더 많은 feature를 추가해도 안전함.
    """
    def join(self, dt_kst: datetime, line: str, station: str) -> dict:
        # 예: 주말/평일, 요일 등 간단 피처부터 시작
        return {
            "weekday": dt_kst.weekday(),            # 0=월 ~ 6=일
            "is_weekend": int(dt_kst.weekday() >= 5)
            # TODO: 공휴일/날씨/이벤트 등 추가
        }

feature_joiner = FeatureJoiner()

# 문자열을 KST 기준 datetime 객체로 변환해서 반환
def parse_datetime_kst(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)

def build_feature_row(dt_kst: datetime, line: str, station: str) -> dict:
    base = {
        "year": dt_kst.year,
        "month": dt_kst.month,
        "hour": dt_kst.hour,
    }
    extra = feature_joiner.join(dt_kst, line, station)
    return {**base, **extra}

def predict_single(line: str, station: str, dt_kst: datetime) -> tuple[int, int, dict]:
    feats = build_feature_row(dt_kst, line, station)

    # 모델 입력에 맞춰 컬럼을 '슬라이스'
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)

    model = load_latest_model(line, station)
    yhat = model.predict(X)[0]  # [gton, gtoff] 가정
    pred_gton = max(0, int(round(yhat[0])))
    pred_gtoff = max(0, int(round(yhat[1])))
    return pred_gton, pred_gtoff, feats