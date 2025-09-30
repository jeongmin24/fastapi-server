from joblib import load
from app.services.preprocessing import preprocess_stats_response
from app.utils.model_loader import get_latest_model_path

# 최신 모델 경로 얻기
model_path = get_latest_model_path()

# 모델 로드
model = load(model_path)

# 테스트용 (실제 API 구조와 유사하게 구성)
row = {
    "SBWY_ROUT_LN_NM": "9호선",
    "SBWY_STNS_NM": "김포공항",
    "GTON_TNOPE": 8000,
    "GTOFF_TNOPE": 6000
}

# 전처리 함수로 x 추출 (y는 예측용이니 무시해도 됨)
result = preprocess_stats_response(row)
if result is None:
    print("❌ 전처리 실패: row가 이상함")
else:
    x, _ = result
    print(f"📊 전처리된 입력값: {x}")

    # 예측
    pred = model.predict([x])
    print(f"🎯 예측 결과: {pred}")