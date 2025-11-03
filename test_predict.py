import pandas as pd
from joblib import load
from app.services.preprocessing import preprocess_stats_time_response
from app.utils.model_loader import load_latest_model, FEATURE_COLUMNS_V1
from app.services.predict import build_feature_row, parse_datetime_kst


# ✅ 테스트 입력 (실제 API 요청과 동일하게 구성)
line = "9호선"
station = "김포공항"
dt_str = "2025-10-06T08:00:00+09:00"  # 테스트용 datetime

# 1️⃣ datetime 문자열을 KST로 변환
dt_kst = parse_datetime_kst(dt_str)

# 2️⃣ feature 생성
feats = build_feature_row(dt_kst, line, station)
print(f"📊 생성된 feature: {feats}")

# 3️⃣ DataFrame으로 변환 (모델 입력 형식 맞추기)
X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)
print(f"📄 모델 입력 X:\n{X}")

# 4️⃣ 최신 모델 불러오기 (자동 캐시 로딩)
model = load_latest_model(line, station)

# 5️⃣ 예측 수행
yhat = model.predict(X)[0]
pred_gton = max(0, int(round(yhat[0])))
pred_gtoff = max(0, int(round(yhat[1])))

# 6️⃣ 결과 출력
print(f"🎯 예측 결과:")
print(f"   🚇 승차 인원 예측: {pred_gton}")
print(f"   🚉 하차 인원 예측: {pred_gtoff}")
