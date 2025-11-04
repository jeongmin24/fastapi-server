import joblib
import pandas as pd
from datetime import datetime
import numpy as np

# 모델 로드
bundle = joblib.load("models/lines_CardSubwayTime_model_20251104.pkl")
model = bundle["model"]
line_encoder = bundle["line_encoder"]
station_encoder = bundle["station_encoder"]

# 사용자 입력
line = "8호선"
station = "석촌"
now = datetime.now()

# 시간 입력
year = now.year
month = now.month
hour = now.hour

# 인코딩
line_encoded = line_encoder.transform([line])[0]
station_encoded = station_encoder.transform([station])[0]

# 입력 데이터 구성
X = pd.DataFrame([{
    "year": year,
    "month": month,
    "hour": hour,
    "line_encoded": line_encoded,
    "station_encoded": station_encoded
}])

# 예측 수행
yhat = model.predict(X)[0]

# 결과 해석
if isinstance(yhat, (list, tuple, np.ndarray)) and len(yhat) == 2:
    boarding, alighting = yhat
elif isinstance(yhat, (list, tuple, np.ndarray)) and len(yhat) == 1:
    boarding, alighting = yhat[0], None
else:
    boarding, alighting = yhat, None

# numpy 배열일 경우 첫 원소만 추출
if isinstance(boarding, (np.ndarray, list)):
    boarding = boarding[0]
if isinstance(alighting, (np.ndarray, list)):
    alighting = alighting[0]

# 혼잡도 계산 (예시)
if alighting is not None:
    max_capacity = 1300
    congestion = (boarding - alighting) / max_capacity * 100
    congestion = max(0, min(congestion, 200))
else:
    congestion = None

# 출력
print(f"예측 승차인원: {boarding:.0f}")
if alighting is not None:
    print(f"예측 하차인원: {alighting:.0f}")
if congestion is not None:
    print(f"예상 혼잡도: {congestion:.1f}%")