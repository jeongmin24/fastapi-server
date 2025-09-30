import os

import requests
import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor #회귀모델
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from sklearn.multioutput import MultiOutputRegressor

from app.common.fetch import fetch_api
from app.config.settings import GENERAL_KEY
from app.services.preprocessing import preprocess_stats_response

print("🔥 train.py 시작됨")

# 1. 최근 7일치 날짜 리스트 구하기
def get_recent_dates(n_days: int = 7, start_days_ago: int = 3) -> list[str]:
    today = datetime.today() - timedelta(days=start_days_ago)
    dates = [
        (today - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(1, n_days + 1)
    ]
    print(f"✅ 수집할 날짜 목록: {dates}")  # ← 이 줄이 핵심
    return dates



# 2. 특정 날짜 지하철 승하차 인원을 JSON 형태 -> pandas DataFrame으로 변환
# 주어진 날짜에 해당하는 서울열린데이터 API에서 수집한 DataFrame을 반환
def build_dataset_for_date(date: str, line: str = None, station: str = None):
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayStatsNew/1/1000/{date}"
    if line:
        url += f"/{line}"
    if station:
        url += f"/{station}"
    raw = fetch_api(url)
    rows = raw.get("CardSubwayStatsNew", {}).get("row", [])
    df = pd.DataFrame(rows)
    return df

# 학습 전체 파이프라인
# dates: 예측 모델 학습에 사용할 날짜 리스트
def train_for_dates(dates: list[str]):
    x_list = []
    y_list = []
    for date in dates:
        print(f"📅 {date} 데이터 수집 중...")
        df = build_dataset_for_date(date) # df: 날짜별 승하차 데이터 DataFrame

        if df.empty:
            print(f"⚠️ {date} 데이터 없음. 건너뜀")
            continue

        print(f"➡️ {len(df)}개의 행이 로드됨")
        for _, row in df.iterrows():
            result = preprocess_stats_response(row) # 각 row를 모델 입력 형식에 맞춰 전처리 (x,y) 튜플로 반환 -> 예측 모델이 학습할 수 있도록 데이터를 정제함
            if result is None:
                continue
            x, y = result
            x_list.append(x)
            y_list.append(y)

    # x_list: 입력 데이터(features), 예측할 때 입력으로 들어갈 특징
    # y_lisy: 타겟(target)값, 예측해야하는 실제 정답
    x = pd.DataFrame(x_list)
    y = pd.DataFrame(y_list)

    # 학습/검증 데이터를 8:2로 나눔
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"📊 최종 학습 데이터 크기: X={len(x)}, Y={len(y)}")

    # MultiOutputRegressor로 학습 진행
    model = MultiOutputRegressor(RandomForestRegressor())
    model.fit(X_train, y_train) #이런 feature(x)가 주어졌을때 y를 예측하는 법 학습

    print("✅ 모델 학습 완료!")
    # 모델 평가 및 저장
    evaluate_model(model, X_test, y_test)
    save_model(model)


# 모델 저장
def save_model(model):
    today = datetime.today().strftime("%Y%m%d")
    os.makedirs("models", exist_ok=True)
    path = f"models/stats_model_{today}.pkl" # models/ 폴더에 날짜 기반으로 .pkl 파일 저장, 이후 API 추론시 이 파일을 로딩하여 사용
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# 모델 성능 평가
def evaluate_model(model, X_test, y_test):
    print("🔍 모델 평가 중...")
    pred = model.predict(X_test) # predict(X_test)로 예측한 값과 실제 y_test를 비교
    rmse = np.sqrt(mean_squared_error(y_test, pred)) #예측값 vs 실제값 비교하여 RMSE(평균 제곱근 오차)를 출력
    print(f"[{datetime.today()}] RMSE: {rmse:.2f}")

    # logs 폴더 없으면 생성
    os.makedirs("logs", exist_ok=True)

    # 로그 파일 저장
    with open("logs/eval.log", "a") as f:
        f.write(f"{datetime.today()} RMSE: {rmse:.2f}\n")


if __name__ == "__main__":
    # 최근 7일치 데이터로 하습
    dates = get_recent_dates(7, start_days_ago=3)
    train_for_dates(dates)