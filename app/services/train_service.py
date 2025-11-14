from datetime import datetime, timedelta
import os, joblib, numpy as np, pandas as pd

from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from app.config.settings import GENERAL_KEY
from app.services.preprocessing import preprocess_stats_time_response
from app.common.fetch import fetch_api

# 1. 최근 6개월 리스트 구하기
def get_recent_months(n_months: int = 6) -> list[str]:
    today = datetime.today()
    months = [
        (today - relativedelta(months=i)).strftime("%Y%m")
        for i in range(n_months)
    ]
    print(f"수집할 월 목록: {months}")
    return months



# 2. 특정 날짜 지하철 승하차 인원을 JSON 형태 -> pandas DataFrame으로 변환
# API: 서울 열린데이터 광장 지하철 호선별 역별 시간대별 승객 현황 조회
def build_dataset_for_date(date: str, line: str = None, station: str = None):
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayTime/1/1000/{date}"
    if line:
        url += f"/{line}" # 특정 노선만 학습할때
    if station:
        url += f"/{station}" # 특정 역만 학습할때
    raw = fetch_api(url)
    rows = raw.get("CardSubwayTime", {}).get("row", [])
    df = pd.DataFrame(rows)
    return df

# 학습 전체 파이프라인
# 역 호선별 개별 모델
# dates: 예측 모델 학습에 사용할 날짜 리스트
def train_for_station_line(months: list[str], line: str, station: str):
    x_list = []
    y_list = []
    for m in months:
        print(f"{m} 데이터 수집 중...")
        # 날짜 하나씩 API로 호출해서 DataFrame 얻기
        df = build_dataset_for_date(m, line=line, station=station)

        if df.empty:
            print(f"{m} 데이터 없음. 건너뜀")
            continue
        print(f"{len(df)}개의 행이 로드됨")

        # row를 preprocess_stats_response(row)(전처리함수)에 전달
        for _, row in df.iterrows():
            results = preprocess_stats_time_response(row) # 각 row를 모델 입력 형식에 맞춰 전처리 (x,y) 튜플로 반환
            if not results:
                continue
            for x,y in results: # 시간대 별로 분해된 샘플들
                x_list.append(x)
                y_list.append(y)

    # x_list: 입력 데이터(features), 예측할 때 입력으로 들어갈 특징
    # y_lisy: 타겟(target)값, 예측해야하는 실제 정답
    x = pd.DataFrame(x_list, columns=["year", "month", "hour"])
    y = pd.DataFrame(y_list, columns=["gton", "gtoff"])

    # 학습/검증 데이터를 8:2로 나눔
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"최종 학습 데이터 크기: X={len(x)}, Y={len(y)}")

    # MultiOutputRegressor로 학습 진행
    model = MultiOutputRegressor(RandomForestRegressor())
    model.fit(X_train, y_train) #이런 feature(x)가 주어졌을때 y를 예측하는 법 학습

    print("모델 학습 완료!")
    # 모델 평가 및 저장
    evaluate_model(model, X_test, y_test)
    save_model(model, line, station)


# 모델 저장 (역,호선별로 따로)
def save_model(model, line, station):
    today = datetime.today().strftime("%Y%m%d")
    os.makedirs("models", exist_ok=True)
    safe_line = line.replace("호선","")
    safe_station = station.replace(" ","_")
    path = f"models/{safe_line}_{safe_station}_{today}.pkl" # models/ 폴더에 날짜 기반으로 .pkl 파일 저장, 이후 API 추론시 이 파일을 로딩하여 사용
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# 모델 성능 평가
def evaluate_model(model, X_test, y_test):
    print("모델 평가 중...")
    pred = model.predict(X_test) # predict(X_test)로 예측한 값과 실제 y_test를 비교
    rmse = np.sqrt(mean_squared_error(y_test, pred)) #예측값 vs 실제값 비교하여 RMSE(평균 제곱근 오차)를 출력
    print(f"[{datetime.today()}] RMSE: {rmse:.2f}")

    # logs 폴더 없으면 생성
    os.makedirs("logs", exist_ok=True)

    # 로그 파일 저장
    with open("logs/eval.log", "a") as f:
        f.write(f"{datetime.today()} RMSE: {rmse:.2f}\n")
