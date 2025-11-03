import os
import requests
import pandas as pd
import joblib
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# 프로젝트 구조에 따라 경로 수정 필요
from app.common.fetch import fetch_api
from app.config.settings import GENERAL_KEY
from app.services.preprocessing import preprocess_stats_time_response

# ----------------------------------------------------------------------
# ⚡️ 통합 모델 학습 상수
# ----------------------------------------------------------------------
INTEGRATED_FEATURES = ["year", "month", "hour", "line_station"]  # 통합 모델에 사용할 특징
TARGET_API_ROW_LIMIT = 1000  # API 호출 시 한 번에 가져올 최대 행 수 (서울시 API 기준)


# ----------------------------------------------------------------------
# 1. 지원하는 모든 역/호선 리스트 구하기 (자동 수집)
# ----------------------------------------------------------------------
def get_all_active_stations() -> list[tuple[str, str]]:
    """
    API를 호출하여 현재 운영 중인 모든 역과 호선 정보를 가져옵니다.
    """
    # 🚨 API 경로는 실제 지하철 역 정보 API 엔드포인트로 변경해야 합니다.
    # 서울시 지하철역 정보 API (예시)
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/SearchSTNBySubwayLineInfo/1/{TARGET_API_ROW_LIMIT}/"

    try:
        raw = fetch_api(url)
        rows = raw.get("SearchSTNBySubwayLineInfo", {}).get("row", [])
    except Exception as e:
        print(f"🚨 지하철 역 정보 API 호출 오류: {e}")
        return []

    station_list = []

    for row in rows:
        line = row.get("LINE_NUM")  # 예: '2호선'
        station = row.get("STN_NM")  # 예: '강남'

        if line and station:
            # 일반적으로 1호선~9호선과 같은 정식 노선만 포함
            if line.replace("호선", "").isdigit() or line in ["신분당선", "경의중앙선"]:
                station_list.append((line, station))

    # 중복 제거 (예: 서울역은 여러 호선에 존재하므로)
    unique_stations = sorted(list(set(station_list)))
    print(f"✅ 총 {len(unique_stations)}개의 고유 역/호선 쌍 수집 완료.")
    return unique_stations


# ----------------------------------------------------------------------
# 2. 최근 6개월 리스트 구하기
# ----------------------------------------------------------------------
def get_recent_months(n_months: int = 6) -> list[str]:
    today = datetime.today()
    months = [
        (today - relativedelta(months=i)).strftime("%Y%m")
        for i in range(n_months)
    ]
    print(f"✅ 수집할 월 목록: {months}")
    return months


# 3. 특정 날짜 지하철 승하차 인원을 JSON 형태 -> pandas DataFrame으로 변환
def build_dataset_for_date(date: str, line: str = None, station: str = None):
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayTime/1/{TARGET_API_ROW_LIMIT}/{date}"
    if line:
        url += f"/{line}"
    if station:
        url += f"/{station}"

    raw = fetch_api(url)
    rows = raw.get("CardSubwayTime", {}).get("row", [])
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# 4. 통합 모델 학습 파이프라인
# ----------------------------------------------------------------------
def train_integrated_model(months: list[str], target_stations: list[tuple[str, str]]):
    x_list = []
    y_list = []

    for line, station in target_stations:  # 모든 역/호선 쌍을 순회
        line_station_key = f"{line}_{station}"  # 고유 키: 2호선_강남

        for m in months:
            # print(f"📅 {m} [{line_station_key}] 데이터 수집 중...") # 로그가 너무 길어질 수 있음

            # 특정 역의 데이터만 API로 호출해서 DataFrame 얻기 (API 효율을 위해)
            df = build_dataset_for_date(m, line=line, station=station)

            if df.empty:
                continue

            for _, row in df.iterrows():
                results = preprocess_stats_time_response(row)
                if not results:
                    continue

                for x, y in results:  # 시간대 별로 분해된 샘플들
                    # 🚨 특징 확장: 역/호선 특징 추가
                    x['line_station'] = line_station_key
                    x_list.append(x)
                    y_list.append(y)

    # DataFrame 생성 및 One-Hot Encoding 적용
    x_combined = pd.DataFrame(x_list, columns=INTEGRATED_FEATURES)
    y = pd.DataFrame(y_list, columns=["gton", "gtoff"])

    # 🚨 One-Hot Encoding 적용 🚨
    # 'line_station' 컬럼을 OHE하여 모든 역 정보를 수치형 특징으로 변환
    X_final = pd.get_dummies(x_combined, columns=['line_station'], prefix='station')

    # 학습/검증 데이터를 8:2로 나눔
    # OHE 후 컬럼 수가 크게 증가하므로, 메모리 관리가 필요할 수 있습니다.
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    print(f"📊 최종 통합 학습 데이터 크기: X={len(X_final)}, 특징(컬럼) 수: {len(X_final.columns)}")
    print(f"📊 최종 타겟 데이터 크기: Y={len(y)}")

    # MultiOutputRegressor로 학습 진행
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1))  # n_jobs=-1로 병렬 처리
    model.fit(X_train, y_train)

    print("✅ 통합 모델 학습 완료!")

    # 모델 평가 및 저장
    evaluate_model(model, X_test, y_test)
    save_integrated_model(model, X_final.columns.tolist())  # 특징 컬럼 목록도 저장 (예측 시 필요)


# ----------------------------------------------------------------------
# 5. 모델 저장 (단일 파일) 및 평가
# ----------------------------------------------------------------------
def save_integrated_model(model, feature_columns: list[str]):
    # 이제 모든 예측 데이터를 담은 단일 파일로 저장
    os.makedirs("models", exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    path = f"models/integrated_stats_model_{today}.pkl"

    # 모델 객체뿐만 아니라, 예측 시 OHE에 필요한 특징 컬럼 리스트도 함께 저장합니다.
    joblib.dump({
        'model': model,
        'feature_columns': feature_columns
    }, path)

    print(f"Model saved to {path}")


def evaluate_model(model, X_test, y_test):
    # ... (모델 평가 함수는 기존과 동일하게 유지)
    print("🔍 모델 평가 중...")
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"[{datetime.today()}] 통합 모델 RMSE: {rmse:.2f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/eval.log", "a") as f:
        f.write(f"{datetime.today()} (Integrated) RMSE: {rmse:.2f}\n")


# ----------------------------------------------------------------------
# 6. 실행 (Main)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("🔥 train.py 시작됨: 통합 모델 학습 모드")

    # 1. 모든 역/호선 정보 자동 수집
    TARGET_STATIONS = get_all_active_stations()

    if not TARGET_STATIONS:
        print("🚨 오류: 학습할 역 정보가 API에서 수집되지 않아 학습을 중단합니다.")
    else:
        # 2. 통합 모델 학습 실행
        months = get_recent_months(n_months=6)
        train_integrated_model(months, TARGET_STATIONS)