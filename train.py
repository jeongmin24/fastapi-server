import os
import pandas as pd
import joblib
import numpy as np
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from app.common.fetch import fetch_api
from app.config.settings import GENERAL_KEY
from app.services.preprocessing import preprocess_stats_time_response


# 1. 최근 6개월 리스트 구하기
def get_recent_months(n_months: int = 6) -> list[str]:
    today = datetime.today()
    months = [
        (today - relativedelta(months=i)).strftime("%Y%m")
        for i in range(n_months)
    ]
    print(f"✅ 수집할 월 목록: {months}")
    return months


# 2. 특정 날짜 지하철 승하차 인원을 JSON 형태 -> pandas DataFrame으로 변환
# API: 서울 열린데이터 광장 지하철 호선별 역별 시간대별 승객 현황 조회
def build_dataset_for_date(date: str):
    # 특정 노선, 역 필터 없이 최대 1000개의 데이터를 가져옴
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayTime/1/1000/{date}"
    raw = fetch_api(url)
    rows = raw.get("CardSubwayTime", {}).get("row", [])
    df = pd.DataFrame(rows)
    return df


# 학습 전체 파이프라인 (모든 역/호선 통합 학습)
def train_all_lines_and_stations(months: list[str]):
    all_dfs = []
    for m in months:
        print(f"📅 {m} 데이터 수집 중...")
        # 모든 노선/역 데이터를 로드
        df = build_dataset_for_date(m)

        if df.empty:
            print(f"⚠️ {m} 데이터 없음. 건너뜀")
            continue
        print(f"➡️ {len(df)}개의 행이 로드됨")
        all_dfs.append(df)

    if not all_dfs:
        print("🚨 학습할 데이터가 없습니다.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    # --- 특징 엔지니어링: 노선과 역 이름을 숫자로 변환 (Label Encoding) ---
    line_encoder = LabelEncoder()
    station_encoder = LabelEncoder()

    # Key Error 발생 문제를 해결하기 위해 실제 컬럼명으로 수정
    LINE_COL = 'SBWY_ROUT_LN_NM'
    STATION_COL = 'STTN'

    print(f"📊 master_df 컬럼 목록: {master_df.columns.tolist()}")

    # 1. 컬럼 존재 확인 및 에러 핸들링
    required_cols = [LINE_COL, STATION_COL]

    # 필요한 컬럼이 master_df에 모두 존재하는지 확인
    missing_cols = [col for col in required_cols if col not in master_df.columns]

    if missing_cols:
        print(f"❌ DataFrame에 필수 컬럼이 없습니다: {missing_cols}")
        print("API 응답 스키마를 확인하십시오.")
        return

    print(f"✅ 필수 컬럼 확인 완료: {LINE_COL}, {STATION_COL}")

    # 2. 결측치(NaN) 방지 및 인코딩 실행
    master_df[LINE_COL] = master_df[LINE_COL].fillna('UnknownLine')
    master_df[STATION_COL] = master_df[STATION_COL].fillna('UnknownStation')

    master_df['LINE_NUM_ENCODED'] = line_encoder.fit_transform(master_df[LINE_COL])
    master_df['STATION_NAME_ENCODED'] = station_encoder.fit_transform(master_df[STATION_COL])

    print(f"⚙️ 총 {len(line_encoder.classes_)}개 호선, {len(station_encoder.classes_)}개 역 인코딩 완료.")
    # -----------------------------------------------------------------

    x_list = []
    y_list = []

    # 전처리 함수 호출 시 인코딩된 값도 함께 전달
    for _, row in master_df.iterrows():
        # 인코딩된 값을 딕셔너리 형태로 전달하여 preprocessing 함수에서 사용할 수 있게 함
        row_with_encoded = row.to_dict()
        row_with_encoded['LINE_NUM_ENCODED'] = row['LINE_NUM_ENCODED']
        row_with_encoded['STATION_NAME_ENCODED'] = row['STATION_NAME_ENCODED']

        # preprocess_stats_time_response는 [year, month, hour]를 반환할 것임
        results = preprocess_stats_time_response(row_with_encoded)

        if not results:
            continue

        # 시간대 별 샘플 (x, y)를 분해하고 인코딩된 특징을 x에 추가
        line_enc = row['LINE_NUM_ENCODED']
        station_enc = row['STATION_NAME_ENCODED']

        for x_base, y in results:  # x_base = [year, month, hour]
            # 최종 입력 특징: [year, month, hour, line_encoded, station_encoded]
            x_final = x_base + [line_enc, station_enc]
            x_list.append(x_final)
            y_list.append(y)

    # x_list: 입력 데이터(features)
    X_cols = ["year", "month", "hour", "line_encoded", "station_encoded"]
    y_cols = ["gton", "gtoff"]

    x = pd.DataFrame(x_list, columns=X_cols)
    y = pd.DataFrame(y_list, columns=y_cols)

    if x.empty:
        print("🚨 전처리 후 학습 데이터가 생성되지 않았습니다.")
        return

    # 학습/검증 데이터를 8:2로 나눔
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"📊 최종 학습 데이터 크기: X={len(x)}, Y={len(y)}")

    # MultiOutputRegressor로 학습 진행
    model = MultiOutputRegressor(RandomForestRegressor())
    model.fit(X_train, y_train)

    print("✅ 모델 학습 완료!")
    # 모델 평가 및 저장
    evaluate_model(model, X_test, y_test)
    # 모델과 인코더를 함께 저장
    save_model_and_encoders(model, line_encoder, station_encoder)


# 모델 및 인코더 저장 (하나의 pkl 파일로)
def save_model_and_encoders(model, line_encoder, station_encoder):
    today = datetime.today().strftime("%Y%m%d")
    os.makedirs("models", exist_ok=True)

    # 모델, 노선 인코더, 역 인코더를 딕셔너리로 묶어 하나의 파일에 저장
    full_model_package = {
        "model": model,
        "line_encoder": line_encoder,
        "station_encoder": station_encoder
    }

    # 파일 이름을 통합 모델임을 나타내도록 변경
    path = f"models/lines_CardSubwayTime_model_{today}.pkl"
    joblib.dump(full_model_package, path)
    print(f"Model and Encoders saved to {path}")


# 모델 성능 평가
def evaluate_model(model, X_test, y_test):
    print("🔍 모델 평가 중...")
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"[{datetime.today()}] RMSE: {rmse:.2f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/eval.log", "a") as f:
        f.write(f"[{datetime.today()}] Unified Model RMSE: {rmse:.2f}\n")


if __name__ == "__main__":
    print("🔥 train.py 시작됨 (통합 학습 모드)")
    # 최근 6개월 데이터로 학습
    months = get_recent_months(n_months=9)
    train_all_lines_and_stations(months)
