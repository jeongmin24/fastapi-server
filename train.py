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

# TODO: app.common.fetch와 app.config.settings 경로가 로컬 환경에 맞는지 확인하세요.
# Render 환경에서는 API 호출 시 GENERAL_KEY가 환경 변수로 설정되어야 합니다.
from app.common.fetch import fetch_api
from app.config.settings import GENERAL_KEY
from app.services.preprocessing import preprocess_stats_time_response


# 1. 최근 N개월 리스트 구하기
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
            print(f"⚠️ {m} 데이터 없음. 건너뜔")
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

    # 실제 컬럼명
    LINE_COL = 'SBWY_ROUT_LN_NM'
    STATION_COL = 'STTN'

    print(f"📊 master_df 컬럼 목록: {master_df.columns.tolist()}")

    # 1. 컬럼 존재 확인 및 에러 핸들링
    required_cols = [LINE_COL, STATION_COL]
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
        row_with_encoded = row.to_dict()
        row_with_encoded['LINE_NUM_ENCODED'] = row['LINE_NUM_ENCODED']
        row_with_encoded['STATION_NAME_ENCODED'] = row['STATION_NAME_ENCODED']

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

    # MultiOutputRegressor로 학습 진행 (RandomForestRegressor에 하이퍼파라미터 적용)
    # n_estimators=50, max_depth=15 적용
    base_estimator = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_estimator)
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
    # .pkl.gz 확장자를 사용하여 압축되었음을 명시 (선택 사항)
    path = f"models/lines_CardSubwayTime_model_{today}.pkl"

    # joblib의 compress 인자를 사용하여 GZIP 압축 레벨 7로 저장
    # compress=('gzip', 7) 적용
    joblib.dump(full_model_package, path, compress=('gzip', 7))

    # 저장 후 파일 크기 출력
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Model and Encoders saved to {path}. Compressed size: {size_mb:.2f} MB")


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
    # 최근 9개월 데이터로 학습
    months = get_recent_months(n_months=9)
    train_all_lines_and_stations(months)
