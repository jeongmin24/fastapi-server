import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# uvicorn은 터미널에서 실행할 거라 import는 안 해도 되지만, 관습상 두었습니다.
# import uvicorn

# ======================================================================
# 0. 초기 설정 및 모델 로딩
# ======================================================================

# 모델 파일 경로 설정 (train_model.py와 동일한 위치에 있어야 함)
# 모델 파일 경로 설정
MODEL_FILE_NAME = 'train_congestion_model.pkl'

# 모델이 저장된 'models' 폴더를 지정
MODEL_DIR = os.path.join(os.getcwd(), 'models')

# 모델 파일의 전체 경로 지정
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME) # <--- 이 부분이 수정되었습니다.
congestion_model = None

# 서버 띄우기 전에 모델 파일을 한 번만 로드해야 메모리를 아낄 수 있음.
try:
    with open(MODEL_PATH, 'rb') as f:
        congestion_model = pickle.load(f)
    print(f"[!] 모델 로드 성공: '{MODEL_FILE_NAME}'. 예측 준비 완료.")
except FileNotFoundError:
    print(f"[ERROR] 모델 파일 '{MODEL_FILE_NAME}'을 못 찾았습니다. 경로를 확인하거나 학습을 먼저 진행해주세요.")
    # 모델 없으면 예측 요청 시 503 에러를 낼 거임.
except Exception as e:
    print(f"[CRITICAL ERROR] 모델 로딩 중 문제 발생: {e}")
    congestion_model = None

# 모델 학습에 사용했던 기준 역 목록 (이것들은 변하지 않음)
TRANSFER_STATION_LIST = ['신도림역', '사당역', '을지로입구역', '불암산역']
BUSINESS_STATION_LIST = ['강남역', '여의도역', '서울역', '선릉역']
# 학습에 사용된 모든 노선 더미 변수 목록 (학습 코드의 결과와 정확히 일치해야 함!)
# 만약 학습 데이터에 1호선부터 9호선까지 있었다면 이 목록을 모두 적어줘야 함.
KNOWN_LINE_DUMMIES = [f'Line_{i}호선' for i in range(1, 10)]

app = FastAPI(title="지하철 혼잡도 예측 API (학생 프로젝트 버전)")


# ======================================================================
# 1. API 요청 데이터 스키마
# ======================================================================

class PredictionRequest(BaseModel):
    """API 요청으로 받을 데이터 정의. Pydantic 사용."""
    start_station: str  # 출발역
    end_station: str  # 도착역
    line_name: str  # 노선 (예: '2호선', '1호선')
    hour_of_day: int  # 시간 (시 단위, 0~23)


# ======================================================================
# 2. 예측 특징 생성 및 예측 로직
# ======================================================================

def create_features_for_prediction(req_data: PredictionRequest, car_num: int):
    """
    하나의 칸에 대한 요청 데이터를 OLS 모델 입력 형태로 변환하는 함수.
    """

    # 1. 기본 특징 딕셔너리 생성
    feature_dict = {
        'AM_Peak': 1 if req_data.hour_of_day == 8 else 0,
        'PM_Peak': 1 if req_data.hour_of_day in [18, 19] else 0,

        'Start_Transfer': 1 if req_data.start_station in TRANSFER_STATION_LIST else 0,
        'Start_Business': 1 if req_data.start_station in BUSINESS_STATION_LIST else 0,

        'Arrival_Transfer': 1 if req_data.end_station in TRANSFER_STATION_LIST else 0,
        'Arrival_Business': 1 if req_data.end_station in BUSINESS_STATION_LIST else 0,

        '칸_번호': car_num  # 칸 번호는 반복문에서 받아옴
    }

    # 2. 노선 더미 변수 처리 (학습 데이터 컬럼 순서를 맞추기 위해 필수)
    line_dummies = {col: 0 for col in KNOWN_LINE_DUMMIES}
    line_key_to_set = f'Line_{req_data.line_name}'

    if line_key_to_set in line_dummies:
        line_dummies[line_key_to_set] = 1
    # else: 학습에 없던 노선은 모두 0으로 처리됨.

    feature_dict.update(line_dummies)

    # 3. 모델 입력 데이터프레임 및 상수항 추가
    X_predict_df = pd.DataFrame([feature_dict])
    X_predict_with_const = sm.add_constant(X_predict_df, has_constant='add')

    # 4. 최종 컬럼 순서 맞추기 (가장 중요한 부분: 모델 파라미터 순서 유지)
    # 로드된 모델의 독립 변수 이름 순서를 그대로 사용
    model_param_names = ['const'] + list(congestion_model.params.index[1:])

    # 순서에 맞게 재정렬하고, 빠진 컬럼은 0으로 채움 (fill_value=0)
    X_final = X_predict_with_const.reindex(columns=model_param_names, fill_value=0)

    return X_final


# ======================================================================
# 3. API 엔드포인트
# ======================================================================

@app.post("/predict/train_congestion")
def get_prediction_results(request: PredictionRequest):
    """
    POST 요청을 받아 1호차부터 10호차까지의 예상 혼잡도를 반환.
    """

    if congestion_model is None:
        # 모델 로드 실패 시 503 Service Unavailable 에러 반환
        raise HTTPException(status_code=503, detail="모델이 로드되지 않아 예측할 수 없습니다.")

    all_car_results = []
    total_predicted_passengers = 0.0

    # 1호차부터 10호차까지 반복하며 예측
    for car_num in range(1, 11):
        try:
            # 1. 특징 데이터 준비
            X_data = create_features_for_prediction(request, car_num)

            # 2. 예측 실행 (결과값은 승객 수)
            predicted_passengers = congestion_model.predict(X_data.iloc[0]).iloc[0]

            # OLS 모델의 단점: 음수 예측 방지 -> 최소 0
            passenger_count = max(0.0, predicted_passengers)

            # 3. 혼잡도 변환 (혼잡도 = 승객 수 / 1.6)
            congestion_percent = (passenger_count / 1.6)

            all_car_results.append({
                "car_number": car_num,
                "predicted_passengers": round(passenger_count, 2),
                "predicted_congestion_percent": round(congestion_percent, 2)
            })

            total_predicted_passengers += passenger_count

        except Exception as e:
            # 예상치 못한 디버깅 오류가 발생하면 500 에러를 반환해야 함.
            print(f"[CRITICAL] 예측 중 알 수 없는 오류: {e}")
            raise HTTPException(status_code=500, detail=f"예측 처리 중 오류: {e}")

    # 최종 응답 데이터 구성
    return {
        "status": "success",
        "request_info": request.dict(),
        "total_predicted_passengers": round(total_predicted_passengers, 2),
        "car_congestion_details": all_car_results
    }


# 가상의 테스트 시나리오 설정 (예: 강남역 -> 사당역, 2호선, 저녁 6시)
TEST_DEPARTURE = '강남역'
TEST_ARRIVAL = '사당역'
TEST_TIME = 18
TEST_LINE = '2호선'

print("\n" + "=" * 50)
print(f"       [테스트 시나리오: {TEST_DEPARTURE} -> {TEST_ARRIVAL} ({TEST_LINE}) {TEST_TIME}시]")
print("=" * 50)

# ======================================================================
# 4. 스크립트 자체 테스트 실행 블록 (IF __NAME__ == '__MAIN__' 역할)
# ======================================================================

if __name__ == '__main__':

    # 가상의 테스트 시나리오 설정
    TEST_DEPARTURE = '강남역'
    TEST_ARRIVAL = '사당역'
    TEST_TIME = 18
    TEST_LINE = '2호선'

    print("\n" + "=" * 50)
    print(f"       [테스트 시나리오: {TEST_DEPARTURE} -> {TEST_ARRIVAL} ({TEST_LINE}) {TEST_TIME}시]")
    print("=" * 50)

    # 1. 모델 로드 확인
    if congestion_model is None:
        print("모델 로드 실패 상태입니다. 예측을 건너뜁니다.")
        # 모델 로드 오류 시 예측 중단
    else:
        try:
            # FastAPI의 Pydantic 모델을 사용하여 입력 데이터 객체 생성
            request_data = PredictionRequest(
                start_station=TEST_DEPARTURE,
                end_station=TEST_ARRIVAL,
                line_name=TEST_LINE,
                hour_of_day=TEST_TIME
            )

            all_car_results = []
            total_predicted_passengers = 0.0

            # 1호차부터 10호차까지 반복하며 예측 (FastAPI 엔드포인트 로직과 동일)
            for car_num in range(1, 11):
                # 1. 특징 데이터 준비 (위에 정의된 create_features_for_prediction 사용)
                X_data = create_features_for_prediction(request_data, car_num)

                # 2. 예측 실행
                predicted_passengers = congestion_model.predict(X_data.iloc[0]).iloc[0]
                passenger_count = max(0.0, predicted_passengers)
                congestion_percent = (passenger_count / 1.6)

                all_car_results.append({
                    "car_number": car_num,
                    "predicted_passengers": round(passenger_count, 2),
                    "predicted_congestion_percent": round(congestion_percent, 2)
                })
                total_predicted_passengers += passenger_count

            # 3. 예측 결과를 보기 좋게 출력
            print(f"총 예측 승객 수: {round(total_predicted_passengers, 2)}명")
            print("\n| 칸 번호 | 예측 승객수 (명) | 예측 혼잡도 (%) |")
            print("|-------|----------------|----------------|")
            for r in all_car_results:
                print(
                    f"| {r['car_number']:7} | {r['predicted_passengers']:14.2f} | {r['predicted_congestion_percent']:14.2f} |")
            print("-" * 50)

        except Exception as e:
            print(f"[ERROR] 예측 실행 중 치명적인 오류 발생: {e}")
