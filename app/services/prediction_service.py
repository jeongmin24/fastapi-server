import pickle
import numpy as np
from typing import List
from datetime import datetime
from fastapi import HTTPException
# from ..config.settings import MODEL_PATH
from ..schemas.predict import CongestionDetail, CongestionResponse, CongestionRequest, PredictRequest, PredictResponse, \
    RouteResponse, SectionResponse, Section, SectionSummary, StationResponse
from ..utils.feature_utils import create_features_for_prediction
from huggingface_hub import hf_hub_download

# 1. Hugging Face 저장소 정보
HF_REPO_ID = "gcanoca/SubwayCongestionPkl"
# 2. 저장소에 있는 모델 파일의 정확한 이름
MODEL_FILENAME = "train_congestion_model.pkl"

# 모델 로드 (서버 시작 시 한 번만)
congestion_model = None

try:
    # ------------------------------------------------------------------
    # 3. hf_hub_download를 사용하여 모델 파일을 로컬 임시 경로로 다운로드
    # ------------------------------------------------------------------
    MODEL_PATH_LOCAL = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        # 다운로드할 로컬 캐시 폴더 (선택 사항)
        # cache_dir="/tmp/hf_cache"
        repo_type="dataset"
    )

    # 4. 다운로드된 로컬 경로를 사용하여 파일 열기 및 로드
    with open(MODEL_PATH_LOCAL, 'rb') as f:
        congestion_model = pickle.load(f)

    print(f"[Service] 모델 로드 성공: {MODEL_PATH_LOCAL} (Hub: {HF_REPO_ID}/{MODEL_FILENAME})")

    # (선택적) 모델 로드 후 임시 파일 삭제
    # os.remove(MODEL_PATH_LOCAL)

except ImportError:
    print("[Service CRITICAL ERROR] 'huggingface_hub' 라이브러리가 설치되지 않았습니다. pip install huggingface_hub 필요.")
    congestion_model = None
except Exception as e:
    print(f"[Service CRITICAL ERROR] Hugging Face 모델 로드 실패 또는 파일 문제: {e}")
    congestion_model = None


def get_congestion_prediction(request: CongestionRequest) -> CongestionResponse:
    """
    1호차부터 10호차까지의 혼잡도를 예측하고 결과를 포맷하는 서비스 로직
    """
    if congestion_model is None:
        raise HTTPException(status_code=503, detail="예측 모델이 로드되지 않았습니다. 서버 설정을 확인하세요.")

    all_car_results = []
    total_predicted_passengers = 0.0

    # 1호차부터 10호차까지 반복하며 예측
    for car_num in range(1, 11):
        try:
            # 특징 생성 (utils 사용)
            X_data = create_features_for_prediction(request, car_num, congestion_model)

            # 예측 실행
            predicted_passengers = congestion_model.predict(X_data.iloc[0]).iloc[0]
            passenger_count = max(0.0, predicted_passengers)
            congestion_percent = (passenger_count / 1.6)  # 기준 혼잡도 1.6

            detail = CongestionDetail(
                car_number=car_num,
                predicted_passengers=round(passenger_count, 2),
                predicted_congestion_percent=round(congestion_percent, 2)
            )
            all_car_results.append(detail)
            total_predicted_passengers += passenger_count

        except Exception as e:
            print(f"[CRITICAL] 예측 중 알 수 없는 오류: {e}")
            raise HTTPException(status_code=500, detail=f"예측 처리 중 오류: {e}")

    return CongestionResponse(
        status="success",
        request_info=request,
        total_predicted_passengers=round(total_predicted_passengers, 2),
        car_congestion_details=all_car_results
    )

def predict_routes_congestion(request: PredictRequest, hour: int | None = None) -> PredictResponse:
    if hour is None:
        # 현재 시각 기준
        hour = datetime.now().hour

    if congestion_model is None:
        raise HTTPException(status_code=503, detail="예측 모델이 로드되지 않았습니다.")

    route_responses = []

    for route in request.routes:
        section_responses = []

        for section in route.sections:

            # Section → 기존 CongestionRequest 변환
            req_data = convert_section_to_congestion_request(section, hour)

            if req_data is None:
                # None이면 예측 불가 구간 → sectionSummary 없이 응답 생성
                section_responses.append(
                    SectionResponse(
                        trafficType=section.trafficType,
                        lanes=section.lanes,
                        distance=section.distance,
                        sectionTime=section.sectionTime,
                        stationCount=section.stationCount,
                        way=section.way,
                        wayCode=section.wayCode,
                        startName=section.startName,
                        startX=section.startX,
                        startY=section.startY,
                        endName=section.endName,
                        endX=section.endX,
                        endY=section.endY,
                        sectionSummary=None,
                        passStopList=None
                    )
                )
                continue

            car_details = []
            total_passengers = 0
            congestion_values = []

            # get_congestion_prediction(1호차~10호차 반복)
            for car_num in range(1, 11):
                X_data = create_features_for_prediction(req_data, car_num, congestion_model)

                pred = congestion_model.predict(X_data.iloc[0]).iloc[0]
                passenger = max(0, pred)
                congestion_percent = passenger / 1.6

                car_details.append({
                    "car": car_num,
                    "passengers": passenger,
                    "congestion": congestion_percent
                })

                total_passengers += passenger
                congestion_values.append(congestion_percent)

            # passStopList 기반 승차 하차 예측
            pass_stop_responses = []

            if section.passStopList:  # ← PredictRequest가 passStopList 보내는 경우
                for stop in section.passStopList:
                    # 정류장 단위 승하차 예측은 “해당 구간 전체 수요의 일정 비율”로 단순화
                    # 별도 모델 호출 가능
                    expected_boarding = int(total_passengers * 0.6 / len(section.passStopList))
                    expected_alighting = int(total_passengers * 0.4 / len(section.passStopList))

                    pass_stop_responses.append(
                        StationResponse(
                            index=stop.index,
                            stationID=stop.stationID,
                            stationName=stop.stationName,
                            expectedBoarding=expected_boarding,
                            expectedAlighting=expected_alighting,
                            predictedCongestionCar = [round(c, 2) for c in congestion_values]

                        )
                    )
            else:
                pass_stop_responses = None

            # SectionSummary 생성
            summary = SectionSummary(
                startStation={
                    "name": section.startName,
                    "expectedBoarding": int(total_passengers * 0.6),
                    "expectedAlighting": 0
                },
                endStation={
                    "name": section.endName,
                    "expectedBoarding": 0,
                    "expectedAlighting": int(total_passengers * 0.4)
                },
                avgCongestion=round(sum(congestion_values)/len(congestion_values), 2),
                maxCongestion=round(max(congestion_values), 2),
                totalExpectedBoarding=int(total_passengers * 0.6),
                totalExpectedAlighting=int(total_passengers * 0.4)
            )

            # SectionResponse 생성
            section_responses.append(
                SectionResponse(
                    trafficType=section.trafficType,
                    lanes=section.lanes,
                    distance=section.distance,
                    sectionTime=section.sectionTime,
                    stationCount=section.stationCount,
                    way=section.way,
                    wayCode=section.wayCode,
                    startName=section.startName,
                    startX=section.startX,
                    startY=section.startY,
                    endName=section.endName,
                    endX=section.endX,
                    endY=section.endY,
                    sectionSummary=summary,
                    passStopList=pass_stop_responses
                )
            )

        route_responses.append(
            RouteResponse(
                routeType=route.routeType,
                sections=section_responses
            )
        )

    return PredictResponse(
        message="success",
        routes=route_responses
    )

def convert_section_to_congestion_request(section: Section, hour: int) -> CongestionRequest:
    """
    새 DTO의 section → 기존 CongestionRequest 형태로 변환
    """
    # 도보 구간은 예측 대상이 아님 → None 반환
    if section.trafficType == 3:
        return None

    # way(방면)를 라인명처럼 사용
    # 예: "강남방면", "사당방면"
    if section.way:
        line_name = section.way
        # way가 없으면 노선명으로 fallback
    elif section.lanes and section.lanes[0].name:
        line_name = section.lanes[0].name
    else:
        line_name = "Unknown"

    # start/end 보정
    start_name = section.startName
    end_name = section.endName

    return CongestionRequest(
        hour_of_day=hour,
        line_name=line_name,
        start_station=start_name,
        end_station=end_name
    )
