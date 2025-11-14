# from app.services.model import load_model
from app.services.preprocessing import preprocess_stats_time_response
import random
from datetime import datetime
import pandas as pd
from app.config.settings import KST
from app.utils.model_loader import load_latest_model, FEATURE_COLUMNS_V1
from typing import List
from app.schemas.predict import PredictResponse, PredictRequest, RouteResponse, SectionResponse, StationResponse, \
    StartAndEndStationResponse, SectionSummary


FEATURE_COLUMNS_V1 = [
    "year",
    "month",
    "hour",
    "line_encoded",
    "station_encoded"
]

# ---- Feature 확장 훅 ----
class FeatureJoiner:
    """
    서버 내부에서 feature를 점진적으로 확장.
    모델에 필요한 컬럼만 슬라이스해서 넣기 때문에
    여기서 더 많은 feature를 추가해도 안전함.
    """
    def join(self, dt_kst: datetime, line: str, station: str) -> dict:
        # 예: 주말/평일, 요일 등 간단 피처부터 시작
        return {
            "weekday": dt_kst.weekday(),            # 0=월 ~ 6=일
            "is_weekend": int(dt_kst.weekday() >= 5)
            # 공휴일/날씨/이벤트 등 추가
        }

feature_joiner = FeatureJoiner()

# 문자열을 KST 기준 datetime 객체로 변환해서 반환
def parse_datetime_kst(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)

def build_feature_row(dt_kst, line, station, line_encoder, station_encoder):
    return {
        "year": dt_kst.year,
        "month": dt_kst.month,
        "hour": dt_kst.hour,
        "line_encoded": int(line_encoder.transform([line])[0]),
        "station_encoded": int(station_encoder.transform([station])[0])
    }


def predict_single(line: str, station: str, dt_kst: datetime, model, line_encoder, station_encoder) -> tuple[
    int, int, dict]:
    feats = build_feature_row(dt_kst, line, station, line_encoder, station_encoder)  # 인코더를 인수로 추가

    # 모델 입력에 맞춰 컬럼을 '슬라이스'
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)


    # 전달받은 통합 모델(model)을 사용하여 예측
    yhat = model.predict(X)[0]
    pred_gton = max(0, int(round(yhat[0])))
    pred_gtoff = max(0, int(round(yhat[1])))
    return pred_gton, pred_gtoff, feats

def generate_mock_predictions(num_cars: int = 10):
    """칸별 혼잡도 가짜 데이터 생성"""
    base = random.uniform(40, 100)
    return [round(max(0, min(160, base + random.uniform(-15, 15))), 1) for _ in range(num_cars)]


def predict_congestion_service(request: PredictRequest) -> PredictResponse:
    """
    요청(PredictRequest)을 받아 예측값(PredictResponse)을 생성.
    실제 모델이 있다면 이 부분에서 호출.
    """
    routes_response: List[RouteResponse] = []

    for route in request.routes:
        section_responses: List[SectionResponse] = []

        for section in route.sections:
            # 도보(3)은 혼잡도 예측 대상 제외
            if section.trafficType == 3:
                data = section.model_dump()
                data.pop("passStopList", None) #원본제거

                section_responses.append(
                    SectionResponse(
                        **data,
                        passStopList=None,
                        sectionSummary=None
                    )
                )
                continue

            # === passStopList 처리 ===
            station_responses: List[StationResponse] = []
            if section.passStopList:
                for s in section.passStopList:
                    station_responses.append(
                        StationResponse(
                            **s.dict(),
                            expectedBoarding=random.randint(0, 50),
                            expectedAlighting=random.randint(0, 50)
                        )
                    )

            # === sectionSummary 생성 ===
            start_station = StartAndEndStationResponse(
                name=section.startName or "UnknownStart",
                expectedBoarding=random.randint(10, 60),
                expectedAlighting=0
            )

            end_station = StartAndEndStationResponse(
                name=section.endName or "UnknownEnd",
                expectedBoarding=0,
                expectedAlighting=random.randint(10, 60)
            )

            section_summary = SectionSummary(
                startStation=start_station.dict(),
                endStation=end_station.dict(),
                avgCongestion=round(random.uniform(60, 95), 2),
                maxCongestion=round(random.uniform(95, 120), 2),
                totalExpectedBoarding=start_station.expectedBoarding,
                totalExpectedAlighting=end_station.expectedAlighting
            )

            # === SectionResponse 완성 ===
            section_response = SectionResponse(
                **data,
                sectionSummary=section_summary,
                passStopList=station_responses
            )


            section_responses.append(section_response)

        # === RouteResponse 완성 ===
        route_response = RouteResponse(
            routeType=route.routeType,
            sections=section_responses
        )
        routes_response.append(route_response)

    # === 최종 PredictResponse ===
    return PredictResponse(
        message="success",
        routes=routes_response
    )