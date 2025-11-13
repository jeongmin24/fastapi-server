import random
from datetime import datetime
import pandas as pd
from typing import List

from app.config.settings import KST
from app.schemas.predict import (
    PredictResponse, PredictRequest, RouteResponse, SectionResponse,
    StationResponse, StartAndEndStationResponse, SectionSummary,
    SubwayLane, BusLane
)

FEATURE_COLUMNS_V1 = ["year", "month", "hour", "line_encoded", "station_encoded"]

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

def predict_single(line: str, station: str, dt_kst: datetime, model, line_encoder, station_encoder):
    feats = build_feature_row(dt_kst, line, station, line_encoder, station_encoder)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS_V1]], columns=FEATURE_COLUMNS_V1)

    yhat = model.predict(X)[0]
    pred_gton = max(0, int(round(yhat[0])))
    pred_gtoff = max(0, int(round(yhat[1])))
    return pred_gton, pred_gtoff, feats

def generate_mock_train_congestion(num_cars: int = 10):
    base = random.uniform(40, 100)
    return [round(max(0, min(160, base + random.uniform(-15, 15))), 1) for _ in range(num_cars)]

# ==============================
#  경로 전체 혼잡도 예측
# ==============================
def predict_congestion_service(request: PredictRequest, model=None, line_encoder=None, station_encoder=None) -> PredictResponse:
    routes_response: List[RouteResponse] = []

    for path in request.result.path:
        section_responses: List[SectionResponse] = []

        for sub in path.subPath:
            if sub.trafficType == 3:  # 도보
                section_responses.append(
                    SectionResponse(
                        trafficType=sub.trafficType,
                        distance=sub.distance,
                        sectionTime=sub.sectionTime
                    )
                )
                continue

            # === line name 추출 ===
            if sub.lane:
                lane_info = sub.lane[0]
                if isinstance(lane_info, SubwayLane):
                    line_name = lane_info.name
                elif isinstance(lane_info, BusLane):
                    line_name = lane_info.busNo
                else:
                    line_name = "UnknownLine"
            else:
                line_name = "UnknownLine"

            # === 역 리스트 처리 ===
            station_responses: List[StationResponse] = []
            if sub.passStopList and sub.passStopList.stations:
                for s in sub.passStopList.stations:
                    try:
                        dt_kst = parse_datetime_kst(request.datetime)
                        gton, gtoff, _ = predict_single(
                            line_name, s.stationName, dt_kst,
                            model=model, line_encoder=line_encoder, station_encoder=station_encoder
                        )
                        congestion = generate_mock_train_congestion(10)
                    except Exception:
                        gton = random.randint(0, 50)
                        gtoff = random.randint(0, 50)
                        congestion = generate_mock_train_congestion(10)

                    station_responses.append(
                        StationResponse(
                            stationID=s.stationID,
                            stationName=s.stationName,
                            x=s.x,
                            y=s.y,
                            expectedBoarding=gton,
                            expectedAlighting=gtoff,
                            trainCongestion=congestion
                        )
                    )

            # === 섹션 요약 ===
            start_station = StartAndEndStationResponse(
                name=sub.startName or "UnknownStart",
                expectedBoarding=random.randint(10, 60),
                expectedAlighting=0
            )
            end_station = StartAndEndStationResponse(
                name=sub.endName or "UnknownEnd",
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

            section_response = SectionResponse(
                trafficType=sub.trafficType,
                distance=sub.distance,
                sectionTime=sub.sectionTime,
                stationCount=sub.stationCount,
                lane=[l.dict() for l in sub.lane] if sub.lane else None,
                intervalTime=sub.intervalTime,
                startName=sub.startName,
                endName=sub.endName,
                passStopList=station_responses,
                sectionSummary=section_summary
                # ##TODO: 버스 혼잡도 필드 추가 가능
            )

            section_responses.append(section_response)

        route_response = RouteResponse(
            routeType=path.pathType,
            sections=section_responses
        )
        routes_response.append(route_response)

    return PredictResponse(
        message="success",
        routes=routes_response
    )
