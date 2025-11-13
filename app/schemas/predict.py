from pydantic import BaseModel, Field
from typing import Dict
from typing import List, Optional

"""
혼잡도 요청 응답 dto
"""
class PredictSingleRequest(BaseModel):
    line: str = Field(..., example="9호선")
    station: str = Field(..., example="여의도")
    datetime: str = Field(..., example="2025-10-06T08:00:00+09:00") # 예측 시점

class PredictSingleResponse(BaseModel):
    line: str
    station: str
    datetime: str # 요청시각 (예측 기준 시점)
    pred_gton: int # 예측된 승차인원
    pred_gtoff: int # 예측된 하차인원
    features_used: Dict[str, int] # 모델 입력으로 실제 사용된 feature
    predicted_count: float


## 칸 별 혼잡도용
class CongestionRequest(BaseModel):
    """지하철 칸별 혼잡도 예측 API의 요청 데이터 정의."""
    start_station: str  # 출발역
    end_station: str    # 도착역
    line_name: str      # 노선 (예: '2호선')
    hour_of_day: int    # 시간 (시 단위, 0~23)

class CongestionDetail(BaseModel):
    """각 칸의 예측 결과를 담는 상세 스키마."""
    car_number: int
    predicted_passengers: float
    predicted_congestion_percent: float

class CongestionResponse(BaseModel):
    """최종 응답 데이터 구조."""
    status: str
    request_info: CongestionRequest
    total_predicted_passengers: float
    car_congestion_details: list[CongestionDetail]

"""
경로 기준 혼잡도 요청 & 응답
"""

class Station(BaseModel):
    index: int
    stationID: int
    stationName: str
    x: str
    y: str

class PassStopList(BaseModel):
    stations: List[Station]

class Lane(BaseModel):
    name: str
    subwayCode: int
    subwayCityCode: int

class SubPath(BaseModel):
    trafficType: int
    distance: int
    sectionTime: int
    stationCount: Optional[int] = None
    lane: Optional[List[Lane]] = None
    intervalTime: Optional[int] = None
    startName: Optional[str] = None
    endName: Optional[str] = None
    passStopList: Optional[PassStopList] = None

class Info(BaseModel):
    trafficDistance: float
    totalWalk: int
    totalTime: int
    payment: int
    subwayTransitCount: int
    firstStartStation: str
    lastEndStation: str
    totalStationCount: int
    totalIntervalTime: Optional[int] = None

class Path(BaseModel):
    pathType: int
    info: Info
    subPath: List[SubPath]

class Result(BaseModel):
    searchType: int
    subwayCount: int
    path: List[Path]

class PredictRequest(BaseModel):
    result: Result
    datetime: Optional[str] = None  # 예측 시간 정보가 필요하다면 포함

class StartAndEndStationResponse(BaseModel):
    name: str
    expectedBoarding: int
    expectedAlighting: int

class SectionSummary(BaseModel):
    startStation: dict
    endStation: dict
    avgCongestion: float
    maxCongestion: float
    totalExpectedBoarding: int
    totalExpectedAlighting: int

class StationResponse(BaseModel):
    stationID: int
    stationName: str
    x: str
    y: str
    expectedBoarding: int
    expectedAlighting: int
    trainCongestion: List[float]

class SectionResponse(BaseModel):
    trafficType: int
    distance: int
    sectionTime: int
    stationCount: Optional[int] = None
    lane: Optional[List[dict]] = None
    intervalTime: Optional[int] = None
    startName: Optional[str] = None
    endName: Optional[str] = None
    passStopList: Optional[List[StationResponse]] = None
    sectionSummary: Optional[SectionSummary] = None

class RouteResponse(BaseModel):
    routeType: int
    sections: List[SectionResponse]

class PredictResponse(BaseModel):
    message: str
    routes: List[RouteResponse]
