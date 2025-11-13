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

# 기본 역 정보
class Station(BaseModel):
    index: int
    stationID: int
    stationName: str
    x: str
    y: str

class PassStopList(BaseModel):
    stations: List[Station]

# 지하철 노선 정보
class Lane(BaseModel):
    name: str
    subwayCode: int
    subwayCityCode: int

# 경로의 세부 구간 정보
class SubPath(BaseModel):
    trafficType: int  # 1: 지하철, 2: 버스, 3: 도보
    distance: int
    sectionTime: int
    stationCount: Optional[int] = None
    lane: Optional[List[Lane]] = None
    intervalTime: Optional[int] = None
    startName: Optional[str] = None
    endName: Optional[str] = None
    passStopList: Optional[PassStopList] = None
    # ##TODO: 버스 혼잡도 처리 필드 추가 예정

# 전체 경로 정보
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

# 최상위 요청 구조
class Result(BaseModel):
    searchType: int
    subwayCount: int
    path: List[Path]

class PredictRequest(BaseModel):
    result: Result
    datetime: Optional[str] = None  # 예측 시간 정보가 필요하다면 포함

# 응답 구조 정의
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
    # ##TODO: 버스 혼잡도 처리 필드 추가 예정

class RouteResponse(BaseModel):
    routeType: int
    sections: List[SectionResponse]

class PredictResponse(BaseModel):
    message: str
    routes: List[RouteResponse]
from pydantic import BaseModel
from typing import List, Optional, Union

# =========================
# 기본 역 정보 및 정차역 리스트
# =========================

class Station(BaseModel):
    index: int
    stationID: int
    stationName: str
    x: str
    y: str

class PassStopList(BaseModel):
    stations: List[Station]

# =========================
# 지하철/버스 노선 정보
# =========================

class SubwayLane(BaseModel):
    name: str
    subwayCode: int
    subwayCityCode: int

class BusLane(BaseModel):
    busNo: str
    type: int
    busID: int
    busLocalBlID: str
    busCityCode: int
    busProviderCode: int

LaneType = Union[SubwayLane, BusLane]

# =========================
# 경로의 세부 구간 정보
# =========================

class SubPath(BaseModel):
    trafficType: int  # 1: 지하철, 2: 버스, 3: 도보
    distance: int
    sectionTime: int
    stationCount: Optional[int] = None
    lane: Optional[List[LaneType]] = None
    intervalTime: Optional[int] = None
    startName: Optional[str] = None
    endName: Optional[str] = None
    passStopList: Optional[PassStopList] = None

# =========================
# 전체 경로 정보
# =========================

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
    datetime: Optional[str] = None

# =========================
# 응답 스키마
# =========================

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
    # ##TODO: 버스 혼잡도 필드 추가 가능

class RouteResponse(BaseModel):
    routeType: int
    sections: List[SectionResponse]

class PredictResponse(BaseModel):
    message: str
    routes: List[RouteResponse]
