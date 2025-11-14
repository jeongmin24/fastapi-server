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
    index: Optional[int] #정류장순서
    stationID: Optional[int] #지하철역 or 버스정류장 ID
    stationName: Optional[str] #정류장 이름

class Lane(BaseModel):
    name: Optional[str] #지하철 노선명
    busNo: Optional[str] #버스 번호
    subwayCode: Optional[int]
    busID: Optional[int]

class Section(BaseModel):
    trafficType: int               # 1=지하철, 2=버스, 3=도보
    lanes: Optional[List[Lane]] #같은 구간을 지나는 노선
    distance: Optional[float] #이동거리
    sectionTime: Optional[int] #이동소요시간
    stationCount: Optional[int] #이동하여 정차하는 정거장수
    way: Optional[str] #방면 (지하철)
    wayCode: Optional[int] #방면 정보 1-상행 2-하행
    startName: Optional[str] #승차정류장/역 명
    startX: Optional[float] #승차정류장/역 X좌표
    startY: Optional[float] #승차정류장/역 Y좌표
    endName: Optional[str] #하차정류장/역 명
    endX: Optional[float] #하차정류장/역 X좌표
    endY: Optional[float] #하차정류장/역 Y좌표
    passStopList: Optional[List[Station]] = None #경로 상세구간

class Route(BaseModel):
    routeType: int                 # 1=지하철, 2=버스, 3=지하철+버스
    sections: List[Section] #한 경로의 구간별 정보 (subPath)

class PredictRequest(BaseModel):
    routes: List[Route] #여러 후보 경로(path)

"""
경로기준 혼잡도 응답 dto
"""
class StationResponse(BaseModel):
    index: Optional[int]  # 정류장순서
    stationID: Optional[int]  # 지하철역 or 버스정류장 ID
    stationName: Optional[str]  # 정류장 이름
    expectedBoarding: Optional[int] = None
    expectedAlighting: Optional[int] = None
    predictedCongestionCar: Optional[List[float]] = None

#승하차 정류장의 승차,하차인원
class StartAndEndStationResponse(BaseModel):
    name: str
    expectedBoarding: Optional[int]  = None
    expectedAlighting: Optional[int]  = None

class SectionSummary(BaseModel):
    startStation: Optional[dict]  = None #승차정류장정보
    endStation: Optional[dict]  = None  #하차정류장정보
    avgCongestion: Optional[float]  = None
    maxCongestion: Optional[float]  = None
    totalExpectedBoarding: Optional[int]  = None
    totalExpectedAlighting: Optional[int]  = None

class SectionResponse(BaseModel):
    trafficType: int  # 1=지하철, 2=버스, 3=도보
    lanes: Optional[List[Lane]]  # 같은 구간을 지나는 노선
    distance: Optional[float]  # 이동거리
    sectionTime: Optional[int]  # 이동소요시간
    stationCount: Optional[int]  # 이동하여 정차하는 정거장수
    way: Optional[str]  # 방면 (지하철)
    wayCode: Optional[int]  # 방면 정보 1-상행 2-하행
    startName: Optional[str]  # 승차정류장/역 명
    startX: Optional[float]  # 승차정류장/역 X좌표
    startY: Optional[float]  # 승차정류장/역 Y좌표
    endName: Optional[str]  # 하차정류장/역 명
    endX: Optional[float]  # 하차정류장/역 X좌표
    endY: Optional[float]  # 하차정류장/역 Y좌표
    sectionSummary: Optional[SectionSummary] = None
    passStopList: Optional[List[StationResponse]] = None

class RouteResponse(BaseModel):
    routeType: int  # 1=지하철, 2=버스, 3=지하철+버스
    sections: List[SectionResponse]

class PredictResponse(BaseModel):
    message: str
    routes: List[RouteResponse]