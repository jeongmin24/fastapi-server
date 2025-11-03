from pydantic import BaseModel

class PredictStatsRequest(BaseModel):
    line: str
    station: str

class PredictStatsResponse(BaseModel):
    line: str
    station: str
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