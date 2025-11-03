from pydantic import BaseModel, Field
from typing import Dict

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