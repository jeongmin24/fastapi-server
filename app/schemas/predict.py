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