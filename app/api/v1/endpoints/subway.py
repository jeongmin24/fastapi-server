# 지하철 관련 API
from fastapi import APIRouter
from app.common.fetch import fetch_api
from app.config.settings import GENERAL_KEY, SUBWAY_KEY

router = APIRouter()

# ========================
# 지하철 역명 검색 (OA-121)
# ========================
@router.get("/{station_name}")
def subway_station(station_name: str):
    """역명 검색 (일반 인증키)"""
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/SearchInfoBySubwayNameService/1/5/{station_name}"
    return fetch_api(url)

# ========================
# 역명 목록 조회
# ========================
@router.get("/subway-list")
def subway_list(start: int = 1, end: int = 1000):
    """서울시 지하철 전체 역 목록 조회 (일반 인증키)"""
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/SearchInfoBySubwayNameService/{start}/{end}/"
    return fetch_api(url)

# ========================
# 실시간 도착 정보
# ========================
@router.get("/realtime/{station_name}")
def realtime_arrival(station_name: str, start: int = 1, end: int = 5):
    url = f"http://swopenapi.seoul.go.kr/api/subway/{SUBWAY_KEY}/json/realtimeStationArrival/{start}/{end}/{station_name}"
    return fetch_api(url)

# ========================
# 일별 승하차 인원
# ========================
@router.get("/subway-stats/{date}")
def subway_stats(date: str, start: int = 1, end: int = 5):
    """
        서울시 지하철 호선별 역별 승하차 인원 조회
        - date: YYYYMMDD 형식 (예: 20220101)
        - start, end: 페이징 범위
        """
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayStatsNew/{start}/{end}/{date}"
    return fetch_api(url)

# ========================
# 시간대별 승하차 인원
# ========================
@router.get("/subway-stats-time/{use_mm}")
def subway_time(use_mm: str, start: int = 1, end: int = 5, line: str = None, station: str = None):
    """
       서울시 지하철 호선별 역별 시간대별 승하차 인원 조회
    - use_mm: YYYYMM 형식 (예: 202201)
    - line: 호선명 (예: 2호선) - 선택
    - station: 지하철역명 (예: 강남) - 선택
       """
    url = f"http://openapi.seoul.go.kr:8088/{GENERAL_KEY}/json/CardSubwayTime/{start}/{end}/{use_mm}"
    if line:
        url += f"/{line}"
    if station:
        url += f"/{station}"
    return fetch_api(url)