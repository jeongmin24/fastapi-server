from pydantic_settings import BaseSettings
from zoneinfo import ZoneInfo

# 전역 설정 클래스: 환경변수, 고정상수 정의
class Settings(BaseSettings):
    API_PREFIX: str = "/api/v1" # API 기본 경로
    TZ: str = "Asia/Seoul" # 서버 시간대
    MODEL_DIR: str = "models" # 모델 파일 보관 폴더

    class Config:
        env_file = ".env" # 필요하면 환경변수로 덮어쓰기

settings = Settings()
KST = ZoneInfo(settings.TZ)
import os

# API 키 등 설정값

# 서울시 열린데이터 API KEY
GENERAL_KEY = "4a646a43776c65653131367a4f594659"
SUBWAY_KEY = "5a647a614c6c65653436786673534e"   # 지하철 실시간 인증키
STATION_KEY = "426f696a5667636139376c54575463"


# 현재 파일(__file__)의 절대 경로를 얻습니다.
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리 (e.g., .../app/config)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)

# 프로젝트 루트 디렉토리 계산 (app/config에서 두 단계 위로 이동)
# .../app/config -> .../app -> .../fastapi-server (프로젝트 루트)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

# 파일 및 경로 설정
## MODEL_FILE_NAME = 'train_congestion_model.pkl'

# 모델 폴더 경로: 프로젝트 루트 아래의 'models' 폴더
## MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# 최종 모델 경로 (절대 경로)
## MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME) # <-- 이제 항상 프로젝트 루트를 기준으로 찾습니다.

# 모델 학습에 사용했던 변수 (환승역, 업무역)
TRANSFER_STATION_LIST = ['신도림역', '사당역', '을지로입구역', '불암산역']
BUSINESS_STATION_LIST = ['강남역', '여의도역', '서울역', '선릉역']

# 노선 더미 변수 목록
KNOWN_LINE_DUMMIES = [f'Line_{i}호선' for i in range(1, 10)]