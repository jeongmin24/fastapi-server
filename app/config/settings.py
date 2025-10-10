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

# 서울시 열린데이터 API KEY
GENERAL_KEY = "4a646a43776c65653131367a4f594659"
SUBWAY_KEY = "5a647a614c6c65653436786673534e"   # 지하철 실시간 인증키