# Python 3.11 slim 버전 사용
FROM python:3.11-slim

# 컨테이너 내부 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 후 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models /app/models

# 소스 코드 복사
COPY . .

# FastAPI 실행 포트
EXPOSE 8000

# 컨테이너 시작 시 실행될 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
