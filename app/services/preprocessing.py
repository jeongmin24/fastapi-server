# fetch.py가 가져온 JSON 데이터 -> 모델이 이해하는 숫자형 feature로 변환
from datetime import datetime
from pyexpat import features


# app/services/preprocessing.py
"""
    시간대별 API(CardSubwayTime) 응답 row -> (x,y) 샘플 여러 개 생성
"""
def preprocess_stats_time_response(row: dict):

    samples = []

    # 1. 월 단위 날짜 정보
    use_mm = row.get("USE_MM", "")
    if not use_mm:
        return None
    try:
        year = int(use_mm[:4])
        month = int(use_mm[4:6])
    except:
        return None


    # 2. 시간대별로 샘플 만들기
    for hour in range(0, 24): # 0시~23시
        try:
            gton = int(row.get(f"HR_{hour}_GET_ON_NOPE", 0)) # 승차
            gtoff = int(row.get(f"HR_{hour}_GET_OFF_NOPE", 0))  # 하차
        except(ValueError, TypeError):
            continue

        # x = 입력 특징
        x = [year, month, hour]

        # y = 정답
        y = [gton, gtoff]

        samples.append((x, y))

        if hour in [0, 12, 23]:
            print(f"✅ {hour}시 샘플 → x: {x}, y: {y}")

    return samples
