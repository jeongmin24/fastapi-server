# app/services/preprocessing.py
"""
    시간대별 API(CardSubwayTime) 응답 row -> (x,y) 샘플 여러 개 생성
"""


def preprocess_stats_time_response(row: dict):
    samples = []

    # 월 단위 날짜 정보
    use_mm = row.get("USE_MM", "")
    if not use_mm:
        return None
    try:
        year = int(use_mm[:4])
        month = int(use_mm[4:6])
    except:
        return None

    # 기존에 학습시켰던 대로 line과 station은 encoded시킴
    line_encoded = row.get("LINE_NUM_ENCODED")
    station_encoded = row.get("STATION_NAME_ENCODED")

    # 인코딩된 값이 누락된 경우 무시
    if line_encoded is None or station_encoded is None:
        return None

    # 시간대별로 샘플 만들기
    for hour in range(0, 24):  # 0시~23시
        try:
            gton = int(row.get(f"HR_{hour}_GET_ON_NOPE", 0))  # 승차
            gtoff = int(row.get(f"HR_{hour}_GET_OFF_NOPE", 0))  # 하차
        except(ValueError, TypeError):
            continue

        # x = 기본 입력 특징 (날짜/시간)
        x_base = [year, month, hour]

        x = x_base  # [year, month, hour]

        # y = 정답
        y = [gton, gtoff]

        samples.append((x, y))


    return samples