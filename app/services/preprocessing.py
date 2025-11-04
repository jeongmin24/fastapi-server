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

    # Line과 Station 인코딩 값은 train.py에서 이미 계산되어 row에 추가되었음
    # 통합 모델은 이 값을 입력 특성으로 사용해야 함
    line_encoded = row.get("LINE_NUM_ENCODED")
    station_encoded = row.get("STATION_NAME_ENCODED")

    # 인코딩된 값이 누락된 경우 (이 경우는 train.py 로직상 발생하지 않아야 함)
    if line_encoded is None or station_encoded is None:
        return None

    # 2. 시간대별로 샘플 만들기
    for hour in range(0, 24):  # 0시~23시
        try:
            gton = int(row.get(f"HR_{hour}_GET_ON_NOPE", 0))  # 승차
            gtoff = int(row.get(f"HR_{hour}_GET_OFF_NOPE", 0))  # 하차
        except(ValueError, TypeError):
            continue

        # x = 기본 입력 특징 (날짜/시간)
        x_base = [year, month, hour]

        # 통합 모델을 위해 train.py에서 이 x_base에 인코딩된 특징을 추가할 것임.
        # (preprocessing.py에서는 기본 특징만 반환)
        x = x_base  # [year, month, hour]

        # y = 정답
        y = [gton, gtoff]

        samples.append((x, y))

        #if hour in [0, 12, 23]:
            # 출력 메세지에 인코딩 정보는 제외하고 기본 정보만 표시
        #    print(f"✅ {hour}시 샘플 → x: {x}, y: {y}")

    return samples