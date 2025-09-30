# fetch.py가 가져온 JSON 데이터 -> 모델이 이해하는 숫자형 feature로 변환

# app/services/preprocessing.py

def preprocess_stats_response(row: dict):
    """
    하나의 통계 응답 row → (feature, label)
    응답 예측 대상은 “승하차 인원 수치” (예: 승차 + 하차 총합 또는 승차만)
    """

    # 필요한 값이 없는 경우 무시
    try:
        gton = int(row.get("GTON_TNOPE", 0))  # 승차인원
        gtoff = int(row.get("GTOFF_TNOPE", 0))  # 하차인원
    except (ValueError, TypeError): # 숫자로 못바꾸는 값은 None리턴하고 해당 row는 학습에서 제외
        print("❌ 전처리 실패: 숫자 변환 오류")
        return None

    # y = 모델이 예측해야할 정답값
    y = [gton, gtoff]

    # feature: 날짜 정보, 호선, 역명 등을 숫자형으로 변환
    # 여기서는 간단 예시만
    # row.get("USE_YMD") 형식 "YYYYMMDD", 예: "20230907" → 시간 정보는 없음
    # 만약 시간 정보 있으면 넣고, 없으면 제외
    # 여기서는 단순히 역명 / 호선 인코딩
    line = row.get("SBWY_ROUT_LN_NM", "") # 호선
    station = row.get("SBWY_STNS_NM", "") # 역명

    # 간단 인코딩 예시 (실제로는 더 정교하게 해야 함)
    # 예: 역명/호선 매핑 dict 필요
    line_map = {
        "2호선": 2,
        "9호선": 9,
        # etc.
    }
    station_map = {
        "김포공항": 0,
        # etc.
    }

    line_encoded = line_map.get(line, -1) # 맵에 없는 호선/역이면 "모름"처리
    station_encoded = station_map.get(station, -1)

    # x = 모델의 입력값
    x = [line_encoded, station_encoded]  # ex. [9,0] = "9호선 김포공항"

    print(f"✅ 전처리 성공 → x: {[line_encoded, station_encoded]}, y: {y}")
    return x, y
