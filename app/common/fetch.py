import requests
from fastapi import HTTPException

# ========================
# 공통 요청 처리 함수
# ========================
def fetch_api(url: str):
    """서울 열린데이터 API 호출 공통 함수"""
    try:
        # 1. HTTP 요청 (5초 안에 응답이 없으면 자동으로 Timeout 예외)
        res = requests.get(url, timeout=5)
        # 2. HTTP 레벨 에러 처리 (200번대 이외 상태코드가 오면 자동으로 예외)
        res.raise_for_status()
        # 3. JSON 응답 파싱
        data = res.json()

        # 4. 서울열린데이터 API의 RESULT 코드 처리
        if "RESULT" in data:
            code = data["RESULT"].get("CODE", "")
            message = data["RESULT"].get("MESSAGE", "")
            if code != "INFO-000":  # 정상 코드가 아니면 예외 발생
                raise HTTPException(status_code=400, detail=f"{code}: {message}")

        # 5. 정상 데이터 반환
        return data

    # ========== 예외 처리 영역 ==========
    except requests.exceptions.Timeout:
        # 요청이 5초 안에 응답하지 않음 → 504 Gateway Timeout
        raise HTTPException(status_code=504, detail="서울열린데이터 API 응답 지연 (Timeout)")
    except requests.exceptions.RequestException as e:
        # 네트워크 오류, 404, 인증키 오류 등 requests 관련 예외
        raise HTTPException(status_code=502, detail=f"서울열린데이터 API 요청 오류: {str(e)}")
    except ValueError:
        # JSON 파싱 실패 (예: 응답이 HTML 에러 페이지일 때)
        raise HTTPException(status_code=500, detail="응답 파싱 실패 (JSON 변환 불가)")