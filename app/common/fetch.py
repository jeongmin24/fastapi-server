import requests
from fastapi import HTTPException

# ========================
# 공통 요청 처리 함수
# ========================
def fetch_api(url: str):
    """서울 열린데이터 / Puzzle API 등 공통 JSON 응답 처리"""
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        # 1️⃣ root key 자동 감지
        root_key = next(iter(data))

        # 2️⃣ RESULT 키가 없을 수도 있음
        result = data[root_key].get("RESULT", None)
        if result:
            code = result.get("CODE", "")
            message = result.get("MESSAGE", "")

            if code == "INFO-000":  # 정상
                return data
            elif code == "INFO-200":  # 데이터 없음
                print(f"⚠️ 데이터 없음: {message}")
                return {root_key: {"row": []}}
            else:
                # 다른 에러 코드 (예: 인증 오류 등)
                raise HTTPException(status_code=400, detail=f"{code}: {message}")
        else:
            # ✅ RESULT 자체가 없으면 정상 데이터로 간주
            return data

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="서울열린데이터 API 응답 지연 (Timeout)")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"서울열린데이터 API 요청 오류: {str(e)}")
    except ValueError:
        raise HTTPException(status_code=500, detail="응답 파싱 실패 (JSON 변환 불가)")
