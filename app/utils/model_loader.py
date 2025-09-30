import os
import glob

# models/ 디렉토리에서 가장 최신 모델 파일 경로를 반환
def get_latest_model_path(models_dir: str = "models") -> str:
    model_files = glob.glob(os.path.join(models_dir, "stats_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("모델 파일이 존재하지 않습니다.")

    # 파일명 기준으로 정렬 (가장 최신 날짜 모델이 맨뒤로)
    model_files.sort()
    latest_model = model_files[-1]
    print(f"최신 모델 로딩: {latest_model}")
    return latest_model