# 모델 로딩
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "9_김포공항_20251103.pkl")
model = joblib.load(MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)


def predict_stats():
    return None