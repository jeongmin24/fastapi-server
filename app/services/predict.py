from app.services.model import load_model
from app.services.preprocessing import preprocess_stats_response

model = load_model()

def predict_stats(line: str, station: str):
    fake_row = {
        "SBWY_ROUT_LN_NM": line,
        "SBWY_STNS_NM": station,
        "GTON_TNOPE": 0,
        "GTOFF_TNOPE": 0,
    }
    result = preprocess_stats_response(fake_row)
    if result is None:
        raise ValueError("전처리 실패")
    x, _ = result
    return model.predict([x])[0]
