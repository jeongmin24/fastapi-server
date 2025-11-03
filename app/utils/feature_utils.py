import pandas as pd
import statsmodels.api as sm
from ..config.settings import TRANSFER_STATION_LIST, BUSINESS_STATION_LIST, KNOWN_LINE_DUMMIES
from ..schemas.predict import CongestionRequest


def create_features_for_prediction(req_data: CongestionRequest, car_num: int, model):
    """
    요청 데이터와 칸 번호를 기반으로 OLS 모델 입력용 DataFrame을 생성
    """

    # 1. 기본 특징 딕셔너리 생성
    feature_dict = {
        'AM_Peak': 1 if req_data.hour_of_day == 8 else 0,
        'PM_Peak': 1 if req_data.hour_of_day in [18, 19] else 0,
        'Start_Transfer': 1 if req_data.start_station in TRANSFER_STATION_LIST else 0,
        'Start_Business': 1 if req_data.start_station in BUSINESS_STATION_LIST else 0,
        'Arrival_Transfer': 1 if req_data.end_station in TRANSFER_STATION_LIST else 0,
        'Arrival_Business': 1 if req_data.end_station in BUSINESS_STATION_LIST else 0,
        '칸_번호': car_num
    }

    # 2. 노선 더미 변수 처리
    line_dummies = {col: 0 for col in KNOWN_LINE_DUMMIES}
    line_key_to_set = f'Line_{req_data.line_name}'
    if line_key_to_set in line_dummies:
        line_dummies[line_key_to_set] = 1

    feature_dict.update(line_dummies)

    # 3. 모델 입력 DataFrame 및 상수항 추가
    X_predict_df = pd.DataFrame([feature_dict])
    X_predict_with_const = sm.add_constant(X_predict_df, has_constant='add')

    # 4. 최종 컬럼 순서 맞추기 (모델 파라미터 순서 유지)
    model_param_names = ['const'] + list(model.params.index[1:])
    X_final = X_predict_with_const.reindex(columns=model_param_names, fill_value=0)

    return X_final