import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import pickle
import sys

# 0. 필수 설정 및 함수 정의 (학습/예측 모두 사용)

# 모델 학습과 예측에 사용할 역 특징 목록을 정의. (TRANSFER_STATIONS = 환승역 / BUSINESS_DISTRICTS = 업무회사역 )
TRANSFER_STATIONS = ['신도림역', '사당역', '을지로입구역', '불암산역'] 
BUSINESS_DISTRICTS = ['강남역', '여의도역', '서울역', '선릉역']

def is_transfer_station(station_name):
    return 1 if station_name in TRANSFER_STATIONS else 0

def is_business_district(station_name):
    return 1 if station_name in BUSINESS_DISTRICTS else 0

def get_train_features(departure_station, arrival_station, time_hour, line_name):
    ## 출발/도착역 정보를 기반으로 특징 딕셔너리 생성
    
    features = { # AM_Peak = 출근시간, PM_Peak = 퇴근시간, Start_Transfer = 환승시작역, Start_Transfer = 업무지구시작역, Arrival_Transfer = 환승도착역, Arrival_Business = 업무도착역
        'AM_Peak': 1 if time_hour == 8 else 0,
        'PM_Peak': 1 if time_hour in [18, 19] else 0,
        'Start_Transfer': is_transfer_station(departure_station),
        'Start_Business': is_business_district(departure_station),
        'Arrival_Transfer': is_transfer_station(arrival_station),
        'Arrival_Business': is_business_district(arrival_station),
    }
    
    line_dummy_name = f'Line_{line_name}'
    features[line_dummy_name] = 1

    return features

# 1.1 데이터 로드 및 통합 

data_folder = 'data_files'
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.xlsx')]

if not all_files:
    print(f"오류: '{data_folder}' 폴더에 .xlsx 파일이 없음. 학습을 건너뛰고 모델 로드를 시도했음.")
    # 파일이 없으면 학습을 할 수 없으므로, 변수 정의 없이 다음 섹션으로 이동함
else:
    list_df = []
    for filename in all_files:
        try:
            df = pd.read_excel(filename)
            list_df.append(df)
        except Exception as e:
            print(f"경고: 파일 {filename} 읽기 실패. 오류: {e}")

    df_raw = pd.concat(list_df, ignore_index=True)
    print(f"총 {len(all_files)}개 파일을 통합해서 {len(df_raw)}개 '열차 운행' 기록을 확보.")

    # ----------------------------------------------------------------------
    # 1.2 데이터 구조 변환 및 독립 변수(X) 생성
    # ----------------------------------------------------------------------
    congestion_cols = {f'혼잡도{i}': f'칸{i}' for i in range(1, 11)}
    df_long = df_raw.melt(
        id_vars=[col for col in df_raw.columns if col not in congestion_cols], 
        value_vars=congestion_cols.keys(), 
        var_name='칸_컬럼명', 
        value_name='혼잡도_퍼센트'
    )

    df_long['칸_번호'] = df_long['칸_컬럼명'].str.extract(r'(\d+)').astype(int)
    df_long['승객_수'] = df_long['혼잡도_퍼센트'] * 1.6
    Y_new = df_long['승객_수']
    print(f"Long Format 변환을 완료했음. 총 {len(df_long)}개 데이터 포인트를 확보.")

    # 특징 변수 생성
    df_long['AM_Peak'] = np.where(df_long['시간(시)'].isin([8]), 1, 0)
    df_long['PM_Peak'] = np.where(df_long['시간(시)'].isin([18, 19]), 1, 0)
    df_long['Start_Transfer'] = np.where(df_long['출발역'].isin(TRANSFER_STATIONS), 1, 0)
    df_long['Start_Business'] = np.where(df_long['출발역'].isin(BUSINESS_DISTRICTS), 1, 0)
    df_long['Arrival_Transfer'] = np.where(df_long['도착역'].isin(TRANSFER_STATIONS), 1, 0)
    df_long['Arrival_Business'] = np.where(df_long['도착역'].isin(BUSINESS_DISTRICTS), 1, 0)

    line_dummies = pd.get_dummies(df_long['노선'], prefix='Line', drop_first=True).astype(int)
    df_features_new = pd.concat([df_long, line_dummies], axis=1)

    X_cols_new = ['AM_Peak', 'PM_Peak', 'Start_Transfer', 'Start_Business', 'Arrival_Transfer', 'Arrival_Business', '칸_번호']
    X_cols_new.extend(line_dummies.columns.tolist())
    X_new = df_features_new[X_cols_new]
    print(f"독립 변수(X) {X_new.shape[1]}개 생성을 완료했음. (출발역/도착역 특징, 칸_번호 포함)")

    # ----------------------------------------------------------------------
    # 1.3 데이터 클리닝 및 모델 학습
    # ----------------------------------------------------------------------
    df_for_model = pd.concat([Y_new.rename('승객_수'), X_new], axis=1)
    df_for_model.dropna(inplace=True) 

    Y_clean = df_for_model['승객_수']
    X_clean = df_for_model.drop('승객_수', axis=1)
    X_clean = X_clean.loc[:, X_clean.apply(pd.Series.nunique) != 1]

    X_cols_clean = X_clean.columns.tolist() # 모델 예측에 필수적인 최종 컬럼 목록을 저장함
    X_with_const_clean = sm.add_constant(X_clean, has_constant='add')
    model_kan = sm.OLS(Y_clean, X_with_const_clean).fit()
    print("\n모델 학습을 완료했음.")
    
    # ----------------------------------------------------------------------
    # 1.4 모델 저장
    # ----------------------------------------------------------------------
    model_filename = 'train_congestion_model.pkl'
    try:
        with open(model_filename, 'wb') as file:
            pickle.dump(model_kan, file)
        print(f"모델 저장 완료: '{model_filename}'")
    except Exception as e:
        print(f"오류: 모델 저장 실패: {e}")

# ----------------------------------------------------------------------
# 2. 모델 로드 및 예측 (모델 파일만 있을 때 실행 가능)
# ----------------------------------------------------------------------

# 2.1 모델 로드
model_filename = 'train_congestion_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"\n모델 로드를 성공했음: '{model_filename}'")
except FileNotFoundError:
    print(f"\n오류: 모델 파일 '{model_filename}'을 찾을 수 없음. 학습 데이터를 넣어주거나 모델 파일을 확인해야 함.")
    sys.exit() # 모델이 없으면 예측할 수 없으므로 종료했음.

# 2.2 X_cols_clean 수동 정의 (학습이 생략되었을 경우 대비)
# 모델 로드만 하는 경우 X_cols_clean 변수가 정의되어 있어야 예측할 수 있음.
if 'X_cols_clean' not in locals() and 'X_cols_clean' not in globals():
    print("\n경고: X_cols_clean 변수가 정의되지 않아 수동으로 정의했음.")
    # 이 목록은 실제 학습 데이터의 노선 변수들을 기반으로 직접 채워야 함.
    X_cols_clean = [
        'AM_Peak', 'PM_Peak', 
        'Start_Transfer', 'Start_Business', 
        'Arrival_Transfer', 'Arrival_Business', 
        '칸_번호',
        # 학습 데이터에 있던 모든 노선 변수를 여기에 채워야 함.
        'Line_2호선', 'Line_4호선', 'Line_5호선', 'Line_6호선', 'Line_7호선'
    ]


# 2.3 예측 함수 정의 (모델 로드 후 사용)
def predict_car_congestion_with_stations(model, X_cols, departure_station, arrival_station, time_hour, line_name):
    """로드된 모델과 X_cols를 사용하여 칸 별 혼잡도를 예측함."""
    
    train_features = get_train_features(departure_station, arrival_station, time_hour, line_name)
    final_features = train_features.copy()
    
    # 예측 시에는 모델이 학습한 모든 변수(X_cols)가 필요함. (없으면 0으로 채움)
    for col in X_cols:
        if col not in final_features:
            final_features[col] = 0

    car_numbers = list(range(1, 11))
    input_data_list = []
    feature_names = ['const'] + X_cols # 상수항 + 학습된 변수 목록 (순서 중요)
    
    for car_num in car_numbers:
        input_row = {name: 0 for name in feature_names}
        input_row['const'] = 1 
        
        # 특징 변수를 모델이 기대하는 순서대로 입력 데이터에 담았음.
        for key in X_cols:
            if key in final_features:
                input_row[key] = final_features[key]

        input_row['칸_번호'] = car_num
        ordered_input_values = [input_row[name] for name in feature_names]
        input_data_list.append(ordered_input_values)

    X_predict = np.array(input_data_list)
    predictions_passengers = model.predict(X_predict)
    
    # 결과를 데이터프레임으로 만들었음.
    df_results = pd.DataFrame({
        '칸 번호': car_numbers,
        '예상 승객 수': predictions_passengers.round(1).astype(str) + '명',
        '예상 혼잡도': (predictions_passengers / 1.6).round(1).astype(str) + '%'
    })
    
    return df_results, final_features

# ----------------------------------------------------------------------
# 3.3 예측 실행: 경복궁 -> 독립문 (3호선, 8시 AM)
# ----------------------------------------------------------------------
departure = '경복궁역'
arrival = '독립문역'
time = 8 
line = '3호선'

df_results, debug_features = predict_car_congestion_with_stations(
    loaded_model, 
    X_cols_clean, # 학습 시 사용된 변수 목록을 전달함.
    departure, 
    arrival, 
    time, 
    line
)

print("\n" + "="*50)
print(f"      [최종 예측 결과: {departure} -> {arrival} ({line}선)]")
print("="*50)
print(f"운행 정보: {line}선, {time}:00 AM (AM Peak)")
print(f"출발역: {departure} (환승: {debug_features.get('Start_Transfer', 0)}, 업무: {debug_features.get('Start_Business', 0)})")
print(f"도착역: {arrival} (환승: {debug_features.get('Arrival_Transfer', 0)}, 업무: {debug_features.get('Arrival_Business', 0)})")
print(df_results)
print("--------------------------------------------------")
