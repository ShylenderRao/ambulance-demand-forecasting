from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import holidays
from sklearn.base import BaseEstimator, TransformerMixin
from lib_file import lib_path

app = Flask(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ie_holidays = holidays.Ireland()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['datetime'] = pd.to_datetime(X[['year', 'month', 'day', 'hour']])
        X['day_of_week'] = X['datetime'].dt.dayofweek
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        X['is_holiday'] = X['datetime'].apply(lambda x: x in self.ie_holidays).astype(int)
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] <= 6)).astype(int)
        X['is_morning'] = ((X['hour'] >= 7) & (X['hour'] <= 11)).astype(int)
        X['is_afternoon'] = ((X['hour'] >= 12) & (X['hour'] <= 17)).astype(int)
        X['is_evening'] = ((X['hour'] >= 18) & (X['hour'] <= 23)).astype(int)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['season'] = X['month'] % 12 // 3 + 1
        X = X.drop(columns=['datetime', 'day_of_week'])
        return X

model_path = "models/xgb.pkl"
encoder_path = "models/station_id_label_encoder.pkl"
gbr_model = joblib.load(model_path)
station_id_encoder = joblib.load(encoder_path)

STATIONS = {
    0: "Balbriggan Fire Station",
    1: "Blanchardstown Fire Station",
    2: "Dolphins Barn Fire Station",
    3: "Donnybrook Fire Station",
    4: "Dun Laoghaire Fire Station",
    5: "Dunshaughlin",
    6: "Finglas Fire Station",
    7: "Kilbarrack Fire Station",
    8: "MH14",
    9: "MH17",
    10: "North Strand Fire Station",
    11: "Phibsborough Fire Station",
    12: "Rathfarnham Fire Station",
    13: "Skerries Fire Station",
    14: "Swords Fire Station",
    15: "Tallaght Fire Station",
    16: "Tara Street Fire Station",
    17: "Unknown Station"
}

def predict_calls(station_name, year, month, day, hour):
    input_df = pd.DataFrame({
        'station_id': [station_name],
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour]
    })
    input_df['station_id'] = station_id_encoder.transform(input_df['station_id'])
    fe = FeatureEngineer()
    input_eng = fe.transform(input_df)
    prediction = gbr_model.predict(input_eng)[0]
    return int(round(prediction))

@app.route('/')
def index():
    return render_template('index.html', stations=STATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        station_name = data['station']
        year = int(data['year'])
        month = int(data['month'])
        day = int(data['day'])
        hour = int(data['hour'])
        result = predict_calls(station_name, year, month, day, hour)
        return jsonify({'success': True, 'prediction': result, 'station': station_name, 'hour': hour})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)