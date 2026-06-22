# ambulance-demand-forecasting
MSc dissertation project — ML-based hourly ambulance demand forecasting for Dublin Fire Brigade
# Ambulance Demand Forecasting — Dublin Fire Brigade

MSc Computing Science dissertation project at Griffith College Dublin. A machine 
learning system that predicts hourly ambulance call volumes across 18 Dublin Fire 
Brigade stations, enabling emergency services to proactively allocate resources and 
reduce response times.

## Overview

Emergency Medical Services face fluctuating demand across different times and 
locations, making staffing and deployment decisions challenging. This project applies 
spatio-temporal machine learning to forecast hourly ambulance demand using historical 
incident data — without relying on external data sources like weather or traffic.

## Dataset

- **Source**: Dublin Fire Brigade Open Data, published by Dublin City Council
- **Period**: 2013 – 2023
- **Records**: 856,153 real ambulance incident records
- **Licence**: Creative Commons Attribution 4.0 International
- **Link**: https://data.gov.ie/dataset/fire-brigade-and-ambulance

> Note: Raw CSV data files are not included in this repository due to size. 
> Download links are provided above to reproduce the dataset locally.

## Methodology

1. **Data Preprocessing** — Merged 5 years of incident-level CSV files, cleaned and 
   standardised station names and timestamps, aggregated into hourly call counts per 
   station.
2. **Feature Engineering** — Engineered 16 features including cyclical hour/month 
   encoding, time-of-day flags, weekend indicators, Irish public holidays, and season.
3. **Model Training** — Trained and compared three models:
   - Random Forest Regressor
   - XGBoost Regressor
   - TensorFlow Multilayer Perceptron (Neural Network)
4. **Evaluation** — Assessed using MAE, RMSE and R² on a chronological 80/20 
   train-test split.
5. **Deployment** — Best model deployed as a real-time Flask web application.

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | 3.72 | 5.13 | 0.790 |
| **XGBoost** | **3.32** | **4.53** | **0.836** |
| Neural Network (MLP) | 4.12 | 5.42 | 0.765 |

**XGBoost** was selected as the best-performing model, explaining 83.6% of demand 
variation with an average prediction error of 3.32 calls per hour.

## Project Structure
Shylendar_Ambulence_Call_Volume/
├── 1_Preprocessing_File.ipynb     # Data cleaning & feature engineering
├── 2_Model-Training.ipynb         # Model training & evaluation
├── 3_inference.ipynb              # Inference & forecasting examples
├── SplittedData/                  # Train/test split files
└── models/                        # Saved trained models (.pkl)
Shylendar_Ambulence_Call_Volume_FE/
├── app.py                         # Flask backend
├── templates/index.html           # Web interface
└── models/                        # Models used by the web app
## Web Application

The trained XGBoost model is deployed via a Flask web app that allows users to select 
a fire station, date, and hour, and receive a real-time predicted call volume along 
with an operational resource recommendation.

### Running locally

```bash
cd Shylendar_Ambulence_Call_Volume_FE
pip install flask xgboost scikit-learn pandas numpy holidays joblib
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Tech Stack

**Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow  
**Web Framework**: Flask  
**Visualisation**: Matplotlib, Seaborn  
**Tools**: Jupyter Notebook, Git

## Author

**Shylender Rao Manpuri**  
MSc Computing Science, Griffith College Dublin  
[LinkedIn](https://www.linkedin.com/in/shylender-rao-manpuri-275341307/) · [GitHub](https://github.com/ShylenderRao)

## Licence

This project uses publicly available open data under Creative Commons Attribution 
4.0 International Licence. Code in this repository is provided for academic and 
portfolio purposes.
