*# ☀️ Day-Ahead Solar Generation Forecasting

An end-to-end Machine Learning project for forecasting next-day solar power generation using historical solar plant data, live weather forecasts, and solar irradiance calculations.

The application predicts hourly solar generation for the upcoming day and provides an interactive dashboard built with Streamlit.

---

## 🚀 Live Demo

🌐 https://solarforecast-dafzc4wrargewnbheaqkyd.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/rajyash8/SolarForecast

---

## ✨ Features

- 📊 Day-ahead solar generation forecasting
- ☀️ Live weather forecasting using Open-Meteo API
- 🌍 Solar position calculation using pvlib
- 🤖 XGBoost Machine Learning Model
- 📈 Interactive Plotly visualization
- 📋 Hourly prediction table
- 📥 Download prediction as CSV
- 📱 Responsive Streamlit dashboard

---

## 🛠 Tech Stack

### Machine Learning
- XGBoost
- Scikit-Learn
- Joblib

### Data Processing
- Pandas
- NumPy

### APIs
- Open-Meteo API

### Solar Engineering
- pvlib

### Visualization
- Plotly

### Deployment
- Streamlit Cloud

---

## 📊 Dataset

Source:
Kaggle Solar Power Generation Dataset

Files used:

- Plant Generation Data
- Weather Sensor Data

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Live Weather Integration
8. Solar Irradiance Calculation
9. Day-Ahead Prediction
10. Streamlit Deployment

---

## 📈 Feature Engineering

Features used by the model include:

- Irradiation
- Ambient Temperature
- Module Temperature
- Hour
- Month
- Day of Year
- Hour Sin
- Hour Cos
- Previous Day Generation

---

## 🤖 Machine Learning Model

Algorithm Used:

**XGBoost Regressor**

Evaluation Metrics:

- MAE
- RMSE
- nRMSE

---

## 🌤 Live Weather Forecast

The application fetches live weather using:

Open-Meteo API

Weather parameters:

- Temperature
- Humidity
- Wind Speed
- Cloud Cover
- Solar Radiation

---

## ☀️ Solar Calculations

Using pvlib the application computes:

- Solar Elevation
- Solar Zenith
- Clear Sky GHI
- Clear Sky DNI
- Clear Sky DHI

These features improve prediction accuracy.

---

## 📷 Dashboard

The Streamlit dashboard displays:

- Current Weather
- Peak Power
- Estimated Energy
- Peak Generation Time
- Hourly Prediction Graph
- Download CSV

---

## 📁 Project Structure

```
SolarForecast/
│
├── app.py
├── xgboost_model.pkl
├── hourly_generation.csv
├── requirements.txt
├── README.md
└── utils.py
```

---

## 📌 Future Improvements

- LSTM based forecasting
- Multi-day prediction
- Solar plant comparison
- Battery storage estimation
- PowerBI Dashboard
- Docker Deployment
- AWS Deployment

---

## 👨‍💻 Author

**Yash Raj**

Machine Learning Intern @ Reliance Industries Limited

Electronics & Communication Engineering

Nitte Meenakshi Institute of Technology

GitHub:
https://github.com/rajyash8

LinkedIn:
(Add your LinkedIn URL)

---

## ⭐ If you like this project

Please consider giving the repository a ⭐.**
