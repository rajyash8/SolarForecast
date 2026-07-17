import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import pvlib
from datetime import datetime, timedelta

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Solar Forecast Dashboard",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------

st.title("☀️ Day Ahead Solar Generation Forecasting")

st.markdown(
"""
### Reliance Industries Limited • Jamnagar Solar Plant

Live Weather Forecast • Machine Learning • XGBoost
"""
)

st.divider()

# -----------------------------
# Constants
# -----------------------------

LATITUDE = 22.4707
LONGITUDE = 70.0577
TIMEZONE = "Asia/Kolkata"

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("xgboost_model.pkl")

# -----------------------------
# Load Yesterday Generation
# -----------------------------

gen_df = pd.read_csv("hourly_generation.csv")
metrics_df = pd.read_csv("evaluation_metrics.csv")

# -----------------------------
# Open Meteo API
# -----------------------------

url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&hourly=temperature_2m,"
    f"relative_humidity_2m,"
    f"cloud_cover,"
    f"wind_speed_10m,"
    f"shortwave_radiation"
    f"&forecast_days=2"
    f"&timezone=Asia/Kolkata"
)

response = requests.get(url).json()

weather = pd.DataFrame({

    "DATE_TIME": response["hourly"]["time"],
    "TEMPERATURE": response["hourly"]["temperature_2m"],
    "HUMIDITY": response["hourly"]["relative_humidity_2m"],
    "CLOUD_COVER": response["hourly"]["cloud_cover"],
    "WIND_SPEED": response["hourly"]["wind_speed_10m"],
    "RADIATION": response["hourly"]["shortwave_radiation"]

})

weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

weather = weather[
    weather["DATE_TIME"].dt.date == tomorrow
].reset_index(drop=True)

# -----------------------------
# Current Weather
# -----------------------------

current = response["hourly"]

c1, c2, c3, c4 = st.columns(4)

c1.metric("🌡 Temperature", f"{current['temperature_2m'][0]} °C")
c2.metric("☁ Cloud Cover", f"{current['cloud_cover'][0]} %")
c3.metric("💨 Wind Speed", f"{current['wind_speed_10m'][0]} km/h")
c4.metric("💧 Humidity", f"{current['relative_humidity_2m'][0]} %")

# -----------------------------
# Predict Button
# -----------------------------

predict = st.button(
    "🚀 Predict Tomorrow",
    use_container_width=True
)

if predict:

    st.info("Fetching tomorrow's weather...")

    # -----------------------------
    # Solar Position
    # -----------------------------

    site = pvlib.location.Location(
        LATITUDE,
        LONGITUDE,
        TIMEZONE
    )

    solar = site.get_solarposition(weather["DATE_TIME"])

    weather["SOLAR_ELEVATION"] = solar["elevation"].values
    weather["SOLAR_ZENITH"] = solar["zenith"].values

    clearsky = site.get_clearsky(
        pd.DatetimeIndex(weather["DATE_TIME"])
    )

    weather["CLEAR_SKY_GHI"] = clearsky["ghi"].values
    weather["CLEAR_SKY_DNI"] = clearsky["dni"].values
    weather["CLEAR_SKY_DHI"] = clearsky["dhi"].values

    # -----------------------------
    # Time Features
    # -----------------------------

    weather["HOUR"] = weather["DATE_TIME"].dt.hour
    weather["MONTH"] = weather["DATE_TIME"].dt.month
    weather["DAY_OF_YEAR"] = weather["DATE_TIME"].dt.dayofyear

    weather["HOUR_SIN"] = np.sin(
        2*np.pi*weather["HOUR"]/24
    )

    weather["HOUR_COS"] = np.cos(
        2*np.pi*weather["HOUR"]/24
    )

    # -----------------------------
    # Previous Day Generation
    # -----------------------------

    weather = weather.merge(
        gen_df,
        on="HOUR",
        how="left"
    )

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    weather["IRRADIATION"] = weather["RADIATION"]

    weather["AMBIENT_TEMPERATURE"] = weather["TEMPERATURE"]

    weather["MODULE_TEMPERATURE"] = (
        weather["TEMPERATURE"]
        + weather["RADIATION"]*0.03
    )

    features = [

       "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "HOUR",
        "MONTH",
        "DAY_OF_YEAR",
        "HOUR_SIN",
        "HOUR_COS",
        "GEN_YESTERDAY"
    ]

    X = weather[features]

    # -----------------------------
    # Prediction
    # -----------------------------

    weather["PREDICTED_AC_POWER"] = model.predict(X)

    weather["PREDICTED_AC_POWER"] = weather[
        "PREDICTED_AC_POWER"
    ].clip(lower=0)

    # -----------------------------
    # Summary
    # -----------------------------

    peak_power = weather["PREDICTED_AC_POWER"].max()

    total_energy = weather["PREDICTED_AC_POWER"].sum()/1000

    peak_time = weather.loc[
        weather["PREDICTED_AC_POWER"].idxmax(),
        "DATE_TIME"
    ]

    st.success("Prediction Completed Successfully!")

    # -----------------------------
    # Model Metrics
    # -----------------------------

    st.subheader("📊 Model Performance")

    r2 = metrics_df.loc[0, "R2"]
    rmse = metrics_df.loc[0, "RMSE"]
    mae = metrics_df.loc[0, "MAE"]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    m2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    m3.metric(
        "MAE",
        f"{mae:.2f} kW"
    )

    # -----------------------------
    # Forecast Summary
    # -----------------------------

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "⚡ Peak Power",
        f"{peak_power:.2f} kW"
    )

    c2.metric(
        "🔋 Total Energy",
        f"{total_energy:.2f} MWh"
    )

    c3.metric(
        "🕒 Peak Time",
        peak_time.strftime("%H:%M")
    )

    # -----------------------------
    # Forecast Graph
    # -----------------------------

    fig = px.line(

        weather,

        x="DATE_TIME",

        y="PREDICTED_AC_POWER",

        markers=True,

        title="24-Hour Solar Generation Forecast"

    )

    fig.update_layout(

        template="plotly_dark",

        xaxis_title="Time",

        yaxis_title="Predicted AC Power (kW)",

        height=550

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------
    # Feature Importance
    # -----------------------------

    st.subheader("⭐ Feature Importance")

    st.image(

        "feature_importance.png",

        use_container_width=True

    )

    # -----------------------------
    # Model Information
    # -----------------------------

    st.subheader("🤖 Model Information")

    st.info("""

Algorithm : XGBoost Regressor

Forecast Horizon : 24 Hours

Target : AC Power (kW)

Weather API : Open-Meteo

Solar Calculations : pvlib

Location : Reliance Industries Ltd. - Jamnagar

""")

    # -----------------------------
    # Tomorrow Weather
    # -----------------------------

    st.subheader("🌦 Tomorrow Weather Forecast")

    st.dataframe(

        weather[

            [

                "DATE_TIME",

                "TEMPERATURE",

                "HUMIDITY",

                "WIND_SPEED",

                "CLOUD_COVER",

                "RADIATION"

            ]

        ],

        use_container_width=True

    )

    # -----------------------------
    # Hourly Prediction
    # -----------------------------

    st.subheader("📋 Hourly Solar Forecast")

    st.dataframe(

        weather[

            [

                "DATE_TIME",

                "PREDICTED_AC_POWER"

            ]

        ],

        use_container_width=True

    )

    # -----------------------------
    # Download CSV
    # -----------------------------

    csv = weather.to_csv(index=False)

    st.download_button(

        "⬇ Download Forecast CSV",

        csv,

        "Tomorrow_Solar_Generation_Forecast.csv",

        "text/csv"

    )

    st.divider()

    st.caption(
        "Developed by Yash Raj | Machine Learning Intern @ Reliance Industries Ltd."
    )
    
