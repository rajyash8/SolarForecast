import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import pvlib
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Solar Forecast Dashboard",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    font-size:48px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#A0AEC0;
    font-size:20px;
}

hr{
    margin-top:10px;
    margin-bottom:20px;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------

st.markdown(
"""
<div class='title'>
☀️ Day Ahead Solar Generation Forecasting
</div>

<div class='subtitle'>
Reliance Industries Limited • Jamnagar Solar Plant
</div>
""",
unsafe_allow_html=True
)

st.divider()
st.markdown(
"""
<div class='title'>
☀ 24-Hour Solar Generation Forecast Dashboard
</div>

<div class='subtitle'>
Live Weather Forecast • Machine Learning • XGBoost • Reliance Industries Ltd.
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ============================
# PART B : WEATHER + MODEL
# ============================

import joblib
import pvlib
import numpy as np
from datetime import datetime, timedelta

LATITUDE = 22.4707
LONGITUDE = 70.0577
TIMEZONE = "Asia/Kolkata"

# Load model
model = joblib.load("xgboost_model.pkl")

# Load yesterday generation lookup
gen_df = pd.read_csv("hourly_generation.csv")

# -------------------------------
# Open Meteo Forecast
# -------------------------------

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

    "DATE_TIME":response["hourly"]["time"],
    "TEMPERATURE":response["hourly"]["temperature_2m"],
    "HUMIDITY":response["hourly"]["relative_humidity_2m"],
    "CLOUD_COVER":response["hourly"]["cloud_cover"],
    "WIND_SPEED":response["hourly"]["wind_speed_10m"],
    "RADIATION":response["hourly"]["shortwave_radiation"]

})

weather["DATE_TIME"]=pd.to_datetime(weather["DATE_TIME"])

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

weather = weather[
    weather["DATE_TIME"].dt.date == tomorrow
].reset_index(drop=True)

# ----------------------------
# Show today's weather
# ----------------------------

current = response["hourly"]

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric(
        "🌡 Temperature",
        f"{current['temperature_2m'][0]} °C"
    )

with col2:
    st.metric(
        "☁ Cloud Cover",
        f"{current['cloud_cover'][0]} %"
    )

with col3:
    st.metric(
        "💨 Wind Speed",
        f"{current['wind_speed_10m'][0]} km/h"
    )

with col4:
    st.metric(
        "💧 Humidity",
        f"{current['relative_humidity_2m'][0]} %"
    )

# ---------------------------------
# Predict Button
# ---------------------------------

predict = st.button(
    "🚀 Predict Tomorrow",
    use_container_width=True
)

if predict:

    st.info("Fetching tomorrow's weather...")

    # Solar position
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

    weather["HOUR"] = weather["DATE_TIME"].dt.hour
    weather["MONTH"] = weather["DATE_TIME"].dt.month
    weather["DAY_OF_YEAR"] = weather["DATE_TIME"].dt.dayofyear

    weather["HOUR_SIN"] = np.sin(2*np.pi*weather["HOUR"]/24)
    weather["HOUR_COS"] = np.cos(2*np.pi*weather["HOUR"]/24)

    weather = weather.merge(
        gen_df,
        on="HOUR",
        how="left"
    )

    weather["IRRADIATION"] = weather["RADIATION"]
    weather["AMBIENT_TEMPERATURE"] = weather["TEMPERATURE"]
    weather["MODULE_TEMPERATURE"] = (
        weather["TEMPERATURE"] +
        weather["RADIATION"]*0.03
    )

    # -----------------------------
    # Model Features
    # -----------------------------

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
    weather["PREDICTED_AC_POWER"] = weather["PREDICTED_AC_POWER"].clip(lower=0)

    # -----------------------------
    # Summary Metrics
    # -----------------------------

    peak_power = weather["PREDICTED_AC_POWER"].max()
    total_energy = weather["PREDICTED_AC_POWER"].sum() / 1000

    peak_time = weather.loc[
        weather["PREDICTED_AC_POWER"].idxmax(),
        "DATE_TIME"
    ]

    st.success("Prediction Completed Successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "⚡ Peak Power",
        f"{peak_power:.2f} kW"
    )

    col2.metric(
        "🔋 Total Energy",
        f"{total_energy:.2f} MWh"
    )

    col3.metric(
        "🕒 Peak Time",
        peak_time.strftime("%H:%M")
    )

    # -----------------------------
    # Plot
    # -----------------------------

    fig = px.line(
        weather,
        x="DATE_TIME",
        y="PREDICTED_AC_POWER",
        title="24-Hour Solar Generation Forecast",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Predicted AC Power (kW)",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Forecast Table
    # -----------------------------

    st.subheader("Hourly Prediction")

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
    # Download Forecast
    # -----------------------------

    csv = weather.to_csv(index=False)

    st.download_button(
        label="⬇ Download Forecast CSV",
        data=csv,
        file_name="Tomorrow_Solar_Generation_Forecast.csv",
        mime="text/csv"
    )