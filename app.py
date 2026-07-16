import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import pvlib
from datetime import datetime, timedelta

if predict:

    st.info("Fetching tomorrow's weather...")

    # ---------------------------------
    # Solar Features
    # ---------------------------------

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

    # ---------------------------------
    # Time Features
    # ---------------------------------

    weather["HOUR"] = weather["DATE_TIME"].dt.hour
    weather["MONTH"] = weather["DATE_TIME"].dt.month
    weather["DAY_OF_YEAR"] = weather["DATE_TIME"].dt.dayofyear

    weather["HOUR_SIN"] = np.sin(
        2 * np.pi * weather["HOUR"] / 24
    )

    weather["HOUR_COS"] = np.cos(
        2 * np.pi * weather["HOUR"] / 24
    )

    # ---------------------------------
    # Previous Day Generation
    # ---------------------------------

    weather = weather.merge(
        gen_df,
        on="HOUR",
        how="left"
    )

    # ---------------------------------
    # Model Features
    # ---------------------------------

    weather["IRRADIATION"] = weather["RADIATION"]

    weather["AMBIENT_TEMPERATURE"] = weather["TEMPERATURE"]

    weather["MODULE_TEMPERATURE"] = (
        weather["TEMPERATURE"]
        + weather["RADIATION"] * 0.03
    )

    weather["SEASON_Monsoon"] = (
        weather["MONTH"].isin([6, 7, 8, 9])
    ).astype(int)

    weather["SEASON_Summer"] = (
        weather["MONTH"].isin([3, 4, 5])
    ).astype(int)

    features = [

        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "HOUR",
        "MONTH",
        "DAY_OF_YEAR",
        "HOUR_SIN",
        "HOUR_COS",
        "GEN_YESTERDAY",
        "SEASON_Monsoon",
        "SEASON_Summer"

    ]

    X = weather[features]

    # ---------------------------------
    # Prediction
    # ---------------------------------

    weather["PREDICTED_AC_POWER"] = model.predict(X)

    weather["PREDICTED_AC_POWER"] = (
        weather["PREDICTED_AC_POWER"]
        .clip(lower=0)
    )

    # ---------------------------------
    # Summary
    # ---------------------------------

    peak_power = weather["PREDICTED_AC_POWER"].max()

    total_energy = (
        weather["PREDICTED_AC_POWER"].sum() / 1000
    )

    peak_time = weather.loc[
        weather["PREDICTED_AC_POWER"].idxmax(),
        "DATE_TIME"
    ]

    st.success("Prediction Completed Successfully!")

    # ---------------------------------
    # Model Performance
    # ---------------------------------

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "R² Score",
        "0.9966"
    )

    c2.metric(
        "RMSE",
        "480.31"
    )

    c3.metric(
        "MAE",
        "242.74 kW"
    )

    # ---------------------------------
    # Forecast Summary
    # ---------------------------------

    c1, c2, c3 = st.columns(3)

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

    # ---------------------------------
    # Forecast Plot
    # ---------------------------------

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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # Feature Importance
    # ---------------------------------

    st.subheader("⭐ Feature Importance")

    st.image(
        "feature_importance.png",
        caption="Top Features Used by XGBoost",
        use_container_width=True
    )

    # ---------------------------------
    # Model Information
    # ---------------------------------

    st.subheader("🤖 Model Information")

    st.markdown("""
**Model:** XGBoost Regressor

**Forecast Horizon:** 24 Hours

**Target Variable:** AC Power (kW)

**Weather Source:** Open-Meteo API

**Solar Calculations:** pvlib

**Location:** Reliance Industries Ltd., Jamnagar
""")

    # ---------------------------------
    # Tomorrow Weather
    # ---------------------------------

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

    # ---------------------------------
    # Hourly Prediction
    # ---------------------------------

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

    # ---------------------------------
    # Download CSV
    # ---------------------------------

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