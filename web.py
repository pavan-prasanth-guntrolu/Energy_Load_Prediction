import streamlit as st
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Energy Load Prediction", page_icon="🔋")

# Load the trained model and scaler
svr = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Energy Load Prediction")

# User input fields
st.sidebar.header("Input Features")

day_of_week = st.sidebar.selectbox("Day of the Week (0=Monday, 6=Sunday)", list(range(7)))
is_weekend = 1 if day_of_week >= 5 else 0

hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

month = st.sidebar.slider("Month", 1, 12, 6)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# Set min and max values for load-related inputs
lag24 = st.sidebar.number_input("Load 24 Hours Ago (MW)", min_value=17461, max_value=51714, value=17461)
lag168 = st.sidebar.number_input("Load 168 Hours Ago (MW)", min_value=17461, max_value=51714, value=17461)
rolling_mean_24 = st.sidebar.number_input("Rolling Mean (24h) Load (MW)", min_value=21194, max_value=44828,value=21194)

is_holiday = st.sidebar.checkbox("Is Holiday?")
is_holiday = 1 if is_holiday else 0

if st.sidebar.button("Predict"):
    # Prepare input features
    features = np.array([[day_of_week, is_weekend, is_holiday, hour_sin, hour_cos,
                          month_sin, month_cos, lag24, lag168, rolling_mean_24]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = svr.predict(features_scaled)[0]

    st.subheader("Predicted Energy Load (MW):")
    st.title(f"{prediction:,.2f} MW")
