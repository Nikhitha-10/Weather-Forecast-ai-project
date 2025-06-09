import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('rain_prediction_model.pkl')

st.title("Rain Prediction Web App")

# Input fields matching your features
rain = st.number_input("Rain (mm)", min_value=0.0, max_value=500.0, value=0.0)
temp_max = st.number_input("Temperature Max (°C)", min_value=-50.0, max_value=60.0, value=25.0)
temp_min = st.number_input("Temperature Min (°C)", min_value=-50.0, max_value=60.0, value=15.0)
rain_lag1 = st.number_input("Rain Lag 1 Day (mm)", min_value=0.0, max_value=500.0, value=0.0)
tempmax_lag1 = st.number_input("Temp Max Lag 1 Day (°C)", min_value=-50.0, max_value=60.0, value=25.0)
tempmin_lag1 = st.number_input("Temp Min Lag 1 Day (°C)", min_value=-50.0, max_value=60.0, value=15.0)
rain_avg3 = st.number_input("Rain 3-Day Avg (mm)", min_value=0.0, max_value=500.0, value=0.0)
tempmax_avg3 = st.number_input("Temp Max 3-Day Avg (°C)", min_value=-50.0, max_value=60.0, value=25.0)
tempmin_avg3 = st.number_input("Temp Min 3-Day Avg (°C)", min_value=-50.0, max_value=60.0, value=15.0)
day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
dayofyear = st.number_input("Day of Year", min_value=1, max_value=366, value=1)

if st.button("Predict Rain"):
    input_features = np.array([[rain, temp_max, temp_min, rain_lag1, tempmax_lag1,
                                tempmin_lag1, rain_avg3, tempmax_avg3, tempmin_avg3,
                                day, month, dayofyear]])
    prediction = model.predict(input_features)[0]
    
    if prediction == 1:
        st.success("Prediction: It will rain ☔")
    else:
        st.info("Prediction: No rain ☀️")
