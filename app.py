import streamlit as st
import joblib
import pandas as pd

# load model
model = joblib.load('model.pkl')
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bike Demand Prediction", layout="centered")

st.title("🚲 Bike Demand Prediction App")

st.write("Enter details below to predict bike demand.")

year = st.selectbox("Year", [2023, 2024, 2025])
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week (0 -> Monday)", 0, 6, 3)
hour = st.slider("Hour", 5, 23, 12)

is_weekend = st.selectbox("Is Weekend?", [0, 1])
season = st.selectbox("Season", ["Spring", "Monsoon", "Summer", "Autumn", "Winter"])
weather = st.selectbox("Weather", ["Clear", "Cloudy", "heavy Rain", "Light Rain", "Mist"])
temperature_c = st.slider("Temperature (°C)", 0.0, 55.0, 27.0)
humidity = st.slider("Humidity", 30, 95, 60)
windspeed_kmh = st.slider("Wind Speed (km/h)", 5.0, 30.0, 15.0)
air_quality_index = st.number_input("Air Quality Index", min_value=30, max_value=300, value=150)

holiday = st.selectbox("Holiday?", [0, 1])
working_day = st.selectbox("Working Day?", [0, 1])
fuel_price = st.number_input("Fuel Price", min_value=50.0, max_value=200.0, value=95.0)
traffic_index = st.slider("Traffic Index", 10.0, 100.0, 60.0)
app_active_users = st.number_input("App Active Users", min_value=100, max_value=100000, value=25000)

promotion_active = st.selectbox("Promotion Active?", [0, 1])
city_zone = st.selectbox("City Zone", ["Residential", "Tourist Area", "University Area", "Business District", "Commercial"])
near_metro_station = st.selectbox("Near Metro Station?", [0, 1])
special_event = st.selectbox("Special Event?", [0, 1])
bike_station_capacity = st.number_input("Bike Station Capacity", min_value=50, max_value=500, value=275)

input_data = pd.DataFrame({
    "year": [year],
    "month": [month],
    "day_of_Week": [day_of_week],
    "hour": [hour],
    "is_weekend": [is_weekend],
    "season": [season],
    "weather": [weather],
    "temperature_c": [temperature_c],
    "humidity": [humidity],
    "windspeed_kmh": [windspeed_kmh],
    "air_quality_index": [air_quality_index],
    "holiday": [holiday],
    "working_day": [working_day],
    "fuel_price": [fuel_price],
    "traffic_index": [traffic_index],
    "app_active_users": [app_active_users],
    "promotion_active": [promotion_active],
    "city_zone": [city_zone],
    "near_metro_station": [near_metro_station],
    "special_event": [special_event],
    "bike_station_capacity": [bike_station_capacity]
})

# Encoding with integer dummy values
input_data = pd.get_dummies(input_data, dtype=int)

# Match columns with training data
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Scale input data using saved scaler
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Bike Demand"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Bike Demand: {int(prediction[0])} bikes")