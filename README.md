# Bike Share Demand Prediction

## Project Overview

This project predicts bike rental demand using historical data and machine learning. The goal is to analyze factors such as weather conditions, traffic, holidays, promotions, and operational metrics to estimate the expected number of bike rentals.

A Linear Regression model was developed and deployed through a Streamlit web application, allowing users to enter input values and receive real-time demand predictions.

---

## Dataset

The dataset contains information related to bike-sharing operations, including:

* Date and time information
* Weather conditions
* Temperature and humidity
* Wind speed
* Air quality index
* Holiday and working day indicators
* Fuel prices
* Traffic index
* App active users
* Promotions and special events
* Metro station proximity
* Bike station capacity

**Target Variable:**

* `bike_demand`

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

---

## Project Workflow

### 1. Data Preprocessing

* Removed unnecessary columns
* Handled categorical features using One-Hot Encoding
* Split data into training and testing sets
* Applied feature scaling using StandardScaler

### 2. Model Development

* Trained a Linear Regression model
* Evaluated model performance using:

  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * R² Score

### 3. Model Deployment

* Saved the trained model using Joblib
* Saved the scaler and feature columns
* Built an interactive Streamlit application for real-time predictions

---

## Model Performance

| Metric   | Value   |
| -------- | ------- |
| MAE      | 65.51   |
| MSE      | 9525.48 |
| RMSE     | 97.60   |
| R² Score | 0.76    |

Training R² Score: **0.78**

Testing R² Score: **0.76**

The small difference between training and testing performance indicates that the model generalizes well and does not show significant overfitting.

---

## Project Structure

```text
Bike-Demand-Prediction/
│
├── app.py
├── train.py
├── Bike_Share_Demand.csv
├── model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

## Features

* Interactive user interface
* Real-time bike demand prediction
* Automated preprocessing pipeline
* Feature scaling using saved scaler
* Consistent prediction workflow using saved model and feature columns

---

## Author

Saurabh Bisht

