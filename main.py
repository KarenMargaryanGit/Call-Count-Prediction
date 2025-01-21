import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "updated_dataset_with_weekdays_and_holidays.csv"  # Replace with your file path
data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

# Ensure all relevant columns are numeric
features = [
    "answered_calls", "missed_calls", "unique_numbers", "call_duration",
    "time_to_next_call", "number_of_unique_logins", "working_time",
    "calls_from_clients", "share_of_answered_calls",
    "share_of_calls_from_registered_number", "number_of_new_clients",
    "number_of_new_clients_last_7_days", "number_of_new_clients_last_30_days",
    "clients_count"
]
data[features] = data[features].apply(pd.to_numeric, errors="coerce")
data["total_calls"] = pd.to_numeric(data["total_calls"], errors="coerce")

# Aggregate daily total_calls for prediction
time_series = data["total_calls"].resample("D").sum()

# Check stationarity (Augmented Dickey-Fuller test)
def check_stationarity(series):
    result = adfuller(series.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing may be required.")

check_stationarity(time_series)

# If not stationary, apply differencing (uncomment if needed)
# time_series = time_series.diff().dropna()

# Train-test split (last 3 months for validation)
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Train-test split for exogenous variables
exog_train = data[features][:train_size]
exog_test = data[features][train_size:]

# SARIMA parameter grid search
p = d = q = range(0, 3)
seasonal_p = seasonal_d = seasonal_q = range(0, 3)
seasonal_period = [7]  # Weekly seasonality

# Generate all parameter combinations
sarima_params = list(itertools.product(p, d, q))
seasonal_params = list(itertools.product(seasonal_p, seasonal_d, seasonal_q, seasonal_period))

best_aic = float("inf")
best_order = None
best_seasonal_order = None

for param in sarima_params:
    for seasonal_param in seasonal_params:
        try:
            model = SARIMAX(
                train, exog=exog_train, order=param, seasonal_order=seasonal_param,
                enforce_stationarity=False, enforce_invertibility=False
            )
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = param
                best_seasonal_order = seasonal_param
        except:
            continue

print("Best SARIMA Order:", best_order)
print("Best Seasonal Order:", best_seasonal_order)

# Fit the best SARIMA model
final_model = SARIMAX(
    train, exog=exog_train, order=best_order, seasonal_order=best_seasonal_order,
    enforce_stationarity=False, enforce_invertibility=False
)
final_results = final_model.fit(disp=False)

# Forecast for 3 months
forecast_steps = 90  # 3 months
future_exog = data[features].iloc[-forecast_steps:]  # Provide future values if known
forecast = final_results.get_forecast(steps=forecast_steps, exog=future_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Calculate total predicted calls for the next 3 months
total_forecast_calls = forecast_mean.sum()

# Evaluate the model
test_forecast = final_results.get_forecast(steps=len(test), exog=exog_test)
test_forecast_mean = test_forecast.predicted_mean
rmse = np.sqrt(mean_squared_error(test, test_forecast_mean))
print(f"RMSE on validation set: {rmse}")

# Plot the forecast
plt.figure(figsize=(12, 8))
plt.plot(train, label="Training Data")
plt.plot(test, label="Test Data", color="orange")
plt.plot(forecast_mean, label="3-Month Forecast", color="green")
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="green", alpha=0.2)
plt.legend(loc="upper left")
plt.title("Total Calls Forecast (Next 3 Months)")
plt.show()

print(f"Total forecasted calls for the next 3 months: {total_forecast_calls}")

