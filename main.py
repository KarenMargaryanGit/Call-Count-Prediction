import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


# Load data
data = pd.read_csv('call_center_data_example.csv', parse_dates=['date'], index_col='date')
data = data.sort_index()

data.head()

plt.figure(figsize=(12, 6))
plt.plot(data['calls'], label='Daily Calls')
plt.title('Daily Call Volume')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(data['calls'], model='additive', period=7)
decomposition.plot()
plt.show()

# Stationarity test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] > 0.05:
        print("The series is not stationary.")
    else:
        print("The series is stationary.")


adf_test(data['calls'])

# Differencing to make the data stationary if needed
data['calls_diff'] = data['calls'].diff().dropna()
adf_test(data['calls_diff'].dropna())

# Train-Test Split
train = data[:int(0.8 * len(data))]
test = data[int(0.8 * len(data)):]  

# SARIMA Model
def train_sarima(train, test, order, seasonal_order):
    model = SARIMAX(train['calls'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)
    
    # Forecast
    forecast = sarima_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
    
    # Metrics
    mae = mean_absolute_error(test['calls'], forecast)
    rmse = np.sqrt(mean_squared_error(test['calls'], forecast))
    print(f"MAE: {mae}, RMSE: {rmse}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train['calls'], label='Train Data')
    plt.plot(test['calls'], label='Test Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title('SARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Calls')
    plt.legend()
    plt.show()
    
    return sarima_fit, forecast

# Example SARIMA Parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)  # Weekly seasonality

sarima_model, forecast = train_sarima(train, test, order, seasonal_order)

# Future Forecast
future_steps = 30
future_forecast = sarima_model.get_forecast(steps=future_steps)
future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='D')[1:]
future_forecast_mean = future_forecast.predicted_mean

plt.figure(figsize=(12, 6))
plt.plot(data['calls'], label='Historical Data')
plt.plot(future_index, future_forecast_mean, label='Future Forecast', linestyle='--')
plt.title('30-Day Call Volume Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend()
plt.show()

# Save the results
future_forecast_mean.to_csv('forecasted_calls.csv', index=True)