import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from datetime import datetime, timedelta

def prepare_data(df):
    """
    Prepare the data for modeling by creating necessary features and encoding
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create copy of DataFrame with date as column for Prophet
    df = df.copy()
    df['date'] = df.index
    
    # Create time-based features
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    
    # Create seasonality features
    df['quarter'] = df.index.quarter
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # One-hot encode day_of_week if it exists as categorical
    if 'day_of_week' in df.columns and df['day_of_week'].dtype == 'O':
        encoder = OneHotEncoder(sparse=False, drop='first')
        day_of_week_encoded = encoder.fit_transform(df[['day_of_week']])
        day_of_week_cols = [f'day_{i}' for i in range(day_of_week_encoded.shape[1])]
        df[day_of_week_cols] = day_of_week_encoded
        df.drop('day_of_week', axis=1, inplace=True)
    
    # Convert is_holiday to numeric if it isn't already
    if 'is_holiday' in df.columns and df['is_holiday'].dtype == 'O':
        df['is_holiday'] = df['is_holiday'].astype(int)
    
    return df

def train_prophet_model(df):
    """
    Train a Prophet model with enhanced seasonality
    """
    # Prepare data for Prophet
    prophet_df = df[['date', 'total_calls']].rename(columns={
        'date': 'ds',
        'total_calls': 'y'
    })
    
    # Initialize Prophet with detailed seasonality
    model = Prophet(
        yearly_seasonality=20,  # More complex yearly seasonality
        weekly_seasonality=7,   # Detailed weekly patterns
        daily_seasonality=True,
        holidays_prior_scale=10,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative'  # Often better for call center data
    )
    
    # Add custom seasonalities
    model.add_seasonality(
        name='quarterly',
        period=365.25/4,
        fourier_order=5
    )
    
    # Add holiday effects
    if 'is_holiday' in df.columns:
        holidays = df[df['is_holiday'] == 1][['date']].rename(columns={'date': 'ds'})
        holidays['holiday'] = 'custom_holiday'
        model.add_holidays(holidays)
    
    model.fit(prophet_df)
    return model

def train_xgboost_model(df):
    """
    Train an XGBoost model using enhanced features
    """
    # Prepare features
    feature_cols = [
        'month', 'year', 'day_of_month', 'week_of_year',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'quarter', 'is_holiday', 'number_of_unique_logins', 'working_time'
    ]
    
    # Add one-hot encoded day_of_week columns if they exist
    day_cols = [col for col in df.columns if col.startswith('day_') and col != 'day_of_month']
    feature_cols.extend(day_cols)
    
    # Remove any features that don't exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['total_calls']
    
    # Train XGBoost with parameters tuned for time series
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    return model, feature_cols

def generate_future_dates(last_date, periods):
    """
    Generate future dates for prediction
    """
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=periods,
        freq='D'
    )
    return future_dates

def make_predictions(df, forecast_days=90):
    """
    Make predictions using both Prophet and XGBoost models
    """
    # Prepare data with enhanced features
    df_prepared = prepare_data(df)
    
    # Train Prophet model
    prophet_model = train_prophet_model(df_prepared)
    
    # Generate future dates for Prophet
    future_dates = generate_future_dates(df_prepared['date'].max(), forecast_days)
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Make Prophet predictions
    prophet_forecast = prophet_model.predict(future_df)
    
    # Train XGBoost model
    xgb_model, feature_cols = train_xgboost_model(df_prepared)
    
    # Prepare future data for XGBoost
    future_features = pd.DataFrame(index=future_dates)
    future_features['date'] = future_dates
    future_features['month'] = future_dates.month
    future_features['year'] = future_dates.year
    future_features['day_of_month'] = future_dates.day
    future_features['week_of_year'] = future_dates.isocalendar().week
    future_features['quarter'] = future_dates.quarter
    future_features['month_sin'] = np.sin(2 * np.pi * future_dates.month / 12)
    future_features['month_cos'] = np.cos(2 * np.pi * future_dates.month / 12)
    future_features['day_of_week_sin'] = np.sin(2 * np.pi * future_dates.dayofweek / 7)
    future_features['day_of_week_cos'] = np.cos(2 * np.pi * future_dates.dayofweek / 7)
    
    # Add holiday information if available
    if 'is_holiday' in df.columns:
        future_features['is_holiday'] = 0  # Default to non-holiday
    
    # Add other features if available
    if 'number_of_unique_logins' in df.columns:
        future_features['number_of_unique_logins'] = df_prepared['number_of_unique_logins'].mean()
    if 'working_time' in df.columns:
        future_features['working_time'] = df_prepared['working_time'].mean()
    
    # Add one-hot encoded day of week if present in training data
    day_cols = [col for col in feature_cols if col.startswith('day_') and col != 'day_of_month']
    if day_cols:
        for col in day_cols:
            future_features[col] = 0
        for i, date in enumerate(future_dates):
            day_num = date.dayofweek
            if f'day_{day_num}' in day_cols:
                future_features.loc[future_features.index[i], f'day_{day_num}'] = 1
    
    # Make XGBoost predictions
    xgb_forecast = xgb_model.predict(future_features[feature_cols])
    
    # Combine predictions with weighted ensemble
    # Give more weight to Prophet for longer-term predictions
    weights = np.linspace(0.4, 0.6, forecast_days)  # Gradually increase Prophet's weight
    final_forecast = pd.DataFrame({
        'date': future_dates,
        'prophet_forecast': prophet_forecast['yhat'],
        'xgboost_forecast': xgb_forecast,
        'ensemble_forecast': weights * prophet_forecast['yhat'] + (1 - weights) * xgb_forecast
    })
    
    return final_forecast

def evaluate_models(df):
    """
    Evaluate model performance using historical data
    """
    # Use last 30 days as test set
    test_size = 30
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    
    # Make predictions
    forecast = make_predictions(train_df, forecast_days=test_size)
    
    # Calculate metrics
    actual_values = test_df['total_calls'].values
    predicted_values = forecast['ensemble_forecast'].values
    
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    
    return mae, rmse, mape

# Example usage:
"""
# Assuming your data is in a pandas DataFrame called 'df' with a datetime index
results = make_predictions(df, forecast_days=90)
mae, rmse, mape = evaluate_models(df)

print(f"Model Performance Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))
plt.plot(results['date'], results['ensemble_forecast'], label='Forecast')
plt.title('Call Center 90-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Total Calls')
plt.legend()
plt.grid(True)
plt.show()
"""
