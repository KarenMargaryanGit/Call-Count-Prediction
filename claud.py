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
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        day_of_week_encoded = encoder.fit_transform(df[['day_of_week']])
        day_of_week_cols = [f'day_{i}' for i in range(day_of_week_encoded.shape[1])]
        df[day_of_week_cols] = day_of_week_encoded
        df.drop('day_of_week', axis=1, inplace=True)
    
    # Convert boolean is_holiday to numeric
    if 'is_holiday' in df.columns:
        df['is_holiday'] = df['is_holiday'].astype(int)
    
    return df

def create_recurring_holidays(df):
    """
    Create a DataFrame of recurring holidays from historical data
    """
    # Get dates that are holidays
    holiday_dates = df[df['is_holiday'] == 1]['date']
    
    # Extract month and day for each holiday
    holiday_patterns = pd.DataFrame({
        'month': holiday_dates.dt.month,
        'day': holiday_dates.dt.day
    }).drop_duplicates()
    
    # Create holidays for the forecast period
    last_date = df.index.max()
    forecast_end = last_date + pd.DateOffset(months=4)  # Add buffer for forecast period
    
    all_holidays = []
    for year in range(df.index.min().year, forecast_end.year + 1):
        for _, holiday in holiday_patterns.iterrows():
            try:
                holiday_date = pd.Timestamp(year=year, month=holiday['month'], day=holiday['day'])
                all_holidays.append(holiday_date)
            except ValueError:
                continue  # Skip invalid dates (e.g., Feb 29 in non-leap years)
    
    holidays_df = pd.DataFrame({
        'ds': all_holidays,
        'holiday': 'recurring_holiday',
        'lower_window': 0,
        'upper_window': 0
    })
    
    return holidays_df

def train_prophet_model(df):
    """
    Train a Prophet model with enhanced seasonality
    """
    # Prepare data for Prophet
    prophet_df = df[['date', 'total_calls']].rename(columns={
        'date': 'ds',
        'total_calls': 'y'
    })
    
    # Create holidays DataFrame
    if 'is_holiday' in df.columns:
        holidays_df = create_recurring_holidays(df)
    else:
        holidays_df = None
    
    # Initialize Prophet with detailed seasonality
    model = Prophet(
        yearly_seasonality=20,     # More complex yearly seasonality
        weekly_seasonality=7,      # Detailed weekly patterns
        daily_seasonality=True,
        holidays=holidays_df,      # Add our recurring holidays
        holidays_prior_scale=10,   # Increase impact of holidays
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative'  # Often better for call center data
    )
    
    # Add custom seasonalities
    model.add_seasonality(
        name='quarterly',
        period=365.25/4,
        fourier_order=5
    )
    
    model.fit(prophet_df)
    return model

# Rest of the functions remain the same as before...
[Previous code from 'train_xgboost_model' onwards remains exactly the same]

# Example usage:
"""
# Make sure your DataFrame has a datetime index and boolean is_holiday column
# Example data preparation:
df['is_holiday'] = df['is_holiday'].astype(bool)  # Ensure is_holiday is boolean
df.index = pd.to_datetime(df.index)  # Ensure datetime index

# Get predictions
forecast_results = make_predictions(df, forecast_days=90)

# Evaluate model
mae, rmse, mape = evaluate_models(df)
"""
