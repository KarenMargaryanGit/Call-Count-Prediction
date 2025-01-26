import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_xgboost_model(df):
    """
    Train an XGBoost model with hyperparameter tuning using time series cross-validation
    """
    # Ensure 'date' column is converted to timestamp
    if 'date' in df.columns:
        df['date'] = df['date'].apply(lambda x: x.timestamp())
    
    # Prepare features
    feature_cols = df.columns.tolist()
    
    # Remove target column from features
    feature_cols.remove('total_calls')
    
    X = df[feature_cols]
    y = df['total_calls']
    
    # Define the base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Print best parameters for reference
    print("Best Parameters:", grid_search.best_params_)
    
    # Optional: Feature importance
    feature_importance = best_model.feature_importances_
    feature_importance_dict = dict(zip(feature_cols, feature_importance))
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")
    
    return best_model, feature_cols

from fbprophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def train_prophet_model(df, cv=True):
    """
    Train a Prophet model with comprehensive hyperparameter optimization
    """
    # Prepare data for Prophet
    prophet_df = df[['date', 'total_calls']].rename(columns={
        'date': 'ds',
        'total_calls': 'y'
    })

    # Expanded parameter grid for tuning
    param_grid = {
        'yearly_seasonality': [10, 15, 20, 25],
        'weekly_seasonality': [5, 7, 10, 12],
        'daily_seasonality': [True, False],
        'holidays_prior_scale': [1, 5, 10, 15],
        'seasonality_prior_scale': [1, 5, 10, 15],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'changepoint_range': [0.8, 0.9, 1.0]
    }

    # Cross-validation and model selection
    if cv:
        best_mse = float('inf')
        best_params = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Nested loop for comprehensive grid search
        for yearly in param_grid['yearly_seasonality']:
            for weekly in param_grid['weekly_seasonality']:
                for daily in param_grid['daily_seasonality']:
                    for holiday_scale in param_grid['holidays_prior_scale']:
                        for seasonality_scale in param_grid['seasonality_prior_scale']:
                            for mode in param_grid['seasonality_mode']:
                                for changepoint_scale in param_grid['changepoint_prior_scale']:
                                    for changepoint_range in param_grid['changepoint_range']:
                                        # Create holidays DataFrame
                                        holidays_df = create_recurring_holidays(df) if 'is_holiday' in df.columns else None
                                        
                                        # Initialize Prophet with current parameters
                                        model = Prophet(
                                            yearly_seasonality=yearly,
                                            weekly_seasonality=weekly,
                                            daily_seasonality=daily,
                                            holidays=holidays_df,
                                            holidays_prior_scale=holiday_scale,
                                            seasonality_prior_scale=seasonality_scale,
                                            seasonality_mode=mode,
                                            changepoint_prior_scale=changepoint_scale,
                                            changepoint_range=changepoint_range
                                        )
                                        
                                        # Perform cross-validation
                                        cv_results = []
                                        for train_index, test_index in tscv.split(prophet_df):
                                            train_df = prophet_df.iloc[train_index]
                                            test_df = prophet_df.iloc[test_index]
                                            
                                            model.fit(train_df)
                                            forecast = model.predict(test_df)
                                            
                                            # Calculate MSE
                                            mse = np.mean((test_df['y'] - forecast['yhat'])**2)
                                            cv_results.append(mse)
                                        
                                        # Update best model if current performs better
                                        avg_mse = np.mean(cv_results)
                                        if avg_mse < best_mse:
                                            best_mse = avg_mse
                                            best_params = {
                                                'yearly_seasonality': yearly,
                                                'weekly_seasonality': weekly,
                                                'daily_seasonality': daily,
                                                'holidays_prior_scale': holiday_scale,
                                                'seasonality_prior_scale': seasonality_scale,
                                                'seasonality_mode': mode,
                                                'changepoint_prior_scale': changepoint_scale,
                                                'changepoint_range': changepoint_range
                                            }
        
        # Print best parameters
        print("Best Parameters:", best_params)
        print("Best Cross-Validation MSE:", best_mse)
        
        # Retrain with best parameters
        final_model = Prophet(
            yearly_seasonality=best_params['yearly_seasonality'],
            weekly_seasonality=best_params['weekly_seasonality'],
            daily_seasonality=best_params['daily_seasonality'],
            holidays=create_recurring_holidays(df) if 'is_holiday' in df.columns else None,
            holidays_prior_scale=best_params['holidays_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            seasonality_mode=best_params['seasonality_mode'],
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            changepoint_range=best_params['changepoint_range']
        )
        final_model.fit(prophet_df)
        
        return final_model
    
    # If no cross-validation, use default optimized parameters
    else:
        model = Prophet(
            yearly_seasonality=20,
            weekly_seasonality=7,
            daily_seasonality=True,
            holidays=create_recurring_holidays(df) if 'is_holiday' in df.columns else None,
            holidays_prior_scale=10,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            changepoint_range=0.9
        )
        model.fit(prophet_df)
        
        return model


def make_predictions(df, forecast_days=90):
    """
    Make predictions using optimized Prophet and XGBoost models
    """
    # Prepare data with enhanced features
    df_prepared = prepare_data(df)
    
    # Train Prophet model with cross-validation
    prophet_model = train_prophet_model(df_prepared, cv=True)
    
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
    
    # Define Armenian holidays 
    armenian_holidays = pd.to_datetime([
        '2022-01-01','2023-01-01','2024-01-01','2025-01-01','2026-01-01',
        # ... (rest of the holidays remain the same)
        '2022-12-31','2023-12-31','2024-12-31','2025-12-31','2026-12-31',
    ])
    
    # Add holiday column
    future_features["is_holiday"] = future_features.index.isin(armenian_holidays)
    
    # Additional features to predict
    additional_features = [
        'working_time', 'number_of_unique_logins', 'clients',
        'answered_calls', 'missed_calls', 'unique_numbers', 'Call duration',
        'time_to_next_call', 'calls_from_clients', 'share_of_answered_calls',
        'share_of_calls_from_registered_number', 'number_of_new_clients',
        'number_of_new_clients_last_7_days', 'number_of_new_clients_last_30_days'
    ]
    
    df['date'] = df.index
    
    # Predict additional features with optimized Prophet models
    for feature in additional_features:
        if feature in df.columns:
            # Predict using optimized Prophet model
            prophet_df = df[['date', feature]].rename(columns={
                'date': 'ds',
                feature: 'y'
            })
            
            # Use the same tuning approach as main model
            model = train_prophet_model(
                prophet_df.rename(columns={'ds': 'date', 'y': 'total_calls'}), 
                cv=True
            )
            
            # Forecast the feature
            future_feature_forecast = model.predict(future_df)
            future_features[feature] = future_feature_forecast['yhat'].to_list()
    
    # Add one-hot encoded day of week
    day_cols = [col for col in feature_cols if col.startswith('day_') and col != 'day_of_month']
    if day_cols:
        for col in day_cols:
            future_features[col] = 0
        for i, date in enumerate(future_dates):
            day_num = date.dayofweek
            if f'day_{day_num}' in day_cols:
                future_features.loc[future_features.index[i], f'day_{day_num}'] = 1
    
    # Convert date to timestamp
    future_features['date'] = future_features['date'].apply(lambda x: x.timestamp())
    
    # Make XGBoost predictions
    xgb_forecast = xgb_model.predict(future_features[feature_cols])
    
    # Combine predictions with weighted ensemble
    weights = np.linspace(0.4, 0.6, forecast_days)
    final_forecast = pd.DataFrame({
        'date': future_dates,
        'prophet_forecast': prophet_forecast['yhat'],
        'xgboost_forecast': xgb_forecast,
        'ensemble_forecast': weights * prophet_forecast['yhat'] + (1 - weights) * xgb_forecast
    })
    
    return final_forecast
