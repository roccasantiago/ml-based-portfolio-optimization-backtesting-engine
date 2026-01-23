import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pypfopt import expected_returns
from sklearn.model_selection import GridSearchCV

def get_baseline_expected_returns(price_data):
    """
    Baseline model expected returns
    """
    return expected_returns.mean_historical_return(price_data)

def get_ml_expected_returns(dataset_ml, features):
    """
    Machine learning model expected returns
    """
    tickers = dataset_ml['ticker'].unique()
    predictions = {}

    # Hyperparameters grid 
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [50, 100]
    }

    for ticker in tickers:
        df_ticker = dataset_ml[dataset_ml['ticker'] == ticker].sort_index()
        X = df_ticker[features]
        y = df_ticker['target']
        
        X_train, y_train = X.iloc[:-1], y.iloc[:-1]
        X_latest = X.iloc[[-1]] 

        # Crossvalidation
        xgb = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions[ticker] = best_model.predict(X_latest)[0]
        
    return pd.Series(predictions)



