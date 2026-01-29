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

def get_ml_expected_returns(dataset_ml, features, device = 'cpu'):
    """
    Machine learning model expected returns
    """
    train_data = dataset_ml.dropna(subset=['target'])
    
    latest_data = dataset_ml.groupby('ticker').tail(1).set_index('ticker')
    valid_tickers = latest_data.index.tolist()
    
    X_train = train_data[features]
    y_train = train_data['target']
    X_predict = latest_data[features]

    if device == 'cuda':
        tree_method = 'hist'
        dtype = 'float32'
    else:
        tree_method = 'auto'
        dtype = 'float64'
        
    X_train_np = X_train.astype(dtype).values
    y_train_np = y_train.astype(dtype).values
    X_predict_np = X_predict.astype(dtype).values

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,     
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=tree_method,
        device=device,
        random_state=42,
        verbosity=0
    )

    xgb.fit(X_train_np, y_train_np)

    preds = xgb.predict(X_predict_np)
    return pd.Series(preds, index=valid_tickers)



