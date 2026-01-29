import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
import os

def get_financial_data(tickers, start_date, end_date):
    """
    Download and cleaning of initial data
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    
    # Way around for the multiindex issue
    if isinstance(data.columns, pd.MultiIndex):
        if len(tickers) > 1:
            data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            data.columns = data.columns.get_level_values(0)
            data['Ticker'] = tickers[0]
            data = data.reset_index()
    else:
        if 'Ticker' not in data.columns:
            data['Ticker'] = tickers[0]
        data = data.reset_index()
    
    data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]
    return data

def clean_outliers(data, sigma_threshold=3):
    """
    Filters outliers
    """
    # Z value
    returns = data['adj_close'].pct_change(fill_method=None) 
    mu = returns.mean()
    sigma = returns.std()
    
    #Bit mask
    outliers = (returns < mu - sigma_threshold * sigma) | (returns > mu + sigma_threshold * sigma)

    data.loc[outliers, 'adj_close'] = np.nan
    data['adj_close'] = data['adj_close'].ffill()
    return data

def get_macro_data(start_date, end_date):
    """
    Get the data of the panic index and 10-Year Treasury Yield of the stated period
    """
    macro_tickers = ['^VIX', '^TNX']
    data = yf.download(macro_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    data = data.rename(columns={'^VIX': 'vix', '^TNX': 'tnx'})
    
    # Shifted so the model could use yesterdays closure
    return data.shift(1).ffill()

def pipeline_preprocess(tickers, start_date, end_date, save_to_csv=False):
    """
    Pipeline for the data extraction
    """
    raw_data = get_financial_data(tickers, start_date, end_date)
    macro_data = get_macro_data(start_date, end_date) 
    processed_list = []

    for ticker in tickers:
        asset = raw_data[raw_data['ticker'] == ticker].copy()

        if asset.empty:
            print(f"Skipping {ticker}: No data found.")
            continue
        
        cols_to_numeric = ['adj_close', 'high', 'low', 'open', 'close', 'volume']
        for col in cols_to_numeric:
            if col in asset.columns:
                asset[col] = pd.to_numeric(asset[col], errors='coerce')
        
        asset = asset.dropna(subset=['adj_close'])
        
        if len(asset) < 30:
            print(f"Skipping {ticker}: Insufficient data points ({len(asset)}).")
            continue

        asset.set_index('date', inplace=True)
        asset.sort_index(inplace=True)
        asset = clean_outliers(asset)

        #get features
        asset['log_ret'] = np.log(asset['adj_close'] / asset['adj_close'].shift(1))
        asset['target'] = asset['log_ret'].shift(-1)
        asset['rsi'] = ta.rsi(asset['adj_close'], length=14)
        asset['atr'] = ta.atr(asset['high'], asset['low'], asset['adj_close'], length=14)
        asset['mom_5'] = asset['log_ret'].rolling(5).sum()
        asset['mom_21'] = asset['log_ret'].rolling(21).sum()
        asset['sma_20'] = ta.sma(asset['adj_close'], length=20)
        asset['sma_dist'] = (asset['adj_close'] / asset['sma_20']) - 1
        

        # macro features
        asset = asset.join(macro_data, how='left').ffill()
        asset['vix_vol'] = asset['vix'].pct_change().rolling(10).std()
        
        asset = asset.dropna()

        if not asset.empty:
            processed_list.append(asset)

    if not processed_list:
        print("ERROR: No assets processed correctly.")
        return pd.DataFrame()    
    
    final_data = pd.concat(processed_list)
    
    if save_to_csv:
        os.makedirs("data", exist_ok=True)
        now = datetime.now()
        date_str = now.strftime(r"%Y-%m-%d_%H-%M")
        file_path = os.path.join("data", f"dataset_{date_str}.csv")
        final_data.to_csv(file_path)
    return final_data