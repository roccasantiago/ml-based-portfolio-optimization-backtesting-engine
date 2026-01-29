
CONFIG = {
    'TICKERS': [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AFL', 'AIG', 
    'AMD', 'AMGN', 'AMT', 'AMZN', 'ELV', 'AON', 'APD', 'ASML', 'AVGO', 'AXP', 
    'BA', 'BABA', 'BAC', 'BEN', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C', 
    'CAT', 'CHTR', 'CL', 'CMCSA', 'CMA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 
    'CVS', 'CVX', 'DD', 'DE', 'DHR', 'DIS', 'DUK', 'EMR', 'EXC', 
    'F', 'META', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS',
    'HON', 'IBM', 'INTC', 'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 
    'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'MMM', 
    'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 
    'PFE', 'PG', 'PM', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'T', 'TGT', 
    'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 
    'WFC', 'WMT', 'XOM'],
    'START_DATE': '2021-01-01',
    'END_DATE': '2022-01-01',
    'FEATURES': ['rsi', 'atr', 'mom_5', 'mom_21', 'sma_dist', 'vix', 'tnx', 'vix_vol'],
    'BROKER_COMISSION': 0.001,
    'SAVE_TO_CSV': False,
    'SAVE_DASHBOARD': True,
    'DEVICE': 'cuda'
}