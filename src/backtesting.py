import pandas as pd
from ingestion import pipeline_preprocess
from model import get_ml_expected_returns, get_baseline_expected_returns
from optimization import optimize_portfolio, get_robust_covariance

def run_backtest_pipeline(tickers, start_date, end_date, features, freq='ME', fee=0.001, save_to_csv = False, device = 'cpu'):
    """
    Backtesting Walk-Forward: ML vs Baseline
    """
    # data prep
    df_long = pipeline_preprocess(tickers, start_date, end_date, save_to_csv)
    if df_long.empty:
        print("[ERROR] No data for backtest.")
        return pd.DataFrame()

    # Price matrix
    df_prices = df_long.pivot_table(index=df_long.index, columns='ticker', values='adj_close')
    
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    results = []
    last_weights_ml = pd.Series(0, index=df_prices.columns)
    last_weights_classic = pd.Series(0, index=df_prices.columns)

    print(f"Starting Backtest: {tickers}")

    weights = None

    #Walk-Forward Sim
    for i in range(len(rebalance_dates) - 1):
        t_current = rebalance_dates[i]      
        t_next = rebalance_dates[i+1]       

        #Filtering historical data to avoid survivorship bias
        train_df = df_long[df_long.index <= t_current]
        hist_prices = df_prices[df_prices.index <= t_current]

        if len(hist_prices) < 20: 
            print(f"Skipping {t_current.date()}: Insufficient history ({len(hist_prices)}/20 days)")
            continue

        try:
            # risk 
            S = get_robust_covariance(hist_prices)
            
            # ml
            mu_ml_daily = get_ml_expected_returns(train_df, features, device)
            mu_ml_annual = mu_ml_daily * 252 
            common_ml = S.index.intersection(mu_ml_annual.index)

            if len(common_ml)>0:
                S_ml = S.loc[common_ml, common_ml]
                mu_ml_aligned = mu_ml_annual.loc[common_ml]
                w_ml_raw = optimize_portfolio(mu_ml_aligned, S_ml)
                w_ml = pd.Series(w_ml_raw).reindex(df_prices.columns).fillna(0)
            else:
                w_ml = pd.Series(0, index=df_prices.columns)
            
            # baseline
            mu_classic = get_baseline_expected_returns(hist_prices) 
            common_cl = S.index.intersection(mu_classic.index)
            
            if len(common_cl) > 0:
                S_cl = S.loc[common_cl, common_cl]
                mu_cl_aligned = mu_classic.loc[common_cl]
                w_classic_raw = optimize_portfolio(mu_cl_aligned, S_cl)
                w_classic = pd.Series(w_classic_raw).reindex(df_prices.columns).fillna(0)
            else:
                 w_classic = pd.Series(0, index=df_prices.columns)

            weights = pd.concat([w_ml, w_classic], axis = 1,keys = ['ml','classic'])

            # returns and Costs
            future_returns = df_prices.loc[t_current:t_next].ffill().pct_change(fill_method=None).dropna()
            
            if not future_returns.empty:
                # gross returns
                step_ret_ml = (future_returns * w_ml).sum(axis=1)
                step_ret_classic = (future_returns * w_classic).sum(axis=1)
                
                # turnover costs
                cost_ml = (w_ml - last_weights_ml).abs().sum() * fee
                cost_classic = (w_classic - last_weights_classic).abs().sum() * fee
                
                # broker comisison 
                step_ret_ml.iloc[0] -= cost_ml
                step_ret_classic.iloc[0] -= cost_classic
                
                results.append(pd.DataFrame({
                    'ML_Strategy': step_ret_ml,
                    'Classic_Strategy': step_ret_classic
                }))
            
            last_weights_ml = w_ml
            last_weights_classic = w_classic
            print(f"Event: {t_current.date()} | OK")

        except Exception as e:
            print(f"Error in window {t_current.date()}: {e}")

    if not results:
        return pd.DataFrame(), pd.DataFrame()
        
    return pd.concat(results), weights