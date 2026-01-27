import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from ingestion import *
from model import * 
from optimization import *
from backtesting import run_backtest_pipeline
from config import CONFIG

# CONFIG
TICKERS = CONFIG['TICKERS']
START_DATE = CONFIG['START_DATE']
END_DATE = CONFIG['END_DATE']
FEATURES = CONFIG['FEATURES']
BROKER_COMISSION = CONFIG['BROKER_COMISSION']
SAVE_TO_CSV = CONFIG['SAVE_TO_CSV']

def get_feature_importance(tickers, features, start, end):
    """
    Train a baseline model to extract feature importance
    """
    df_long = pipeline_preprocess(tickers, start, end, save_to_csv=False)

    df_sample = df_long[df_long['ticker'] == tickers[0]].sort_index()
    X, y = df_sample[features], df_sample['target']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    return pd.Series(model.feature_importances_, index=features).sort_values()

def get_metrics(serie, benchmark_rets):

    df_metrics = pd.concat([serie, benchmark_rets], axis=1).dropna()
    rets_port = df_metrics.iloc[:, 0]
    rets_bench = df_metrics.iloc[:, 1]

    cum_rets = (1 + rets_port).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    sharpe = (rets_port.mean() / rets_port.std()) * (252**0.5)
    max_dd = ((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()
    
    # beta
    covariance_matrix = np.cov(rets_port, rets_bench)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    
    # alpha
    alpha = (rets_port.mean() * 252) - (beta * (rets_bench.mean() * 252))

    return total_ret, sharpe, max_dd, cum_rets, beta, alpha, rets_port, rets_bench

if __name__ == "__main__":

    results, weights = run_backtest_pipeline(
        tickers=TICKERS, start_date=START_DATE, end_date=END_DATE,
        features=FEATURES, fee=BROKER_COMISSION, save_to_csv=SAVE_TO_CSV
    )

    print(f"Benchmark and get feature importance")
    df_spy = pipeline_preprocess(['SPY'], START_DATE, END_DATE, save_to_csv=False)
    ret_spy = (np.exp(df_spy['log_ret'].reindex(results.index)) - 1).fillna(0)
    
    feature_importance_list = get_feature_importance(TICKERS, FEATURES, START_DATE, END_DATE)
    
    #Visualisation
    if not results.empty:
        # Metrics
        # Get metrics
        metrics_ml = get_metrics(results['ML_Strategy'], ret_spy)
        metrics_cl = get_metrics(results['Classic_Strategy'], ret_spy)
        metrics_sp = get_metrics(ret_spy, ret_spy)

        # Graph initialization
        fig = plt.figure(figsize=(15, 16)) 
        gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 1, 1, 1, 1])
    
        # Returns graph
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(metrics_ml[3], label=f"ML Strategy (Sharpe: {metrics_ml[1]:.2f}, Beta: {metrics_ml[4]:.2f})", color='forestgreen', lw=2)
        ax1.plot(metrics_cl[3], label=f"Classic (Sharpe: {metrics_cl[1]:.2f}, Beta: {metrics_cl[4]:.2f})", color='royalblue', ls='--')
        ax1.plot(metrics_sp[3], label="S&P 500 (Benchmark)", color='gray', alpha=0.5)
        ax1.set_title("Cumulative Performance: Risk and Return Analysis", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
    
        # Feature importance graph
        ax2 = fig.add_subplot(gs[1, 0])
        feature_importance_list.plot(kind='barh', ax=ax2, color='teal', edgecolor='black')
        ax2.set_title("Feature Importance")
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
        # Covariance matrix graph
        ax3 = fig.add_subplot(gs[1, 1])
        df_cov = pd.concat([results['ML_Strategy'], ret_spy], axis=1).dropna().cov() * 252
        df_cov.columns = ['ML', 'SPY']
        im = ax3.imshow(df_cov, cmap='RdYlGn_r')
        ax3.set_xticks([0, 1]); ax3.set_xticklabels(['ML', 'SPY'])
        ax3.set_yticks([0, 1]); ax3.set_yticklabels(['ML', 'SPY'])
        for (i, j), val in np.ndenumerate(df_cov):
            ax3.text(j, i, f'{val:.5f}', ha='center', va='center', weight='bold', size=12)
        ax3.set_title("Covariance Matrix (Annualized Risk)")
        plt.colorbar(im, ax=ax3)
    
        # Drawdown graph
        ax4 = fig.add_subplot(gs[2, :])
        dd_ml = (metrics_ml[3] - metrics_ml[3].cummax()) / metrics_ml[3].cummax()
        dd_cl = (metrics_cl[3] - metrics_cl[3].cummax()) / metrics_cl[3].cummax()
        ax4.fill_between(dd_ml.index, dd_ml, color='forestgreen', alpha=0.3, label='ML Drawdown')
        ax4.fill_between(dd_cl.index, dd_cl, color='royalblue', alpha=0.2, label='Classic Drawdown')
        ax4.set_title("Risk Profile: Maximum Drawdown")
        ax4.legend()
        ax4.grid(alpha=0.3)
    
        # Scatter Plot
        ax5 = fig.add_subplot(gs[3, :])
        x_bench = metrics_ml[7]
        y_strat = metrics_ml[6]
        ax5.scatter(x_bench, y_strat, alpha=0.5, color='forestgreen', s=15)
        
        # Regression line
        beta_val = metrics_ml[4]
        line = beta_val * x_bench
        ax5.plot(x_bench, line, color='darkred', lw=2, label=f'Beta: {beta_val:.2f}')
        
        ax5.set_title("Sensitivity Analysis (Beta vs Benchmark)")
        ax5.set_xlabel("S&P 500 Daily Returns")
        ax5.set_ylabel("ML Strategy Daily Returns")
        ax5.legend()
        ax5.grid(alpha=0.2)
        # Piecharts
        colors = plt.cm.Paired(range(len(TICKERS)))

        ax6 = fig.add_subplot(gs[4, 0])
        wedges6, texts6 = ax6.pie(weights['ml'].sort_values(), 
                        startangle=90, 
                        colors=colors)
        ax6.set_title("ML Weights")

        ax6.legend(wedges6, TICKERS,
          title="Tickers",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1)
        )

        ax7 = fig.add_subplot(gs[4, 1])
        wedges7, texts7 = ax7.pie(weights['classic'].sort_values(), 
                        startangle=90, 
                        colors=colors)
        ax7.set_title("Classic Model Weights")

        ax7.legend(wedges7, TICKERS,
          title="Tickers",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1)
        )

        print(weights)
        
        plt.tight_layout()
        plt.show()

        # Clean printing of results (awful)
        print("\n" + "="*65)
        print(f"{'METRIC':<20} | {'ML STRATEGY':<12} | {'CLASSIC':<12} | {'S&P 500':<12}")
        print("-" * 65)
        print(f"{'Total Return':<20} | {metrics_ml[0]*100:>10.2f}% | {metrics_cl[0]*100:>10.2f}% | {metrics_sp[0]*100:>10.2f}%")
        print(f"{'Sharpe Ratio':<20} | {metrics_ml[1]:>11.2f} | {metrics_cl[1]:>11.2f} | {metrics_sp[1]:>11.2f}")
        print(f"{'Max Drawdown':<20} | {metrics_ml[2]*100:>10.2f}% | {metrics_cl[2]*100:>10.2f}% | {metrics_sp[2]*100:>10.2f}%")
        print(f"{'Beta (vs Market)':<20} | {metrics_ml[4]:>11.2f} | {metrics_cl[4]:>11.2f} | {metrics_sp[4]:>11.2f}")
        print(f"{'Alpha (Excess)':<20} | {metrics_ml[5]*100:>10.2f}% | {metrics_cl[5]*100:>10.2f}% | {'-':>12}")
        print("="*65 + "\n")
        
        print(weights)