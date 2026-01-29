import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from ingestion import *
from model import * 
from optimization import *
from backtesting import run_backtest_pipeline
from config import CONFIG
import time

# CONFIG
TICKERS = CONFIG['TICKERS']
START_DATE = CONFIG['START_DATE']
END_DATE = CONFIG['END_DATE']
FEATURES = CONFIG['FEATURES']
BROKER_COMISSION = CONFIG['BROKER_COMISSION']
SAVE_TO_CSV = CONFIG['SAVE_TO_CSV']
SAVE_DASHBOARD = CONFIG['SAVE_DASHBOARD']
DEVICE = CONFIG['DEVICE']



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

def get_metrics(series, benchmark_rets):

    df_metrics = pd.concat([series, benchmark_rets], axis=1).dropna()
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

    #6 month warmup
    trading_start = pd.to_datetime(START_DATE)
    download_start = trading_start - pd.DateOffset(months=6)
    download_start_str = download_start.strftime('%Y-%m-%d')

    print(f"Trading sart: {START_DATE}")
    print(f"\nDate of data download (Warm-up): {download_start_str}")

    results, weights = run_backtest_pipeline(
        tickers=TICKERS, start_date=download_start_str, end_date=END_DATE,
        features=FEATURES, fee=BROKER_COMISSION, save_to_csv=SAVE_TO_CSV, device = DEVICE 
    )

    print(f"Benchmark and get feature importance")
    df_spy = pipeline_preprocess(['SPY'], START_DATE, END_DATE, save_to_csv=False)
    ret_spy = (np.exp(df_spy['log_ret'].reindex(results.index)) - 1)
    
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

        im = ax3.imshow(df_cov, cmap='BuGn', alpha=0.6)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['ML', 'SPY'], color='#555555') 
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['ML', 'SPY'], color='#555555')

        for (i, j), val in np.ndenumerate(df_cov):
            ax3.text(j, i, f'{val:.5f}', ha='center', va='center', 
                     weight='bold', size=12, color='#2c3e50') 

        ax3.set_title("Covariance Matrix (Annualized Risk)", pad=15, color='#333333')


        cbar = plt.colorbar(im, ax=ax3)
        cbar.outline.set_visible(False) 
        cbar.ax.tick_params(labelsize=10, colors='#555555')

        for spine in ax3.spines.values():
            spine.set_visible(False)
    
        # Drawdown graph
        ax4 = fig.add_subplot(gs[2, :])
        dd_ml = (metrics_ml[3] - metrics_ml[3].cummax()) / metrics_ml[3].cummax()
        dd_cl = (metrics_cl[3] - metrics_cl[3].cummax()) / metrics_cl[3].cummax()
        dd_sp = (metrics_sp[3] - metrics_sp[3].cummax()) / metrics_sp[3].cummax()

        ax4.fill_between(dd_ml.index, dd_ml, color='forestgreen', alpha=0.3, label='ML Drawdown')
        ax4.fill_between(dd_cl.index, dd_cl, color='royalblue', alpha=0.2, label='Classic Drawdown')
        ax4.plot(dd_sp.index, dd_sp, color='gray', linestyle='--', lw=1.5, label='S&P 500 Drawdown', alpha=0.8)

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
        cmap = plt.cm.tab20c
        
        ax6 = fig.add_subplot(gs[4, 0])
        
        w_ml = weights['ml']
        w_ml_clean = w_ml[w_ml > 0.01].sort_values(ascending=False)
        
        colors_ml = cmap(np.linspace(0, 1, len(w_ml_clean)))
        
        wedges6, texts6 = ax6.pie(
            w_ml_clean, 
            startangle=90, 
            colors=colors_ml,
            pctdistance=0.85,
            textprops={'fontsize': 9}
        )
        
        ax6.add_artist(plt.Circle((0,0), 0.70, fc='white'))
        ax6.set_title("ML Strategy Weights", fontsize=12, fontweight='bold')

        ax6.legend(
            wedges6, 
            w_ml_clean.index,
            title="Tickers (>1%)",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False,
            fontsize=8
        )

        ax7 = fig.add_subplot(gs[4, 1])
        
        w_cl = weights['classic']
        w_cl_clean = w_cl[w_cl > 0.01].sort_values(ascending=False)
        
        colors_cl = cmap(np.linspace(0, 1, len(w_cl_clean)))
        
        wedges7, texts7 = ax7.pie(
            w_cl_clean, 
            startangle=90, 
            colors=colors_cl,
            pctdistance=0.85,
            textprops={'fontsize': 9}
        )
        
        ax7.add_artist(plt.Circle((0,0), 0.70, fc='white'))
        ax7.set_title("Classic Model Weights", fontsize=12, fontweight='bold')

        ax7.legend(
            wedges7, 
            w_cl_clean.index,
            title="Tickers (>1%)",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False,
            fontsize=8
        )
        
        plt.tight_layout()

        if SAVE_DASHBOARD:
            os.makedirs("dashboards", exist_ok=True)
            now = datetime.now()
            date_str = now.strftime(r"%Y-%m-%d_%H-%M")
            graph_path = os.path.join("dashboards", f"dashboard{date_str}.png")
            plt.savefig(graph_path, bbox_inches="tight", dpi=300)

        plt.show()
        
        result = {
            "ML Strategy": [f"{metrics_ml[0]:.2%}", metrics_ml[1], f"{metrics_ml[2]:.2%}", metrics_ml[4]],
            "Classic": [f"{metrics_cl[0]:.2%}", metrics_cl[1], f"{metrics_cl[2]:.2%}", metrics_cl[4]],
            "S&P 500": [f"{metrics_sp[0]:.2%}", metrics_sp[1], f"{metrics_sp[2]:.2%}", metrics_sp[4]]
        }

        df_result = pd.DataFrame(result, index=["Return", "Sharpe", "Drawdown", "Beta"])

        print(df_result)

       
  