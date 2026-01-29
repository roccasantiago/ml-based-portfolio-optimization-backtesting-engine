from pypfopt import risk_models, EfficientFrontier,objective_functions


def get_robust_covariance(price_data):
    """
    Calculates Ledoit Wold covariance matrix
    """
    return risk_models.CovarianceShrinkage(price_data).ledoit_wolf()

def optimize_portfolio(expected_returns, cov_matrix):
    #Markowitz (its a qudratic problem)
    ef = EfficientFrontier(expected_returns, cov_matrix)

    try:
        #try to maximize max sharpe rate if thea assets get the risk_free_rate
        ef.max_sharpe(risk_free_rate=0.02)
    except:
        print("Optimization Warning: Falling back to Min Volatility")
        ef.min_volatility()
        
    return ef.clean_weights()

