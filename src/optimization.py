from pypfopt import risk_models, EfficientFrontier,objective_functions


def get_robust_covariance(price_data):
    """
    Calculates Ledoit Wold covariance matrix
    """
    return risk_models.CovarianceShrinkage(price_data).ledoit_wolf()

def optimize_portfolio(expected_returns, cov_matrix, target="max_sharpe"):
    #Markowitz (its a qudratic problem)
    ef = EfficientFrontier(expected_returns, cov_matrix)
    
    #possibility to use L2 and add constrains (bibliography recommendation .2 < w <  .3 )
    # L2
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    # for every asset the weight asigned must be below .3
    #ef.add_constraint(lambda w: w <= 0.30)
    
    if target == "min_vol":
        ef.min_volatility()
    elif target == "max_sharpe":
        ef.max_sharpe(risk_free_rate=0.02)
        
    return ef.clean_weights()

