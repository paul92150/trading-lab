import numpy as np
from gplearn.fitness import make_fitness
from scipy.stats import rankdata
from src.simulate import simulate_portfolio_fast

def create_neg_sharpe_fitness(template_df):
    """
    Returns a fitness function for gplearn that minimizes the negative Sharpe ratio.
    
    Args:
        template_df (pd.DataFrame): DataFrame with ['Date', 'Ticker', 'Return_1d'].
                                    'Alpha' will be added dynamically during evaluation.

    Returns:
        gplearn-compatible fitness function (minimize -Sharpe).
    """
    def neg_sharpe_metric(y_true, y_pred, sample_weight):
        try:
            if len(y_pred) != len(template_df):
                print(f"❌ Length mismatch: {len(y_pred)} vs {len(template_df)}")
                return 1e6

            df_temp = template_df.copy()
            df_temp["Alpha"] = y_pred

            # Sanity checks
            if df_temp["Alpha"].isna().any():
                print("⚠️ Alpha contains NaNs")
                return 1e6
            if np.std(y_pred) < 1e-6:
                return 1e6

            _, sharpe = simulate_portfolio_fast(df_temp)

            if np.isnan(sharpe) or np.isinf(sharpe):
                print("⚠️ Sharpe is NaN or Inf")
                return 1e6

            return -sharpe

        except Exception as e:
            print("⚠️ simulate_portfolio_fast failed:", e)
            print("🧪 Sample Alpha:", y_pred[:5])
            return 1e6

    return make_fitness(function=neg_sharpe_metric, greater_is_better=False, wrap=False)


def create_spearman_rank_fitness(valid_df):
    """
    Returns a fitness function that minimizes negative Spearman rank correlation
    between Alpha and Target on the validation set.

    Args:
        valid_df (pd.DataFrame): DataFrame with ['Date', 'Ticker', 'Target'] columns.

    Returns:
        gplearn-compatible fitness function (minimize -IC).
    """
    grouped = valid_df.groupby("Date")
    rank_target_by_day = {}
    index_by_day = {}

    for date, group in grouped:
        target_vals = group["Target"].values
        if len(group) >= 3 and np.std(target_vals) > 1e-6:
            rank_target_by_day[date] = rankdata(target_vals)
            index_by_day[date] = group.index

    valid_index = valid_df.index

    def rank_correlation_fast(y_true, y_pred, sample_weight):
        if len(y_pred) != len(valid_index):
            return 1.0

        y_pred_arr = np.asarray(y_pred)
        correlations = []

        for date in rank_target_by_day:
            idx = index_by_day[date]
            alpha = y_pred_arr[idx]

            if np.std(alpha) < 1e-6:
                continue

            alpha_ranks = rankdata(alpha)
            target_ranks = rank_target_by_day[date]

            rho = np.corrcoef(alpha_ranks, target_ranks)[0, 1]
            if not np.isnan(rho):
                correlations.append(rho)

        if not correlations:
            return 1.0
        return -np.mean(correlations)

    return make_fitness(function=rank_correlation_fast, greater_is_better=False, wrap=False)
