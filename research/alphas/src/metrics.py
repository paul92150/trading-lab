import numpy as np
from scipy.stats import spearmanr

def compute_sharpe(L, annualize=True, periods_per_year=252):
    Returns = [(L.iloc[i] - L.iloc[i - 1]) / L.iloc[i - 1] for i in range(1, len(L))]
    mean_returns = np.mean(Returns)
    std_returns = np.std(Returns)
    if std_returns == 0:
        return 0
    sharpe = mean_returns / std_returns
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    return sharpe


def compute_turnover(position_df):
    """
    Computes daily average turnover.
    Expects a dataframe with columns: Date, Ticker, Position
    """
    position_df = position_df.copy()
    position_df["AbsWeight"] = position_df["Position"].abs()

    # Pivot to Ticker x Date matrix
    pivot = position_df.pivot(index="Date", columns="Ticker", values="Position").fillna(0)

    # Turnover is sum of absolute daily changes in position weights
    daily_turnover = pivot.diff().abs().sum(axis=1)
    return daily_turnover.mean()

def compute_information_coefficient(df):
    """
    Computes daily Spearman rank correlation between Alpha and future Return (IC).
    Expects: Date, Ticker, Alpha, Target
    """
    ic_list = []

    for date, group in df.groupby("Date"):
        valid = group.dropna(subset=["Alpha", "Target_5d"])
        if len(valid) < 5:
            continue
        ic, _ = spearmanr(valid["Alpha"], valid["Target_5d"])
        ic_list.append(ic)

    return np.mean(ic_list)