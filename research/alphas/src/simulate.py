import pandas as pd
import numpy as np
from src.metrics import compute_sharpe 

def alpha_to_portfolio(df, alpha_col="Alpha", capital=1_000):
    df = df.copy()
    df["Alpha_centered"] = df[alpha_col] - df[alpha_col].mean()
    df["Weight"] = df["Alpha_centered"] / df["Alpha_centered"].abs().sum()
    df["Allocated_$"] = df["Weight"] * capital
    return df[["Ticker", "Weight"]]

def simulate_portfolio(df, alpha_list, capital_start=1_000):
    """
    df: full dataframe with columns ['Date', 'Ticker', 'Return_1d']
    alpha_list: list of DataFrames, each with columns ['Ticker', 'Alpha'], one per day (same order as sorted unique dates)
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    unique_dates = sorted(df["Date"].unique())
    assert len(unique_dates) == len(alpha_list), "Mismatch between number of days and alphas"

    capital_history = []
    capital = capital_start

    for i, date in enumerate(unique_dates[:-1]):
        alpha_df = alpha_list[i].copy()
        alpha_df["Date"] = date

        next_day = unique_dates[i + 1]
        returns_df = df[df["Date"] == next_day][["Ticker", "Return_1d"]]

        merged = pd.merge(alpha_df, returns_df, on="Ticker", how="inner")
        if merged.empty:
            capital_history.append(capital)
            continue

        weighted = alpha_to_portfolio(merged, alpha_col="Alpha", capital=capital)
        merged = pd.merge(merged, weighted, on="Ticker")
        merged["PnL"] = merged["Weight"] * capital * merged["Return_1d"]

        daily_pnl = merged["PnL"].sum()
        capital += daily_pnl
        capital_history.append(capital)

    portfolio_series = pd.Series(capital_history, index=unique_dates[:-1], name="PortfolioValue")
    sharpe = compute_sharpe(portfolio_series) 

    return portfolio_series, sharpe

def simulate_portfolio_fast(df, capital_start=1_000):
    """
    Ultra-fast vectorized backtest:
    - df must have columns: ['Date', 'Ticker', 'Return_1d', 'Alpha']
    - Returns a Series of portfolio value and the Sharpe ratio
    """
    df = df.sort_values(["Date", "Ticker"])
    df["NextReturn"] = df.groupby("Ticker")["Return_1d"].shift(-1)
    df = df.dropna(subset=["NextReturn"])

    # Vectorized portfolio weighting
    date_groups = df.groupby("Date")
    alpha = df["Alpha"]
    centered = alpha - date_groups["Alpha"].transform("mean")
    weights = centered / date_groups["Alpha"].transform(lambda x: x.abs().sum())
    df["Weight"] = weights

    df["DailyPnL"] = df["Weight"] * df["NextReturn"]
    daily_returns = df.groupby("Date")["DailyPnL"].sum()

    returns_array = np.array([
        float(np.ravel(x)[0]) if isinstance(x, (np.ndarray, list, tuple)) else float(x)
        for x in daily_returns
    ])

    portfolio_array = np.cumprod(1 + returns_array) * capital_start
    portfolio = pd.Series(portfolio_array, index=daily_returns.index, name="PortfolioValue")
    sharpe = compute_sharpe(portfolio)

    return portfolio, sharpe

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=5, freq="D").tolist() * 3,
        "Ticker": ["A", "B", "C"] * 5,
        "Return_1d": [0.01, -0.02, 0.03] * 5
    }
    df = pd.DataFrame(data)

    # Example alpha list (one DataFrame per day)
    alpha_list = [
        pd.DataFrame({"Ticker": ["A", "B", "C"], "Alpha": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"Ticker": ["A", "B", "C"], "Alpha": [0.2, 0.1, 0.4]}),
        pd.DataFrame({"Ticker": ["A", "B", "C"], "Alpha": [0.3, 0.4, 0.2]}),
        pd.DataFrame({"Ticker": ["A", "B", "C"], "Alpha": [0.4, 0.3, 0.1]}),
        pd.DataFrame({"Ticker": ["A", "B", "C"], "Alpha": [0.5, 0.6, 0.7]})
    ]

    portfolio_value, sharp = simulate_portfolio(df, alpha_list)
    print(portfolio_value, sharp)