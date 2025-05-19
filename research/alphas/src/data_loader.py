import pandas as pd

def load_mini_features(
    path="mini_features.csv",
    start_date="2013-01-01",
    end_date="2017-12-31",
    top_n_tickers=50
):
    """
    Load and filter the mini_features.csv dataset.

    Args:
        path (str): Path to the CSV file.
        start_date (str): Start date for filtering.
        end_date (str): End date for filtering.
        top_n_tickers (int): Keep the N most liquid tickers by average volume.

    Returns:
        pd.DataFrame: Cleaned and filtered dataframe.
    """
    df = pd.read_csv(path, parse_dates=["Date"])

    # Filter by date
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Keep the top N tickers by average trading volume
    top_tickers = (
        df.groupby("Ticker")["Volume"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_tickers)
        .index
    )
    df = df[df["Ticker"].isin(top_tickers)]

    # Final clean-up
    df = df.dropna()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return df
