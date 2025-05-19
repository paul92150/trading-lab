import pandas as pd


def rank(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    """
    Compute a normalized cross-sectional rank per date (similar to WorldQuant Brain's `rank()`).

    Args:
        df (pd.DataFrame): DataFrame containing 'Date' and the target column.
        column (str): Column to rank.
        ascending (bool): If True, lower values get lower ranks. 
                          If False, higher values get lower ranks (inverse ranking).

    Returns:
        pd.Series: Cross-sectional rank between 0 and 1 (normalized by number of assets per day).
    """
    return df.groupby("Date")[column].transform(
        lambda x: x.rank(method="average", pct=True, ascending=ascending)
    )
