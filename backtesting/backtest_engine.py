# backtesting/backtest_engine.py

import numpy as np
import pandas as pd

def backtest_strategies(price_df, signal_dfs, log_return_column='log_return'):
    """
    Backtest multiple strategies based on log-returns and signal DataFrames.

    Args:
        price_df (pd.DataFrame): DataFrame with 'log_return' and price index.
        signal_dfs (dict): Dictionary {strategy_name: pd.DataFrame with signal column}
        log_return_column (str): Name of the log return column in price_df.

    Returns:
        pd.DataFrame: price_df enriched with strategy return columns and performance summaries.
    """

    df = price_df.copy()
    if log_return_column not in df.columns:
        raise ValueError(f"'{log_return_column}' column is required in price_df.")

    cumulative_returns = []

    for strat_name, signal_df in signal_dfs.items():
        # Ensure alignment
        signal_series = signal_df.iloc[:, 0].reindex(df.index).fillna(0)
        
        # Simulate: shift signal by 1 day to avoid lookahead bias
        strat_daily_return = signal_series.shift(1) * df[log_return_column]
        strat_cum_return = np.exp(strat_daily_return.cumsum())
        
        # Add columns to main DataFrame
        df[f'{strat_name}_ret'] = strat_daily_return
        df[f'{strat_name}_cum'] = strat_cum_return
        
        cumulative_returns.append(f'{strat_name}_cum')

    # Buy & Hold for comparison
    df['buy_hold_cum'] = np.exp(df[log_return_column].cumsum())

    # Average strategy cumulative return
    df['avg_strategy_cum'] = df[cumulative_returns].mean(axis=1)

    return df
