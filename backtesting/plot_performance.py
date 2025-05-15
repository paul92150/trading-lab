import matplotlib.pyplot as plt

def plot_cumulative_returns(df, strategies=None, figsize=(14, 8)):
    """
    Plot cumulative returns for each strategy and buy & hold.

    Args:
        df (pd.DataFrame): DataFrame returned by backtest_strategies.
        strategies (list): List of strategy names as in the signal dict keys. If None, inferred from columns.
        figsize (tuple): Size of the plot.
    """
    if strategies is None:
        # Infer strategies by detecting *_cum columns except buy & hold and avg
        strategies = [col.replace('_cum', '') for col in df.columns if col.endswith('_cum') and not col.startswith(('buy_hold', 'avg_strategy'))]

    plt.figure(figsize=figsize)

    # Plot Buy & Hold baseline
    if 'buy_hold_cum' in df.columns:
        plt.plot(df.index, df['buy_hold_cum'], label='Buy & Hold', color='black', linewidth=2)

    # Plot each strategy
    for strat in strategies:
        cum_col = f'{strat}_cum'
        if cum_col in df.columns:
            plt.plot(df.index, df[cum_col], label=strat)

    # Plot average if available
    if 'avg_strategy_cum' in df.columns:
        plt.plot(df.index, df['avg_strategy_cum'], label='Average Strategy', linestyle='--', color='brown', linewidth=2)

    plt.title("Cumulative Strategy Performance vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
