import numpy as np
import warnings

from data_loader import load_crypto_data

from strategies import (
    trend_following,
    mean_reversion,
    momentum,
    breakout_volatility,
    market_making,
    derivative_signal
)
from backtest_engine import backtest_strategies
from plot_performance import plot_cumulative_returns

warnings.filterwarnings("ignore")


def compute_log_returns(df, price_column='Close'):
    df['log_return'] = np.log(df[price_column] / df[price_column].shift(1))
    return df.dropna()


def main():
    print("Fetching Bitcoin price data...")
    df = load_crypto_data(limit=2000)
    df = compute_log_returns(df)

    print("Applying strategies...")
    signals = {
        'TrendFollowing': trend_following.apply_trend_following(df),
        'MeanReversion': mean_reversion.apply_mean_reversion(df),
        'Momentum': momentum.apply_momentum(df),
        'BreakoutVolatility': breakout_volatility.apply_breakout_volatility(df),
        'MarketMaking': market_making.apply_market_making(df),
        'DerivativeSignal': derivative_signal.apply_derivative_signal(df)
    }

    print("Running backtest...")
    results_df = backtest_strategies(df, signals)

    print("Plotting results...")
    plot_cumulative_returns(results_df)


    buy_hold = results_df["buy_hold_cum"].iloc[-1]
    avg = results_df["avg_strategy_cum"].iloc[-1]
    print(f"Buy & Hold: {buy_hold:.2f}x")
    print(f"Average Strategy: {avg:.2f}x")


if __name__ == "__main__":
    main()
