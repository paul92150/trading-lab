import numpy as np
import ta

def apply_breakout_volatility(df):
    df = df.copy()

    df['SMA20'] = ta.trend.sma_indicator(df["Close"], window=20)
    bollinger = ta.volatility.BollingerBands(df["Close"])
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_width'] = (df["bb_upper"] - df["bb_lower"]) / df['SMA20']
    df['bb_width_shift'] = df['bb_width'].shift(1)
    df['breakout_signal'] = np.where(df['bb_width'] > 1.5 * df['bb_width_shift'], 1, 0)

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"])
    df['ATR'] = atr.average_true_range()
    df['ATR_SMA'] = ta.trend.sma_indicator(df['ATR'], window=14)
    df['atr_expansion_signal'] = np.where(df['ATR'] > 1.5 * df['ATR_SMA'], 1, 0)

    df['breakout_volatility_signal'] = np.where(
        (df['breakout_signal'] == 1) | (df['atr_expansion_signal'] == 1), 1, 0
    )

    return df[['breakout_volatility_signal']]
