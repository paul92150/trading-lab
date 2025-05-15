import numpy as np
import ta

def apply_mean_reversion(df):
    df = df.copy()

    df['RSI'] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df['mr_rsi_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))

    bollinger = ta.volatility.BollingerBands(df["Close"])
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_upper'] = bollinger.bollinger_hband()
    df['mr_bb_signal'] = np.where(df["Close"] < df["bb_lower"], 1, np.where(df["Close"] > df["bb_upper"], -1, 0))

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df['stoch'] = stoch.stoch()
    df['mr_stoch_signal'] = np.where(df['stoch'] < 20, 1, np.where(df['stoch'] > 80, -1, 0))

    df['mean_reversion_signal'] = np.sign(df['mr_rsi_signal'] + df['mr_bb_signal'] + df['mr_stoch_signal'])

    return df[['mean_reversion_signal']]
