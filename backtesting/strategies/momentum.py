import numpy as np
import ta

def apply_momentum(df):
    df = df.copy()

    df['momentum_simple'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)

    df['volume_SMA20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    df['abnormal_volume'] = np.where(df['Volume'] > 1.2 * df['volume_SMA20'], 1, 0)

    df['momentum_signal'] = np.where(df['abnormal_volume'] == 1, df['momentum_simple'], 0)

    return df[['momentum_signal']]
