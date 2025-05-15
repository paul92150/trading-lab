import numpy as np

def apply_market_making(df):
    df = df.copy()

    df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    df['market_making_signal'] = np.where(df['daily_range'] < 0.02, 1, -1)

    return df[['market_making_signal']]
