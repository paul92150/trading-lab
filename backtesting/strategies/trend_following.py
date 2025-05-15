import numpy as np
import ta

def apply_trend_following(df):
    df = df.copy()

    df['SMA200'] = ta.trend.sma_indicator(df["Close"], window=200)
    df['tf_mm200_signal'] = np.where(df["Close"] > df["SMA200"], 1, -1)

    macd = ta.trend.MACD(df["Close"])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['tf_macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

    ichimoku = ta.trend.IchimokuIndicator(high=df["High"], low=df["Low"])
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['tf_ichimoku_signal'] = np.where(
        df["Close"] > np.maximum(df["ichimoku_conversion"], df["ichimoku_base"]),
        1, -1
    )

    df['trend_following_signal'] = np.sign(
        df['tf_mm200_signal'] + df['tf_macd_signal'] + df['tf_ichimoku_signal']
    )

    return df[['trend_following_signal']]
