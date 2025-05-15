def load_crypto_data(symbol="BTC", fiat="USD", days=2000):
    import pandas as pd
    import numpy as np
    import cryptocompare
    import ta

    print(f"Téléchargement des données {symbol}/{fiat} sur {days} jours...")
    data = cryptocompare.get_historical_price_day(symbol, fiat, limit=days)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={
        "close": "Close", 
        "open": "Open", 
        "high": "High", 
        "low": "Low", 
        "volumefrom": "VolumeCrypto", 
        "volumeto": "VolumeFiat"
    }, inplace=True)
    df["Volume"] = df["VolumeCrypto"]

    # === Fenêtres de chaque indicateur ===
    windows = {
        "LogReturns": 1,
        "RSI": 14,
        "SMA_10": 10,
        "SMA_50": 50,
        "MACD_diff": 26,
        "BB_width": 20
    }

    # === Calcul des indicateurs ===
    df["LogReturns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=windows["RSI"]).rsi()
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=windows["SMA_10"])
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=windows["SMA_50"])
    macd = ta.trend.MACD(df["Close"], window_slow=windows["MACD_diff"])
    df["MACD_diff"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=windows["BB_width"])
    df["BB_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["Close"]

    max_window = max(windows.values())
    df = df.iloc[max_window:].reset_index(drop=True)


    # === Sélection finale ===
    return df[["Close", "LogReturns", "RSI", "SMA_10", "SMA_50", "MACD_diff", "BB_width"]]

if __name__ == "__main__":
    df = load_crypto_data()
    print(df.head())