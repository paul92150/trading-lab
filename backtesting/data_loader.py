import pandas as pd
import numpy as np
import cryptocompare

def load_crypto_data(symbol="BTC", currency="USD", limit=2000):
    data = cryptocompare.get_historical_price_day(symbol, currency, limit=limit)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit='s')
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
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    return df
