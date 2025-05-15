import cryptocompare
import pandas as pd
from fft_cycles import (
    compute_log_returns,
    compute_fft,
    detect_dominant_cycles,
    plot_fft_spectrum,
    plot_price_series
)

def fetch_btc_data(days=2000):
    data = cryptocompare.get_historical_price_day("BTC", "USD", limit=days)
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

    return df[["Close"]].dropna()

def main():
    print("Fetching BTC data...")
    df = fetch_btc_data()

    print("Computing log returns and FFT...")
    log_returns = compute_log_returns(df)
    freqs, amps = compute_fft(log_returns)

    print("Detecting dominant cycles...")
    peak_freqs, peak_amps = detect_dominant_cycles(freqs, amps)

    print("Plotting results...")
    plot_price_series(df)
    plot_fft_spectrum(freqs, amps, peak_freqs, peak_amps)

if __name__ == "__main__":
    main()
