# engine.py

import time
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import pandas as pd


class SimulationEngine:
    def __init__(self):
        self.prices = {
            "AAPL": 150.0,
            "GOOG": 2800.0,
            "AMZN": 3300.0,
            "TSLA": 700.0,
            "VT": 200.0,
            "BTC": 30000.0,
            "ETH": 2000.0,
            "XRP": 0.5,
            "DOGE": 0.1,
            "BNB": 400.0
        }
        self.running = True

        self.stock_tickers = ["AAPL", "GOOG", "AMZN", "TSLA", "VT"]
        self.crypto_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "BNB": "binancecoin"
        }

    def update_prices(self):
        cg = CoinGeckoAPI()
        while self.running:
            self._update_stock_prices()
            self._update_crypto_prices(cg)
            print('sleeping')
            time.sleep(60)  # Update every minute

    def _update_stock_prices(self):
        try:
            print('Updating stock prices...')
            for ticker in self.stock_tickers:
                try:
                    df = yf.download(ticker, period="1d", interval="5m", progress=False)
                    if not df.empty:
                        latest_price = df["Close"].dropna().iloc[-1]
                        self.prices[ticker] = float(latest_price)
                        print(f"Updated {ticker}: {latest_price:.2f} USD")
                    else:
                        print(f"No data returned for {ticker}")
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                time.sleep(2)  # Wait 2 seconds between requests to avoid rate limiting
        except Exception as e:
            print("Erreur générale lors de la mise à jour des actions :", e)

    def _update_crypto_prices(self, cg):
        try:
            crypto_ids = list(self.crypto_mapping.values())
            crypto_data = cg.get_price(ids=crypto_ids, vs_currencies="usd")
            for ticker, coin_id in self.crypto_mapping.items():
                if coin_id in crypto_data and "usd" in crypto_data[coin_id]:
                    self.prices[ticker] = crypto_data[coin_id]["usd"]
        except Exception as e:
            print("Erreur lors de la récupération des cryptos :", e)
