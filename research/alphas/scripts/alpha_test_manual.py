"""
Test a manually defined alpha signal using simulate_portfolio_fast.
Useful for quick experimentation and baseline comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_mini_features
from src.utils_rank import rank
from src.simulate import simulate_portfolio_fast

# --- Config ---
CSV_PATH = "mini_features.csv"

# --- Step 1: Load dataset ---
df = load_mini_features(path=CSV_PATH)

# --- Step 2: Define alpha manually ---
# Example: Reversal alpha = rank of 5-day return (lowest returns → highest rank)
df["Alpha"] = rank(df, "Return_5d", ascending=True)

# Optional: Try other alpha ideas here
# df["Alpha"] = rank(df, "Volatility_5d", ascending=False)

# --- Step 3: Backtest alpha signal ---
portfolio, sharpe = simulate_portfolio_fast(df)

# --- Step 4: Results ---
print("\n📈 Alpha Test Results:")
print(f"Sharpe Ratio: {sharpe:.4f}")

# --- Step 5: Plot portfolio value ---
portfolio.plot(title="Manual Alpha Portfolio Value")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.tight_layout()
plt.show()
