"""
Train a symbolic alpha using Genetic Programming (GP) with Sharpe ratio as the fitness function.
"""

import time
import numpy as np
from gplearn.genetic import SymbolicRegressor

# Local imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_mini_features
from src.preprocessing import split_data, scale_features
from src.fitness import create_neg_sharpe_fitness

# --- Config ---
CSV_PATH = "mini_features.csv"
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d", "Volatility_5d", "Volatility_20d",
    "Volume_Relative", "Range", "Close_to_Low", "Close_to_High",
    "Close_vs_Open", "ZScore_Price_20d"
]

# --- Step 1: Load and prepare dataset ---
df = load_mini_features(path=CSV_PATH)
df["Target"] = df.groupby("Ticker")["Return_1d"].shift(-1)
df = df.dropna()

# Use the full train DataFrame as template (features + Date, Ticker, Return_1d)
train, val, test = split_data(df)
template_df = train.copy()

# Scale features
X_train, _, _, _ = scale_features(train, val, test, FEATURES)

# --- Step 2: Create fitness function ---
neg_sharpe_fitness = create_neg_sharpe_fitness(template_df)

# --- Step 3: Train Symbolic Regressor ---
model = SymbolicRegressor(
    function_set=["add", "sub", "mul", "div", "log", "abs", "neg"],
    population_size=300,
    generations=15,
    parsimony_coefficient=0.0001,
    max_samples=0.9,
    p_point_mutation=0.1,
    p_crossover=0.7,
    verbose=1,
    metric=neg_sharpe_fitness,
    random_state=42,
    stopping_criteria=-np.inf
)

print("🧠 Training GP model with Sharpe fitness...")
start = time.time()
model.fit(X_train, train["Target"].values)
print(f"✅ Training completed in {time.time() - start:.2f} seconds\n")

# --- Step 4: Output discovered formula ---
print("📈 Best symbolic alpha discovered:\n")
print(model._program)