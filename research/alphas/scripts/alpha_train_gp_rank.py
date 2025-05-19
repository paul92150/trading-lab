"""
Train a symbolic alpha using Genetic Programming (GP) 
with cross-sectional Spearman correlation (IC) on the validation set.
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
from src.fitness import create_spearman_rank_fitness

# --- Config ---
CSV_PATH = "mini_features.csv"
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d", "Volatility_5d", "Volatility_20d",
    "Volume_Relative", "Range", "Close_to_Low", "Close_to_High",
    "Close_vs_Open", "ZScore_Price_20d"
]

# --- Step 1: Load dataset and create Target column ---
df = load_mini_features(path=CSV_PATH)
df["Target"] = df.groupby("Ticker")["Return_1d"].shift(-1)
df = df.dropna()

train, val, test = split_data(df)
X_val, _, _, _ = scale_features(val, val, val, FEATURES)  # use only val set for IC fitness

# --- Step 2: Create Spearman-based fitness function ---
spearman_fitness = create_spearman_rank_fitness(val)

# --- Step 3: Train GP model ---
model = SymbolicRegressor(
    function_set=["add", "sub", "mul", "div", "log", "abs", "neg"],
    population_size=300,
    generations=15,
    parsimony_coefficient=0.0001,
    max_samples=0.9,
    p_point_mutation=0.1,
    p_crossover=0.7,
    verbose=1,
    metric=spearman_fitness,
    random_state=42,
    stopping_criteria=-np.inf
)

print("🧠 Training GP model with Spearman rank correlation (IC) fitness...")
start = time.time()
model.fit(X_val, val["Target"].values)
print(f"✅ Training completed in {time.time() - start:.2f} seconds\n")

# --- Step 4: Output result ---
print("📈 Best symbolic alpha discovered:\n")
print(model._program)