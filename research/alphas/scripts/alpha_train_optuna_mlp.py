"""
Train a Multi-Layer Perceptron (MLP) to generate alpha signals.
Use Optuna to tune hyperparameters based on validation Sharpe ratio.
"""

import optuna
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Local imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_mini_features
from src.preprocessing import split_data
from src.simulate import simulate_portfolio_fast

# --- Config ---
CSV_PATH = "mini_features.csv"
FEATURES = [
    "Return_5d", "Volatility_5d", "Volatility_20d", "Volume_Relative",
    "Range", "Close_to_Low", "Close_to_High", "Close_vs_Open", "ZScore_Price_20d"
]

# --- Step 1: Load and split dataset ---
df = load_mini_features(path=CSV_PATH)
df["Target"] = df.groupby("Ticker")["Return_1d"].shift(-1)
df = df.dropna()

train, val, test = split_data(df)

X_train = train[FEATURES].values
y_train = train["Target"].values
X_val = val[FEATURES].values
y_val = val["Target"].values
X_test = test[FEATURES].values
y_test = test["Target"].values

# --- Step 2: Normalize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Step 3: Define Optuna objective ---
def objective(trial):
    hidden_layer_sizes = (
        trial.suggest_int("n_units_l1", 32, 256),
        trial.suggest_int("n_units_l2", 32, 256)
    )
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "adam"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1.0, log=True)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    val_copy = val.copy()
    val_copy["Alpha"] = model.predict(X_val_scaled)

    alpha_list_val = [
        group[["Ticker", "Alpha"]] for _, group in val_copy.groupby("Date")
    ]
    _, sharpe_val = simulate_portfolio_fast(val_copy)
    return -sharpe_val  # Optuna minimizes

# --- Step 4: Run optimization ---
study = optuna.create_study(direction="minimize")
print("🔍 Running Optuna optimization...")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("✅ Best hyperparameters found:", best_params)

# --- Step 5: Retrain on train + val ---
X_trainval = pd.concat([train, val])[FEATURES].values
y_trainval = pd.concat([train, val])["Target"].values
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(best_params["n_units_l1"], best_params["n_units_l2"]),
    activation=best_params["activation"],
    solver=best_params["solver"],
    alpha=best_params["alpha"],
    learning_rate_init=best_params["learning_rate_init"],
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
model.fit(X_trainval_scaled, y_trainval)

# --- Step 6: Save model and config ---
joblib.dump(model, "mlp_alpha_model.joblib")
joblib.dump(scaler, "mlp_alpha_scaler.joblib")
with open("mlp_alpha_features.json", "w") as f:
    json.dump(FEATURES, f)
print("💾 MLP model and config saved.")

# --- Step 7: Evaluate on val and test ---
val = val.copy()
test = test.copy()
val["Alpha"] = model.predict(X_val_scaled)
test["Alpha"] = model.predict(X_test_scaled)

alpha_list_val = [group[["Ticker", "Alpha"]] for _, group in val.groupby("Date")]
alpha_list_test = [group[["Ticker", "Alpha"]] for _, group in test.groupby("Date")]

portfolio_val, sharpe_val = simulate_portfolio_fast(val)
portfolio_test, sharpe_test = simulate_portfolio_fast(test)

print(f"\n📊 Sharpe Ratio on validation (2016): {sharpe_val:.4f}")
print(f"📈 Sharpe Ratio on test (2017): {sharpe_test:.4f}")

# --- Step 8: Plot ---
plt.figure(figsize=(10, 5))
portfolio_val.plot(label="Validation 2016")
portfolio_test.plot(label="Test 2017")
plt.title("MLP Alpha Portfolio Value (Val & Test)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
