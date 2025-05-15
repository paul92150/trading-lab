import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from data_loader import load_crypto_data

def backtest_svm(symbol="BTC", days=2000):
    print(f"📥 Chargement des données {symbol}/{symbol} sur {days} jours...")
    df = load_crypto_data(symbol=symbol, days=days)

    # Créer une cible binaire : 1 si le prix augmente le lendemain, 0 sinon
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    # Séparer les features et la cible
    features = df.drop(columns=["Close", "Target"])
    target = df["Target"]

    # Normaliser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Split train/test (80/20)
    split = int(0.8 * len(df))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = target[:split], target[split:]
    prices_test = df["Close"].iloc[split:].reset_index(drop=True)

    # Entraîner le modèle SVM
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    # Prédictions
    probs = model.predict_proba(X_test)
    preds = (probs[:, 1] > 0.55).astype(int)  # Seuil ajustable

    # Backtest
    balance = 1000.0
    crypto = 0.0
    values = []

    for i in range(len(preds) - 1):
        price_today = prices_test[i]
        price_tomorrow = prices_test[i + 1]

        if preds[i] == 1:  # BUY signal
            if balance > 0:
                crypto = balance / price_today
                balance = 0.0
        else:  # SELL signal
            if crypto > 0:
                balance = crypto * price_today
                crypto = 0.0

        portfolio_value = balance + crypto * price_today
        values.append(portfolio_value)

    # Final sell (liquidate)
    final_value = balance + crypto * prices_test.iloc[-1]
    print(f"\n💰 Valeur finale du portefeuille (SVM) : ${final_value:.2f}")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(values, label="Portefeuille SVM")
    plt.xlabel("Pas de temps (test)")
    plt.ylabel("Valeur du portefeuille ($)")
    plt.title("Backtest de la stratégie SVM")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    backtest_svm()
