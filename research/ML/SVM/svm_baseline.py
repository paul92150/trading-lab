import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_crypto_data

# Charger les données prétraitées
df = load_crypto_data(symbol="BTC", days=2000)

# Générer une étiquette simplifiée : 1 si le prix monte demain, 0 sinon
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Supprimer la dernière ligne (target NaN)
df = df.dropna()

X = df.drop(columns=["Close", "Target"])
y = df["Target"]

# Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Entraînement SVM
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# Score
accuracy = clf.score(X_test, y_test)
print(f"Accuracy sur le jeu de test : {accuracy:.2%}")
