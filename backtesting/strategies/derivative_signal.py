import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def apply_derivative_signal(df, window_size=11, poly_order=3, threshold=1e-5):
    """
    Génère des signaux d'achat/vente basés sur les dérivées premières et secondes
    d'une courbe de prix lissée avec un filtre Savitzky-Golay.
    
    Retourne :
        pd.Series avec les signaux (-1, 0, +1)
    """
    df = df.copy()
    df['Smoothed'] = savgol_filter(df['Close'], window_size, poly_order, mode='nearest')
    df['First_Derivative'] = np.gradient(df['Smoothed'])
    df['Second_Derivative'] = np.gradient(df['First_Derivative'])
    df['Signal'] = 0

    holding = False
    for i in range(len(df)):
        if not holding and df['First_Derivative'].iloc[i] > threshold and df['Second_Derivative'].iloc[i] > threshold:
            df.at[df.index[i], 'Signal'] = 1  # Acheter
            holding = True
        elif holding and df['First_Derivative'].iloc[i] < -threshold:
            df.at[df.index[i], 'Signal'] = -1  # Vendre
            holding = False

    return df[['Signal']]
