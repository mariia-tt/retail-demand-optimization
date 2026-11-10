"""Моделі прогнозу. Кожна функція: train (pd.Series) -> np.array з h прогнозами."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import H, SEASON


# --- наївні бенчмарки
def naive(train: pd.Series, h: int = H):
    return np.repeat(train.iloc[-1], h)


def seasonal_naive(train: pd.Series, h: int = H):
    # значення рік тому для кожного з наступних h тижнів (вони вже є в train)
    return train.iloc[-SEASON:-SEASON + h].values


def moving_average(train: pd.Series, window: int = 4, h: int = H):
    return np.repeat(train.iloc[-window:].mean(), h)


# --- перші робочі моделі
def ets_holt(train: pd.Series, h: int = H):
    """Holt із затухаючим трендом; сезонну компоненту не беремо — даних лише на 2 сезони."""
    model = ExponentialSmoothing(train, trend="add", damped_trend=True).fit()
    return model.forecast(h).values


def ridge_fourier(features: pd.DataFrame, alpha: float = 1.0, h: int = H):
    """Повертає функцію прогнозу: Ridge на ознаках з gold (level_lag4 + Фур'є).
    Тестові тижні — h тижнів одразу після останнього тижня train."""
    def forecast(train: pd.Series):
        X_train = features.loc[train.index].dropna()
        y_train = train.loc[X_train.index]
        start = features.index.get_loc(train.index[-1]) + 1
        X_test = features.iloc[start:start + h]

        mean, std = X_train.mean(), X_train.std()        # нормалізація тільки по train
        model = Ridge(alpha=alpha).fit((X_train - mean) / std, y_train)
        return model.predict((X_test - mean) / std)
    return forecast
