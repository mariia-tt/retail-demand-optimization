"""Ознаки для моделей. Усі мають лаг >= H або є календарем — на момент прогнозу вони відомі."""
import numpy as np
import pandas as pd

from .config import H, SEASON

FEATURES = ["level_lag4", "sin1", "cos1", "sin2", "cos2"]


def make_features(series: pd.Series, h: int = H) -> pd.DataFrame:
    """level_lag4 — середнє за тижні t-h-3 … t-h; sin/cos — Фур'є-члени від номера тижня.

    `series` може містити NaN у майбутніх тижнях: ознаки з лагом >= h все одно рахуються.
    """
    week = series.index.isocalendar().week.values.astype(float)
    feats = pd.DataFrame(index=series.index)
    feats["level_lag4"] = series.shift(h).rolling(4).mean()
    for k in [1, 2]:
        feats[f"sin{k}"] = np.sin(2 * np.pi * k * week / SEASON)
        feats[f"cos{k}"] = np.cos(2 * np.pi * k * week / SEASON)
    return feats
