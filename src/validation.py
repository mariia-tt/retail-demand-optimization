"""Rolling-origin backtesting (expanding window) і метрики."""
import numpy as np
import pandas as pd

from .config import H, N_FOLDS, STEP
from .features import make_features


def make_origins(n_weeks: int, h: int = H, step: int = STEP, n_folds: int = N_FOLDS) -> list:
    """Позиції origin: модель вчиться на series[:o] і прогнозує series[o:o+h]. Останній фолд закінчується в кінці ряду."""
    first = n_weeks - h - step * (n_folds - 1)
    return list(range(first, n_weeks - h + 1, step))


def week_span(week_end: pd.Timestamp) -> str:
    """Мітка тижня — неділя (кінець тижня); показуємо повний діапазон Пн–Нд."""
    start = week_end - pd.Timedelta(days=6)
    return f"{start:%d.%m.%Y}–{week_end:%d.%m.%Y}"


def calc_metrics(actual, pred, mase_scale) -> dict:
    error = actual - pred
    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "WAPE, %": np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100,
        "MASE": np.mean(np.abs(error)) / mase_scale,
        "Bias, %": np.sum(error) / np.sum(actual) * 100,   # > 0 — модель занижує прогноз
    }


def rolling_forecasts(forecast_fn, train_series: pd.Series, test_series: pd.Series, origins: list, h: int = H):
    """Прогнози по всіх фолдах підряд (фолд 0, фолд 1, ...).
    Повертає (факт, прогноз, середній MAE наївного прогнозу на train — масштаб для MASE)."""
    all_actual, all_pred, scales = [], [], []
    for o in origins:
        train = train_series.iloc[:o]
        all_actual.append(test_series.iloc[o:o + h].values)
        all_pred.append(forecast_fn(train))
        scales.append(np.mean(np.abs(np.diff(train.values))))
    return np.concatenate(all_actual), np.concatenate(all_pred), np.mean(scales)


def backtest(forecast_fn, train_series: pd.Series, test_series: pd.Series, origins: list, h: int = H):
    """train_series — на чому вчимося (quantity_clean), test_series — з чим порівнюємо (сира quantity).
    Повертає (метрики по всіх фолдах, прогнози підряд)."""
    actual, pred, mase_scale = rolling_forecasts(forecast_fn, train_series, test_series, origins, h)
    return calc_metrics(actual, pred, mase_scale), pred


def backtest_panel(forecast_fn, panel: pd.DataFrame, origins: list, h: int = H):
    """Backtest окремо для кожного товару (колонки panel), метрики — по всіх товарах разом."""
    actual_all, pred_all, preds = [], [], {}
    for sku in panel.columns:
        actual, pred, _ = rolling_forecasts(forecast_fn, panel[sku], panel[sku], origins, h)
        actual_all.append(actual)
        pred_all.append(pred)
        preds[sku] = pred
    error = np.concatenate(actual_all) - np.concatenate(pred_all)
    total_actual = np.concatenate(actual_all).sum()
    metrics = {"WAPE, %": np.abs(error).sum() / total_actual * 100,
               "Bias, %": error.sum() / total_actual * 100}
    return metrics, preds


def features_leak_free(series: pd.Series, features: pd.DataFrame, origins: list, h: int = H) -> bool:
    """Перераховуємо ознаки тестових тижнів ТІЛЬКИ з train і порівнюємо з тими, що лежать у gold."""
    for o in origins:
        train_only = series.iloc[:o]
        future = pd.Series(np.nan, index=series.index[o:o + h])
        recomputed = make_features(pd.concat([train_only, future]), h).iloc[-h:]
        if not np.allclose(recomputed.values, features.iloc[o:o + h].values):
            return False
    return True
