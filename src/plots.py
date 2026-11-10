"""Графіки для ноутбука — щоб клітинки ноутбука були короткими."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

MAIN_COLOR = "#2C5F2D"
SECOND_COLOR = "#B85042"
# для порівняння політик: пара, що розрізняється і при дальтонізмі (перевірено валідатором палітри)
POLICY_COLORS = {"замовити = прогноз": "#b5b4ad", "ЛП": "#2a78d6", "запас під 95% сервісу": "#eb6834"}
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]


def setup_style():
    plt.rcParams["figure.figsize"] = (12, 4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def daily_sales(daily: pd.Series):
    by_weekday = daily.groupby(daily.index.dayofweek).mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].plot(daily.index, daily.values, color=MAIN_COLOR, lw=0.7)
    axes[0].set_title("Продажі по днях, од.")
    axes[1].bar(WEEKDAYS, by_weekday.values, color=MAIN_COLOR)
    axes[1].set_title("Середні продажі за день тижня")
    plt.tight_layout()
    plt.show()


def revenue_and_countries(monthly_revenue: pd.Series, country_share: pd.Series, top: int = 6):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].bar(monthly_revenue.index, monthly_revenue.values / 1e6, width=20, color=MAIN_COLOR)
    axes[0].set_title("Виручка по місяцях, £ млн")
    top_countries = country_share.head(top)[::-1]
    axes[1].barh(top_countries.index, top_countries.values, color=MAIN_COLOR)
    axes[1].set_title("Частка виручки за країнами, %")
    plt.tight_layout()
    plt.show()


def weekly_anomalies(gold_weekly: pd.DataFrame):
    flagged = gold_weekly[gold_weekly["is_spike"] | gold_weekly["is_closed_week"]]
    plt.plot(gold_weekly.index, gold_weekly["quantity"], color="lightgray", label="quantity (сирий)")
    plt.plot(gold_weekly.index, gold_weekly["quantity_clean"], color=MAIN_COLOR, label="quantity_clean")
    plt.scatter(flagged.index, flagged["quantity"], color=SECOND_COLOR, zorder=3, label="аномалії |z| > 3.5")
    plt.title("Тижневі продажі (одиниць)")
    plt.legend()
    plt.show()


def seasonality_and_acf(series: pd.Series, lags: int = 60):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # накладаємо роки по номеру тижня; ISO-рік, бо тиждень з 1 січня може належати до минулого року
    iso = series.index.isocalendar()
    for year, color in [(2010, "lightgray"), (2011, MAIN_COLOR)]:
        in_year = (iso.year == year).values
        axes[0].plot(iso.week[in_year], series[in_year], color=color, label=str(year), lw=2)
    axes[0].set_title("Сезонність: продажі за тижнем року")
    axes[0].set_xlabel("тиждень року")
    axes[0].legend()
    plot_acf(series, lags=lags, ax=axes[1], color=MAIN_COLOR)
    axes[1].set_title("ACF тижневого ряду")
    plt.tight_layout()
    plt.show()


def pareto(total_quantity: pd.Series):
    cum_share = total_quantity.sort_values(ascending=False).cumsum() / total_quantity.sum()
    x = np.arange(1, len(cum_share) + 1) / len(cum_share) * 100
    plt.figure(figsize=(7, 3.5))
    plt.plot(x, cum_share.values * 100, color=MAIN_COLOR)
    plt.axhline(80, color=SECOND_COLOR, ls="--", lw=1)
    plt.xlabel("% асортименту")
    plt.ylabel("% обсягу продажів")
    plt.title("Крива Парето (ABC)")
    plt.show()


def backtest_folds(origins: list, h: int, start: int = 60):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, o in enumerate(origins):
        ax.barh(i, o - start, left=start, color="lightgray", height=0.6)   # train показуємо з тижня start
        ax.barh(i, h, left=o, color=MAIN_COLOR, height=0.6)
    ax.set_yticks(range(len(origins)))
    ax.set_yticklabels([f"фолд {i + 1}" for i in range(len(origins))], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"номер тижня (сірий — train, зелений — test на {h} тижні)")
    ax.set_title("Rolling-origin backtesting, expanding window")
    plt.show()


def backtest_forecasts(actual: pd.Series, predictions: dict, origins: list, h: int, start: int = 60):
    """Прогнози на непересічних фолдах (кожен другий), щоб кожен тиждень було видно один раз."""
    plt.plot(actual.index[start:], actual.iloc[start:], color="black", lw=1.5, label="факт")
    styles = [("Seasonal Naive", "lightgray"), ("ETS (Holt, damped)", SECOND_COLOR), ("Ridge + Fourier", MAIN_COLOR)]
    for name, color in styles:
        preds = predictions[name]
        for j in range(0, len(origins), 2):
            o = origins[j]
            plt.plot(actual.index[o:o + h], preds[j * h:(j + 1) * h], color=color, lw=2,
                     label=name if j == 0 else None)
    plt.axvline(actual.index[origins[0]], color="gray", ls=":")
    plt.title(f"Прогнози на {h} тижні вперед у backtest (загальний обсяг)")
    plt.legend()
    plt.show()


def policy_costs_by_fold(policy_costs: pd.DataFrame):
    """Фактичні витрати трьох політик на кожному фолді (згруповані стовпці)."""
    policies = list(POLICY_COLORS)
    x = np.arange(len(policy_costs))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, name in enumerate(policies):
        ax.bar(x + (j - 1) * width, policy_costs[name] / 1e3, width=width, color=POLICY_COLORS[name],
               edgecolor="white", linewidth=1.5, label=name)
    labels = [f"фолд {k}\n{d:%d.%m}" for k, d in zip(policy_costs.index, policy_costs["початок"])]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("фактичні витрати, £ тис.")
    ax.set_title("Фактичні витрати на кожному фолді (топ-30 товарів, план на 4 тижні)")
    ax.legend(frameon=False)
    plt.show()


def lp_sensitivity(sensitivity: pd.DataFrame, value_col: str, label_col: str):
    """Сумарний виграш ЛП проти «замовити = прогноз» залежно від критичного відношення p/(p+h)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ratios = sensitivity.index.values
    values = sensitivity[value_col].values
    ax.axhline(0, color="#8a8980", lw=1)
    ax.text(ratios.max(), 1.2, "0 = витрати «замовити = прогноз»", fontsize=8, color="#5e5d58", ha="right")
    ax.plot(ratios, values, color=POLICY_COLORS["ЛП"], lw=2, marker="o", markersize=8)
    for i, (r, v, lab) in enumerate(zip(ratios, values, sensitivity[label_col])):
        text = f"{v:+.0f}%  ({lab} фолдів)".replace("-", "−")
        if i == len(values) - 1:                 # остання точка — підпис знизу
            offset, ha = (0, -22), "center"
        elif i > 0 and values[i - 1] > v:        # лінія приходить згори — підпис праворуч
            offset, ha = (10, 6), "left"
        else:
            offset, ha = (0, 12), "center"
        ax.annotate(text, (r, v), textcoords="offset points", xytext=offset, ha=ha, fontsize=8, color="#3d3c38")
    ax.set_xlabel("p / (p + h) — наскільки дефіцит дорожчий за залишок")
    ax.set_ylabel("ЛП vs прогноз, %")
    ax.set_title("Чутливість: сумарний виграш ЛП за всі фолди")
    ax.set_ylim(min(values.min(), 0) - 12, max(values.max(), 0) + 12)
    plt.show()
