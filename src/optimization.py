"""Стохастична задача поповнення складу (sample average approximation, ЛП у PuLP)."""
import numpy as np
import pandas as pd
import pulp

from . import config as cfg


def relative_errors(sku_preds: dict, panel: pd.DataFrame, origins: list, plan_origin: int,
                    h: int = cfg.H, min_forecast: float = cfg.MIN_FORECAST) -> np.ndarray:
    """Відносні помилки 4-тижневого прогнозу (факт - прогноз) / прогноз з попередніх фолдів.

    Беремо лише фолди, чий тестовий період повністю закінчився до plan_origin (інакше leakage).
    Майже нульові прогнози відкидаємо: 7 шт. прогноз і 600 факт дають +8600% — безглуздо."""
    errors = []
    for j, o in enumerate(origins):
        if o + h > plan_origin:
            continue
        for sku in panel.columns:
            f = sku_preds[sku][j * h:(j + 1) * h].sum()
            a = panel[sku].iloc[o:o + h].sum()
            if f >= min_forecast:
                errors.append((a - f) / f)
    return np.array(errors)


def price_asof(sku_prices: pd.DataFrame, skus, before_week: pd.Timestamp) -> pd.Series:
    """Медіанна ціна кожного товару за всі рядки продажів у тижнях до before_week (без майбутніх цін)."""
    past = sku_prices[(sku_prices["week"] < before_week) & sku_prices["StockCode"].isin(skus)]
    return past.groupby("StockCode").apply(
        lambda g: np.median(np.repeat(g["Price"].values, g["n_lines"].values)), include_groups=False)


def cost_params(price: pd.Series, holding_share: float = cfg.HOLDING_SHARE) -> pd.DataFrame:
    """holding_share — частка закупівельної ціни, яку втрачаємо на кожній непроданій одиниці:
    0.15 — залишок переходить у наступний період (зберігання, заморожені гроші);
    1.0  — непроданий товар пропадає повністю (класичний newsvendor без переносу)."""
    cost = cfg.COST_SHARE * price
    return pd.DataFrame({
        "cost": cost,
        "shortage": cfg.SHORTAGE_SHARE * price,   # втрачена маржа
        "holding": holding_share * cost,
    })


def critical_ratio(holding_share: float) -> float:
    """p / (p + h) — оптимальний квантиль попиту в newsvendor (однаковий для всіх товарів)."""
    p, h = cfg.SHORTAGE_SHARE, holding_share * cfg.COST_SHARE
    return p / (p + h)


def make_scenarios(forecast: pd.Series, rel_errors: np.ndarray,
                   n: int = cfg.N_SCENARIOS, seed: int = cfg.RANDOM_SEED) -> dict:
    """Попит у сценарії = прогноз × (1 + випадкова помилка з backtest), не менше 0."""
    rng = np.random.default_rng(seed)
    return {sku: np.maximum(forecast[sku] * (1 + rng.choice(rel_errors, size=n)), 0)
            for sku in forecast.index}


def solve_replenishment(scenarios: dict, shortage_cost: pd.Series, holding_cost: pd.Series,
                        capacity: float = None):
    """min (1/S) Σ_s Σ_i (p_i·u_is + h_i·o_is),  u_is >= D_is - q_i,  o_is >= q_i - D_is,  Σ q_i <= V.

    Повертає (план q, очікувані витрати, тіньова ціна місткості або None)."""
    skus = list(scenarios)
    n_scen = len(next(iter(scenarios.values())))

    model = pulp.LpProblem("replenishment", pulp.LpMinimize)
    q = {i: pulp.LpVariable(f"q_{i}", lowBound=0) for i in skus}
    u = {(i, s): pulp.LpVariable(f"short_{i}_{s}", lowBound=0) for i in skus for s in range(n_scen)}
    ov = {(i, s): pulp.LpVariable(f"over_{i}_{s}", lowBound=0) for i in skus for s in range(n_scen)}

    model += pulp.lpSum((shortage_cost[i] * u[i, s] + holding_cost[i] * ov[i, s]) / n_scen
                        for i in skus for s in range(n_scen))
    for i in skus:
        for s in range(n_scen):
            model += u[i, s] >= scenarios[i][s] - q[i]
            model += ov[i, s] >= q[i] - scenarios[i][s]
    if capacity is not None:
        model += pulp.lpSum(q.values()) <= capacity, "capacity"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    plan = pd.Series({i: q[i].value() for i in skus})
    shadow_price = -model.constraints["capacity"].pi if capacity is not None else None
    return plan, pulp.value(model.objective), shadow_price


def realized_cost(plan: pd.Series, actual: pd.Series, shortage_cost: pd.Series, holding_cost: pd.Series) -> float:
    """Фактичні витрати плану, коли попит уже відомий."""
    shortage = np.maximum(actual - plan, 0)
    leftover = np.maximum(plan - actual, 0)
    return (shortage_cost * shortage + holding_cost * leftover).sum()


# =============================================================== перевірка по всіх фолдах
FORECAST = "замовити = прогноз"
SERVICE = "запас під 95% сервісу"
LP = "ЛП"
LP_CAP = "ЛП з місткістю"


def fold_inputs(k: int, sku_preds: dict, panel: pd.DataFrame, origins: list, h: int = cfg.H):
    """Для фолду k: сумарний прогноз на h тижнів, факт і відносні помилки попередніх фолдів."""
    o = origins[k]
    forecast = pd.Series({sku: sku_preds[sku][k * h:(k + 1) * h].sum() for sku in panel.columns})
    actual = panel.iloc[o:o + h].sum()
    errors = relative_errors(sku_preds, panel, origins, plan_origin=o, h=h)
    return forecast, actual, errors


def compare_policies(forecast: pd.Series, actual: pd.Series, scenarios: dict, costs: pd.DataFrame,
                     with_capacity: bool = True) -> dict:
    """Фактичні витрати кожної політики замовлення на одному фолді.

    - замовити = прогноз;
    - запас під 95% сервісу: 95-й перцентиль сценаріїв — типова евристика, яка не дивиться на витрати;
    - ЛП: оптимум при заданих p і h (без обмеження = newsvendor для кожного товару);
    - ЛП з місткістю: склад вміщує CAPACITY_SHARE від плану без обмежень."""
    plans = {
        FORECAST: forecast,
        SERVICE: pd.Series({sku: np.quantile(scenarios[sku], cfg.SERVICE_LEVEL) for sku in forecast.index}),
    }
    plans[LP], _, _ = solve_replenishment(scenarios, costs["shortage"], costs["holding"])
    if with_capacity:
        capacity = round(cfg.CAPACITY_SHARE * plans[LP].sum(), -3)
        plans[LP_CAP], _, _ = solve_replenishment(scenarios, costs["shortage"], costs["holding"], capacity)
    return {name: realized_cost(plan, actual, costs["shortage"], costs["holding"]) for name, plan in plans.items()}


def backtest_policies(sku_preds: dict, panel: pd.DataFrame, origins: list, sku_prices: pd.DataFrame,
                      weeks: pd.DatetimeIndex, holding_share: float = cfg.HOLDING_SHARE,
                      with_capacity: bool = True, min_errors: int = cfg.MIN_SCENARIO_ERRORS,
                      h: int = cfg.H) -> pd.DataFrame:
    """Rolling-origin перевірка оптимізації: на кожному фолді плануємо h тижнів і рахуємо фактичні витрати.
    Фолди, де з попередніх фолдів набирається менше min_errors помилок, пропускаємо — сценаріям нема з чого будуватись."""
    rows = []
    for k, o in enumerate(origins):
        forecast, actual, errors = fold_inputs(k, sku_preds, panel, origins, h)
        if len(errors) < min_errors:
            continue
        price = price_asof(sku_prices, panel.columns, before_week=weeks[o])
        costs = cost_params(price, holding_share)
        scenarios = make_scenarios(forecast, errors)
        row = {"фолд": k + 1, "початок": weeks[o] - pd.Timedelta(days=6),
               "прогноз, од.": forecast.sum(), "факт, од.": actual.sum(), "помилок": len(errors)}
        row.update(compare_policies(forecast, actual, scenarios, costs, with_capacity))
        rows.append(row)
    return pd.DataFrame(rows).set_index("фолд")


def savings_summary(policy_costs: pd.DataFrame, baseline: str = FORECAST) -> pd.DataFrame:
    """Сумарні фактичні витрати по фолдах і виграш відносно baseline."""
    policies = [c for c in policy_costs.columns if c in (FORECAST, SERVICE, LP, LP_CAP)]
    costs = policy_costs[policies]
    return pd.DataFrame({
        "витрати за всі фолди, £": costs.sum(),
        f"vs «{baseline}», %": (costs.sum() / costs[baseline].sum() - 1) * 100,
        "медіана по фолдах, %": ((costs.div(costs[baseline], axis=0) - 1) * 100).median(),
        "фолдів, де краще за baseline": (costs.lt(costs[baseline], axis=0)).sum(),
    })
