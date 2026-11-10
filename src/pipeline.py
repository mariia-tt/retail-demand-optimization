"""Bronze -> Silver -> Gold.

Запуск з кореня проєкту:  python -m src.pipeline            (bronze береться з кешу)
                          python -m src.pipeline --rebuild  (перечитати Excel, ~40 с)
"""
import argparse

import numpy as np
import pandas as pd

from . import config as cfg
from .checks import ORIGINAL_COLUMNS, PRODUCT_CODE, validate_bronze, validate_gold, validate_silver
from .features import FEATURES, make_features


# =============================================================== Bronze
def build_bronze(rebuild: bool = False) -> pd.DataFrame:
    """Обидва аркуші Excel без змін + колонка source_sheet. Кешується в parquet."""
    if cfg.BRONZE_PATH.exists() and not rebuild:
        return pd.read_parquet(cfg.BRONZE_PATH)

    # dtype=str, бо в Invoice/StockCode є букви (C489449, 85123A)
    sheets = pd.read_excel(cfg.RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str})
    bronze = pd.concat([sheet.assign(source_sheet=name) for name, sheet in sheets.items()],
                       ignore_index=True)
    cfg.BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bronze.to_parquet(cfg.BRONZE_PATH)
    return bronze


def bronze_profile(bronze: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "тип": bronze.dtypes.astype(str),
        "пропусків": bronze.isna().sum(),
        "% пропусків": bronze.isna().mean() * 100,
        "унікальних": bronze.nunique(),
    })


def bronze_issues(bronze: pd.DataFrame) -> pd.DataFrame:
    """Проблеми якості, які вирішуються в silver."""
    stock_code = bronze["StockCode"]
    issues = pd.Series({
        "дублікати рядків (без source_sheet)": bronze.duplicated(subset=ORIGINAL_COLUMNS).sum(),
        "скасування (Invoice на C)": bronze["Invoice"].str.startswith("C").sum(),
        "Quantity <= 0": (bronze["Quantity"] <= 0).sum(),
        "Price <= 0": (bronze["Price"] <= 0).sum(),
        "службові коди (POST, DOT, M, AMAZONFEE...)": (~stock_code.str.match(PRODUCT_CODE)).sum(),
        "StockCode з малими літерами (85123a vs 85123A)": stock_code.str.contains("[a-z]").sum(),
        "пропущений Description": bronze["Description"].isna().sum(),
        "пропущений Customer ID": bronze["Customer ID"].isna().sum(),
        "Country = Unspecified": (bronze["Country"] == "Unspecified").sum(),
    })
    return pd.DataFrame({"рядків": issues, "% від усіх": issues / len(bronze) * 100})


# =============================================================== Silver
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()
    return df


def find_fake_orders(df: pd.DataFrame, min_qty: int = cfg.BIG_CANCEL_QTY):
    """Пари «велике замовлення <-> його скасування»: той самий клієнт, товар і кількість,
    замовлення не пізніше за скасування. Повертає (індекси замовлень, кількість великих скасувань).
    Customer ID використовується лише тут і далі в silver не потрапляє."""
    cancel_mask = df["Invoice"].str.startswith("C")
    big_cancels = df[cancel_mask & (df["Quantity"] <= -min_qty)].copy()
    big_cancels["Quantity"] = -big_cancels["Quantity"]

    orders = df[~cancel_mask & (df["Quantity"] > 0)].reset_index()
    pairs = orders.merge(big_cancels[["Customer ID", "StockCode", "Quantity", "InvoiceDate"]],
                         on=["Customer ID", "StockCode", "Quantity"], suffixes=("", "_cancel"))
    pairs = pairs[pairs["InvoiceDate"] <= pairs["InvoiceDate_cancel"]]
    return pairs["index"].unique(), len(big_cancels)


def build_silver(bronze: pd.DataFrame):
    """Очищені транзакції + журнал очищення (скільки рядків прибрав кожен крок)."""
    silver = normalize_text(bronze)
    fake_orders, _ = find_fake_orders(silver)

    log = []

    def apply_step(df, keep_mask, step_name):
        log.append({"крок": step_name, "видалено рядків": int((~keep_mask).sum())})
        return df[keep_mask]

    silver = apply_step(silver, ~silver.index.isin(fake_orders), "помилкові мега-замовлення")
    silver = apply_step(silver, ~silver.duplicated(subset=ORIGINAL_COLUMNS), "дублікати (перекриття аркушів)")
    silver = apply_step(silver, ~silver["Invoice"].str.startswith("C"), "скасування (Invoice на C)")
    silver = apply_step(silver, silver["StockCode"].str.match(PRODUCT_CODE), "службові коди")
    silver = apply_step(silver, silver["Quantity"] > 0, "Quantity <= 0 (складські коригування)")
    silver = apply_step(silver, silver["Price"] > 0, "Price <= 0 (безкоштовні позиції)")

    # нова колонка + прибираємо те, що далі не потрібно (зокрема персональні дані)
    silver["Revenue"] = silver["Quantity"] * silver["Price"]
    silver = silver[["InvoiceDate", "Invoice", "StockCode", "Description", "Quantity", "Price", "Revenue", "Country"]]
    silver = silver.sort_values("InvoiceDate").reset_index(drop=True)

    log = pd.DataFrame(log)
    log["% від bronze"] = log["видалено рядків"] / len(bronze) * 100
    log.loc[len(log)] = ["РАЗОМ залишилось у silver", len(silver), len(silver) / len(bronze) * 100]
    return silver, log


# =============================================================== Gold
def detect_weekly_anomalies(quantity: pd.Series) -> pd.DataFrame:
    """Робастний z-score: відхилення від ковзної медіани, поділене на MAD.
    z < -поріг — тижні, коли склад закритий (лише прапорець);
    z > +поріг — разові гуртові піки, обрізаються до медіана + поріг·MAD у quantity_clean."""
    rolling_median = quantity.rolling(cfg.ANOMALY_WINDOW, center=True, min_periods=5).median()
    deviation = quantity - rolling_median
    mad = deviation.abs().median() * 1.4826
    robust_z = deviation / mad

    out = pd.DataFrame(index=quantity.index)
    out["is_spike"] = robust_z > cfg.ANOMALY_Z
    out["is_closed_week"] = robust_z < -cfg.ANOMALY_Z
    out["quantity_clean"] = np.where(out["is_spike"], rolling_median + cfg.ANOMALY_Z * mad, quantity)
    return out


def build_gold(silver: pd.DataFrame) -> dict:
    # --- загальний тижневий ряд
    weekly_all = silver.set_index("InvoiceDate").resample(cfg.WEEK_FREQ).agg(
        {"Quantity": "sum", "Revenue": "sum", "Invoice": "nunique", "StockCode": "nunique"})
    weekly_all.columns = ["quantity", "revenue", "invoices", "active_skus"]

    gold_weekly = weekly_all.iloc[1:-1].copy()     # перший і останній тижні неповні
    gold_weekly.index.name = "week"
    gold_weekly["quantity"] = gold_weekly["quantity"].astype(float)
    gold_weekly = gold_weekly.join(detect_weekly_anomalies(gold_weekly["quantity"]))
    gold_weekly[FEATURES] = make_features(gold_weekly["quantity_clean"], cfg.H)

    # --- матриця тиждень x товар (тижні без продажів = 0)
    gold_sku = (silver.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq=cfg.WEEK_FREQ)])["Quantity"].sum()
                      .unstack("StockCode")
                      .reindex(gold_weekly.index)
                      .fillna(0))

    # --- довідник товарів: найчастіший опис, медіанна ціна, обсяг, ABC-клас
    dim_product = silver.groupby("StockCode").agg(
        description=("Description", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        median_price=("Price", "median"),
        total_quantity=("Quantity", "sum"),
    ).sort_values("total_quantity", ascending=False)
    cum_share = dim_product["total_quantity"].cumsum() / dim_product["total_quantity"].sum()
    dim_product["abc_class"] = np.select([cum_share <= 0.8, cum_share <= 0.95], ["A", "B"], default="C")
    dim_product["zero_weeks_share"] = (gold_sku == 0).mean().reindex(dim_product.index)

    # --- ціни по тижнях: скільки рядків продано за кожною ціною.
    # Дає точну медіану ціни «станом на дату» без звернення до silver.
    sku_prices = (silver.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq=cfg.WEEK_FREQ), "Price"])
                        .size().rename("n_lines").reset_index()
                        .rename(columns={"InvoiceDate": "week"}))

    return {"weekly": gold_weekly, "sku": gold_sku, "product": dim_product, "prices": sku_prices}


# =============================================================== читання шарів
def _with_week_freq(df: pd.DataFrame) -> pd.DataFrame:
    # parquet не зберігає частоту індексу; без неї statsmodels сипле попередженнями.
    # Якщо тижні не підряд — тут буде помилка.
    df.index = pd.DatetimeIndex(df.index, freq=cfg.WEEK_FREQ, name=df.index.name)
    return df


def load_silver() -> pd.DataFrame:
    return pd.read_parquet(cfg.SILVER_PATH)


def load_cleaning_log() -> pd.DataFrame:
    return pd.read_csv(cfg.CLEANING_LOG_PATH)


def load_gold() -> dict:
    return {
        "weekly": _with_week_freq(pd.read_parquet(cfg.GOLD_WEEKLY_PATH)),
        "sku": _with_week_freq(pd.read_parquet(cfg.GOLD_SKU_PATH)),
        "product": pd.read_parquet(cfg.GOLD_PRODUCT_PATH),
        "prices": pd.read_parquet(cfg.GOLD_PRICES_PATH),
    }


def gold_exists() -> bool:
    paths = [cfg.SILVER_PATH, cfg.CLEANING_LOG_PATH, cfg.GOLD_WEEKLY_PATH,
             cfg.GOLD_SKU_PATH, cfg.GOLD_PRODUCT_PATH, cfg.GOLD_PRICES_PATH]
    return all(p.exists() for p in paths)


# =============================================================== запуск
def run(rebuild_bronze: bool = False):
    print("— Bronze")
    bronze = build_bronze(rebuild_bronze)
    validate_bronze(bronze)

    print("— Silver")
    silver, log = build_silver(bronze)
    validate_silver(silver)
    cfg.SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(cfg.SILVER_PATH)
    log.to_csv(cfg.CLEANING_LOG_PATH, index=False)

    print("— Gold")
    gold = build_gold(silver)
    validate_gold(gold["weekly"], gold["sku"], gold["product"])
    cfg.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold["weekly"].to_parquet(cfg.GOLD_WEEKLY_PATH)
    gold["sku"].to_parquet(cfg.GOLD_SKU_PATH)
    gold["product"].to_parquet(cfg.GOLD_PRODUCT_PATH)
    gold["prices"].to_parquet(cfg.GOLD_PRICES_PATH)

    print(f"\nbronze {bronze.shape} -> silver {silver.shape} -> "
          f"gold weekly {gold['weekly'].shape}, sku {gold['sku'].shape}, product {gold['product'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bronze -> Silver -> Gold")
    parser.add_argument("--rebuild", action="store_true", help="перечитати Excel замість кешу bronze")
    run(rebuild_bronze=parser.parse_args().rebuild)
