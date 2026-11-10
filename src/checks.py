"""Перевірки якості даних: якщо умова не виконується — пайплайн/ноутбук зупиняється."""
import numpy as np
import pandas as pd

from .config import H, SPIKE_FREE_FROM
from .features import FEATURES

ORIGINAL_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "Price", "Customer ID", "Country"]
PRODUCT_CODE = r"^\d{5}"   # справжні товари мають код з 5 цифр


def check(condition, message):
    if not condition:
        raise AssertionError("ПОМИЛКА: " + message)
    print("✓", message)


def validate_bronze(bronze: pd.DataFrame):
    check(list(bronze.columns) == ORIGINAL_COLUMNS + ["source_sheet"], "схема bronze: 8 колонок джерела + source_sheet")
    check(pd.api.types.is_datetime64_any_dtype(bronze["InvoiceDate"]), "InvoiceDate має тип дата")
    check(pd.api.types.is_integer_dtype(bronze["Quantity"]), "Quantity — ціле число")
    check(bronze["Invoice"].notna().all() and bronze["StockCode"].notna().all(), "Invoice і StockCode без пропусків")


def validate_silver(silver: pd.DataFrame):
    check(silver["Quantity"].gt(0).all(), "усі Quantity > 0")
    check(silver["Price"].gt(0).all(), "усі Price > 0")
    check(not silver["Invoice"].str.startswith("C").any(), "немає скасувань")
    check(silver["StockCode"].str.match(PRODUCT_CODE).all(), "тільки товарні коди")
    check(not silver["StockCode"].str.contains("[a-z]").any(), "StockCode у верхньому регістрі")
    check(not silver.duplicated().any(), "немає дублікатів")
    check("Customer ID" not in silver.columns, "персональних даних немає")


def validate_gold(gold_weekly: pd.DataFrame, gold_sku: pd.DataFrame, dim_product: pd.DataFrame):
    week_steps = gold_weekly.index.to_series().diff().dropna()
    check(week_steps.eq(pd.Timedelta(days=7)).all(), "тижні йдуть підряд, без пропусків")
    check(gold_weekly["quantity"].notna().all(), "немає пропусків у цільовій змінній")
    check(gold_weekly.index.equals(gold_sku.index), "gold_weekly і gold_sku мають однакові тижні")
    check(np.isclose(gold_sku.values.sum(), gold_weekly["quantity"].sum()), "сума по товарах = загальна сума (gold узгоджений)")
    check(gold_sku.columns.isin(dim_product.index).all(), "кожен товар з gold_sku є в довіднику")
    check(gold_weekly.loc[gold_weekly.index >= SPIKE_FREE_FROM, "is_spike"].sum() == 0,
          "обрізання піків не зачіпає 2011 рік (тестовий період)")
    check(gold_weekly[FEATURES].iloc[2 * H:].notna().all().all(), "ознаки заповнені (крім перших тижнів, де ще немає лагу)")
