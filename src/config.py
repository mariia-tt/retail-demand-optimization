"""Шляхи до шарів даних і всі параметри проєкту в одному місці."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# --- шари даних
RAW_XLSX = DATA_DIR / "raw" / "online_retail_II.xlsx"

BRONZE_PATH = DATA_DIR / "bronze" / "transactions.parquet"

SILVER_PATH = DATA_DIR / "silver" / "transactions.parquet"
CLEANING_LOG_PATH = DATA_DIR / "silver" / "cleaning_log.csv"

GOLD_DIR = DATA_DIR / "gold"
GOLD_WEEKLY_PATH = GOLD_DIR / "weekly_total.parquet"      # загальний тижневий ряд + ознаки
GOLD_SKU_PATH = GOLD_DIR / "weekly_sku.parquet"           # матриця тиждень x товар
GOLD_PRODUCT_PATH = GOLD_DIR / "dim_product.parquet"      # довідник товарів
GOLD_PRICES_PATH = GOLD_DIR / "sku_weekly_prices.parquet" # ціни по тижнях (для оптимізації as-of)

# --- часовий ряд
WEEK_FREQ = "W-SUN"   # тижні Пн-Нд, мітка тижня = неділя (кінець тижня)
SEASON = 52

# --- очищення та аномалії
BIG_CANCEL_QTY = 1000      # скасування від цієї кількості шукаємо в парі з замовленням
ANOMALY_WINDOW = 9         # ковзна медіана, тижнів
ANOMALY_Z = 3.5            # поріг робастного z-score
SPIKE_FREE_FROM = "2011-01-01"  # з цієї дати піки не обрізаємо (тестовий період)

# --- валідація
H = 4          # горизонт прогнозу, тижнів
STEP = 2       # крок зсуву origin, тижнів
N_FOLDS = 13
RANDOM_SEED = 42

# --- оптимізація (припущення: в даних цього немає)
TOP_N_SKUS = 30
COST_SHARE = 0.60       # закупівельна ціна = 60% ціни продажу
SHORTAGE_SHARE = 0.40   # штраф за дефіцит = втрачена маржа = 40% ціни
HOLDING_SHARE = 0.15    # витрати на залишок = 15% закупівельної ціни за період
N_SCENARIOS = 200
MIN_FORECAST = 100      # відносні помилки беремо лише для прогнозів >= 100 од. за 4 тижні
CAPACITY_SHARE = 0.85   # склад вміщує на ~15% менше, ніж хоче план без обмежень
SERVICE_LEVEL = 0.95    # евристика для порівняння: запас під 95-й перцентиль сценаріїв
MIN_SCENARIO_ERRORS = 50  # фолд оцінюємо, лише якщо з попередніх фолдів є >= 50 помилок для сценаріїв
# чутливість: частка закупівельної ціни, втрачена на непроданій одиниці (0.15 — база, 1.0 — товар пропадає)
HOLDING_SHARE_GRID = [0.15, 1 / 3, 2 / 3, 1.0]
