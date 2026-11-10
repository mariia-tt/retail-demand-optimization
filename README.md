# Прогнозування попиту для оптимізації складу

Проєкт з курсу «Прогнозування та дослідження операцій». Дані — [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii): транзакції британського онлайн-гуртовика, 12.2009 – 12.2011.

Логіка: прогноз тижневого попиту → оцінка невизначеності через backtest → стохастична оптимізація замовлення → (далі) симуляція сценаріїв складу.

## Структура

```
├── data/
│   ├── raw/online_retail_II.xlsx      # джерело (не змінюється)
│   ├── bronze/                        # Excel як є + мітка аркуша          ┐
│   ├── silver/                        # очищені транзакції + журнал очищення │ генеруються
│   └── gold/                          # вітрини для моделей                 ┘ python -m src.pipeline
├── src/
│   ├── config.py        # шляхи, горизонт, фолди, припущення про витрати — усі параметри тут
│   ├── pipeline.py      # Bronze -> Silver -> Gold
│   ├── checks.py        # перевірки якості даних (19 штук)
│   ├── features.py      # ознаки для моделей (лаг >= H + Фур'є)
│   ├── validation.py    # rolling-origin backtest, метрики, перевірка leakage
│   ├── models.py        # наївні бенчмарки, ETS, Ridge + Fourier
│   ├── optimization.py  # сценарії попиту, ЛП поповнення складу (PuLP)
│   └── plots.py         # графіки для ноутбука
├── notebooks/
│   └── project_progress.ipynb   # Project Progress: 4 блоки, читає тільки готові шари
└── reports/
    ├── presentation.pptx
    └── speech.md                # текст до відеоогляду
```

Правило шарів: ML-код (`validation`, `models`, `optimization`) читає **тільки gold**.

## Запуск

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline             # будує bronze/silver/gold (bronze кешується)
python -m src.pipeline --rebuild   # перечитати Excel заново (~40 с)
```

Потім відкрити `notebooks/project_progress.ipynb`. Якщо шари ще не побудовані, ноутбук сам запустить пайплайн.

## Як додавати наступні етапи

- нова модель → функція `train -> np.array(h)` у `src/models.py`, оцінка тим самим `validation.backtest`;
- нова ознака → `src/features.py` + колонка в gold (`pipeline.build_gold`);
- новий параметр або припущення → `src/config.py`.
