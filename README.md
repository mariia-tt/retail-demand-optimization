# Прогнозування попиту для оптимізації складу

Дані — [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii): транзакції британського онлайн-гуртовика.

Логіка: прогноз тижневого попиту - оцінка невизначеності через backtest - стохастична оптимізація замовлення - симуляція сценаріїв складу.

## Структура

```
├── data/
│   ├── raw/online_retail_II.xlsx      
│   ├── bronze/                        
│   ├── silver/                      
│   └── gold/                          
├── src/
│   ├── config.py        # шляхи, горизонт, фолди, припущення про витрати — усі параметри тут
│   ├── pipeline.py      # Bronze -> Silver -> Gold
│   ├── checks.py        # перевірки якості даних 
│   ├── features.py      # ознаки для моделей 
│   ├── validation.py    # rolling-origin backtest, метрики, перевірка leakage
│   ├── models.py        # наївні бенчмарки, ETS, Ridge + Fourier
│   ├── optimization.py  # сценарії попиту, ЛП поповнення складу 
│   └── plots.py         # графіки для ноутбука
├── notebooks/
   └── project_progress.ipynb   
```

Правило шарів: ML-код (`validation`, `models`, `optimization`) читає тільки gold.

## Запуск

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline            
python -m src.pipeline --rebuild  
```


