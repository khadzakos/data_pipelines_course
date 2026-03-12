import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Пути к таблицам
DELTA_PATH = os.path.join(DATA_DIR, "delta_table")
ICEBERG_PATH = os.path.join(DATA_DIR, "iceberg_warehouse")
PARQUET_PATH = os.path.join(DATA_DIR, "parquet_baseline")

# Параметры генерации данных
NUM_ROWS = 1_000_000
NUM_PARTITIONS = 10

# Объемы для тестов записи
WRITE_FRACTIONS = [0.1, 0.2, 0.5, 1.0]

# Количество итераций для усреднения
NUM_ITERATIONS = 3

# Количество воркеров для конкурентной записи
NUM_WORKERS = 4
ROWS_PER_WORKER = 10_000
