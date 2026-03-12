import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, NUM_ROWS


def generate_test_data(num_rows: int = NUM_ROWS, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Sports',
                  'Home', 'Beauty', 'Toys', 'Auto', 'Garden']

    start_date = datetime(2023, 1, 1)

    data = {
        'id': range(1, num_rows + 1),
        'user_id': np.random.randint(1, 100_000, num_rows),
        'product_id': np.random.randint(1, 50_000, num_rows),
        'category': np.random.choice(categories, num_rows),
        'amount': np.round(np.random.uniform(1.0, 10000.0, num_rows), 2),
        'quantity': np.random.randint(1, 100, num_rows),
        'transaction_date': [
            start_date + timedelta(days=int(x))
            for x in np.random.randint(0, 365, num_rows)
        ],
        'created_at': [
            datetime.now() - timedelta(seconds=int(x))
            for x in np.random.randint(0, 86400 * 30, num_rows)
        ],
        'is_returned': np.random.choice([True, False], num_rows, p=[0.05, 0.95]),
        'description': [
            f"Transaction {i} for product"
            for i in range(num_rows)
        ]
    }

    return pd.DataFrame(data)


def save_source_data(df: pd.DataFrame, filename: str = "source_data.parquet"):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)

    # Используем PyArrow для более эффективного сохранения
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression='snappy')

    print(f"Данные сохранены: {filepath}")
    print(f"Размер: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    print(f"Строк: {len(df):,}")
    return filepath


def load_source_data(filename: str = "source_data.parquet") -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, filename)
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    print("Генерация тестовых данных...")
    print(f"Количество строк: {NUM_ROWS:,}")

    df = generate_test_data()

    print("\nПример данных:")
    print(df.head())
    print("\nТипы данных:")
    print(df.dtypes)
    print("\nСтатистика:")
    print(df.describe())

    save_source_data(df)
