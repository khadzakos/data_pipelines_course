import os
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from deltalake import DeltaTable, write_deltalake

from pyiceberg.catalog import load_catalog
import pyarrow as pa

from config import DATA_DIR, RESULTS_DIR, NUM_WORKERS, ROWS_PER_WORKER


def clean_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def generate_batch_data(batch_id: int, num_rows: int = ROWS_PER_WORKER) -> pd.DataFrame:
    np.random.seed(batch_id * 100)

    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Sports']
    start_date = datetime(2023, 1, 1)

    data = {
        'id': range(batch_id * num_rows + 1, (batch_id + 1) * num_rows + 1),
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
        'description': [f"Batch {batch_id} Transaction {i}" for i in range(num_rows)],
        'batch_id': [batch_id] * num_rows
    }

    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].astype('datetime64[us]')

    return df


# ============================================================================
# DELTA LAKE - Concurrent Write Test
# ============================================================================

def delta_concurrent_worker(batch_id: int, path: str) -> Dict:
    result = {
        "batch_id": batch_id,
        "format": "Delta Lake",
        "status": "unknown",
        "error": None,
        "time_sec": 0,
        "rows_written": 0,
        "retries": 0
    }

    try:
        df = generate_batch_data(batch_id)

        start_time = time.time()

        # Append к существующей таблице
        # delta-rs автоматически обрабатывает конкурентность
        write_deltalake(
            path,
            df,
            mode="append"
        )

        result["time_sec"] = round(time.time() - start_time, 3)
        result["status"] = "success"
        result["rows_written"] = len(df)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:200]

    return result


def test_delta_concurrent(num_workers: int = NUM_WORKERS) -> List[Dict]:
    path = os.path.join(DATA_DIR, "delta_concurrent")
    clean_directory(path)

    # Создаем начальную таблицу
    init_df = generate_batch_data(0, 1000)
    write_deltalake(path, init_df, mode="overwrite")

    print(f"\nТестирование конкурентной записи Delta Lake ({num_workers} воркеров)...")
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(delta_concurrent_worker, i, path): i
            for i in range(1, num_workers + 1)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["status"] == "success" else "FAIL"
            print(f"  Batch {result['batch_id']}: [{status}] {result['time_sec']:.3f}s")
            if result["error"]:
                print(f"    Error: {result['error'][:80]}...")

    # Проверяем итоговое количество записей
    try:
        dt = DeltaTable(path)
        total_rows = len(dt.to_pandas())
        print(f"  Итого записей в таблице: {total_rows:,}")
    except Exception as e:
        print(f"  Не удалось прочитать итог: {e}")

    return results


# ============================================================================
# ICEBERG - Concurrent Write Test
# ============================================================================

def iceberg_concurrent_worker(batch_id: int, warehouse_path: str, table_name: str) -> Dict:
    result = {
        "batch_id": batch_id,
        "format": "Iceberg",
        "status": "unknown",
        "error": None,
        "time_sec": 0,
        "rows_written": 0,
        "retries": 0
    }

    try:
        catalog = load_catalog(
            "default",
            **{
                "type": "sql",
                "uri": f"sqlite:///{warehouse_path}/catalog.db",
                "warehouse": f"file://{warehouse_path}",
            }
        )

        df = generate_batch_data(batch_id)
        arrow_table = pa.Table.from_pandas(df)

        start_time = time.time()

        table = catalog.load_table(f"db.{table_name}")
        table.append(arrow_table)

        result["time_sec"] = round(time.time() - start_time, 3)
        result["status"] = "success"
        result["rows_written"] = len(df)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:200]

    return result


def test_iceberg_concurrent(num_workers: int = NUM_WORKERS) -> List[Dict]:
    warehouse_path = os.path.join(DATA_DIR, "iceberg_concurrent")
    table_name = "concurrent_test"
    clean_directory(warehouse_path)
    os.makedirs(warehouse_path, exist_ok=True)

    # Создаем каталог и начальную таблицу
    catalog = load_catalog(
        "default",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    try:
        catalog.create_namespace("db")
    except Exception:
        pass

    init_df = generate_batch_data(0, 1000)
    arrow_table = pa.Table.from_pandas(init_df)

    try:
        catalog.drop_table(f"db.{table_name}")
    except Exception:
        pass

    iceberg_table = catalog.create_table(f"db.{table_name}", schema=arrow_table.schema)
    iceberg_table.append(arrow_table)

    print(f"\nТестирование конкурентной записи Iceberg ({num_workers} воркеров)...")
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(iceberg_concurrent_worker, i, warehouse_path, table_name): i
            for i in range(1, num_workers + 1)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["status"] == "success" else "FAIL"
            print(f"  Batch {result['batch_id']}: [{status}] {result['time_sec']:.3f}s")
            if result["error"]:
                print(f"    Error: {result['error'][:80]}...")

    # Проверяем итоговое количество записей
    try:
        table = catalog.load_table(f"db.{table_name}")
        total_rows = len(table.scan().to_pandas())
        print(f"  Итого записей в таблице: {total_rows:,}")
    except Exception as e:
        print(f"  Не удалось прочитать итог: {e}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_concurrent_tests(num_workers: int = NUM_WORKERS) -> pd.DataFrame:
    all_results = []

    print("="*70)
    print("ТЕСТИРОВАНИЕ КОНКУРЕНТНОЙ ЗАПИСИ")
    print("="*70)
    print(f"Параметры: {num_workers} воркеров, {ROWS_PER_WORKER:,} строк/воркер")
    print("\nОсобенности механизмов конкурентности:")
    print("- Delta Lake: OCC (Optimistic Concurrency Control)")
    print("- Iceberg: OCC (Optimistic Concurrency Control)")
    print("- Hudi: NBCC (Non-Blocking Concurrency Control) - требует Spark")

    # Delta Lake
    try:
        results = test_delta_concurrent(num_workers)
        all_results.extend(results)
    except Exception as e:
        print(f"  Delta Lake общая ошибка: {e}")

    # Iceberg
    try:
        results = test_iceberg_concurrent(num_workers)
        all_results.extend(results)
    except Exception as e:
        print(f"  Iceberg общая ошибка: {e}")

    # Сохраняем результаты
    df = pd.DataFrame(all_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "concurrent_write_results.csv"), index=False)

    # Выводим сводку
    print("\n" + "="*70)
    print("ИТОГИ КОНКУРЕНТНОЙ ЗАПИСИ")
    print("="*70)

    if len(df) > 0:
        summary = df.groupby('format').agg({
            'status': lambda x: (x == 'success').sum(),
            'time_sec': 'mean',
            'rows_written': 'sum'
        }).reset_index()
        summary.columns = ['Format', 'Successful', 'Avg Time (sec)', 'Total Rows']
        print(summary.to_string(index=False))

        # Анализ ошибок
        failed = df[df['status'] == 'failed']
        if len(failed) > 0:
            print("\nОшибки при записи:")
            for _, row in failed.iterrows():
                print(f"  {row['format']} Batch {row['batch_id']}: {row['error'][:60]}...")

    return df


if __name__ == "__main__":
    run_all_concurrent_tests()
