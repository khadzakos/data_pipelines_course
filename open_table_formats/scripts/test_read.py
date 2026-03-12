import os
import time
from typing import Dict, List
import pandas as pd
import pyarrow.parquet as pq
import duckdb

from deltalake import DeltaTable

from pyiceberg.catalog import load_catalog

from config import (
    DATA_DIR, RESULTS_DIR, DELTA_PATH, ICEBERG_PATH, PARQUET_PATH, NUM_ITERATIONS
)


# ============================================================================
# PARQUET
# ============================================================================

def read_parquet_full(path: str) -> Dict:
    filepath = os.path.join(path, "data.parquet")

    start_time = time.time()
    df = pd.read_parquet(filepath)
    count = len(df)
    read_time = time.time() - start_time

    return {
        "format": "Parquet",
        "operation": "full_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_parquet_filtered(path: str, category: str = "Electronics") -> Dict:
    filepath = os.path.join(path, "data.parquet")

    start_time = time.time()
    # Используем predicate pushdown через PyArrow
    table = pq.read_table(
        filepath,
        filters=[('category', '=', category)]
    )
    count = table.num_rows
    read_time = time.time() - start_time

    return {
        "format": "Parquet",
        "operation": "filtered_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_parquet_aggregation(path: str) -> Dict:
    filepath = os.path.join(path, "data.parquet")

    start_time = time.time()
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT category, COUNT(*) as cnt, AVG(amount) as avg_amount
        FROM '{filepath}'
        GROUP BY category
    """).fetchdf()
    read_time = time.time() - start_time
    con.close()

    return {
        "format": "Parquet",
        "operation": "aggregation",
        "read_time_sec": round(read_time, 4),
        "rows": len(result)
    }


# ============================================================================
# DELTA LAKE
# ============================================================================

def read_delta_full(path: str) -> Dict:
    start_time = time.time()
    dt = DeltaTable(path)
    df = dt.to_pandas()
    count = len(df)
    read_time = time.time() - start_time

    return {
        "format": "Delta Lake",
        "operation": "full_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_delta_filtered(path: str, category: str = "Electronics") -> Dict:
    start_time = time.time()
    dt = DeltaTable(path)
    # Delta-rs поддерживает predicate pushdown
    df = dt.to_pandas(filters=[("category", "=", category)])
    count = len(df)
    read_time = time.time() - start_time

    return {
        "format": "Delta Lake",
        "operation": "filtered_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_delta_aggregation(path: str) -> Dict:
    start_time = time.time()
    con = duckdb.connect()
    # DuckDB нативно поддерживает Delta Lake
    result = con.execute(f"""
        SELECT category, COUNT(*) as cnt, AVG(amount) as avg_amount
        FROM delta_scan('{path}')
        GROUP BY category
    """).fetchdf()
    read_time = time.time() - start_time
    con.close()

    return {
        "format": "Delta Lake",
        "operation": "aggregation",
        "read_time_sec": round(read_time, 4),
        "rows": len(result)
    }


# ============================================================================
# ICEBERG
# ============================================================================

def read_iceberg_full(warehouse_path: str, table_name: str = "test_100") -> Dict:
    catalog = load_catalog(
        "default",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    start_time = time.time()
    table = catalog.load_table(f"db.{table_name}")
    df = table.scan().to_pandas()
    count = len(df)
    read_time = time.time() - start_time

    return {
        "format": "Iceberg",
        "operation": "full_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_iceberg_filtered(warehouse_path: str, table_name: str = "test_100",
                          category: str = "Electronics") -> Dict:
    from pyiceberg.expressions import EqualTo

    catalog = load_catalog(
        "default",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    start_time = time.time()
    table = catalog.load_table(f"db.{table_name}")
    # Используем row filter для predicate pushdown
    df = table.scan(row_filter=EqualTo("category", category)).to_pandas()
    count = len(df)
    read_time = time.time() - start_time

    return {
        "format": "Iceberg",
        "operation": "filtered_scan",
        "read_time_sec": round(read_time, 4),
        "rows": count
    }


def read_iceberg_aggregation(warehouse_path: str, table_name: str = "test_100") -> Dict:
    """Агрегационный запрос к Iceberg (через pandas)"""
    catalog = load_catalog(
        "default",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    start_time = time.time()
    table = catalog.load_table(f"db.{table_name}")
    df = table.scan().to_pandas()
    result = df.groupby('category').agg({'id': 'count', 'amount': 'mean'})
    read_time = time.time() - start_time

    return {
        "format": "Iceberg",
        "operation": "aggregation",
        "read_time_sec": round(read_time, 4),
        "rows": len(result)
    }


# ============================================================================
# MAIN
# ============================================================================

def run_read_tests(num_iterations: int = NUM_ITERATIONS) -> List[Dict]:
    all_results = []

    print(f"\n{'='*60}")
    print(f"Тесты чтения ({num_iterations} итераций)")
    print(f"{'='*60}")

    for iteration in range(num_iterations):
        print(f"\n--- Итерация {iteration + 1}/{num_iterations} ---")

        # Parquet
        if os.path.exists(os.path.join(PARQUET_PATH, "data.parquet")):
            print("\n[1/3] Parquet...")
            for test_func in [read_parquet_full, read_parquet_filtered, read_parquet_aggregation]:
                try:
                    result = test_func(PARQUET_PATH)
                    result["iteration"] = iteration
                    all_results.append(result)
                    print(f"  {result['operation']}: {result['read_time_sec']:.4f} сек")
                except Exception as e:
                    print(f"  {test_func.__name__}: Ошибка - {e}")

        # Delta Lake
        if os.path.exists(DELTA_PATH):
            print("\n[2/3] Delta Lake...")
            for test_func in [read_delta_full, read_delta_filtered, read_delta_aggregation]:
                try:
                    result = test_func(DELTA_PATH)
                    result["iteration"] = iteration
                    all_results.append(result)
                    print(f"  {result['operation']}: {result['read_time_sec']:.4f} сек")
                except Exception as e:
                    print(f"  {test_func.__name__}: Ошибка - {e}")

        # Iceberg
        if os.path.exists(os.path.join(ICEBERG_PATH, "catalog.db")):
            print("\n[3/3] Iceberg...")
            for test_func in [read_iceberg_full, read_iceberg_filtered, read_iceberg_aggregation]:
                try:
                    result = test_func(ICEBERG_PATH)
                    result["iteration"] = iteration
                    all_results.append(result)
                    print(f"  {result['operation']}: {result['read_time_sec']:.4f} сек")
                except Exception as e:
                    print(f"  {test_func.__name__}: Ошибка - {e}")

    return all_results


def save_results(results: List[Dict], filename: str = "read_results.csv") -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\nРезультаты сохранены: {filepath}")

    # Выводим сводку
    if len(df) > 0:
        summary = df.groupby(['format', 'operation'])['read_time_sec'].agg(['mean', 'std']).reset_index()
        summary.columns = ['format', 'operation', 'mean_sec', 'std_sec']
        print("\n" + "="*60)
        print("СРЕДНИЕ РЕЗУЛЬТАТЫ ЧТЕНИЯ")
        print("="*60)
        print(summary.to_string(index=False))

    return df


if __name__ == "__main__":
    results = run_read_tests()
    save_results(results)
