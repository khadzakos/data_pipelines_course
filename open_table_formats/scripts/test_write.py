import os
import time
import shutil
from typing import Dict, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import deltalake
from deltalake import DeltaTable, write_deltalake

from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField, LongType, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType
)

from config import (
    DATA_DIR, RESULTS_DIR, DELTA_PATH, ICEBERG_PATH, PARQUET_PATH, WRITE_FRACTIONS
)
from generate_data import load_source_data


def clean_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def get_directory_size(path: str) -> float:
    if not os.path.exists(path):
        return 0
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / 1024 / 1024


# ============================================================================
# PARQUET (Baseline)
# ============================================================================

def write_parquet(df: pd.DataFrame, path: str) -> Dict:
    clean_directory(path)
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, "data.parquet")

    start_time = time.time()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression='snappy')
    write_time = time.time() - start_time

    size_mb = get_directory_size(path)

    return {
        "format": "Parquet",
        "write_time_sec": round(write_time, 3),
        "size_mb": round(size_mb, 2),
        "rows": len(df)
    }


# ============================================================================
# DELTA LAKE (using delta-rs)
# ============================================================================

def write_delta(df: pd.DataFrame, path: str) -> Dict:
    clean_directory(path)

    # Конвертируем timestamp колонки в формат, совместимый с Delta
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].astype('datetime64[us]')

    start_time = time.time()
    write_deltalake(path, df_copy, mode="overwrite")
    write_time = time.time() - start_time

    size_mb = get_directory_size(path)

    return {
        "format": "Delta Lake",
        "write_time_sec": round(write_time, 3),
        "size_mb": round(size_mb, 2),
        "rows": len(df)
    }


# ============================================================================
# ICEBERG (using PyIceberg with SQLite catalog)
# ============================================================================

def write_iceberg(df: pd.DataFrame, warehouse_path: str, table_name: str = "test_table") -> Dict:
    clean_directory(warehouse_path)
    os.makedirs(warehouse_path, exist_ok=True)

    # Создаем SQLite catalog
    catalog = load_catalog(
        "default",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    # Создаем namespace
    try:
        catalog.create_namespace("db")
    except Exception:
        pass

    # Конвертируем DataFrame в PyArrow Table
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].astype('datetime64[us]')

    arrow_table = pa.Table.from_pandas(df_copy)

    start_time = time.time()

    # Создаем таблицу и записываем данные
    try:
        catalog.drop_table(f"db.{table_name}")
    except Exception:
        pass

    iceberg_table = catalog.create_table(
        f"db.{table_name}",
        schema=arrow_table.schema,
    )
    iceberg_table.append(arrow_table)

    write_time = time.time() - start_time

    size_mb = get_directory_size(warehouse_path)

    return {
        "format": "Iceberg",
        "write_time_sec": round(write_time, 3),
        "size_mb": round(size_mb, 2),
        "rows": len(df)
    }


# ============================================================================
# MAIN
# ============================================================================

def run_write_tests(sample_fraction: float = 1.0) -> List[Dict]:
    results = []
    source_df = load_source_data()

    if sample_fraction < 1.0:
        source_df = source_df.sample(frac=sample_fraction, random_state=42)

    num_rows = len(source_df)
    print(f"\n{'='*60}")
    print(f"Тест записи: {sample_fraction*100:.0f}% данных ({num_rows:,} строк)")
    print(f"{'='*60}")

    # Parquet
    print("\n[1/3] Тестирование Parquet...")
    try:
        result = write_parquet(source_df, PARQUET_PATH)
        result["sample_fraction"] = sample_fraction
        results.append(result)
        print(f"  Время: {result['write_time_sec']:.3f} сек, Размер: {result['size_mb']:.2f} MB")
    except Exception as e:
        print(f"  Ошибка: {e}")

    # Delta Lake
    print("\n[2/3] Тестирование Delta Lake...")
    try:
        result = write_delta(source_df, DELTA_PATH)
        result["sample_fraction"] = sample_fraction
        results.append(result)
        print(f"  Время: {result['write_time_sec']:.3f} сек, Размер: {result['size_mb']:.2f} MB")
    except Exception as e:
        print(f"  Ошибка: {e}")

    # Iceberg
    print("\n[3/3] Тестирование Iceberg...")
    try:
        table_name = f"test_{int(sample_fraction*100)}"
        result = write_iceberg(source_df, ICEBERG_PATH, table_name)
        result["sample_fraction"] = sample_fraction
        results.append(result)
        print(f"  Время: {result['write_time_sec']:.3f} сек, Размер: {result['size_mb']:.2f} MB")
    except Exception as e:
        print(f"  Ошибка: {e}")
        import traceback
        traceback.print_exc()

    return results


def save_results(results: List[Dict], filename: str = "write_results.csv"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\nРезультаты сохранены: {filepath}")
    return df


if __name__ == "__main__":
    all_results = []

    for fraction in WRITE_FRACTIONS:
        results = run_write_tests(sample_fraction=fraction)
        all_results.extend(results)

    df = save_results(all_results)

    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЗАПИСИ")
    print("="*60)
    print(df.to_string(index=False))
