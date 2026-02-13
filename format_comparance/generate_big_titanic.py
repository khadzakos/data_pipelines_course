import os
import json
import time
import math
import string
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import writer as avro_writer


def folder_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def rand_ascii(n: int, rng: np.random.Generator) -> str:
    # Быстрая генерация “не сильно сжимаемой” строки
    alphabet = np.frombuffer(
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        dtype=np.uint8
    )
    idx = rng.integers(0, len(alphabet), size=n, dtype=np.int32)
    return bytes(alphabet[idx]).decode("ascii")


def normalize_titanic_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Приведём типы к более стабильным (важно для Avro/Parquet)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string")
    for c in ["PassengerId", "Survived", "Pclass", "SibSp", "Parch"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["Age", "Fare"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def make_avro_schema_from_df(df: pd.DataFrame, name="titanic") -> dict:
    def field_type_for_series(s: pd.Series):
        # Avro schema: делаем nullable поля через ["null", type]
        if pd.api.types.is_integer_dtype(s.dtype):
            return ["null", "long"]
        if pd.api.types.is_float_dtype(s.dtype):
            return ["null", "double"]
        if pd.api.types.is_bool_dtype(s.dtype):
            return ["null", "boolean"]
        # строки
        return ["null", "string"]

    fields = [{"name": col, "type": field_type_for_series(df[col])} for col in df.columns]
    return {"type": "record", "name": name, "fields": fields}


def df_to_records(df: pd.DataFrame):
    # pandas nullable Int64 -> Python int/None
    recs = []
    for row in df.itertuples(index=False, name=None):
        rec = {}
        for col, val in zip(df.columns, row):
            if pd.isna(val):
                rec[col] = None
            else:
                # numpy types -> python types
                if isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    rec[col] = float(val)
                else:
                    rec[col] = str(val) if isinstance(val, pd.StringDtype().type) else val
        recs.append(rec)
    return recs


def generate(
    input_csv: str,
    out_dir: str = "out_data",
    target_gb: float = 10.0,
    payload_bytes: int = 200,
    chunk_rows: int = 200_000,
    parquet_compression: str = "snappy",
    seed: int = 42,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(input_csv)
    base = normalize_titanic_schema(base)

    rng = np.random.default_rng(seed)

    # Папки под форматы
    csv_path = out / "titanic_big.csv"
    parquet_dir = out / "parquet"
    avro_dir = out / "avro"

    parquet_dir.mkdir(exist_ok=True)
    avro_dir.mkdir(exist_ok=True)

    # Очистим старые файлы
    for p in [csv_path]:
        if p.exists():
            p.unlink()
    for d in [parquet_dir, avro_dir]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    f.unlink()

    target_bytes = int(target_gb * 1024**3)

    # Добавим payload колонку (будет генериться на лету)
    cols = list(base.columns) + ["payload"]
    base_cols = list(base.columns)

    # Для Parquet writer
    parquet_writer = None
    parquet_part = 0

    # Для Avro — будем писать “part файлы” (проще, чем аппендить контейнер)
    avro_schema = make_avro_schema_from_df(pd.DataFrame({c: base[c] for c in base_cols}).assign(payload=pd.Series(dtype="string")))

    # Заголовки для CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

    written_rows = 0
    start_total = time.perf_counter()

    n_base = len(base)

    # Пишем, пока CSV (как “самый большой” обычно) не достигнет target
    # Можно заменить на max по всем форматам, но цель задания — датасет ~10GB.
    while csv_path.stat().st_size < target_bytes:
        # Сформируем chunk_rows строк, повторяя base
        reps = math.ceil(chunk_rows / n_base)
        chunk = pd.concat([base] * reps, ignore_index=True).iloc[:chunk_rows].copy()

        # Сделаем PassengerId уникальнее (если есть)
        if "PassengerId" in chunk.columns:
            chunk["PassengerId"] = (chunk["PassengerId"].astype("Int64") + written_rows).astype("Int64")

        # payload — случайная строка фиксированной длины (увеличивает размер и снижает сжатие)
        payload = [rand_ascii(payload_bytes, rng) for _ in range(len(chunk))]
        chunk["payload"] = pd.Series(payload, dtype="string")

        # --- CSV append ---
        t0 = time.perf_counter()
        chunk.to_csv(csv_path, mode="a", header=False, index=False)
        t_csv = time.perf_counter() - t0

        # --- Parquet append (part files) ---
        t0 = time.perf_counter()
        table = pa.Table.from_pandas(chunk[cols], preserve_index=False)
        part_path = parquet_dir / f"part-{parquet_part:06d}.parquet"
        pq.write_table(
            table,
            part_path,
            compression=parquet_compression,
            use_dictionary=True
        )
        parquet_part += 1
        t_parquet = time.perf_counter() - t0

        # --- Avro append (part files) ---
        t0 = time.perf_counter()
        avro_path = avro_dir / f"part-{parquet_part:06d}.avro"
        records = df_to_records(chunk[cols])
        with open(avro_path, "wb") as fo:
            avro_writer(fo, avro_schema, records)
        t_avro = time.perf_counter() - t0

        written_rows += len(chunk)

        print(
            f"rows={written_rows:,} | CSV={csv_path.stat().st_size/1024**3:.2f}GB "
            f"| write_s: csv={t_csv:.2f}, parquet={t_parquet:.2f}, avro={t_avro:.2f}"
        )

    total_s = time.perf_counter() - start_total
    print("\nFinished")
    print(f"Total rows: {written_rows:,}")
    print(f"Total time: {total_s:.1f}s")
    print(f"CSV size:    {folder_size_bytes(csv_path)/1024**3:.2f} GB")
    print(f"Parquet dir: {folder_size_bytes(parquet_dir)/1024**3:.2f} GB")
    print(f"Avro dir:    {folder_size_bytes(avro_dir)/1024**3:.2f} GB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to titanic.csv")
    ap.add_argument("--out", default="out_data")
    ap.add_argument("--target-gb", type=float, default=10.0)
    ap.add_argument("--payload-bytes", type=int, default=200)
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--parquet-compression", default="snappy", choices=["snappy", "gzip", "zstd", "none"])
    args = ap.parse_args()

    generate(
        input_csv=args.input,
        out_dir=args.out,
        target_gb=args.target_gb,
        payload_bytes=args.payload_bytes,
        chunk_rows=args.chunk_rows,
        parquet_compression=None if args.parquet_compression == "none" else args.parquet_compression,
    )
