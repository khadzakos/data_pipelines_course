import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from fastavro import parse_schema
from fastavro.write import Writer


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def rm_path(path: Path):
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
    else:
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            path.rmdir()
        except OSError:
            pass


def src_relation(con: duckdb.DuckDBPyConnection, src_fmt: str, src_path: str) -> str:
    src_fmt = src_fmt.lower()
    if src_fmt == "csv":
        return f"read_csv_auto('{src_path}')"
    if src_fmt == "parquet":
        # может быть файл или glob типа out_data/parquet/*.parquet
        return f"read_parquet('{src_path}')"
    if src_fmt == "avro":
        con.execute("INSTALL avro;")
        con.execute("LOAD avro;")
        return f"read_avro('{src_path}')"
    raise ValueError("src_fmt must be one of: csv, parquet, avro")


def avro_schema_from_arrow_schema(schema: pa.Schema, name="dataset") -> dict:
    def avro_type(t: pa.DataType):
        # nullable оборачиваем с ["null", ...]
        if pa.types.is_boolean(t):
            return ["null", "boolean"]
        if pa.types.is_integer(t):
            return ["null", "long"]
        if pa.types.is_floating(t):
            return ["null", "double"]
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return ["null", "string"]
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            return ["null", "bytes"]
        # на всякий случай всё остальное как string
        return ["null", "string"]

    fields = [{"name": f.name, "type": avro_type(f.type)} for f in schema]
    return {"type": "record", "name": name, "fields": fields}


def write_csv(con, src_rel: str, out_csv: Path) -> float:
    rm_path(out_csv)
    t0 = time.perf_counter()
    con.execute(f"COPY (SELECT * FROM {src_rel}) TO '{out_csv.as_posix()}' (FORMAT CSV, HEADER true);")
    return time.perf_counter() - t0


def write_jsonl(con, src_rel: str, out_jsonl: Path) -> float:
    rm_path(out_jsonl)
    t0 = time.perf_counter()
    # DuckDB пишет JSON Lines (по одной JSON-записи на строку) — то что обычно и надо
    con.execute(f"COPY (SELECT * FROM {src_rel}) TO '{out_jsonl.as_posix()}' (FORMAT JSON);")
    return time.perf_counter() - t0


def write_parquet(con, src_rel: str, out_parquet: Path, compression="snappy") -> float:
    rm_path(out_parquet)
    t0 = time.perf_counter()
    comp = compression.upper()
    con.execute(
        f"COPY (SELECT * FROM {src_rel}) TO '{out_parquet.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION '{comp}');"
    )
    return time.perf_counter() - t0


def write_avro_streaming(con, src_rel: str, out_avro: Path, batch_size: int = 100_000, codec: str = "null") -> float:
    """
    Пишем Avro в 1 файл потоково: DuckDB -> Arrow batches -> fastavro Writer
    """
    rm_path(out_avro)

    t0 = time.perf_counter()

    # Arrow reader (потоково) — fetch_arrow_reader есть у DuckDBPyRelation, не у connection
    reader = con.sql(f"SELECT * FROM {src_rel}").fetch_arrow_reader(batch_size=batch_size)

    first_batch = None
    it = iter(reader)
    try:
        first_batch = next(it)
    except StopIteration:
        # пустой датасет
        out_avro.write_bytes(b"")
        return time.perf_counter() - t0

    schema = avro_schema_from_arrow_schema(first_batch.schema, name="titanic_big")
    parsed = parse_schema(schema)

    out_avro.parent.mkdir(parents=True, exist_ok=True)
    with open(out_avro, "wb") as fo:
        avro_writer = Writer(fo, parsed, codec=codec)
        # первая пачка — fastavro Writer пишет по одному record через write()
        for record in first_batch.to_pylist():
            avro_writer.write(record)
        # остальные пачки
        for batch in it:
            for record in batch.to_pylist():
                avro_writer.write(record)
        avro_writer.flush()

    return time.perf_counter() - t0


def main(src_fmt: str, src_path: str, out_dir: str, parquet_compression="snappy", avro_codec="null", avro_batch_size=100_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    rel = src_relation(con, src_fmt, src_path)

    outputs = {
        "CSV": out_dir / "data.csv",
        "Parquet": out_dir / "data.parquet",
        "Avro": out_dir / "data.avro",
    }

    rows = []

    # CSV
    dt = write_csv(con, rel, outputs["CSV"])
    rows.append({"format": "CSV", "write_seconds": dt, "size_gb": size_bytes(outputs["CSV"]) / 1024**3})

    # Parquet
    dt = write_parquet(con, rel, outputs["Parquet"], compression=parquet_compression)
    rows.append({"format": "Parquet", "write_seconds": dt, "size_gb": size_bytes(outputs["Parquet"]) / 1024**3})

    # Avro
    dt = write_avro_streaming(con, rel, outputs["Avro"], batch_size=avro_batch_size, codec=avro_codec)
    rows.append({"format": "Avro", "write_seconds": dt, "size_gb": size_bytes(outputs["Avro"]) / 1024**3})

    con.close()

    df = pd.DataFrame(rows).sort_values("write_seconds")
    print(df.to_string(index=False))

    out_csv = out_dir / "benchmark_write_results.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv.as_posix())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-fmt", required=True, choices=["csv", "parquet", "avro"])
    ap.add_argument("--src", required=True, help="Путь или glob (для parquet/avro можно *.parquet/*.avro)")
    ap.add_argument("--out-dir", default="out_bench_write")
    ap.add_argument("--parquet-compression", default="snappy", choices=["snappy", "gzip", "zstd", "uncompressed"])
    ap.add_argument("--avro-codec", default="null", choices=["null", "deflate", "snappy"])
    ap.add_argument("--avro-batch-size", type=int, default=100_000)
    args = ap.parse_args()

    main(
        src_fmt=args.src_fmt,
        src_path=args.src,
        out_dir=args.out_dir,
        parquet_compression=args.parquet_compression,
        avro_codec=args.avro_codec,
        avro_batch_size=args.avro_batch_size,
    )
