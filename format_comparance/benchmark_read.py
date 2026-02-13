import time
from pathlib import Path
import duckdb
import pandas as pd


def folder_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def time_query(con, sql: str, repeats: int = 1):
    # Можно сделать прогрев (warmup) отдельно, если хотите
    best = None
    last_df = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_df = con.execute(sql).df()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, last_df


def benchmark(data_dir="out_data", repeats=1):
    data_dir = Path(data_dir)
    csv_path = data_dir / "titanic_big.csv"
    parquet_glob = str((data_dir / "parquet" / "*.parquet").as_posix())
    avro_glob = str((data_dir / "avro" / "*.avro").as_posix())

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")  # настройте под ваш CPU
    con.execute("PRAGMA enable_progress_bar=false")

    # Avro extension
    has_avro = True
    try:
        con.execute("INSTALL avro;")
        con.execute("LOAD avro;")
    except Exception:
        has_avro = False

    sources = [
        ("CSV",   f"read_csv_auto('{csv_path.as_posix()}')", folder_size_bytes(csv_path)),
        ("Parquet", f"read_parquet('{parquet_glob}')", folder_size_bytes(data_dir / "parquet")),
    ]
    if has_avro:
        sources.append(("Avro", f"read_avro('{avro_glob}')", folder_size_bytes(data_dir / "avro")))

    queries = {
        "full_scan_count": "SELECT count(*) AS cnt FROM SRC",
        "filter_count": "SELECT count(*) AS cnt FROM SRC WHERE Sex='female' AND Fare > 30",
        "agg_avg_fare": "SELECT Pclass, avg(Fare) AS avg_fare FROM SRC GROUP BY Pclass ORDER BY Pclass",
    }

    rows = []
    for fmt, src_sql, size_b in sources:
        for qname, qtpl in queries.items():
            sql = qtpl.replace("SRC", src_sql)
            dt, df = time_query(con, sql, repeats=repeats)
            rows.append({
                "format": fmt,
                "query": qname,
                "seconds_best": dt,
                "size_gb": size_b / 1024**3
            })

    res = pd.DataFrame(rows).sort_values(["query", "seconds_best"])
    print(res)
    res.to_csv(data_dir / "benchmark_results.csv", index=False)
    print("\nSaved:", (data_dir / "benchmark_results.csv").as_posix())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="out_data")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    benchmark(args.data_dir, repeats=args.repeats)
