from __future__ import annotations
import argparse
import time
import json
from typing import Dict, Any, Optional, Tuple

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, LongType, StringType
    from pyspark.sql.functions import broadcast
except Exception as e:
    print("Ошибка импорта pyspark. Установите pyspark: python -m pip install pyspark")
    raise

try:
    import requests
except Exception:
    print("Установите requests: python -m pip install requests")
    raise

def build_spark(app_name: str, extra_conf: Optional[Dict[str, str]] = None) -> SparkSession:
    builder = SparkSession.builder.master("local[*]").appName(app_name)
    base_conf = {
        "spark.ui.showConsoleProgress": "false",
        "spark.sql.shuffle.partitions": "200",
    }
    if extra_conf:
        base_conf.update(extra_conf)
    for k, v in base_conf.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def now_ts() -> float:
    return time.time()

def fetch_spark_stage_metrics(spark, app_id: Optional[str] = None, ui_url: Optional[str] = None) -> Dict[str, Any]:
    try:
        sc = spark.sparkContext
        if app_id is None:
            app_id = sc.applicationId
        if ui_url is None:
            try:
                ui_candidate = sc._jsc.sc().uiWebUrl().get()
                if ui_candidate:
                    ui_url = ui_candidate
            except Exception:
                ui_url = "http://localhost:4040"
        if not ui_url:
            ui_url = "http://localhost:4040"
        stages_url = f"{ui_url.rstrip('/')}/api/v1/applications/{app_id}/stages"
        resp = requests.get(stages_url, timeout=5)
        resp.raise_for_status()
        stages = resp.json()
        summary = {
            "shuffleReadBytes": 0,
            "shuffleWriteBytes": 0,
            "memoryBytesSpilled": 0,
            "diskBytesSpilled": 0,
            "executorRunTime": 0,
            "inputBytes": 0,
            "numTasks": 0
        }
        for s in stages:
            task_metrics = s.get("taskMetrics") or {}
            for key in ["shuffleReadBytes", "shuffleWriteBytes", "memoryBytesSpilled", "diskBytesSpilled", "executorRunTime", "inputBytes"]:
                if key in s and isinstance(s[key], (int, float)):
                    summary[key] = summary.get(key, 0) + s.get(key, 0)
                elif key in task_metrics and isinstance(task_metrics[key], (int, float)):
                    summary[key] = summary.get(key, 0) + task_metrics.get(key, 0)
            summary["numTasks"] += s.get("numTasks", 0) or 0
        return summary
    except Exception as e:
        return {}

def run_and_time(action_callable):
    t0 = now_ts()
    res = action_callable()
    t1 = now_ts()
    return res, (t1 - t0)

def generate_orders_and_countries(spark: SparkSession, n_orders: int = 2_000_000, n_countries: int = 200) -> Tuple[DataFrame, DataFrame]:
    orders = spark.range(0, n_orders).withColumnRenamed("id", "order_id")
    orders = orders.withColumn("country_id", (F.col("order_id") % n_countries).cast(IntegerType()))
    orders = orders.withColumn("amount", (F.sha2(F.concat(F.col("order_id").cast(StringType())), 256).substr(1, 8).cast("long") % 10000).cast("int"))
    orders = orders.withColumn("ts", (F.col("order_id") % 1_000_000).cast(LongType()))

    countries = spark.range(0, n_countries).withColumnRenamed("id", "country_id")
    countries = countries.withColumn("country_name", F.concat(F.lit("country_"), F.col("country_id").cast(StringType())))
    return orders, countries

def generate_skewed_dataset(spark: SparkSession, n_rows: int = 5_000_000, skew_fraction: float = 0.9) -> DataFrame:
    df = spark.range(0, n_rows).withColumnRenamed("id", "row_id")
    df = df.withColumn("rand_mod", (F.col("row_id") % 10000).cast(IntegerType()))
    threshold = int(skew_fraction * 10000)
    df = df.withColumn("key", F.when(F.col("rand_mod") < threshold, F.lit("HOT_KEY")).otherwise(F.concat(F.lit("K_"), (F.col("rand_mod") % 1000).cast(StringType()))))
    df = df.withColumn("value", (F.col("row_id") % 1000).cast(IntegerType()))
    df = df.drop("rand_mod")
    return df

def experiment_broadcast_join(spark: SparkSession, n_orders: int = 2_000_000, n_countries: int = 200):
    print("\n--- EXPERIMENT 1: Broadcast vs Shuffle join ---")
    orders, countries = generate_orders_and_countries(spark, n_orders=n_orders, n_countries=n_countries)
    orders.createOrReplaceTempView("orders")
    countries.createOrReplaceTempView("countries")
    print(f"Сгенерировано orders={n_orders}, countries={n_countries}")

    results = {}
    def baseline_job():
        return orders.join(countries, on="country_id").agg(F.count("*").alias("cnt")).collect()

    print("Запуск baseline (shuffle join)...")
    _, baseline_time = run_and_time(baseline_job)
    baseline_metrics = fetch_spark_stage_metrics(spark)
    results["baseline"] = {"time_s": baseline_time, "metrics": baseline_metrics}
    print(f"Baseline time: {baseline_time:.2f}s, metrics (REST): {baseline_metrics}")

    def broadcast_job():
        return orders.join(broadcast(countries), on="country_id").agg(F.count("*").alias("cnt")).collect()

    print("Запуск optimized (broadcast join)...")
    _, opt_time = run_and_time(broadcast_job)
    opt_metrics = fetch_spark_stage_metrics(spark)
    results["broadcast"] = {"time_s": opt_time, "metrics": opt_metrics}
    print(f"Broadcast time: {opt_time:.2f}s, metrics (REST): {opt_metrics}")

    print("\nСравнение:")
    print(f"Time baseline / broadcast = {baseline_time:.2f}s / {opt_time:.2f}s")
    return results

def experiment_skew_and_salting(spark: SparkSession, n_rows: int = 3_000_000, skew_fraction: float = 0.9, salts: int = 10):
    print("\n--- EXPERIMENT 2: Skew & Salting ---")
    df = generate_skewed_dataset(spark, n_rows=n_rows, skew_fraction=skew_fraction)
    df.createOrReplaceTempView("skewed")
    print(f"Сгенерировано строк: {n_rows}, skew_fraction={skew_fraction}, salts={salts}")

    results = {}

    def baseline_job():
        return df.groupBy("key").agg(F.count("*").alias("cnt")).count()

    print("Запуск baseline (groupBy без salting)...")
    _, baseline_time = run_and_time(baseline_job)
    baseline_metrics = fetch_spark_stage_metrics(spark)
    results["baseline"] = {"time_s": baseline_time, "metrics": baseline_metrics}
    print(f"Baseline time: {baseline_time:.2f}s, metrics: {baseline_metrics}")

    def salted_job():
        salted = df.withColumn("salt", (F.hash(F.col("row_id")) % salts).cast(IntegerType()))
        salted = salted.withColumn("salted_key", F.concat(F.col("key"), F.lit("_"), F.col("salt").cast(StringType())))
        agg1 = salted.groupBy("salted_key", "key").agg(F.count("*").alias("cnt_partial"))
        agg2 = agg1.groupBy("key").agg(F.sum("cnt_partial").alias("cnt"))
        return agg2.count()

    print("Запуск salted (salting -> partial agg -> merge)...")
    _, salted_time = run_and_time(salted_job)
    salted_metrics = fetch_spark_stage_metrics(spark)
    results["salted"] = {"time_s": salted_time, "metrics": salted_metrics}
    print(f"Salted time: {salted_time:.2f}s, metrics: {salted_metrics}")

    return results

def experiment_partitions_repartition_coalesce(
    spark: "SparkSession",
    n_rows: int = 5_000_000,
    group_count: int = 100,
    partitions_list: Optional[list] = None,
    test_coalesce: bool = True
) -> Dict[str, Any]:
    from pyspark.sql import functions as F

    print("\n--- EXPERIMENT 3: repartition / coalesce impact on performance ---")
    print(f"Параметры: n_rows={n_rows}, group_count={group_count}, test_coalesce={test_coalesce}")

    default_parallelism = spark.sparkContext.defaultParallelism or 4
    if partitions_list is None:
        candidates = sorted({
            2,
            max(1, default_parallelism // 2),
            default_parallelism,
            default_parallelism * 2,
            50,
            100,
            200,
            500,
            1000
        })
        partitions_list = [p for p in candidates if p <= max(1, n_rows // 1000)]
        if not partitions_list:
            partitions_list = candidates[:5]
    else:
        partitions_list = sorted(set(int(p) for p in partitions_list if int(p) > 0))

    print(f"default_parallelism (Spark): {default_parallelism}")
    print(f"partitions_list for testing: {partitions_list}")

    results: Dict[str, Any] = {"baseline": {}, "repartition": {}, "coalesce": {}}

    df = (
        spark.range(n_rows)
        .withColumnRenamed("id", "row_id")
        .withColumn("group_id", (F.col("row_id") % group_count).cast("int"))
        .withColumn("value", F.rand(seed=42))
    )

    print("Warm-up: quick count() для инициализации Spark UI и JIT ...")
    try:
        df.limit(1000).count()
    except Exception as e:
        print("Warning: warm-up count failed:", e)

    def groupby_job(df_local):
        return df_local.groupBy("group_id").agg(F.sum("value").alias("total")).count()

    print("\nRunning baseline (as-is, без repartition/coalesce)...")
    try:
        _, baseline_time = run_and_time(lambda: groupby_job(df))
        baseline_metrics = fetch_spark_stage_metrics(spark)
    except Exception as e:
        print("Baseline job error:", e)
        baseline_time = None
        baseline_metrics = {}
    results["baseline"] = {
        "time_s": baseline_time,
        "metrics": baseline_metrics,
        "num_partitions": df.rdd.getNumPartitions()
    }
    print(f"Baseline: time_s={baseline_time}, num_partitions={results['baseline']['num_partitions']}, metrics={baseline_metrics}")

    for p in partitions_list:
        print(f"\n--- Test: repartition({p}) ---")
        try:
            df_re = df.repartition(p)
            actual_p = df_re.rdd.getNumPartitions()
            print(f"Actual partitions after repartition: {actual_p}")
            _, t = run_and_time(lambda: groupby_job(df_re))
            metrics = fetch_spark_stage_metrics(spark)
            results["repartition"][str(p)] = {
                "time_s": t,
                "metrics": metrics,
                "num_partitions": actual_p
            }
            print(f"repartition({p}) => time_s={t:.3f}s, metrics={metrics}")
        except Exception as e:
            print(f"repartition({p}) failed: {e}")
            results["repartition"][str(p)] = {"error": str(e)}

    if test_coalesce:
        base_p = results["baseline"].get("num_partitions", df.rdd.getNumPartitions())
        print(f"\nTesting coalesce - baseline partitions = {base_p}")
        coalesce_candidates = [p for p in partitions_list if p < base_p]
        if not coalesce_candidates:
            coalesce_candidates = sorted(set([max(1, base_p // 2), max(1, base_p // 4)]))
        for p in sorted(set(coalesce_candidates)):
            print(f"\n--- Test: coalesce({p}) ---")
            try:
                df_co = df.coalesce(p)
                actual_p = df_co.rdd.getNumPartitions()
                print(f"Actual partitions after coalesce: {actual_p}")
                _, t = run_and_time(lambda: groupby_job(df_co))
                metrics = fetch_spark_stage_metrics(spark)
                results["coalesce"][str(p)] = {
                    "time_s": t,
                    "metrics": metrics,
                    "num_partitions": actual_p
                }
                print(f"coalesce({p}) => time_s={t:.3f}s, metrics={metrics}")
            except Exception as e:
                print(f"coalesce({p}) failed: {e}")
                results["coalesce"][str(p)] = {"error": str(e)}

    print("\n--- Partitioning experiment finished ---")
    print("Summary (baseline + repartition + coalesce):")
    try:
        print(f"Baseline time_s: {results['baseline']['time_s']}")
        for p, info in results["repartition"].items():
            if "time_s" in info:
                print(f"repartition({p}) time_s: {info['time_s']}")
            else:
                print(f"repartition({p}) error: {info.get('error')}")
        for p, info in results["coalesce"].items():
            if "time_s" in info:
                print(f"coalesce({p}) time_s: {info['time_s']}")
            else:
                print(f"coalesce({p}) error: {info.get('error')}")
    except Exception:
        pass

    return results

def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark optimization experiments (broadcast, skew, repartition).")
    parser.add_argument("--experiment", choices=["broadcast", "skew", "repartition", "all"], default="all", help="Which experiment to run")
    parser.add_argument("--orders", type=int, default=2_000_000, help="Number of orders for broadcast experiment")
    parser.add_argument("--countries", type=int, default=200, help="Number of countries for broadcast experiment")
    parser.add_argument("--skew_rows", type=int, default=3_000_000, help="Rows for skew experiment")
    parser.add_argument("--skew_fraction", type=float, default=0.9, help="Fraction for hot key in skew experiment")
    parser.add_argument("--salts", type=int, default=10, help="Number of salts for salting")
    parser.add_argument("--repartition_rows", type=int, default=1_000_000, help="Rows for repartition experiment")
    args = parser.parse_args(argv)

    spark = build_spark("spark_opt_experiments")
    try:
        summary = {}
        if args.experiment in ("broadcast", "all"):
            summary["broadcast"] = experiment_broadcast_join(spark, n_orders=args.orders, n_countries=args.countries)
        if args.experiment in ("skew", "all"):
            summary["skew"] = experiment_skew_and_salting(spark, n_rows=args.skew_rows, skew_fraction=args.skew_fraction, salts=args.salts)
        if args.experiment in ("repartition", "all"):
            summary["repartition"] = experiment_partitions_repartition_coalesce(spark, n_rows=args.repartition_rows)

        out_path = "results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"\nРезультаты записаны в {out_path}")
        print("Spark UI (http://localhost:4040) для подробной информации по стадиям/таскам.")
        input()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()