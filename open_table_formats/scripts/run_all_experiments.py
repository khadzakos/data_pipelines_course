import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NUM_ROWS, WRITE_FRACTIONS


def main():
    print("="*70)
    print("LAKEHOUSE BENCHMARKS: Parquet vs Delta Lake vs Iceberg")
    print("="*70)

    print("\n" + "="*70)
    print("[1/5] ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    print("="*70)
    from generate_data import generate_test_data, save_source_data
    df = generate_test_data(num_rows=NUM_ROWS)
    save_source_data(df)

    # Шаг 2: Тесты записи
    print("\n" + "="*70)
    print("[2/5] ТЕСТЫ ЗАПИСИ")
    print("="*70)
    from test_write import run_write_tests, save_results as save_write_results
    all_write_results = []
    for fraction in WRITE_FRACTIONS:
        results = run_write_tests(sample_fraction=fraction)
        all_write_results.extend(results)
    write_df = save_write_results(all_write_results)
    print("\nИтоговая таблица записи:")
    print(write_df.to_string(index=False))

    # Шаг 3: Тесты чтения
    print("\n" + "="*70)
    print("[3/5] ТЕСТЫ ЧТЕНИЯ")
    print("="*70)
    from test_read import run_read_tests, save_results as save_read_results
    read_results = run_read_tests()
    save_read_results(read_results)

    # Шаг 4: Тесты конкурентной записи
    print("\n" + "="*70)
    print("[4/5] ТЕСТЫ КОНКУРЕНТНОЙ ЗАПИСИ")
    print("="*70)
    from test_concurrent_write import run_all_concurrent_tests
    run_all_concurrent_tests()

    # Шаг 5: Визуализация
    print("\n" + "="*70)
    print("[5/5] ГЕНЕРАЦИЯ ГРАФИКОВ")
    print("="*70)
    from visualize_results import generate_all_plots
    generate_all_plots()

    print("\n" + "="*70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*70)
    print("\nФайлы результатов:")
    print("  - results/write_results.csv")
    print("  - results/read_results.csv")
    print("  - results/concurrent_write_results.csv")
    print("  - results/*.png (графики)")
    print("\nЗаполните отчет: REPORT.md")


if __name__ == "__main__":
    main()
