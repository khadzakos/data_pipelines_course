import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


def load_results():
    results = {}

    files = {
        'write': 'write_results.csv',
        'read': 'read_results.csv',
        'concurrent': 'concurrent_write_results.csv'
    }

    for key, filename in files.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            results[key] = pd.read_csv(filepath)
            print(f"Загружено: {filename}")

    return results


def plot_write_performance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Подготовка данных
    formats = df['format'].unique()
    fractions = sorted(df['sample_fraction'].unique())
    x = np.arange(len(fractions))
    width = 0.25

    colors = {'Parquet': '#2ecc71', 'Delta Lake': '#3498db', 'Iceberg': '#9b59b6'}

    # 1. Время записи
    for i, fmt in enumerate(formats):
        data = df[df['format'] == fmt].sort_values('sample_fraction')
        axes[0].bar(x + i*width, data['write_time_sec'], width,
                   label=fmt, color=colors.get(fmt, '#95a5a6'))

    axes[0].set_xlabel('Доля данных')
    axes[0].set_ylabel('Время (сек)')
    axes[0].set_title('Время записи')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([f'{int(f*100)}%' for f in fractions])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Размер файлов
    for i, fmt in enumerate(formats):
        data = df[df['format'] == fmt].sort_values('sample_fraction')
        axes[1].bar(x + i*width, data['size_mb'], width,
                   label=fmt, color=colors.get(fmt, '#95a5a6'))

    axes[1].set_xlabel('Доля данных')
    axes[1].set_ylabel('Размер (MB)')
    axes[1].set_title('Размер хранилища')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f'{int(f*100)}%' for f in fractions])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Скорость записи (строк/сек)
    df_copy = df.copy()
    df_copy['rows_per_sec'] = df_copy['rows'] / df_copy['write_time_sec']

    for i, fmt in enumerate(formats):
        data = df_copy[df_copy['format'] == fmt].sort_values('sample_fraction')
        axes[2].bar(x + i*width, data['rows_per_sec'] / 1000, width,
                   label=fmt, color=colors.get(fmt, '#95a5a6'))

    axes[2].set_xlabel('Доля данных')
    axes[2].set_ylabel('Тысяч строк/сек')
    axes[2].set_title('Скорость записи')
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels([f'{int(f*100)}%' for f in fractions])
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'write_performance.png'), dpi=150)
    plt.close()
    print(f"Сохранено: write_performance.png")


def plot_read_performance(df: pd.DataFrame):
    # Агрегируем по формату и операции
    summary = df.groupby(['format', 'operation'])['read_time_sec'].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'Parquet': '#2ecc71', 'Delta Lake': '#3498db', 'Iceberg': '#9b59b6'}
    operations = ['full_scan', 'filtered_scan', 'aggregation']
    titles = ['Полное сканирование', 'Фильтрованное чтение', 'Агрегация']

    for idx, (op, title) in enumerate(zip(operations, titles)):
        data = summary[summary['operation'] == op]
        bars = axes[idx].bar(data['format'], data['read_time_sec'],
                            color=[colors.get(f, '#95a5a6') for f in data['format']])

        axes[idx].set_title(title)
        axes[idx].set_ylabel('Время (сек)')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

        # Значения над столбцами
        for bar, val in zip(bars, data['read_time_sec']):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'read_performance.png'), dpi=150)
    plt.close()
    print(f"Сохранено: read_performance.png")


def plot_concurrent_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'Delta Lake': '#3498db', 'Iceberg': '#9b59b6'}

    # 1. Успешные vs неудачные записи
    summary = df.groupby('format')['status'].value_counts().unstack(fill_value=0)

    formats = summary.index.tolist()
    success = summary.get('success', pd.Series([0]*len(formats))).values
    failed = summary.get('failed', pd.Series([0]*len(formats))).values

    x = np.arange(len(formats))
    width = 0.35

    axes[0].bar(x - width/2, success, width, label='Успешно', color='#2ecc71')
    axes[0].bar(x + width/2, failed, width, label='Ошибка', color='#e74c3c')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(formats)
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Результаты конкурентной записи')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Среднее время успешных записей
    success_df = df[df['status'] == 'success']
    if len(success_df) > 0:
        avg_time = success_df.groupby('format')['time_sec'].mean()
        bars = axes[1].bar(avg_time.index, avg_time.values,
                          color=[colors.get(f, '#95a5a6') for f in avg_time.index])

        axes[1].set_ylabel('Время (сек)')
        axes[1].set_title('Среднее время записи (успешные)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, avg_time.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'concurrent_performance.png'), dpi=150)
    plt.close()
    print(f"Сохранено: concurrent_performance.png")


def plot_size_comparison(df: pd.DataFrame):
    full_data = df[df['sample_fraction'] == 1.0].copy()

    if len(full_data) == 0:
        print("Нет данных для 100% записи")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'Parquet': '#2ecc71', 'Delta Lake': '#3498db', 'Iceberg': '#9b59b6'}
    bar_colors = [colors.get(f, '#95a5a6') for f in full_data['format']]

    bars = ax.bar(full_data['format'], full_data['size_mb'], color=bar_colors)

    ax.set_title('Сравнение размера хранилища (1M строк)')
    ax.set_ylabel('Размер (MB)')
    ax.grid(axis='y', alpha=0.3)

    # Значения над столбцами
    for bar, size in zip(bars, full_data['size_mb']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Добавляем относительные значения
    parquet_size = full_data[full_data['format'] == 'Parquet']['size_mb'].values
    if len(parquet_size) > 0:
        parquet_size = parquet_size[0]
        for bar, size in zip(bars, full_data['size_mb']):
            ratio = size / parquet_size
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{ratio:.2f}x', ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'size_comparison.png'), dpi=150)
    plt.close()
    print(f"Сохранено: size_comparison.png")


def generate_all_plots():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = load_results()

    if not results:
        print("Нет данных для визуализации. Сначала запустите тесты.")
        return

    if 'write' in results and len(results['write']) > 0:
        print("\nГенерация графиков записи...")
        plot_write_performance(results['write'])
        plot_size_comparison(results['write'])

    if 'read' in results and len(results['read']) > 0:
        print("Генерация графиков чтения...")
        plot_read_performance(results['read'])

    if 'concurrent' in results and len(results['concurrent']) > 0:
        print("Генерация графиков конкурентной записи...")
        plot_concurrent_results(results['concurrent'])

    print(f"\nВсе графики сохранены в: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
