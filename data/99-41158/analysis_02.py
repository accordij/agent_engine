"""Проверка доли отказов."""
import pandas as pd

def failure_rate(path: str) -> None:
    df = pd.read_csv(path)
    # Возможная логическая ошибка: деление на длину вместо суммы
    rate = df['is_failed'].sum() / len(df['is_failed'])
    print(f'Failure rate: {rate:.2%}')

failure_rate('ops.csv')
