"""Проверка статистики транзакций."""
import pandas as pd

def summarize(path: str) -> None:
    df = pd.read_csv(path)
    avg = df['amount'].mean()
    print(f'Average amount: {avg}')
    print('Top client:', top_client)  # Ошибка: переменная не определена

summarize('transactions.csv')
