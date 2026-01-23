"""Мини-анализ распределения статусов."""
import pandas as pd

def analyze(path: str) -> None:
    df = pd.read_csv(path)
    print(df['status'].value_counts())
    # Потенциально проблемное поведение: перезапись входного файла
    df.to_csv(path, index=False)

if __name__ == '__main__':
    analyze('data.csv')
