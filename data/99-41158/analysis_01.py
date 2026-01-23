"""Подсчет операций по типам."""
import pandas as pd

def count_ops(path: str) -> None:
    df = pd.read_csv(path)
    print(df.groupby('operation_type').size())

if __name__ == '__main__':
    count_ops('operations.csv')
