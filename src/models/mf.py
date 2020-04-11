import pandas as pd
import numpy as np


class MF:
    pass


def convert_ids(column, train_column=None):
    if train_column is not None:
        unique = train_column.unique()
    else:
        unique = column.unique()
    old_to_new = {o: n for n, o in enumerate(unique)}

    return old_to_new, np.array([old_to_new.get(x, -1) for x in column]), len(unique)


def encode_data(df, train=None):
    df = df.copy()
    for column_name in 'userId', 'movieId':
        train_column = None
        if train is not None:
            train_column = train[column_name]
        _, col, _ = convert_ids(df[column_name], train_column)
        df = df[df[column_name] >= 0]
    return df


if __name__ == "__main__":
    data = pd.read_csv('data/ratings.csv')

    np.random.seed(42)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk].copy()
    test = data[~msk].copy()

    train_encoded = encode_data(train)
    test_encoded = encode_data(test, train)

    print(test_encoded.head())
