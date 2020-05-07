from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class MF(nn.Module):

    def __init__(self, users_count, items_count, features_count):
        super().__init__()

        self.users_emb = nn.Embedding(users_count, features_count)
        self.items_emb = nn.Embedding(items_count, features_count)

        self.users_emb.weight.data.uniform_(0, 0.05)
        self.items_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.users_emb(u)
        v = self.items_emb(v)
        return (u * v).sum(1)


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
        df[column_name] = col
        df = df[df[column_name] >= 0]
    return df


def show_losses_plot(train_losses, test_losses):
    plt.title('Mean squared error')
    plt.xlabel('number of epochs')
    plt.ylabel('loss')

    plt.plot(range(len(train_losses)), train_losses, label='Train losses')
    plt.scatter(list(map(lambda x: x[0], test_losses)), list(map(lambda x: x[1], test_losses)), c=[[1, 0, 0]], label='Test losses')

    plt.legend()
    plt.show()


def test_loss(model, test_data):
    model.eval()

    users = torch.LongTensor(test_data['userId'].values)
    items = torch.LongTensor(test_data['movieId'].values)
    ratings = torch.FloatTensor(test_data['rating'].values)

    y_hat = model(users, items)
    loss = nnf.mse_loss(y_hat, ratings)

    return loss


def train_model(model, train_data, epochs=10, learning_rate=0.01, weight_decay=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()

    train_losses = []

    for i in range(epochs):
        users = torch.LongTensor(train_data['userId'].values)
        items = torch.LongTensor(train_data['movieId'].values)
        ratings = torch.FloatTensor(train_data['rating'].values)

        y_hat = model(users, items)

        loss = nnf.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return train_losses


if __name__ == "__main__":
    data = pd.read_csv('data/ratings.csv')

    np.random.seed(42)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk].copy()
    test = data[~msk].copy()

    train_encoded = encode_data(train)
    test_encoded = encode_data(test, train)

    users_count = len(train_encoded['userId'].unique())
    items_count = len(train_encoded['movieId'].unique())

    model = MF(users_count, items_count, 100)
    loss = test_loss(model, test_encoded)
    print(f'Test loss: {loss.item()}')
    print()

    epochs = [10, 15, 15, 15]
    learning_rates = [0.1, 0.01, 0.01, 0.01]
    test_losses = [(0, loss.item())]
    all_train_losses = []

    for i, epoch_count in enumerate(epochs):
        train_losses = train_model(model, train_encoded, epoch_count, learning_rates[i])
        all_train_losses.extend(train_losses)

        loss = test_loss(model, test_encoded)
        print(f'Test loss: {loss.item()}')
        print()

        test_losses.append((sum(epochs[:i + 1]), loss.item()))

    show_losses_plot(all_train_losses, test_losses)
