import csv
from collections import defaultdict


class Dataset:

    def __init__(self, ratings):
        """
        Constructs `Dataset` object used to load data from data sources and
        get train sets and sets for tests

        Arguments:
        `ratings` -- list of tuples (user_id, item_id, rating)
        """
        self.ratings = ratings

    @classmethod
    def from_csv_file(cls, csv_file_name):
        with open(csv_file_name, 'r') as csv_file:
            ratings = [(row[0], row[1], row[2])
                       for row in csv.reader(csv_file)]

        return cls(ratings)

    def split_into_train_and_test_sets(train_share):
        pass

    def get_full_trainset():
        pass


class Trainset:
    def __init__(self, users_ratings, items_ratings):
        """
        Constructs `Trainset`, that is used for training model

        Arguments:
        `users_ratings` -- dict where key is a user_id and value is a list of
        tuples (item_id, rating)
        `items_ratings` -- dict where key is an item_id and value is a list of
        tuples (user_id, rating)
        """
        self.users_ratings = users_ratings
        self.items_ratings = items_ratings

    @classmethod
    def build_from_ratings(cls, ratings):
        users_ratings = defaultdict(list)
        items_ratings = defaultdict(list)

        for (user_id, item_id, rating) in ratings:
            users_ratings[user_id].append((item_id, rating))
            items_ratings[item_id].append((user_id, rating))

        return cls(users_ratings, items_ratings)
