import csv
import random
from collections import defaultdict


class Dataset:

    def __init__(self, ratings):
        """
        Constructs `Dataset` object used to load data from data sources and
        get train sets and sets for tests

        Arguments:
        `ratings` -- list of tuples (user_id, item_id, rating)
        """
        self._ratings = ratings

    @classmethod
    def from_csv_file(cls, csv_file_name):
        with open(csv_file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip headers

            ratings = [(int(row[0]), int(row[1]), float(row[2]))
                       for row in reader]

        return cls(ratings)

    def split_into_train_and_test_sets(self, train_share, random_state=1):
        ratings_count = len(self._ratings)
        trainset_ratings_count = int(train_share * ratings_count)

        random.seed(random_state)
        indexes_for_trainset = random.sample(range(ratings_count),
                                             trainset_ratings_count)
        indexes_for_trainset.sort()

        trainset_ratings = []
        testset_ratings = []

        for i, index_for_trainset in enumerate(indexes_for_trainset):
            trainset_ratings.append(self._ratings[index_for_trainset])

            testset_slice = None
            if i != trainset_ratings_count - 1:
                testset_slice = slice(index_for_trainset + 1,
                                      indexes_for_trainset[i + 1])
            else:
                testset_slice = slice(index_for_trainset + 1, ratings_count)

            testset_ratings.extend(self._ratings[testset_slice])

        return Trainset.build_from_ratings(trainset_ratings), testset_ratings

    def get_full_trainset(self):
        return Trainset.build_from_ratings(self._ratings)


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
        self._users_ratings = users_ratings
        self._items_ratings = items_ratings

    @classmethod
    def build_from_ratings(cls, ratings):
        users_ratings = defaultdict(list)
        items_ratings = defaultdict(list)

        for user_id, item_id, rating in ratings:
            users_ratings[user_id].append((item_id, rating))
            items_ratings[item_id].append((user_id, rating))

        return cls(users_ratings, items_ratings)

    @property
    def users_ratings(self):
        return self._users_ratings

    @property
    def items_ratings(self):
        return self._items_ratings
