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

    @property
    def ratings(self):
        return self._ratings

    def split_into_train_and_test_sets(self, train_share, random_state=1):
        ratings_count = len(self._ratings)
        trainset_ratings_count = int(train_share * ratings_count)

        random.seed(random_state)
        indexes_for_trainset = random.sample(range(ratings_count),
                                             trainset_ratings_count)
        indexes_for_trainset.sort()

        trainset_ratings = [None] * trainset_ratings_count
        testset_ratings = [None] * (ratings_count - trainset_ratings_count)

        trainset_index, testset_index = 0, 0
        for i, index_for_trainset in enumerate(indexes_for_trainset):
            trainset_ratings[trainset_index] = self._ratings[index_for_trainset]
            trainset_index += 1

            testset_slice = None
            items_count = 0
            if i != trainset_ratings_count - 1:
                testset_slice = slice(index_for_trainset + 1,
                                      indexes_for_trainset[i + 1])
                items_count = indexes_for_trainset[i + 1] - \
                    (index_for_trainset + 1)
            else:
                testset_slice = slice(index_for_trainset + 1, ratings_count)
                items_count = ratings_count - (index_for_trainset + 1)

            testset_ratings[testset_index:testset_index + items_count] = \
                self._ratings[testset_slice]
            testset_index += items_count

        return Trainset.build_from_ratings(trainset_ratings), testset_ratings

    def get_full_trainset(self):
        return Trainset.build_from_ratings(self._ratings)


class UnknownIdError(Exception):
    pass


class Trainset:
    def __init__(self, users_ratings, items_ratings, original_to_inner_users, original_to_inner_items):
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

        self._users_count = len(users_ratings)
        self._items_count = len(items_ratings)

        self._original_to_inner_users = original_to_inner_users
        self._original_to_inner_items = original_to_inner_items

        self._inner_to_original_users = {inner_id: original_id
                                         for original_id, inner_id in original_to_inner_users.items()}

        self._inner_to_original_items = {inner_id: original_id
                                         for original_id, inner_id in original_to_inner_items.items()}

    @classmethod
    def build_from_ratings(cls, ratings):
        users_ratings = defaultdict(list)
        items_ratings = defaultdict(list)

        current_id_user, current_id_item = 0, 0
        original_to_inner_users = {}
        original_to_inner_items = {}

        for user_id, item_id, rating in ratings:
            if user_id not in original_to_inner_users:
                original_to_inner_users[user_id] = current_id_user
                current_id_user += 1

            if item_id not in original_to_inner_items:
                original_to_inner_items[item_id] = current_id_item
                current_id_item += 1

            user_inner_id = original_to_inner_users[user_id]
            item_inner_id = original_to_inner_items[item_id]

            users_ratings[user_inner_id].append((item_inner_id, rating))
            items_ratings[item_inner_id].append((user_inner_id, rating))

        return cls(users_ratings, items_ratings, original_to_inner_users, original_to_inner_items)

    @property
    def users_ratings(self):
        return self._users_ratings

    @property
    def items_ratings(self):
        return self._items_ratings

    @property
    def users_count(self):
        return self._users_count

    @property
    def items_count(self):
        return self._items_count

    def to_inner_user_id(self, original_user_id):
        if original_user_id in self._original_to_inner_users:
            return self._original_to_inner_users[original_user_id]
        else:
            raise UnknownIdError(f"User with {original_user_id} wasn't in the train set")

    def to_inner_item_id(self, original_item_id):
        if original_item_id in self._original_to_inner_items:
            return self._original_to_inner_items[original_item_id]
        else:
            raise UnknownIdError(f"Item with {original_item_id} wasn't in the train set")
