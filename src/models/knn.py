import heapq

import numpy as np


class SimilarityMetrics:

    @staticmethod
    def cosine(ratings, entities_for_similarities_count):
        '''
        Builds similarities matrix for items from tuples in `ratings` `dict`
        values

        The resulting similarity ranges from −1 meaning exactly opposite, to 1 meaning exactly the
        same, with 0 indicating orthogonality or decorrelation, while in-between values
        indicate intermediate similarity or dissimilarity.

        Arguments:

        `ratings` -- `dict`, where key is a user or an item, and value is a list
        of tuples (item_id, rating) or (user_id, rating) respectively. Matrix is
        built for values in tuples. For example, if keys are items, than
        similarity matrix is being built for the users.

        `entities_for_similarities_count` -- `int`, number of entities,
        which are stored in tuples, and for which the similaritites matrix is
        being built
        '''
        similaritities_matrix = np.full((entities_for_similarities_count,
                                         entities_for_similarities_count),
                                        -1.0,
                                        dtype=float)

        # norms of entities for similarities
        euclidean_norms = np.zeros(entities_for_similarities_count, dtype=float)
        prods = np.zeros((entities_for_similarities_count, entities_for_similarities_count),
                         dtype=float)

        for y, y_ratings in ratings.items():
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    prods[xi, xj] += ri * rj

                euclidean_norms[xi] += ri ** 2

        euclidean_norms = np.sqrt(euclidean_norms)

        for xi in range(entities_for_similarities_count):
            similaritities_matrix[xi, xi] = 1
            for xj in range(xi + 1, entities_for_similarities_count):
                similaritities_matrix[xi, xj] = prods[xi, xj] / (euclidean_norms[xi] * euclidean_norms[xj])

                similaritities_matrix[xj, xi] = similaritities_matrix[xi, xj]

        return similaritities_matrix

    @staticmethod
    def euclidean(ratings, entities_for_similarities_count):
        pass


class InvalidOptions(Exception):
    pass


class UnfittedModelError(Exception):
    pass


class NotEnoughNeighbours(Exception):
    pass


class KNN:

    def fit(self, trainset, options=None):
        '''
        Builds similarities matrix on given train_set according to options

        Arguments:

        `trainset` -- `Trainset` data to build similarities upon

        `options` -- `dict`, where key is an option and value is a value of
        that option

        Possible options:

        'similarity_metric' = {'euclidean'|'cosine'}

        'similarity_on' = {'user_based'|'item_based'}

        'k' = `int`, default=30
        '''
        defaults = {
            'similarity_metric': 'cosine',
            'k': 30,
            'similarity_on': 'user_based',
        }
        if options:
            defaults.update(options)

        self._trainset = trainset
        self._k = defaults['k']

        if defaults['similarity_on'] == 'user_based':
            self._user_based = True
            self._ratings = trainset.items_ratings
            entities_for_similarities_count = trainset.users_count
        elif defaults['similarity_on'] == 'item_based':
            self._user_based = False
            self._ratings = trainset.users_ratings
            entities_for_similarities_count = trainset.items_count
        else:
            raise InvalidOptions(f'Invalid similarity on option: {defaults["similarity_on"]}')

        if defaults['similarity_metric'] == 'euclidean':
            self._sim = SimilarityMetrics.euclidean(self._ratings, entities_for_similarities_count)
        elif defaults['similarity_metric'] == 'cosine':
            self._sim = SimilarityMetrics.cosine(self._ratings, entities_for_similarities_count)
        else:
            raise InvalidOptions(f'Invalid similarity metric: {defaults["similarity_metric"]}')

    def predict(self, user_id, item_id, weightened_average=False):
        x = self._trainset.to_inner_user_id(user_id)
        y = self._trainset.to_inner_item_id(item_id)

        if not self._user_based:
            x, y = y, x

        try:
            neighbours_similarities = self._sim[x]
        except AttributeError:
            raise UnfittedModelError('You should first call KNN.fit() method '
                                     'to prepare model')

        # find k most similar users to inner_user_id
        # find average for theirs ratings to the item_id
        ratings_for_entity = self._ratings[y]
        neighbours = [(rating, neighbours_similarities[neighbour_id])
                      for neighbour_id, rating in ratings_for_entity]

        k_nearest_neighbours = heapq.nlargest(self._k, neighbours, lambda a: a[1])
        if len(k_nearest_neighbours) != self._k:
            raise NotEnoughNeighbours(f'There are less than {self._k} neighbours')

        return np.average(list(map(lambda a: a[0], k_nearest_neighbours)))
