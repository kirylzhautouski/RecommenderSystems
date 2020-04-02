

class UnfittedModelError(Exception):
    pass


class KNN:
    def fit(self, train_set, options):
        '''
        Builds similarities matrix on given train_set according to options

        Arguments:

        `train_set` -- `Trainset` data to build similarities upon

        `options` -- `dict`, where key is an option and value is a value of
        that option

        Possible options:

        'similarity_metric' = {'euclidean'|'cosine'}

        'rating_prediction' = {'average'|'weighted_average'}

        'similarity_on' = {'user_based'|'item_based'}
        '''
        self.sim = []

    def predict(self, user_id, item_id):
        try:
            pass
        except AttributeError:
            raise UnfittedModelError('You should first call KNN.fit() method '
                                     'to prepare model')
