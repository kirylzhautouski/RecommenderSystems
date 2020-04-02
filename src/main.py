from models.utils import Dataset
from models.knn import KNN


if __name__ == '__main__':
    dataset = Dataset.from_csv_file('data/ratings.csv')

    trainset, testset = dataset.split_into_train_and_test_sets(0.7)

    knn = KNN()
    knn.fit(trainset)
    knn.predict(1, 1)
