from models.utils import Dataset


if __name__ == '__main__':
    dataset = Dataset.from_csv_file('data/ratings.csv')
    train, test = dataset.split_into_train_and_test_sets(0.999)
    print(test)
