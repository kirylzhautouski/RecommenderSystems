from models.utils import Dataset


if __name__ == '__main__':
    dataset = Dataset.from_csv_file('data/ratings.csv')
