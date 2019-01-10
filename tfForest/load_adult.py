from feature_engineering import FeatureParser
import numpy as np


def load_adult(one_hot=True):
    """
    Load UCI ADULT data. If not exists in data/, download it and put into data_base_dir.

    :param one_hot: whether use one-hot encoding
    :return: X_train, y_train, X_test, y_test
    """
    data_dir = "../DeepForestTF_Data/"
    train_path = data_dir + 'adult.data'
    test_path = data_dir + 'adult.test'
    features_path = data_dir + 'features'
    feature_parsers = []
    with open(features_path) as f:
        for row in f.readlines():
            feature_parsers.append(FeatureParser(row))
    x_train, y_train = load_util(train_path, feature_parsers, one_hot)
    x_test, y_test = load_util(test_path, feature_parsers, one_hot)
    return x_train, x_test, y_train, y_test


def load_util(data_path, feature_parsers, one_hot):
    """
    UCI ADULT dataset load utility.

    :param data_path: path of data to load
    :param feature_parsers: feature parsers to parse every single feature
    :param one_hot: whether use one-hot encoding
    :return: X, y
    """
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
        n_train = len(rows)
        if one_hot:
            train_dim = np.sum([f_parser.get_featuredim() for f_parser in feature_parsers])
            X = np.zeros((n_train, train_dim), dtype=np.float32)
        else:
            X = np.zeros((n_train, 1), dtype=np.float32)
        y = np.zeros(n_train, dtype=np.float32)
        for i, row in enumerate(rows):
            assert len(row) != 14, "len(row) wrong, i={}".format(i)
            f_offset = 0
            for j in range(14):
                if one_hot:
                    f_dim = feature_parsers[j].get_featuredim()
                    X[i, f_offset:f_offset + f_dim] = feature_parsers[j].get_data(row[j].strip())
                    f_offset += f_dim
                else:
                    X[i, j] = feature_parsers[j].get_continuous(row[j].strip())
            y[i] = 0 if row[-1].strip().startswith("<=50K") else 1
        return X, y

