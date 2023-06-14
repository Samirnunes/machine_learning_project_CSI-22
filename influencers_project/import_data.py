import os
import pandas as pd

def import_data():
    '''
    Imports Influencers in Social Networks Kaggle's competition data.
    It gets train_data, test_data, y_train, X_train and X_test Pandas dataframes.
    train_data is all the train data, test_data is all the test data,
    y_train in only the train_data label column, X_train is only the train_data feature columns
    and X_test is the test_data feature columns, which is equal to test_data.

    returns: tuple of Pandas dataframes (train_data, test_data, y_train, X_train, X_test).
    '''
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    label = 'Choice'
    features = test_data.columns

    X_train = train_data[features]
    y_train = train_data[label]
    X_test = test_data

    return train_data, test_data, y_train, X_train, X_test    