import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def preprocess_houses():

    # import data
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')

    train_data = pd.read_csv(train_path, index_col = 'Id')
    test_data = pd.read_csv(test_path, index_col = 'Id')

    # separate data

    features = test_data.columns
    label = 'SalePrice'

    X_train = train_data[features].copy()
    y_train = train_data[label].copy()
    X_test = test_data[features].copy()

    categorical = list(X_train.select_dtypes(['object']).columns)
    numerical = list(set(features).difference(set(categorical)))

    # numerical scaling and imputation

    scaler = StandardScaler()

    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    X_test[numerical] = scaler.transform(X_test[numerical])

    imputer = SimpleImputer(strategy = 'mean')

    X_train[numerical] = pd.DataFrame(imputer.fit_transform(X_train[numerical]), columns = numerical, index = X_train.index)
    X_test[numerical] = pd.DataFrame(imputer.transform(X_test[numerical]), columns = numerical, index = X_test.index)

    # categorical imputation

    imputer = SimpleImputer(strategy = 'most_frequent')

    X_train[categorical] = pd.DataFrame(imputer.fit_transform(X_train[categorical]), columns = categorical, index = X_train.index)
    X_test[categorical] = pd.DataFrame(imputer.transform(X_test[categorical]), columns = categorical, index = X_test.index)

    # categorical encoding

    categorical_description = X_train[categorical].describe()

    unique2 = categorical_description.loc['unique'] == 2
    unique2 = unique2[unique2 == True].index

    ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
    X_train[unique2] = ordinal_encoder.fit_transform(X_train[unique2])
    X_test[unique2] = ordinal_encoder.transform(X_test[unique2])

    uniquegr2 = categorical_description.loc['unique'] > 2
    uniquegr2 = uniquegr2[uniquegr2 == True].index

    ordered_classes = ['SaleCondition', 'Fence', 'PoolQC', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu', 'Functional',
                  'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual', 'Condition1', 'Condition2']

    unordered_classes = list(set(uniquegr2).difference(ordered_classes))

    onehot_encoder = OneHotEncoder(handle_unknown = 'infrequent_if_exist', min_frequency = 10)
    onehot_encoder.fit(X_train[unordered_classes])

    train_encoded = onehot_encoder.transform(X_train[unordered_classes]).toarray()
    train_encoded = pd.DataFrame(train_encoded, columns = onehot_encoder.get_feature_names_out(), index = X_train.index)

    test_encoded = onehot_encoder.transform(X_test[unordered_classes]).toarray()
    test_encoded = pd.DataFrame(test_encoded, columns = onehot_encoder.get_feature_names_out(), index = X_test.index)

    X_train = pd.concat([X_train, train_encoded], axis = 1)
    X_test = pd.concat([X_test, test_encoded], axis = 1)

    X_train = X_train.drop(columns = unordered_classes)
    X_test = X_test.drop(columns = unordered_classes)

    ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
    X_train[ordered_classes] = ordinal_encoder.fit_transform(X_train[ordered_classes])
    X_test[ordered_classes] = ordinal_encoder.transform(X_test[ordered_classes])
    # categorical scaling

    scaler = MinMaxScaler()
    to_scale = list(set(X_train.columns).difference(numerical))

    X_train[to_scale] = scaler.fit_transform(X_train[to_scale])
    X_test[to_scale] = scaler.transform(X_test[to_scale])

    return y_train, X_train, X_test