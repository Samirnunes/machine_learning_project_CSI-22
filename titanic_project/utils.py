import os
import pandas as pd

def preprocess_titanic():
    '''
    We will do the following preprocess steps, which doesn't need to be done in order:

    1) Transform 'Sex' column into numerical data through a new column named 'is_male'. If it's 1, then it's male. If it's 0, then it's female.

    2) Standard scaling on 'Age' column.

    3) Standard scaling on 'Parch' column.

    4) Since there are a lot of classes in the 'Ticket' column, it doesn't worth it to one-hot encode. So we will eliminate this column for prediction.

    5) Standard scaling on 'Fare' column.

    6) Since there are a lot of classes in the 'Cabin' column, it doesn't worth it to one-hot encode. So we will eliminate this column for prediction.

    7) Transform 'Embarked' column into numerical data through one-hot encoding.
    
    returns: Tuple with the Pandas dataframes y_train (labels), X_train and X_test (features).
    '''
    
    # get data
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    train_data = pd.read_csv(train_path, index_col = [0])
    test_data = pd.read_csv(test_path, index_col = [0])
    
    # get features and label
    features = list(test_data.columns)
    features.remove('Name')
    label = 'Survived'

    # separate data
    X_train = train_data[features]
    y_train = train_data[label]
    X_test = test_data[features]
    
    # steps 4 and 7
    X_train = X_train.drop(['Ticket', 'Cabin'], axis = 1)
    X_test = X_test.drop(['Ticket', 'Cabin'], axis = 1)
    
    # step 
    X_train['is_male'] = X_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    X_test['is_male'] = X_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    X_train = X_train.drop('Sex', axis = 1)
    X_test = X_test.drop('Sex', axis = 1)
    
    # steps 2, 3 and 5
    age_mean = X_train['Age'].mean()
    age_std = X_train['Age'].std()

    X_train['Age'] = (X_train['Age'] - age_mean) / age_std
    X_test['Age'] = (X_test['Age'] - age_mean) / age_std

    parch_mean = X_train['Parch'].mean()
    parch_std = X_train['Parch'].std()

    X_train['Parch'] = (X_train['Parch'] - parch_mean) / parch_std
    X_test['Parch'] = (X_test['Parch'] - parch_mean) / parch_std

    fare_mean = X_train['Fare'].mean()
    fare_std = X_train['Fare'].std()

    X_train['Fare'] = (X_train['Fare'] - fare_mean) / fare_std
    X_test['Fare'] = (X_test['Fare'] - fare_mean) / fare_std
    
    # step 7 
    embarked_1hot_train = pd.get_dummies(X_train['Embarked'])
    embarked_1hot_test = pd.get_dummies(X_test['Embarked'])

    X_train = pd.concat([X_train, embarked_1hot_train], axis = 1)
    X_train.rename(columns = {'C': 'embarked_cherbourg', 'Q': 'embarked_queenstown', 'S': 'embarked_southampton'}, inplace = True)
    X_test = pd.concat([X_test, embarked_1hot_test], axis = 1)
    X_test.rename(columns = {'C': 'embarked_cherbourg', 'Q': 'embarked_queenstown', 'S': 'embarked_southampton'}, inplace = True)

    X_train = X_train.drop('Embarked', axis = 1)
    X_test = X_test.drop('Embarked', axis = 1)
    
    # renaming columns for standardizing
    
    X_train.rename(columns = {'Pclass': 'pclass', 'Age': 'age', 'SibSp': 'sibsp', 'Parch': 'parch', 'Fare': 'fare'}, inplace = True)
    X_test.rename(columns = {'Pclass': 'pclass', 'Age': 'age', 'SibSp': 'sibsp', 'Parch': 'parch', 'Fare': 'fare'}, inplace = True)
    
    return y_train, X_train, X_test