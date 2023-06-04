import os
import pandas as pd
from sklearn.impute import SimpleImputer
import re

def get_title_from_name(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

def preprocess_titanic():
    '''
    We will do the following preprocess steps, which doesn't need to be done in order:

    -Transform 'Sex' column into numerical data through a new column named 'Is_Male'. If it's 1, then it's male. If it's 0, then it's female.

    -Median imputation for 'Age'.

    -Create bins for 'Age': ['Children','Teenage','Adult','Elder'] and then make one-hot encoding.

    -Create bins for 'Fare': ['Low','Median', 'Average','High'] and then make one-hot encoding. Mode imputation.

    -Add feature 'Titles' which has the passenger's name's titles. Then, one-hot encode them. Mode imputation.

    -Since there are a lot of classes in the 'Cabin' column, it doesn't worth it to one-hot encode. So we will eliminate this column for prediction.

    -Since there are a lot of classes in the 'Ticket' column, it doesn't worth it to one-hot encode. So we will eliminate this column for prediction.

    -Create feature 'Family_Size' = 'SibSp' + 'Parch'. Mode imputation.

    -Transform 'Embarked' column into numerical data through one-hot encoding. Mode imputation.
    
    returns: Tuple with the Pandas dataframes y_train (labels), X_train and X_test (features).
    '''
    
    # get data
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    train_data = pd.read_csv(train_path, index_col = [0])
    test_data = pd.read_csv(test_path, index_col = [0])
    
    # get features and label
    features = list(test_data.columns)
    label = 'Survived'

    # separate data
    X_train = train_data[features].copy()
    y_train = train_data[label].copy()
    X_test = test_data[features].copy()
    
    X_train = X_train.drop(['Ticket', 'Cabin'], axis = 1)
    X_test = X_test.drop(['Ticket', 'Cabin'], axis = 1)
    
    X_train['Is_Male'] = X_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    X_test['Is_Male'] = X_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    X_train['Age'].fillna(X_train['Age'].median(), inplace = True)
    X_test['Age'].fillna(X_train['Age'].median(), inplace = True)

    X_train['Title'] = X_train['Name'].apply(get_title_from_name)
    X_test['Title'] = X_test['Name'].apply(get_title_from_name)

    # Replacing rare titles by 'Rare'.

    X_train['Title'] = X_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    X_test['Title'] = X_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # Transforming titles to their related:
    X_train['Title'] = X_train['Title'].replace('Mlle', 'Miss')
    X_train['Title'] = X_train['Title'].replace('Ms', 'Miss')
    X_train['Title'] = X_train['Title'].replace('Mme', 'Mrs')
    X_test['Title'] = X_test['Title'].replace('Mlle', 'Miss')
    X_test['Title'] = X_test['Title'].replace('Ms', 'Miss')
    X_test['Title'] = X_test['Title'].replace('Mme', 'Mrs')

    X_train['Age_Bin'] = pd.cut(X_train['Age'], bins = [0, 12, 20, 40, 100], labels = ['Child', 'Teen', 'Adult', 'Elder'])
    X_test['Age_Bin'] = pd.cut(X_test['Age'], bins = [0, 12, 20, 40, 100], labels = ['Child', 'Teen', 'Adult', 'Elder'])

    X_train['Fare_Bin'] = pd.cut(X_train['Fare'], bins = [0, 7.91, 14.45, 31, 550], labels = ['Low', 'Median', 'Mean', 'High'])
    X_test['Fare_Bin'] = pd.cut(X_test['Fare'], bins = [0, 7.91, 14.45, 31, 550], labels = ['Low', 'Median', 'Mean', 'High'])
    
    mode_imputer = SimpleImputer(strategy = 'most_frequent')
    impute_cols = ['Title', 'Age_Bin', 'Fare_Bin', 'Embarked']

    X_train[impute_cols] = pd.DataFrame(mode_imputer.fit_transform(X_train[impute_cols]), index = X_train.index, columns = impute_cols) 
    X_test[impute_cols] = pd.DataFrame(mode_imputer.transform(X_test[impute_cols]), index = X_test.index, columns = impute_cols)
    
    X_train = pd.get_dummies(X_train, columns = ['Title', 'Age_Bin', 'Fare_Bin', 'Embarked'],
                         prefix = ['Title', 'Age_Bin', 'Fare_Class', 'Embark_Place'])

    X_test = pd.get_dummies(X_test, columns = ['Title', 'Age_Bin', 'Fare_Bin', 'Embarked'],
                            prefix = ['Title', 'Age_Bin', 'Fare_Class', 'Embark_Place'])
    
    X_train['Family_Size'] = X_train['Parch'] + X_train['SibSp']
    X_test['Family_Size'] = X_test['Parch'] + X_test['SibSp']

    X_train = X_train.drop(['Sex', 'Name', 'Age', 'Fare'], axis = 1)
    X_test = X_test.drop(['Sex', 'Name', 'Age', 'Fare'], axis = 1)

    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean)/train_std
    X_test = (X_test - train_mean)/train_std

    return y_train, X_train, X_test