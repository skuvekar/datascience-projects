import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error



train=pd.read_csv('C:/Users/User/PycharmProjects/pycharm myproject/Black Friday Sales/train.csv')
test=pd.read_csv('C:/Users/User/PycharmProjects/pycharm myproject/Black Friday Sales/test.csv')

train['User_ID'] = train['User_ID'] - 1000000
test['User_ID'] = test['User_ID'] - 1000000

enc = LabelEncoder()
train['User_ID'] = enc.fit_transform(train['User_ID'])
test['User_ID'] = enc.transform(test['User_ID'])

train = train.fillna(0)
test = test.fillna(0)

cat_col = ['Gender', 'City_Category']
num_col = ['Age', 'Occupation', 'Stay_In_Current_City_Years', 'Product_Category_1',
           'Product_Category_2', 'Product_Category_3']

train['Age'] = train['Age'].map({'0-17': 15,
                               '18-25': 21,
                               '26-35': 30,
                               '36-45': 40,
                               '46-50': 48,
                               '51-55': 53,
                               '55+': 60})
test['Age'] = test['Age'].map({'0-17': 15,
                               '18-25': 21,
                               '26-35': 30,
                               '36-45': 40,
                               '46-50': 48,
                               '51-55': 53,
                               '55+': 60})

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].map({'0': 0,
                                                                               '1': 1,
                                                                                '2': 2,
                                                                                '3': 3,
                                                                                '4+': 4})
test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].map({'0': 0,
                                                                               '1': 1,
                                                                                '2': 2,
                                                                                '3': 3,
                                                                                '4+': 4})

encoder = LabelEncoder()

for col in cat_col:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])

X = train.drop(['Purchase','Product_ID'], axis=1)
y = train[['Purchase']]
X_test = test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

scaler = StandardScaler()

for col in num_col:
    train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))

xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print(rmse)






