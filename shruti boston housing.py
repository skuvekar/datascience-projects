import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("C:/Users/User/PycharmProjects/pycharm myproject/archive/HousingData.csv")
# print(df)

lr=LinearRegression()

#hgbm=HistGradientBoostingClassifier()
for col in df.columns:
    df[col].fillna((df[col].mean()), inplace=True)


print(df)

x=df.drop(['MEDV'],axis=1)
print(x)

y=df['MEDV']
print(y)


x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
#print(accuracy_score(y_test,y_pred))
print( "mean squared error = ", mean_squared_error(y_test,y_pred))