import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

x_df = pd.read_csv("C:/Users/User/PycharmProjects/pycharm myproject/IRIS/IRIS.csv")
# print(x_train)
y_df = pd.read_csv("C:/Users/User/PycharmProjects/pycharm myproject/IRIS/IRIS.csv")

x=x_df.drop(columns= ["species"],axis=1)
print(x)

y=y_df['species']
print(y)

model = ExtraTreesClassifier()
model.fit(x,y)

print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(4).plot(kind = 'barh')
#feat_importance.nlargest(4).plot(kind = 'kde')

plt.show()

