import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv("C:/Program Files/JetBrains/python pycharm projects/IRIS/IRIS.csv")

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
nb=MultinomialNB()
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)


X= df.drop(['species'], axis=1)
print(X)

Y= df["species"]
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1, test_size=0.2)

'''Applying algorithm and training dataset'''
rf.fit(X_train, Y_train)
lr.fit(X_train, Y_train)
gbm.fit(X_train, Y_train)
dt.fit(X_train, Y_train)
nb.fit(X_train, Y_train)
sv.fit(X_train, Y_train)
nn.fit(X_train, Y_train)

'''Testing dataset'''
Y_pred1= rf.predict(X_test)
Y_pred2= lr.predict(X_test)
Y_pred3= gbm.predict(X_test)
Y_pred4= dt.predict(X_test)
Y_pred5= nb.predict(X_test)
Y_pred6= sv.predict(X_test)
Y_pred7= nn.predict(X_test)


print("Random Forest: ", accuracy_score(Y_test, Y_pred1))
print("Logistic Regression: ", accuracy_score(Y_test, Y_pred2))
print("Gradient Boosting Classifier: ", accuracy_score(Y_test, Y_pred3))
print("Decision Tree: ", accuracy_score(Y_test, Y_pred4))
print("Naive Bayes: ", accuracy_score(Y_test, Y_pred5))
print("Support Vector: ", accuracy_score(Y_test, Y_pred6))
print("Neural Networks: ", accuracy_score(Y_test, Y_pred7))


