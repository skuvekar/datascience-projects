import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

df = pd.read_csv("C:/Users/User/PycharmProjects/pycharm myproject/Census Income/adult.csv")
df = df.replace({'?': np.nan}).dropna()
le=LabelEncoder()
df['workclass']=le.fit_transform(df['workclass'])
df['education']=le.fit_transform(df['education'])
df['marital.status']=le.fit_transform(df['marital.status'])
df['occupation']=le.fit_transform(df['occupation'])
df['relationship']=le.fit_transform(df['relationship'])
df['race']=le.fit_transform(df['race'])
df['sex']=le.fit_transform(df['sex'])
df['native.country']=le.fit_transform(df['native.country'])
df['income']=le.fit_transform(df['income'])


x=df.drop('income',axis=1)
y=df['income']

# feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(x,y)
features = fit.transform(x)
pca = PCA(n_components=3)
fit = pca.fit(x)
model = ExtraTreesClassifier()
model.fit(x,y )

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.3,random_state=2)
names = ['Logistic Regression ', "GradientBoostingClasifier", "RandomForestClassifier", "Decision_Tree_Classifier","SVC", "MLPClassifier","MultinomialClassifier"]
regressors = [
LogisticRegression(random_state=0),
GradientBoostingClassifier(n_estimators=10),
RandomForestClassifier(random_state=1),
DecisionTreeClassifier(random_state=0),
svm.SVC(),
MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
,MultinomialNB()]

scores = []
for name, clf in zip(names, regressors):
    clf.fit(X_Train,Y_Train)
    score = accuracy_score(Y_Test,clf.predict(X_Test))
    scores.append(score)
scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
print(scores_df.sort_values('accuracy', ascending= False))

