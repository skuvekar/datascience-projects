import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

rf= RandomForestClassifier(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
cv=CountVectorizer()

df = pd.read_json("C:/Users/CC-061/PycharmProjects/pythonProject/whats-cooking/train.json/train.json")
print(df['cuisine'].unique())

d_c = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

x=df['ingredients']
y=df['cuisine'].apply(d_c.index)

z=df['all_ingredients']=df['ingredients'].map(";".join) #here commas are considered to separate two different attributes or features in a list. But here the individual ingredients act as unique values so they need not be represented as a list element. Here, all commmas get replaced with semi colons to avoid this issue. Join function will join the ingredient values and it will be treated like a whole sentence. This helps in fuctions of Count Vectorizer.
print(z)


X = cv.fit_transform(df['all_ingredients'].values)

X_train,X_test,Y_train,Y_test = train_test_split(X,y, random_state=0, test_size=0.2)

dt.fit(X_train,Y_train)
rf.fit(X_train,Y_train)
nb.fit(X_train,Y_train)

y_pred1=dt.predict(X_test)
y_pred2=rf.predict(X_test)
y_pred3=nb.predict(X_test)

print("Decision Tree: ", accuracy_score(Y_test, y_pred1))
print("Random Forest: ", accuracy_score(Y_test, y_pred2))
print("Naive Bayes: ", accuracy_score(Y_test, y_pred3))


'''
output

Decision Tree:  0.6419861722187303
Random Forest:  0.7583909490886235
Naive Bayes:  0.7323695788812068
'''

