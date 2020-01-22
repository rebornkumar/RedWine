import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib




data = pd.read_csv("winequality-red.csv")
print data.head()
data = pd.read_csv("winequality-red.csv",sep=';')
data
print data.head()
data.describe()
print data.shape
Y=data.quality
X=data.drop('quality',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123,stratify=Y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
print X_train_scaled.mean(axis=0)
print "\n"
print X_train_scaled.std(axis=0)
X_test_scaled = scaler.transform(X_test)
print X_test_scaled.mean(axis=0)
print "\n"
print X_test_scaled.std(axis=0)
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, Y_train)
print clf.best_params_
print clf.refit
Y_pred = clf.predict(X_test)
print r2_score(Y_test, Y_pred)
print mean_squared_error(Y_test, Y_pred)
joblib.dump(clf, 'rf_regressor.pkl')
clf2 = joblib.load('rf_regressor.pkl')
clf2.predict(X_test)

